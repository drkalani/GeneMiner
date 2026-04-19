"""Background training jobs."""

from __future__ import annotations

import json
import shutil
import threading
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from app.config import get_settings
from app.schemas.models import ArticleInput, KFoldTrainJobCreate, TrainJobCreate, TrainingConfig
from app.services import job_store
from app.services.project_service import project_dir

from geneminer_core.metrics import binary_classification_metrics, confusion_binary
from geneminer_core.relevance import predict_relevance, train_relevance_classifier


def _fix_articles_to_xy(articles: List[ArticleInput]) -> Tuple[List[str], List[int], List[str]]:
    texts: List[str] = []
    labels: List[int] = []
    pmids: List[str] = []
    for a in articles:
        if a.label is None:
            raise ValueError("Each article must have label (0 or 1) for training")
        texts.append(a.text)
        labels.append(int(a.label))
        pmids.append(str(a.pmid))
    return texts, labels, pmids


def run_single_train_job(
    job_id: str,
    project_id: str,
    body: TrainJobCreate,
) -> None:
    def _run() -> None:
        try:
            job_store.set_job(job_id, "running", "Preparing data…")
            texts, labels, _pmids = _fix_articles_to_xy(body.articles)
            cfg: TrainingConfig = body.config

            if len(set(labels)) < 2:
                raise ValueError("Need at least one sample per class (0 and 1) to train")

            strat = labels
            train_t, val_t, train_y, val_y = train_test_split(
                texts,
                labels,
                test_size=body.validation_split,
                random_state=cfg.seed,
                stratify=strat,
            )

            out_name = f"relevance_{job_id[:8]}"
            out_dir = project_dir(project_id) / "models" / out_name
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            job_store.set_job(job_id, "running", "Fine-tuning BioBERT…")
            _, _tok, eval_metrics = train_relevance_classifier(
                train_t,
                train_y,
                val_t,
                val_y,
                output_dir=out_dir,
                device_kind=cfg.processor,
                base_model=cfg.base_model,
                learning_rate=cfg.learning_rate,
                num_train_epochs=cfg.num_train_epochs,
                per_device_train_batch_size=cfg.per_device_train_batch_size,
                per_device_eval_batch_size=cfg.per_device_eval_batch_size,
                weight_decay=cfg.weight_decay,
                seed=cfg.seed,
                max_length=cfg.max_length,
                fp16=cfg.fp16,
            )

            preds, probs = predict_relevance(
                val_t,
                out_dir,
                device_kind=cfg.processor,
                batch_size=cfg.per_device_eval_batch_size,
                max_length=cfg.max_length,
            )
            pr = (
                np.stack([1 - probs, probs], axis=1)
                if probs is not None and len(probs) == len(val_y)
                else None
            )
            metrics = binary_classification_metrics(val_y, preds.tolist(), pr)

            cm = confusion_binary(val_y, preds.tolist())

            result = {
                "model_id": out_name,
                "path": str(out_dir.relative_to(get_settings().data_dir)),
                "eval_loss": float(eval_metrics.get("eval_loss", 0.0)),
                "metrics": metrics,
                "confusion": cm,
                "eval_sklearn": {
                    "accuracy": float(accuracy_score(val_y, preds)),
                    "precision": float(precision_score(val_y, preds, zero_division=0)),
                    "recall": float(recall_score(val_y, preds, zero_division=0)),
                    "f1": float(f1_score(val_y, preds, zero_division=0)),
                },
            }
            (out_dir / "validation_report.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            job_store.set_job(job_id, "completed", "Done", result=result)
        except Exception as exc:  # noqa: BLE001
            job_store.set_job(job_id, "failed", message=str(exc), result=None)

    threading.Thread(target=_run, daemon=True).start()


def run_kfold_train_job(
    job_id: str,
    project_id: str,
    body: KFoldTrainJobCreate,
) -> None:
    def _run() -> None:
        try:
            job_store.set_job(job_id, "running", "K-fold cross-validation…")
            texts, labels, _ = _fix_articles_to_xy(body.articles)
            cfg = body.config

            if len(set(labels)) < 2:
                raise ValueError("Need both classes in the dataset for stratified k-fold")

            skf = StratifiedKFold(
                n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed
            )
            fold_metrics: List[Dict[str, Any]] = []
            out_root = project_dir(project_id) / "models" / f"kfold_{job_id[:8]}"
            out_root.mkdir(parents=True, exist_ok=True)

            for fold_idx, (train_idx, val_idx) in enumerate(
                skf.split(texts, labels)
            ):
                job_store.update_job(
                    job_id,
                    message=f"Training fold {fold_idx + 1}/{cfg.n_splits}…",
                )
                train_t = [texts[i] for i in train_idx]
                train_y = [labels[i] for i in train_idx]
                val_t = [texts[i] for i in val_idx]
                val_y = [labels[i] for i in val_idx]

                fold_dir = out_root / f"fold_{fold_idx}"
                if fold_dir.exists():
                    shutil.rmtree(fold_dir)
                fold_dir.mkdir(parents=True)

                _, _, eval_metrics = train_relevance_classifier(
                    train_t,
                    train_y,
                    val_t,
                    val_y,
                    output_dir=fold_dir,
                    device_kind=cfg.processor,
                    base_model=cfg.base_model,
                    learning_rate=cfg.learning_rate,
                    num_train_epochs=cfg.num_train_epochs,
                    per_device_train_batch_size=cfg.per_device_train_batch_size,
                    per_device_eval_batch_size=cfg.per_device_eval_batch_size,
                    weight_decay=cfg.weight_decay,
                    seed=cfg.seed + fold_idx,
                    max_length=cfg.max_length,
                    fp16=cfg.fp16,
                )
                preds, probs = predict_relevance(
                    val_t,
                    fold_dir,
                    device_kind=cfg.processor,
                    batch_size=cfg.per_device_eval_batch_size,
                    max_length=cfg.max_length,
                )
                m = binary_classification_metrics(val_y, preds.tolist(), None)
                fold_metrics.append(
                    {
                        "fold": fold_idx,
                        "eval_loss": float(eval_metrics.get("eval_loss", 0.0)),
                        "metrics": m,
                        "confusion": confusion_binary(val_y, preds.tolist()),
                    }
                )

            # aggregate
            f1s = [f["metrics"].get("f1", 0.0) for f in fold_metrics]
            accs = [f["metrics"].get("accuracy", 0.0) for f in fold_metrics]
            summary = {
                "model_bundle": out_root.name,
                "path": str(out_root.relative_to(get_settings().data_dir)),
                "n_splits": cfg.n_splits,
                "folds": fold_metrics,
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s)),
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
            }
            (out_root / "kfold_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            job_store.set_job(job_id, "completed", "K-fold complete", result=summary)
        except Exception as exc:  # noqa: BLE001
            job_store.set_job(job_id, "failed", message=str(exc), result=None)

    threading.Thread(target=_run, daemon=True).start()
