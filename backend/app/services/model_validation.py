"""Model task compatibility checks for HF models."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from app.schemas.models import ModelCompatibilityResult, ModelTaskKind


_TASK_NAME_BY_KEY = {
    "classification": "text-classification",
    "token_classification": "token-classification",
}


def _human_task_name(task: str) -> str:
    return _TASK_NAME_BY_KEY.get(task, task)


def _normalize_model_source(model_id: str) -> str:
    return (model_id or "").strip()


def _architectures_from_config(config: AutoConfig) -> List[str]:
    architectures = config.architectures or []
    return [str(a) for a in architectures]


def _detect_tasks_from_architectures(config: AutoConfig) -> List[str]:
    detected = OrderedDict[str, None]()
    for arch in _architectures_from_config(config):
        low = arch.lower()
        if "forsequenceclassification" in low or "sequenceclassification" in low:
            detected["classification"] = None
        if "fortokenclassification" in low or "tokenclassification" in low:
            detected["token_classification"] = None
    return list(detected.keys())


def _can_load(model_id: str, task: ModelTaskKind) -> bool:
    try:
        if task == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
        else:
            model = AutoModelForTokenClassification.from_pretrained(model_id)
        del model
        return True
    except Exception:
        return False


def detect_model_tasks(model_id: str) -> List[str]:
    """Detect common tasks from config and fallback probing when architecture metadata is absent."""
    normalized = _normalize_model_source(model_id)
    if not normalized:
        return []

    detected: List[str] = []
    try:
        cfg = AutoConfig.from_pretrained(normalized)
        detected = _detect_tasks_from_architectures(cfg)
    except Exception:
        pass

    if not detected:
        if _can_load(normalized, "classification"):
            detected.append("classification")
        if _can_load(normalized, "token_classification"):
            detected.append("token_classification")

    if not detected:
        return []

    # Keep deterministic ordering and uniqueness
    out: List[str] = []
    for task in ("classification", "token_classification"):
        if task in detected and task not in out:
            out.append(task)
    for extra in detected:
        if extra not in out:
            out.append(extra)
    return out


def validate_model_task(model_id: str, expected_task: ModelTaskKind) -> Dict[str, object]:
    model_id = _normalize_model_source(model_id)
    if not model_id:
        return {
            "model_id": model_id,
            "expected_task": expected_task,
            "compatible": False,
            "detected_tasks": [],
            "message": "Model id is required.",
        }

    detected = detect_model_tasks(model_id)
    if not detected:
        return {
            "model_id": model_id,
            "expected_task": expected_task,
            "compatible": False,
            "detected_tasks": [],
            "message": (
                f"Could not detect a valid HF task profile for '{model_id}'. "
                "If this is a custom checkpoint, ensure it contains a valid config and task head."
            ),
        }

    compatible = expected_task in detected
    detected_labels = [_human_task_name(task) for task in detected]
    if compatible:
        message = (
            f"Compatible. '{model_id}' exposes {', '.join(detected_labels)} "
            f"and can be used for {_human_task_name(expected_task)}."
        )
    else:
        message = (
            f"Incompatible model type. '{model_id}' exposes {', '.join(detected_labels)} "
            f"but this step requires {_human_task_name(expected_task)}."
        )

    return {
        "model_id": model_id,
        "expected_task": expected_task,
        "compatible": compatible,
        "detected_tasks": detected_labels,
        "message": message,
    }


def as_result(model_id: str, expected_task: ModelTaskKind) -> ModelCompatibilityResult:
    raw = validate_model_task(model_id, expected_task)
    return ModelCompatibilityResult(**raw)
