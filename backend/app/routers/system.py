"""Device and health endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Set

import torch
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fastapi import APIRouter, HTTPException
from fastapi import Query

from app.schemas.models import BaseModelDownloadRequest, DeviceInfo
from app.schemas.models import ModelCompatibilityResult, ModelTaskKind
from app.services.model_validation import as_result


class _BaseModelDownloadResponse(BaseModel):
    model_id: str
    downloaded: bool
    status: str
    message: str


def _cache_roots() -> Set[Path]:
    roots: Set[Path] = set()

    for candidate in (
        os.environ.get("HF_HOME"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.environ.get("HF_HUB_CACHE"),
    ):
        if not candidate:
            continue
        p = Path(candidate)
        roots.add(p if p.name == "hub" else p / "hub")

    roots.add(Path.home() / ".cache" / "huggingface" / "hub")

    return {r for r in roots if r}


def _repo_from_hf_cache_dir(name: str) -> str | None:
    if not name.startswith("models--"):
        return None
    return name.removeprefix("models--").replace("--", "/")


def _is_downloaded_model_dir(path: Path) -> bool:
    snapshots = path / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(child.is_dir() for child in snapshots.iterdir())


def list_downloaded_hf_models() -> list[str]:
    model_ids: Set[str] = set()
    for root in _cache_roots():
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            model_id = _repo_from_hf_cache_dir(child.name)
            if not model_id:
                continue
            if not _is_downloaded_model_dir(child):
                continue
            model_ids.add(model_id)
    return sorted(model_ids, key=lambda value: value.lower())


def _ensure_model_downloaded(model_id: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    del tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    del model


router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/devices", response_model=DeviceInfo)
def devices() -> DeviceInfo:
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
    if cuda:
        rec = "cuda"
    elif mps:
        rec = "mps"
    else:
        rec = "cpu"
    return DeviceInfo(
        available={"cuda": cuda, "mps": mps, "cpu": True},
        recommended=rec,
    )


@router.get("/models/base")
def list_base_models() -> dict:
    return {"models": list_downloaded_hf_models()}


@router.post("/models/base/download", response_model=_BaseModelDownloadResponse)
def download_base_model(payload: BaseModelDownloadRequest) -> _BaseModelDownloadResponse:
    model_id = payload.model_id.strip()
    if not model_id:
        return _BaseModelDownloadResponse(
            model_id=model_id,
            downloaded=False,
            status="error",
            message="Model id is required",
        )

    try:
        _ensure_model_downloaded(model_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    already = model_id in list_downloaded_hf_models()
    status = "already_downloaded" if already else "downloaded"
    message = (
        f"Model already cached locally: {model_id}"
        if already
        else f"Model cached locally: {model_id}"
    )

    return _BaseModelDownloadResponse(
        model_id=model_id,
        downloaded=True,
        status=status,
        message=message,
    )


@router.get("/models/validate", response_model=ModelCompatibilityResult)
def validate_model(
    model_id: str,
    expected_task: ModelTaskKind = Query(
        ...,
        description="Expected task for compatibility check",
    ),
) -> ModelCompatibilityResult:
    if not model_id.strip():
        raise HTTPException(400, "Model id is required")
    return as_result(model_id, expected_task)
