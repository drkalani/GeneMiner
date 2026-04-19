"""Device and health endpoints."""

from __future__ import annotations

import torch

from fastapi import APIRouter

from app.schemas.models import DeviceInfo

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
