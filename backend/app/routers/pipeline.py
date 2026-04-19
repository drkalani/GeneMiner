"""Pipeline execution (per-step or full)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.models import PipelineRunRequest
from app.services.pipeline_runner import execute_pipeline
from app.services import project_service

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/run")
def run_pipeline(body: PipelineRunRequest) -> dict:
    if not project_service.get_project(body.project_id):
        raise HTTPException(404, "Project not found")
    try:
        return execute_pipeline(body)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
