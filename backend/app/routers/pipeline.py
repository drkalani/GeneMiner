"""Pipeline execution (per-step or full)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.models import PipelineRunRequest
from app.services.pipeline_runner import execute_pipeline
from app.services import project_service
from app.services.model_validation import validate_model_task
from app.services.project_service import project_dir

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/run")
def run_pipeline(body: PipelineRunRequest) -> dict:
    if not project_service.get_project(body.project_id):
        raise HTTPException(404, "Project not found")
    if body.mode in {"classify", "full"}:
        model_path = str(project_dir(body.project_id) / "models" / body.model_id)
        result = validate_model_task(model_path, "classification")
        if not result["compatible"]:
            raise HTTPException(400, result["message"])

    if body.mode in {"ner", "full"}:
        ner_result = validate_model_task(body.ner_model, "token_classification")
        if not ner_result["compatible"]:
            raise HTTPException(400, ner_result["message"])

    try:
        return execute_pipeline(body)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
