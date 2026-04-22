"""Pipeline execution (per-step or full)."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException
from importlib import import_module

from app.config import get_settings
from app.schemas.models import PipelineRunRequest
from app.services.pipeline_runner import execute_pipeline
from app.services import project_service
from app.services.model_validation import validate_model_task
from app.services.project_service import project_dir

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _check_bent_service(service_url: str) -> None:
    endpoint = service_url.rstrip("/")
    if endpoint.endswith("/annotate"):
        endpoint = endpoint[: -len("/annotate")]
    health_url = f"{endpoint}/health"

    request = Request(health_url)
    try:
        with urlopen(request, timeout=6) as response:
            response_body = response.read().decode("utf-8", errors="ignore")
            if response.status >= 400:
                raise RuntimeError(f"Bent service returned status {response.status}: {response_body}")
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Bent service request failed: HTTP {exc.code}: {response_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Bent service request failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Bent service request failed: {exc}") from exc

    try:
        parsed = json.loads(response_body or "{}")
    except ValueError as exc:
        raise RuntimeError(f"Bent service response was not valid JSON: {response_body}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Bent service response has unexpected format: {response_body}")
    if parsed.get("status") != "ok":
        raise RuntimeError(f"Bent service health check reported unhealthy status: {response_body}")


def _ensure_bent_available() -> None:
    service_url = get_settings().bent_service_url.strip()
    if service_url:
        try:
            _check_bent_service(service_url)
        except RuntimeError as exc:
            message = str(exc)
            if (
                "Bent method requires `bent` package" in message
                or "Bent execution failed" in message
            ):
                raise HTTPException(
                    400,
                    "Bent method is selected but the configured Bent service does not have the required runtime. "
                    + message,
                ) from exc
            raise HTTPException(
                503,
                f"Bent service at {service_url} is unavailable or returned an error: {message}",
            ) from exc
        return
    try:
        bt = import_module("bent.annotate")
    except ModuleNotFoundError as exc:
        raise HTTPException(
            400,
            "Bent method is selected but `bent` is not installed in the running backend environment. "
            "Install with `pip install bent==0.0.80` (Python 3.10.x, <=3.10.13) and run from that runtime, "
            "or set `BENT_SERVICE_URL` / run a Bent service for service mode.",
        ) from exc
    except Exception as exc:
        raise HTTPException(500, f"Failed to import bent annotator: {exc}") from exc
    if not hasattr(bt, "annotate"):
        raise HTTPException(500, "Bent package does not expose annotate() entry point.")


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
        if body.ner_method == "transformers":
            ner_result = validate_model_task(body.ner_model, "token_classification")
            if not ner_result["compatible"]:
                raise HTTPException(400, ner_result["message"])
        else:
            _ensure_bent_available()

    try:
        return execute_pipeline(body)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(500, str(e)) from e
