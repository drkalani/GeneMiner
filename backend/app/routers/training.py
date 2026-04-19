"""Fine-tuning and k-fold training."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.models import JobStatus, KFoldTrainJobCreate, TrainJobCreate
from app.services import job_store
from app.services import project_service
from app.services.training_runner import run_kfold_train_job, run_single_train_job

router = APIRouter(prefix="/train", tags=["training"])


@router.post("/{project_id}/relevance", response_model=JobStatus)
def train_relevance(project_id: str, body: TrainJobCreate) -> JobStatus:
    if not project_service.get_project(project_id):
        raise HTTPException(404, "Project not found")
    job_id = job_store.new_job_id()
    job_store.set_job(job_id, "queued", "Single train job queued")
    run_single_train_job(job_id, project_id, body)
    return JobStatus(job_id=job_id, state="queued", message="Started in background")


@router.post("/{project_id}/relevance/kfold", response_model=JobStatus)
def train_relevance_kfold(project_id: str, body: KFoldTrainJobCreate) -> JobStatus:
    if not project_service.get_project(project_id):
        raise HTTPException(404, "Project not found")
    if len(body.articles) < body.config.n_splits * 2:
        raise HTTPException(
            400,
            f"Need at least {body.config.n_splits * 2} articles for stratified k-fold",
        )
    job_id = job_store.new_job_id()
    job_store.set_job(job_id, "queued", "K-fold job queued")
    run_kfold_train_job(job_id, project_id, body)
    return JobStatus(job_id=job_id, state="queued", message="K-fold started in background")


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str) -> JobStatus:
    j = job_store.get_job(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return JobStatus(
        job_id=j["job_id"],
        state=j["state"],
        message=j.get("message", ""),
        result=j.get("result"),
    )
