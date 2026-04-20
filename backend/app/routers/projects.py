"""Projects CRUD."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from app.schemas.models import (
    ProjectCreate,
    ProjectModelCatalog,
    ProjectOut,
)
from app.services import project_service

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=List[ProjectOut])
def list_projects() -> List[ProjectOut]:
    raw = project_service.list_projects()
    return [ProjectOut(**r) for r in raw]


@router.post("", response_model=ProjectOut)
def create_project(body: ProjectCreate) -> ProjectOut:
    meta = project_service.ensure_project(
        body.name, body.disease_key, body.description
    )
    return ProjectOut(**meta)


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: str) -> ProjectOut:
    meta = project_service.get_project(project_id)
    if not meta:
        raise HTTPException(404, "Project not found")
    return ProjectOut(**meta)


@router.delete("/{project_id}")
def delete_project(project_id: str) -> dict:
    if not project_service.delete_project(project_id):
        raise HTTPException(404, "Project not found")
    return {"ok": True}


@router.get("/{project_id}/models")
def list_models(project_id: str) -> dict:
    if not project_service.get_project(project_id):
        raise HTTPException(404, "Project not found")
    return {"models": project_service.list_models(project_id)}


@router.get("/{project_id}/models/catalog", response_model=ProjectModelCatalog)
def list_model_catalog(project_id: str) -> ProjectModelCatalog:
    if not project_service.get_project(project_id):
        raise HTTPException(404, "Project not found")
    return ProjectModelCatalog(models=project_service.list_model_catalog(project_id))
