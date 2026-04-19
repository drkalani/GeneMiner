"""Project filesystem layout."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import get_settings


def projects_root() -> Path:
    return get_settings().data_dir / "projects"


def ensure_project(name: str, disease_key: str, description: str) -> Dict[str, Any]:
    projects_root().mkdir(parents=True, exist_ok=True)
    pid = str(uuid.uuid4())
    pdir = projects_root() / pid
    pdir.mkdir(parents=True, exist_ok=False)
    meta = {
        "id": pid,
        "name": name,
        "disease_key": disease_key,
        "description": description,
    }
    (pdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (pdir / "models").mkdir(exist_ok=True)
    (pdir / "uploads").mkdir(exist_ok=True)
    (pdir / "outputs").mkdir(exist_ok=True)
    return meta


def list_projects() -> List[Dict[str, Any]]:
    out = []
    root = projects_root()
    if not root.exists():
        return out
    for sub in root.iterdir():
        if sub.is_dir() and (sub / "meta.json").exists():
            meta = json.loads((sub / "meta.json").read_text(encoding="utf-8"))
            out.append(meta)
    return sorted(out, key=lambda x: x.get("name", ""))


def get_project(project_id: str) -> Optional[Dict[str, Any]]:
    p = projects_root() / project_id / "meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def project_dir(project_id: str) -> Path:
    return projects_root() / project_id


def list_models(project_id: str) -> List[str]:
    mdir = project_dir(project_id) / "models"
    if not mdir.exists():
        return []
    return sorted([d.name for d in mdir.iterdir() if d.is_dir()])


def delete_project(project_id: str) -> bool:
    p = project_dir(project_id)
    if not p.exists():
        return False
    shutil.rmtree(p)
    return True
