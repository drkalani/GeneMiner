"""In-memory job status (replace with Redis/DB in large deployments)."""

from __future__ import annotations

from datetime import datetime, timezone
import threading
import uuid
from typing import Any, Dict, List, Optional

_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_progress(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def new_job_id() -> str:
    return str(uuid.uuid4())


def set_job(
    job_id: str,
    state: str,
    message: str = "",
    result: Optional[Dict] = None,
    project_id: Optional[str] = None,
    progress: Optional[float] = None,
) -> None:
    with _lock:
        current = _jobs.get(job_id, {})
        next_progress = _coerce_progress(progress)
        if next_progress is None:
            next_progress = current.get("progress")
        _jobs[job_id] = {
            "job_id": job_id,
            "project_id": project_id if project_id is not None else current.get("project_id"),
            "state": state,
            "message": message,
            "result": result,
            "created_at": current.get("created_at") or _timestamp(),
            "updated_at": _timestamp(),
            "progress": next_progress,
        }


def list_jobs(project_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with _lock:
        jobs = list(_jobs.values())
    if project_id:
        jobs = [j for j in jobs if j.get("project_id") == project_id]
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    if limit is not None and limit > 0:
        jobs = jobs[:limit]
    return jobs


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return _jobs.get(job_id)


def update_job(job_id: str, **kwargs: Any) -> None:
    with _lock:
        if job_id in _jobs:
            if "progress" in kwargs:
                kwargs["progress"] = _coerce_progress(
                    kwargs["progress"] if kwargs["progress"] is not None else None
                )
            kwargs.setdefault("updated_at", _timestamp())
            _jobs[job_id].update(kwargs)
