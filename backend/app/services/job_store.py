"""In-memory job status with lightweight file persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import threading
import uuid
from typing import Any, Dict, List, Optional

from app.config import get_settings

_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_MAX_JOBS = 200


def _jobs_file_path() -> Path:
    jobs_dir = get_settings().data_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir / "jobs.json"


def _coerce_state(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"queued", "running", "completed", "failed"}:
        return normalized
    return "running"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    progress = _coerce_progress(record.get("progress"))
    state = _coerce_state(str(record.get("state", "queued")))
    created_at = record.get("created_at") or _timestamp()
    updated_at = record.get("updated_at") or created_at
    return {
        "job_id": str(record.get("job_id", "")),
        "project_id": record.get("project_id"),
        "state": state,
        "message": str(record.get("message", "")),
        "result": record.get("result"),
        "created_at": str(created_at),
        "updated_at": str(updated_at),
        "progress": progress,
    }


def _persist_jobs() -> None:
    path = _jobs_file_path()
    payload = {job_id: _normalize_record(job_state) for job_id, job_state in _jobs.items()}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _trim_jobs() -> None:
    if len(_jobs) <= _MAX_JOBS:
        return
    ordered = sorted(
        _jobs.items(),
        key=lambda pair: str(pair[1].get("updated_at", "")),
        reverse=True,
    )
    for removed_job_id, _ in ordered[_MAX_JOBS:]:
        _jobs.pop(removed_job_id, None)


def _load_jobs() -> None:
    path = _jobs_file_path()
    if not path.exists():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(raw, dict):
        return
    for job_id, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        normalized = _normalize_record({"job_id": job_id, **payload})
        if normalized["state"] in {"running", "queued", "pending", "starting", "started", "in_progress"}:
            normalized["state"] = "failed"
            normalized["message"] = "Server restarted before this job completed."
            normalized["updated_at"] = _timestamp()
        _jobs[job_id] = normalized
    _trim_jobs()


def _coerce_progress(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


_load_jobs()


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
            "state": _coerce_state(state),
            "message": message,
            "result": result,
            "created_at": current.get("created_at") or _timestamp(),
            "updated_at": _timestamp(),
            "progress": next_progress,
        }
        _trim_jobs()
        _persist_jobs()


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
            if "state" in kwargs:
                kwargs["state"] = _coerce_state(kwargs["state"])
            if "progress" in kwargs:
                kwargs["progress"] = _coerce_progress(
                    kwargs["progress"] if kwargs["progress"] is not None else None
                )
            kwargs.setdefault("updated_at", _timestamp())
            _jobs[job_id].update(kwargs)
            _trim_jobs()
            _persist_jobs()
