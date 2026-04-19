"""In-memory job status (replace with Redis/DB in large deployments)."""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, Optional

_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def new_job_id() -> str:
    return str(uuid.uuid4())


def set_job(job_id: str, state: str, message: str = "", result: Optional[Dict] = None) -> None:
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "state": state,
            "message": message,
            "result": result,
        }


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return _jobs.get(job_id)


def update_job(job_id: str, **kwargs: Any) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)
