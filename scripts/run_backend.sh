#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
export PYTHONPATH="$ROOT:$ROOT/backend:${PYTHONPATH:-}"

PY_BIN="python3.10"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "Python 3.10 executable not found. Install Python 3.10 and ensure 'python3.10' is on PATH."
  exit 1
fi

PY_MINOR_VERSION="$(
${PY_BIN} - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [[ "$PY_MINOR_VERSION" != "3.10" ]]; then
  echo "Project runtime is pinned to Python 3.10.x for Bent compatibility."
  echo "Current interpreter: $PY_MINOR_VERSION"
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF token is not set (HF_TOKEN). Model downloads will use unauthenticated access."
else
  echo "HF token detected. Requests to Hugging Face will use authenticated access."
fi
exec ${PY_BIN} -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
