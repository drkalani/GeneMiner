#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
export PYTHONPATH="$ROOT:$ROOT/backend:${PYTHONPATH:-}"

PY_BIN="${PY_BIN:-python3.13}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "Python 3.13 executable not found. Install Python 3.13.7+ and ensure 'python3.13' is on PATH."
  exit 1
fi

PY_VERSION="$(
${PY_BIN} - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
)"

read -r PY_MAJOR PY_MINOR PY_PATCH <<<"${PY_VERSION//./ }"
if [[ "$PY_MAJOR" != "3" || "$PY_MINOR" != "13" ]]; then
  echo "Backend runtime is now using Python 3.13.7+ (Bent remains a separate service)."
  echo "Current interpreter: ${PY_VERSION}"
  exit 1
fi

if (( PY_PATCH < 7 )); then
  echo "Backend runtime needs Python 3.13.7+."
  echo "Current interpreter: ${PY_VERSION}"
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF token is not set (HF_TOKEN). Model downloads will use unauthenticated access."
else
  echo "HF token detected. Requests to Hugging Face will use authenticated access."
fi
exec ${PY_BIN} -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
