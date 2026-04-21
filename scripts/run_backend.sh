#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
export PYTHONPATH="$ROOT:$ROOT/backend:${PYTHONPATH:-}"

ENV_FILE="${ENV_FILE:-}"
if [[ -z "${ENV_FILE}" ]]; then
  if [[ -f "$ROOT/.env" ]]; then
    ENV_FILE="$ROOT/.env"
  elif [[ -f "$ROOT/backend/.env" ]]; then
    ENV_FILE="$ROOT/backend/.env"
  fi
fi

if [[ -n "${ENV_FILE}" && -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$ENV_FILE"
  set +a
  echo "Loaded environment variables from $ENV_FILE."
else
  echo "No .env file found in ${ROOT} or ${ROOT}/backend. Using defaults and shell environment."
fi

# Default optional values to keep local startup predictable.
: "${HF_TOKEN:=}"
: "${HF_HOME:=${HOME}/.cache/huggingface}"
: "${HF_HUB_CACHE:=${HF_HOME}/hub}"
: "${HUGGINGFACE_HUB_CACHE:=${HF_HOME}/hub}"
: "${BENT_SERVICE_URL:=}"
: "${BENT_SERVICE_TIMEOUT_SECONDS:=30}"

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
