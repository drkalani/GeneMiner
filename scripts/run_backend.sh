#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
export PYTHONPATH="$ROOT:$ROOT/backend:${PYTHONPATH:-}"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF token is not set (HF_TOKEN). Model downloads will use unauthenticated access."
else
  echo "HF token detected. Requests to Hugging Face will use authenticated access."
fi
exec uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
