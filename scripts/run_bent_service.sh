#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENT_VENV="${BENT_VENV:-$ROOT/.venv-bent}"
HOST="${BENT_SERVICE_HOST:-0.0.0.0}"
PORT="${BENT_SERVICE_PORT:-8010}"
RUN_SETUP="${RUN_SETUP:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_bent_service.sh

Optional env vars:
  BENT_VENV  Path to Bent runtime venv (default: <repo>/.venv-bent)
  RUN_SETUP  Set to 1 to run bent_setup before serving (default: 0)
  BENT_SERVICE_HOST  Bind host (default: 0.0.0.0)
  BENT_SERVICE_PORT  Bind port (default: 8010)
EOF
}

if [[ "${1-}" != "" ]]; then
  echo "Error: this script does not accept positional arguments."
  echo
  usage
  exit 1
fi

if [[ ! -d "$BENT_VENV" ]]; then
  echo "Bent venv not found: $BENT_VENV"
  echo "Run scripts/setup_bent_runtime.sh to create it first."
  exit 1
fi

python_bin="$BENT_VENV/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "No Python executable found in ${BENT_VENV}."
  exit 1
fi

if [[ "${RUN_SETUP}" == "1" ]]; then
  if "$python_bin" -m bent_setup >/dev/null 2>&1; then
    echo "Ran bent_setup in ${BENT_VENV}."
  else
    echo "Could not run bent_setup in ${BENT_VENV}; continue anyway."
  fi
fi

cat <<EOF
Starting Bent service from ${BENT_VENV}
URL: http://${HOST}:${PORT}/annotate
EOF

PYTHONPATH="$ROOT" "$python_bin" -m bent_service.main --host "$HOST" --port "$PORT"
