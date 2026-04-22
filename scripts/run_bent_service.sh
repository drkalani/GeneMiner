#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENT_VENV="${BENT_VENV:-$ROOT/.venv-bent}"
HOST="${BENT_SERVICE_HOST:-0.0.0.0}"
PORT="${BENT_SERVICE_PORT:-8010}"
RUN_SETUP="${RUN_SETUP:-0}"
BENT_VERSION="${BENT_VERSION:-0.0.80}"
BENT_SPACY_VERSION="${BENT_SPACY_VERSION:-3.7.2}"
BENT_SETUPTOOLS_VERSION="${BENT_SETUPTOOLS_VERSION:-65.5.0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_bent_service.sh

Optional env vars:
  BENT_VENV  Path to Bent runtime venv (default: <repo>/.venv-bent)
  RUN_SETUP  Set to 1 to run bent_setup before serving (default: 0)
  BENT_SERVICE_HOST  Bind host (default: 0.0.0.0)
  BENT_SERVICE_PORT  Bind port (default: 8010)
  BENT_VERSION  Bent package version (default: 0.0.80)
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

python_version="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if [[ "$python_version" != 3.10.* ]]; then
  echo "Bent runtime must use Python 3.10.x (found ${python_version} in ${BENT_VENV})."
  echo "Recreate .venv-bent with PY_BIN=python3.10."
  exit 1
fi

python_patch="$(echo "$python_version" | awk -F. '{print $3}')"
if [[ -z "$python_patch" || "$python_patch" -gt 13 ]]; then
  echo "Bent runtime supports Python 3.10.x up to 3.10.13 only (found ${python_version})."
  echo "Recreate .venv-bent with PY_BIN=python3.10 and delete the current virtualenv if needed."
  exit 1
fi

ensure_bent_package() {
  if "$python_bin" - <<PY
import importlib
from importlib import metadata

importlib.import_module("bent.annotate")
spacy = importlib.import_module("spacy")
importlib.import_module("setuptools")
expected = "${BENT_VERSION}"
installed = metadata.version("bent")
spacy_version = metadata.version("spacy")
setuptools_version = metadata.version("setuptools")
assert installed == expected
assert spacy_version == "${BENT_SPACY_VERSION}"
assert setuptools_version == "${BENT_SETUPTOOLS_VERSION}"
PY
  then
    return 0
  fi

  echo "Bent package is missing in ${BENT_VENV}; installing bent==${BENT_VERSION}."
  "$BENT_VENV/bin/pip" install --upgrade pip
  "$BENT_VENV/bin/pip" install "setuptools==${BENT_SETUPTOOLS_VERSION}" "spacy==${BENT_SPACY_VERSION}" "bent==${BENT_VERSION}"
}

ensure_bent_package

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
