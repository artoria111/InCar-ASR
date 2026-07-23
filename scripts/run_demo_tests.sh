#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${INCAR_ASR_VENV:-$REPOSITORY_ROOT/.venv-demo}"
PYTHON_BIN="${PYTHON:-python3}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPOSITORY_ROOT/.cache/pip}"

cd "$REPOSITORY_ROOT"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check --quiet \
    -r requirements-demo.txt

"$VENV_DIR/bin/python" -m unittest discover -s tests -v
