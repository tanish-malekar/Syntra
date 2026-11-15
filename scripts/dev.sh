#!/usr/bin/env bash
# Start the BloomBee backend for development: creates the venv if needed, seeds data, runs uvicorn with reload.
set -euo pipefail
cd "$(dirname "$0")/../backend"
PY="${PYTHON:-/opt/homebrew/bin/python3.12}"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --disable-pip-version-check -r requirements.txt
.venv/bin/python -m app.seed
exec .venv/bin/uvicorn app.main:app --reload --port "${PORT:-8000}"
