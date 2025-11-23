#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "[run] Missing .venv. Run scripts/setup_backend.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

UVICORN_ARGS=${UVICORN_ARGS:-"backend_server:app --host 0.0.0.0 --port 8000"}
exec uvicorn $UVICORN_ARGS
