#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON_BIN=${PYTHON_BIN:-python3}

if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] Creating virtual environment in .venv"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[setup] ERROR: ffmpeg is not installed or not on PATH. Please install ffmpeg and rerun."
  exit 1
fi

MUSC_DIR="$PROJECT_ROOT/MUSC_violin"
if [ ! -d "$MUSC_DIR/.git" ]; then
  echo "[setup] Cloning MUSC violin transcription repo..."
  if ! git clone https://github.com/MTG/violin-transcription.git "$MUSC_DIR"; then
    echo "[setup] ERROR: Failed to clone MUSC repository. Check network access and rerun." >&2
    exit 1
  fi
else
  echo "[setup] MUSC violin repository already present. Pulling latest changes..."
  git -C "$MUSC_DIR" pull --ff-only || true
fi

if [ -f "$MUSC_DIR/requirements.txt" ]; then
  pip install -r "$MUSC_DIR/requirements.txt"
fi
if [ -f "$MUSC_DIR/requirements_dev.txt" ]; then
  pip install -r "$MUSC_DIR/requirements_dev.txt"
fi

SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
MUSC_PTH="$SITE_PACKAGES/musc_violin.pth"
echo "$MUSC_DIR" > "$MUSC_PTH"
echo "[setup] Wrote $MUSC_PTH so 'import musc' resolves into MUSC_violin"

echo "[setup] Backend dependencies installed. Activate the venv with 'source .venv/bin/activate'."
