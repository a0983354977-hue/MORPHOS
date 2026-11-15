#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

if [ -x "$ROOT/MORPHOS_Studio" ]; then
  exec "$ROOT/MORPHOS_Studio"
fi
if [ -x "$ROOT/dist/MORPHOS_Studio" ]; then
  exec "$ROOT/dist/MORPHOS_Studio"
fi

if [ ! -f "$ROOT/studio/dist/index.html" ]; then
  echo "MORPHOS Studio build is missing." >&2
  echo "Build it first with: npm --prefix studio install && npm --prefix studio run build" >&2
  exit 1
fi

PYTHON=$(command -v python3 || command -v python || true)
if [ -z "$PYTHON" ]; then
  echo "Python 3 is required. Install dependencies with: python -m pip install -r requirements.txt" >&2
  exit 1
fi

exec "$PYTHON" "$ROOT/studio_launcher.py"
