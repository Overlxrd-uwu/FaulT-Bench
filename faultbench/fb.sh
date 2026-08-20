#!/bin/sh
# faultbench launcher: finds the sibling SADE venv's Python, sets PYTHONUTF8,
# and forwards all arguments to `python -m faultbench`.
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$DIR/../SADE-NetworkAgent/.venv/bin/python"
[ -x "$PY" ] || PY="$DIR/../SADE-NetworkAgent/.venv/Scripts/python.exe"
if [ ! -x "$PY" ]; then
  echo "SADE venv not found under $DIR/../SADE-NetworkAgent/.venv -- run the Setup steps in README.md first." >&2
  exit 1
fi
PYTHONUTF8=1 exec "$PY" -m faultbench "$@"
