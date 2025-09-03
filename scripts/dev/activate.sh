#!/usr/bin/env bash
# Activate local venv and set deterministic env knobs
# Usage: source scripts/dev/activate.sh
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -d "$VENV_DIR" ]; then
  python3.11 -m venv "$VENV_DIR" || python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
. "$VENV_DIR/bin/activate"

# Determinism knobs
export PYTHONHASHSEED=0
export CLICOLOR_FORCE=1

# Prefer non-interactive operations
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

# Print summary
python - <<'PY'
import sys,sysconfig,platform
print(f"VENV: {sys.prefix}")
print(f"Python: {platform.python_version()}")
print("OK: environment activated")
PY

