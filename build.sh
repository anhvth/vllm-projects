#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

cd "$REPO_ROOT"

echo "=== Step 0: Remove old venv ==="
if [ -d "$VENV_DIR" ]; then
  rm -rf "$VENV_DIR"
fi

echo "=== Step 1: Create Python 3.12 venv ==="
uv venv --python 3.12 "$VENV_DIR" --seed
source "$VENV_DIR/bin/activate"

echo "=== Step 2: Install locked prebuilt vLLM environment ==="
uv sync --locked --active

echo ""
echo "=== Done! ==="
echo "Installed prebuilt vLLM and local hotload package from: $REPO_ROOT"
echo "Activate with: source $VENV_DIR/bin/activate"
