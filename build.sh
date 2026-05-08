#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

cd "$REPO_ROOT"

echo "=== Creating venv ==="
uv venv --python 3.12 --seed --managed-python "$VENV_DIR" --clear

source "$VENV_DIR/bin/activate"

echo "=== Installing vLLM + Ray ==="
uv pip install vllm --torch-backend=auto
uv pip install 'ray[llm]' 'ray[serve]' boto3

echo "=== Freezing lock ==="
uv pip freeze > "$REPO_ROOT/requirements_lock.txt"

echo ""
echo "=== Done! ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Lock saved to: requirements_lock.txt"
echo "To install from lock (skipping resolution): uv pip sync requirements_lock.txt"
