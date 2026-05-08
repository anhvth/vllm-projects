#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

cd "$REPO_ROOT"

echo "=== Creating venv ==="
uv venv --python 3.12 --seed --managed-python "$VENV_DIR" --clear

source "$VENV_DIR/bin/activate"

echo "=== Installing vLLM + Ray ==="
 # vLLM 0.20.x wheels currently require libcudart.so.13, but this workspace
 # runs against the CUDA 12.9 runtime exposed by the host and Torch build.
uv pip install 'vllm==0.19.1' --torch-backend=auto 'ray[llm,serve]==2.55.1' boto3
# uv pip install 

echo "=== Freezing lock ==="
uv pip freeze > "$REPO_ROOT/requirements_lock.txt"

echo ""
echo "=== Done! ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Lock saved to: requirements_lock.txt"
echo "To install from lock (skipping resolution): uv pip sync requirements_lock.txt"
