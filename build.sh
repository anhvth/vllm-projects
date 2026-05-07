#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VLLM_SRC="$REPO_ROOT/vllm"
VENV_DIR="$REPO_ROOT/.venv"

echo "=== Step 0: Remove old venv ==="
rm -rf "$VENV_DIR"

echo "=== Step 1: Create Python 3.12 venv ==="
uv venv --python 3.12 "$VENV_DIR" --seed
source "$VENV_DIR/bin/activate"

echo "=== Step 2: Install torch with CUDA 12.9 ==="
uv pip install torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
  --index-url https://download.pytorch.org/whl/cu129

echo "=== Step 3: Install editable metadata dependencies ==="
uv pip install \
  "cmake>=3.26.1" \
  ninja \
  "packaging>=24.2" \
  "setuptools>=77.0.3,<81.0.0" \
  "setuptools-scm>=8.0" \
  wheel \
  jinja2

echo "=== Step 4: Install vLLM with precompiled binaries ==="
VLLM_USE_PRECOMPILED=1 uv pip install -e "$VLLM_SRC" --torch-backend=auto --no-build-isolation

echo ""
echo "=== Done! ==="
echo "Activate with: source $VENV_DIR/bin/activate"
