#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
WHEELS_DIR="$REPO_ROOT/wheels"

cd "$REPO_ROOT"

echo "=== Step 0: Remove old venv ==="
if [ -d "$VENV_DIR" ]; then
  rm -rf "$VENV_DIR"
fi

echo "=== Step 1: Create Python 3.12 venv ==="
uv venv --python 3.12 "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "=== Step 2: Install all wheels from $WHEELS_DIR ==="
# Install directly from .whl file paths — uv doesn't need to resolve by name
uv pip install --python "$VENV_DIR/bin/python" \
  --no-index \
  "$WHEELS_DIR"/typing_extensions-*.whl \
  "$WHEELS_DIR"/jinja2-*.whl \
  "$WHEELS_DIR"/markupsafe-*.whl \
  "$WHEELS_DIR"/cuda_toolkit-*.whl \
  "$WHEELS_DIR"/nvidia_*.whl \
  "$WHEELS_DIR"/torch-*.whl \
  "$WHEELS_DIR"/torchaudio-*.whl \
  "$WHEELS_DIR"/torchvision-*.whl \
  "$WHEELS_DIR"/triton-*.whl

# vllm has many transitive deps from pypi — install it with --find-links
uv pip install --python "$VENV_DIR/bin/python" \
  --no-index \
  --find-links "$WHEELS_DIR" \
  "$WHEELS_DIR"/vllm-0.20*.whl

# ray[serve] from PyPI (still accessible directly)
uv pip install --python "$VENV_DIR/bin/python" \
  --default-index https://pypi.org/simple \
  "ray[serve]==2.55.1"

echo ""
echo "=== Step 3: Install local hotload package ==="
uv pip install --python "$VENV_DIR/bin/python" -e .

echo ""
echo "=== Done! ==="
echo "Activate with: source $VENV_DIR/bin/activate"
