#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
VLLM_PATCH_DIR="$REPO_ROOT/vllm_patch"

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

echo "=== Step 3: Enable local vLLM hotpatch overlay ==="
SITE_PACKAGES="$("$VENV_DIR/bin/python" - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)"
cat > "$SITE_PACKAGES/vllm_hotpatch.pth" <<EOF
import sys; p = ${VLLM_PATCH_DIR@Q}; sys.path.remove(p) if p in sys.path else None; sys.path.insert(0, p)
EOF

echo ""
echo "=== Done! ==="
echo "Installed prebuilt vLLM and enabled local overlay at: $VLLM_PATCH_DIR/vllm"
echo "Activate with: source $VENV_DIR/bin/activate"
