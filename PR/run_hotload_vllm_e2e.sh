#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_PATCH_DIR="$BASE_DIR/vllm_patch"
VENV_DIR="$BASE_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
LOG_DIR="$BASE_DIR/PR/logs"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="http://$HOST:$PORT"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-1.7b}"
CHAT_MODEL_PATH="${CHAT_MODEL_PATH:-$HOME/ckpt/hf_models/Qwen/Qwen3-1.7B}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$HOME/ckpt/hf_models/Qwen/Qwen3-1.7B-Base}"
PROMPT="${PROMPT:-Explain SVD using the smallest possible 2D example. Keep it short.}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
SERVER_LOG="$LOG_DIR/hotload_vllm_server.log"
RESULTS_DIR="$LOG_DIR/hotload_vllm_results"
KEEP_SERVER="${KEEP_SERVER:-0}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
READY_TIMEOUT_SECS="${READY_TIMEOUT_SECS:-600}"

SERVER_PID=""

require_path() {
    local path="$1"
    local label="$2"
    if [[ ! -e "$path" ]]; then
        echo "Missing $label: $path" >&2
        exit 1
    fi
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Missing required command: $cmd" >&2
        exit 1
    fi
}

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        if [[ "$KEEP_SERVER" == "1" ]]; then
            echo "Leaving server running on PID $SERVER_PID because KEEP_SERVER=1"
            return
        fi
        echo "Stopping server PID $SERVER_PID"
        kill "$SERVER_PID" >/dev/null 2>&1 || true
        wait "$SERVER_PID" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT

http_get() {
    local path="$1"
    curl --fail --silent --show-error \
        --max-time "$REQUEST_TIMEOUT" \
        "$BASE_URL$path"
}

http_post_json() {
    local path="$1"
    local payload="$2"
    curl --fail --silent --show-error \
        --max-time "$REQUEST_TIMEOUT" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$BASE_URL$path"
}

chat_completion() {
    local output_file="$1"
    http_post_json "/v1/chat/completions" "$(cat <<JSON
{
  "model": "$SERVED_MODEL_NAME",
  "messages": [
    {
      "role": "user",
      "content": "$PROMPT"
    }
  ],
  "temperature": 0,
  "max_tokens": 256,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
JSON
)" | tee "$output_file"
}

wait_for_server() {
    local deadline=$((SECONDS + READY_TIMEOUT_SECS))
    until (( SECONDS >= deadline )); do
        if curl --silent --fail "$BASE_URL/managed/status" >/dev/null 2>&1; then
            return 0
        fi
        if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
            echo "Server exited before becoming ready. Recent log tail:" >&2
            tail -n 80 "$SERVER_LOG" >&2 || true
            exit 1
        fi
        sleep 2
    done

    echo "Timed out waiting for $BASE_URL/managed/status" >&2
    tail -n 80 "$SERVER_LOG" >&2 || true
    exit 1
}

start_server() {
    mkdir -p "$LOG_DIR" "$RESULTS_DIR"
    : > "$SERVER_LOG"

    echo "Starting dummy-weight vLLM server"
    (
        cd "$BASE_DIR"
        export PYTHONPATH="$VLLM_PATCH_DIR"
        unset VLLM_API_KEY
        export VLLM_SERVER_DEV_MODE=1
        export VLLM_ALLOW_INSECURE_SERIALIZATION=1
        exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
            "$CHAT_MODEL_PATH" \
            --host "$HOST" \
            --port "$PORT" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --trust-remote-code \
            --dtype bfloat16 \
            --load-format dummy \
            --weight-transfer-config '{"backend":"ipc"}' \
            --enable-sleep-mode \
            --managed-weight-sync \
            --tensor-parallel-size "$TP_SIZE" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-model-len "$MAX_MODEL_LEN" \
            >"$SERVER_LOG" 2>&1
    ) &
    SERVER_PID="$!"
    echo "Server PID: $SERVER_PID"
    wait_for_server
}

run_endpoint_smoke() {
    echo "Checking /v1/models"
    http_get "/v1/models" | tee "$RESULTS_DIR/v1_models.json"

    echo "Checking /managed/status"
    http_get "/managed/status" | tee "$RESULTS_DIR/managed_status.json"

    echo "Checking /managed/world_size"
    http_get "/managed/world_size" | tee "$RESULTS_DIR/managed_world_size.json"

    echo "Checking pause/resume"
    http_post_json "/managed/pause" '{}' | tee "$RESULTS_DIR/pause.json"
    http_post_json "/managed/resume" '{}' | tee "$RESULTS_DIR/resume.json"

    echo "Checking sleep/wake"
    http_post_json "/managed/sleep" '{"level": 1}' | tee "$RESULTS_DIR/sleep.json"
    http_post_json "/managed/wake" '{"tags": null}' | tee "$RESULTS_DIR/wake.json"

    echo "Checking prepare/finish weight update"
    http_post_json "/managed/prepare_weight_update" '{"sleep_level": 2, "wake_weights": true}' \
        | tee "$RESULTS_DIR/prepare_weight_update.json"
    http_post_json "/managed/finish_weight_update" '{"wake_kv_cache": true, "resume": true}' \
        | tee "$RESULTS_DIR/finish_weight_update.json"
}

run_push_example() {
    local model_path="$1"
    local output_file="$2"

    (
        cd "$BASE_DIR"
        export PYTHONPATH="$VLLM_PATCH_DIR"
        export VLLM_ALLOW_INSECURE_SERIALIZATION=1
        "$PYTHON_BIN" "$VLLM_PATCH_DIR/examples/managed_weight_sync/hf_push_ipc.py" \
            --model-path "$model_path" \
            --base-url "$BASE_URL" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --target-devices "$(seq -s, 0 $((TP_SIZE - 1)))" \
            --skip-before-generate
    ) | tee "$output_file"
}

main() {
    require_path "$VLLM_PATCH_DIR/vllm" "vLLM hotpatch overlay"
    require_path "$VLLM_PATCH_DIR/examples/managed_weight_sync/hf_push_ipc.py" \
        "managed weight-sync IPC example"
    require_path "$PYTHON_BIN" "workspace python interpreter"
    require_path "$CHAT_MODEL_PATH" "chat model path"
    require_path "$BASE_MODEL_PATH" "base model path"
    require_cmd curl

    start_server
    run_endpoint_smoke

    echo "Capturing chat completion before pushing real weights"
    chat_completion "$RESULTS_DIR/chat_before_transfer.json"

    echo "Pushing chat-aligned weights"
    run_push_example "$CHAT_MODEL_PATH" "$RESULTS_DIR/push_chat_model.log"

    echo "Capturing chat completion after chat-model push"
    chat_completion "$RESULTS_DIR/chat_after_chat_push.json"

    echo "Pushing base-model weights without restarting server"
    run_push_example "$BASE_MODEL_PATH" "$RESULTS_DIR/push_base_model.log"

    echo "Capturing chat completion after base-model push"
    chat_completion "$RESULTS_DIR/chat_after_base_push.json"

    echo
    echo "End-to-end flow completed."
    echo "Server log: $SERVER_LOG"
    echo "Result artifacts: $RESULTS_DIR"
}

main "$@"
