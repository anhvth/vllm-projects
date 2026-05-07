---
name: vllm-managed-hotload
description: 'Run a long-lived vLLM serve process in tmux, hot-load or swap Hugging Face model weights into the live server with managed weight sync, smoke-test /v1 and /managed endpoints, and decide when to sleep, wake, or stop the server to free GPUs. Use when the user says host vllm in tmux, keep vllm hot, upload weights, swap checkpoints, hot service, inference after training, or free GPUs between runs.'
argument-hint: '[checkpoint path, model path, or task summary]'
---

# vLLM Managed Hotload

Use this skill when the user wants a persistent vLLM server that stays up while model weights change.
In this workflow, "upload weights" means pushing an in-memory Transformers model into a running managed vLLM server with `examples/managed_weight_sync/hf_push_ipc.py`. It is not a file upload endpoint.

Skill-owned helper scripts:

- `./scripts/mangage_hotload_vllm.py`: Python helper that exposes a reusable `ManagedHotloadClient` for managed control endpoints and weight transfer, plus a small `ManagedHotloadDemo` subclass for starting and stopping the notebook demo server.
- `./scripts/`: add this directory to `sys.path` before importing the helper.

## Python Helper Usage

If the user wants a notebook or a Python script instead of raw curl commands, import the helper from this skill folder.

Notebook or script setup:

```python
import json
import sys
from pathlib import Path

SKILL_SCRIPTS_DIR = Path('/home/anhvth8/vllm_projects/.agents/skills/vllm-managed-hotload/scripts')
if str(SKILL_SCRIPTS_DIR) not in sys.path:
  sys.path.insert(0, str(SKILL_SCRIPTS_DIR))

from mangage_hotload_vllm import demo, describe_demo_config

print(json.dumps(describe_demo_config(), indent=2))
```

Typical helper calls:

```python
demo.start_dummy_service()
demo.pause()
demo.sleep(level=1)
demo.wake(tags=['weights', 'kv_cache'])
demo.resume()
demo.stop()
```

Use `demo.push(model)` when the caller already has an in-memory module with `named_parameters()` and wants to transfer it directly from the current Python process.

For model swaps from notebook code, call `demo.push(model)` after loading the model in the notebook kernel. Keep completion and chat requests in the notebook, not in the helper.

Implement `/v1/completions` and `/v1/chat/completions` calls in the notebook or calling script by using `demo.post_json(...)` or another OpenAI-compatible client. Keep the helper focused on managed control and weight transfer.

## When to Use

- Host `vllm serve` in `tmux` and keep the process alive
- Push a new checkpoint into the same server without restarting
- Swap chat, base, or freshly trained checkpoints and compare behavior
- Run inference immediately after a checkpoint becomes available
- Decide whether to keep the server warm, put it to sleep, or stop it to free GPUs
- The request is single-node or local-dev. For multi-node hosting, use the cluster-specific deployment workflow instead.

## Before Acting

Confirm these inputs if the user has not already given them:

1. Exact model path to host or push. Never guess the latest checkpoint.
2. Served model name, host, port, and tensor parallel size.
3. Whether the goal is hot-swap in-place or full GPU release.
4. Whether the inference target is a quick smoke test or a full downstream job.

## Decision Rules

- If the user needs standard serving only and does not need live weight replacement, use ordinary `vllm serve` without managed-weight flags.
- If the user wants one long-lived server that survives multiple checkpoint uploads, start with dummy weights and `--managed-weight-sync`.
- If the user wants to temporarily quiesce the server between pushes, use `/managed/pause`, `/managed/sleep`, `/managed/wake`, and `/managed/resume`.
- If the user needs to truly free GPU allocations for training or another job, stop the `tmux` session. Sleep mode is a managed transition step, not a guaranteed substitute for full process exit.
- If the request becomes multi-node, hand off to the multi-node deployment workflow instead of improvising.

## Recommended Workflow

### Step 1 - Verify local prerequisites

From `/home/anhvth8/vllm_projects`, confirm:

- `/home/anhvth8/vllm_projects/.venv/bin/python` exists
- `curl` and `tmux` are installed
- the model path exists
- the port is free
- the managed transfer example exists at `vllm/examples/managed_weight_sync/hf_push_ipc.py`

### Step 2 - Start a long-lived hot server in tmux

Use a dedicated session such as `vllm-hot-serve` and keep server logs separate from client jobs:

```bash
tmux new-session -d -s vllm-hot-serve \
  "cd /home/anhvth8/vllm_projects/vllm && \
   export PYTHONPATH=/home/anhvth8/vllm_projects/vllm && \
   export VLLM_SERVER_DEV_MODE=1 && \
   export VLLM_ALLOW_INSECURE_SERIALIZATION=1 && \
   /home/anhvth8/vllm_projects/.venv/bin/python -m vllm.entrypoints.openai.api_server \
     ~/ckpt/hf_models/Qwen/Qwen3-1.7B \
     --host 0.0.0.0 \
     --port 8000 \
     --served-model-name qwen3-1.7b \
     --trust-remote-code \
     --dtype bfloat16 \
     --load-format dummy \
     --weight-transfer-config '{\"backend\":\"ipc\"}' \
     --enable-sleep-mode \
     --managed-weight-sync \
     --tensor-parallel-size 2 \
     --gpu-memory-utilization 0.40 \
     --max-model-len 4096 \
     2>&1 | tee /tmp/vllm-hot-serve.log"
```

Why dummy weights:

- the server starts once
- later checkpoints replace weights without a server restart
- pre-upload outputs may be garbage, which is expected

### Step 3 - Wait until the server is healthy

Poll the managed and OpenAI-compatible endpoints before pushing real weights:

```bash
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/managed/status
curl --fail http://127.0.0.1:8000/managed/world_size
```

If readiness is unclear, inspect the tmux session:

```bash
tmux capture-pane -t vllm-hot-serve -p | tail -n 80
```

### Step 4 - Run one pre-transfer inference check

Use the same prompt every time so behavior changes are easy to compare:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-1.7b",
    "messages": [
      {
        "role": "user",
        "content": "Explain SVD using the smallest possible 2D example. Keep it short."
      }
    ],
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }'
```

Before a real transfer, the output may be incoherent because the server is still using dummy weights.

### Step 5 - Push weights into the live server

In this repo, the canonical single-node client is `vllm/examples/managed_weight_sync/hf_push_ipc.py`:

```bash
cd /home/anhvth8/vllm_projects
export PYTHONPATH=/home/anhvth8/vllm_projects/vllm
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

/home/anhvth8/vllm_projects/.venv/bin/python \
  /home/anhvth8/vllm_projects/vllm/examples/managed_weight_sync/hf_push_ipc.py \
  --model-path ~/ckpt/hf_models/Qwen/Qwen3-1.7B \
  --base-url http://127.0.0.1:8000 \
  --served-model-name qwen3-1.7b \
  --keep-alive
```

What this script does:

- calls `/managed/init_weight_transfer`
- calls `/managed/prepare_weight_update`
- sends parameters via CUDA IPC
- calls `/managed/finish_weight_update`
- runs an after-transfer generation check

### Step 6 - Infer through the normal OpenAI-compatible API

After the push, use the normal `/v1` API. Do not invent a separate server or a special inference path.

Typical smoke test:

- call `/v1/chat/completions` again with the same prompt
- verify the response is coherent and not dummy-weight garbage
- if the user has a downstream eval or data-generation job, point that job at `http://127.0.0.1:8000/v1`

### Step 7 - Swap to a different checkpoint without restarting

When a new checkpoint appears, push it into the same running server:

```bash
/home/anhvth8/vllm_projects/.venv/bin/python \
  /home/anhvth8/vllm_projects/vllm/examples/managed_weight_sync/hf_push_ipc.py \
  --model-path ~/ckpt/hf_models/Qwen/Qwen3-1.7B-Base \
  --base-url http://127.0.0.1:8000 \
  --served-model-name qwen3-1.7b \
  --prompt "Explain SVD using the smallest possible 2D example. Keep it short." \
  --keep-alive
```

Expectations:

- the server process stays up
- `/v1/models` still works
- the response changes in a way consistent with the new weights
- the response should not regress to dummy-weight nonsense

### Step 8 - Choose how to park or stop the server

To prepare for another update without killing the process:

```bash
curl -X POST http://127.0.0.1:8000/managed/pause
curl -X POST http://127.0.0.1:8000/managed/sleep \
  -H 'Content-Type: application/json' \
  -d '{"level": 1}'
curl -X POST http://127.0.0.1:8000/managed/wake \
  -H 'Content-Type: application/json' \
  -d '{"tags": null}'
curl -X POST http://127.0.0.1:8000/managed/resume
```

To cleanly stop and fully release GPUs:

```bash
tmux kill-session -t vllm-hot-serve
```

Recommended rule:

- keep the tmux session alive if another checkpoint will be pushed soon
- kill the tmux session if training or another workload needs the GPUs back

## Training-to-Serve Loop

For iterative checkpoint work, use this loop:

1. Training writes a checkpoint to disk.
2. Confirm the exact checkpoint path.
3. If the hot server is not already running, start it once in tmux with dummy weights.
4. Push the new checkpoint with `hf_push_ipc.py`.
5. Run one smoke prompt through `/v1/chat/completions`.
6. Run the real inference or eval workload.
7. When the next checkpoint arrives, push again into the same server.
8. When GPUs are needed elsewhere, kill the tmux session.

## Completion Checks

Do not claim success until these are true:

- the tmux session exists and the server log shows startup completed
- `/v1/models` returns 200
- `/managed/status` returns 200
- `/managed/world_size` returns 200 when expected for this setup
- pre-transfer output can be poor, but post-transfer output is coherent
- a second transfer changes behavior without restarting the server
- the normal `/v1` API still works after each upload

## Common Mistakes

- Guessing the checkpoint path instead of confirming it
- Treating upload as an HTTP file upload instead of a local weight push
- Restarting the server for every checkpoint when managed sync is the goal
- Using the wrong Python environment instead of `/home/anhvth8/vllm_projects/.venv/bin/python`
- Declaring success before checking `/v1/models` and the post-transfer generation
- Keeping the server alive when the real requirement is to free GPUs completely

## Related Files in This Workspace

- `.agents/skills/vllm-managed-hotload/scripts/mangage_hotload_vllm.py`
- `PR/hotload_vllm.md`
- `vllm_patch/examples/managed_weight_sync/hf_push_ipc.py`
