# PR: Extend `vllm serve` with managed weight-sync control endpoints

## Current implementation status

Last updated: 2026-05-07.

Implementation base:

```text
/home/anhvth8/vllm_projects
```

Repos/workspaces:

```text
/home/anhvth8/vllm_projects
  Self-contained PR workspace. The shared Python environment and project
  metadata live here:
    .venv/
    pyproject.toml
    PR/hotload_vllm.md

/home/anhvth8/vllm_projects/vllm
  Real vLLM implementation checkout. Keep this as the ordinary upstream git
  checkout and put implementation changes here.
```

Current vLLM branch:

```text
feature/managed-weight-sync-serve
```

Current `git status --short --untracked-files=all` from
`/home/anhvth8/vllm_projects/vllm`:

```text
 D pyproject.toml
 M tests/entrypoints/openai/test_cli_args.py
 M vllm/entrypoints/openai/cli_args.py
 M vllm/entrypoints/serve/__init__.py
?? docs/serving/managed_weight_sync.md
?? examples/managed_weight_sync/hf_push_ipc.py
?? examples/managed_weight_sync/hf_push_nccl_skeleton.py
?? tests/entrypoints/openai/test_managed_weight_sync.py
?? vllm/entrypoints/openai/managed_weight_sync.py
```

Note: `pyproject.toml` was intentionally moved from the vLLM checkout to the
self-contained base workspace at `/home/anhvth8/vllm_projects/pyproject.toml`,
alongside `/home/anhvth8/vllm_projects/.venv`.

### Implemented so far

* Added managed weight-sync CLI arguments in
  `vllm/entrypoints/openai/cli_args.py`:
  * `--managed-weight-sync`
  * `--managed-weight-sync-prefix`
  * `--managed-weight-sync-require-dev-mode`
* Added CLI validation that rejects managed mode unless
  `VLLM_SERVER_DEV_MODE=1`, when the dev-mode guard is enabled.
* Added route registration from `vllm/entrypoints/serve/__init__.py` into the
  existing `vllm serve` router stack.
* Added new managed route module:
  `vllm/entrypoints/openai/managed_weight_sync.py`.
* Added intended managed endpoints:
  * `GET /managed/status`
  * `GET /managed/world_size`
  * `POST /managed/pause`
  * `POST /managed/resume`
  * `POST /managed/sleep`
  * `POST /managed/wake`
  * `POST /managed/init_weight_transfer`
  * `POST /managed/prepare_weight_update`
  * `POST /managed/finish_weight_update`
* Added focused mock tests in
  `tests/entrypoints/openai/test_managed_weight_sync.py`.
* Added CLI tests in `tests/entrypoints/openai/test_cli_args.py`.
* Added docs page at `docs/serving/managed_weight_sync.md`.
* Added examples:
  * `examples/managed_weight_sync/hf_push_ipc.py`
  * `examples/managed_weight_sync/hf_push_nccl_skeleton.py`
* Created a shared workspace environment at
  `/home/anhvth8/vllm_projects/.venv` because the fresh vLLM checkout did not
  already have one. Use this env from the base workspace, not from
  `vllm/.venv`.

### Verification so far

Ruff was run on the touched implementation, test, docs/example-adjacent Python
files after installing lightweight tooling into the shared base `.venv`.

Result:

```text
ruff format: completed on touched files
ruff check: clean on touched Python files after minor cleanup
```

Focused pytest was attempted with:

```bash
cd /home/anhvth8/vllm_projects
source .venv/bin/activate
cd vllm
python -m pytest -q \
  tests/entrypoints/openai/test_managed_weight_sync.py \
  tests/entrypoints/openai/test_cli_args.py \
  -q
```

The test environment needed several lightweight dependencies installed into the
shared base `.venv` before collection could proceed:

```text
ruff
cbor2
tblib
cachetools
py-cpuinfo
watchfiles
model-hosting-container-standards
ijson
mistral_common[image]
```

Current focused pytest result:

```text
Not passing yet.
```

Known current blockers:

1. `vllm/entrypoints/openai/managed_weight_sync.py` currently contains a
   duplicate function definition line:

   ```python
   async def _safe_is_sleeping(request: Request) -> bool | None:
   async def _safe_is_sleeping(request: Request) -> bool | None:
   ```

   This should be fixed before rerunning tests because it can cause a syntax or
   import failure.

2. `tests/entrypoints/openai/test_cli_args.py` currently fails at parser fixture
   setup in this environment because vLLM cannot infer a device type:

   ```text
   RuntimeError: Failed to infer device type
   ```

   The failure happens while constructing `AsyncEngineArgs` / `EngineArgs`,
   before the managed CLI assertions can run. The next fix should make these
   parser tests avoid auto device detection, likely by setting/mocking the vLLM
   platform/device in the test fixture rather than requiring a GPU.

3. The full end-to-end acceptance test has not been run yet:
   dummy weights -> Qwen3-1.7B chat weights -> Qwen3-1.7B-Base weights with
   tensor parallel size 2 and IPC weight transfer.

### Definition-of-done progress

```text
[partial] vllm serve --help shows --managed-weight-sync
          CLI argument was added, but help has not been reverified after the
          current syntax/test issues.

[partial] managed endpoints are only registered when enabled
          Route hook and mock test exist, but tests are not passing yet.

[partial] managed mode refuses to start without VLLM_SERVER_DEV_MODE=1
          CLI validation and router attach guard exist, but tests are not
          passing yet.

[not verified] OpenAI API remains unchanged

[not done] IPC example can push HF Qwen3-1.7B weights into dummy vLLM with
           tensor parallel size 2

[not done] end-to-end acceptance test passes for
           dummy -> Qwen3-1.7B chat -> Qwen3-1.7B-Base update

[done] docs include quick start and safety warning
       Added docs/serving/managed_weight_sync.md.

[partial] tests/mocks cover endpoint behavior
          Added mock tests, but they still need to pass.

[done] final/status reporting includes current branch and git status
       Captured above for the current implementation checkout.
```

### Next implementation steps

1. Fix the duplicate `_safe_is_sleeping` definition in
   `vllm/entrypoints/openai/managed_weight_sync.py`.
2. Rerun import/syntax checks for the managed module.
3. Fix the CLI parser test environment so `test_cli_args.py` does not fail on
   device auto-detection in a non-GPU/dev environment.
4. Rerun the focused pytest suite.
5. Recheck `vllm serve --help` for the new flags.
6. Run or explicitly document the GPU acceptance path with Qwen3-1.7B,
   tensor-parallel size 2, dummy load format, and IPC transfer.

### Workspace helper

The self-contained workspace now includes an orchestration script for the
dummy-host + managed end-to-end flow:

```bash
/home/anhvth8/vllm_projects/PR/run_hotload_vllm_e2e.sh
```

It starts the dummy-weight server from the shared base `.venv`, waits for
managed readiness, runs the managed endpoint smoke checks, then pushes
`Qwen3-1.7B` and `Qwen3-1.7B-Base` sequentially with the IPC example while
capturing result artifacts under `PR/logs/`.

## Objective

Clone the vLLM source, create a feature branch, and implement a thin extension to `vllm serve` that exposes managed control endpoints for:

- weight transfer lifecycle
- pause/resume
- sleep/offload/wake
- status/debug
- prepare/finish weight update flow

Do **not** build a separate server that reimplements OpenAI APIs. The goal is to extend the existing `vllm serve` OpenAI-compatible server.

The final UX should look like:

```bash
VLLM_SERVER_DEV_MODE=1 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
uv run vllm serve ~/ckpt/hf_models/Qwen/Qwen3-1.7B/ \
  --served-model-name qwen3-1.7b \
  --trust-remote-code \
  --dtype bfloat16 \
  --load-format dummy \
  --weight-transfer-config '{"backend":"ipc"}' \
  --enable-sleep-mode \
  --managed-weight-sync \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.40 \
  --max-model-len 4096
```

Then external trainer/controller code can call:

```bash
curl http://127.0.0.1:8000/managed/status

curl -X POST http://127.0.0.1:8000/managed/prepare_weight_update \
  -H 'Content-Type: application/json' \
  -d '{"sleep_level": 2, "wake_weights": true}'

# trainer sends weights using IPCWeightTransferEngine / NCCLWeightTransferEngine

curl -X POST http://127.0.0.1:8000/managed/finish_weight_update \
  -H 'Content-Type: application/json' \
  -d '{"wake_kv_cache": true, "resume": true}'
```

## Multi-node serving decision

Decision: one served model replica must fit on one physical node. We do not
support splitting a single model replica across nodes for managed hotload.

For a 3-node, 8-GPU-per-node cluster, the managed service should run as up to
three independent vLLM replicas:

```text
node0: one dummy -> hotload vLLM replica using local GPUs 0..7
node1: one dummy -> hotload vLLM replica using local GPUs 0..7
node2: one dummy -> hotload vLLM replica using local GPUs 0..7
```

This keeps the weight-transfer path local to each machine. The current IPC
hotload mechanism remains a good fit because CUDA IPC handles are
single-machine objects. Cross-node Ray is still useful for cluster lifecycle
and placement, but it should not be used to create one model replica spread
across multiple machines.

User-facing goal:

```text
hotloadctl start --nodes node0,node1,node2 --gpus-per-replica 8
hotloadctl push /path/to/checkpoint
hotloadctl status
hotloadctl sleep
hotloadctl wake
hotloadctl stop
```

The controller handles per-node details. Users should not need to manually run
three different `vllm serve` commands, remember ports, or call every managed
endpoint themselves.

### Public OpenAI endpoint

Application code should see one OpenAI-compatible base URL, not one URL per
replica:

```text
public OpenAI base URL: http://head-node:8000/v1
```

The public endpoint is backed by a small proxy/load balancer that routes normal
OpenAI-compatible requests to healthy per-node vLLM replicas:

```text
node0 replica: http://node0:8100/v1
node1 replica: http://node1:8100/v1
node2 replica: http://node2:8100/v1
```

Client usage stays ordinary:

```python
from openai import OpenAI

client = OpenAI(base_url="http://head-node:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="qwen3-1.7b",
    messages=[{"role": "user", "content": "hello"}],
)
```

Managed control remains private and per-replica:

```text
node0 control: http://node0:8100/managed
node1 control: http://node1:8100/managed
node2 control: http://node2:8100/managed
```

Only the controller should call the managed URLs. Regular inference clients
should use the single public `/v1` endpoint.

`hotloadctl status` should show both views:

```text
public_base_url: http://head-node:8000/v1

replicas:
  node0: http://node0:8100/v1 healthy
  node1: http://node1:8100/v1 healthy
  node2: http://node2:8100/v1 healthy
```

Recommended internal shape:

```text
hotloadctl start
  -> ssh each node if needed
  -> optionally start/join a dedicated Ray cluster on non-default ports
  -> start one vLLM dummy replica per selected node
  -> wait for /managed/status and /v1/models on every replica

hotloadctl push CHECKPOINT
  -> for each replica:
       POST /managed/init_weight_transfer
       POST /managed/prepare_weight_update
       run local IPC weight push on that replica's node
       POST /managed/finish_weight_update
  -> verify /v1/chat/completions or /v1/completions on every replica

hotloadctl status
  -> return one combined view across replicas
```

Suggested first implementation: use SSH/tmux for process launch because it is
already proven in this workspace, and keep the command surface compatible with
Ray so the launcher can later replace raw process management with Ray actors.
Ray should use non-default ports to avoid conflicts with training jobs.

The important product boundary is that users think in terms of a hotloadable
replica group, while the system maintains multiple per-node vLLM servers under
the hood.

### Why not split one model across nodes?

Cross-node model parallelism makes the hotload transport much harder and less
reliable for this PR:

* The existing IPC push helper is machine-local.
* A true multi-node transfer would need one transfer worker per node plus extra
  coordination to map parameters to distributed ranks.
* Tensor/pipeline parallelism across machines increases operational coupling
  and makes failures harder to recover from.
* The target deployment has 8 GPUs per node, which is a natural boundary for
  one vLLM replica.

If a future model cannot fit inside one 8-GPU node, that should be treated as a
separate feature: multi-node model-parallel hotload with a different transfer
design, not a small extension of the current IPC path.

## Superseded multi-node notes

The notes below are kept as background only. They are not the chosen product
direction for this PR because they imply one served model may be split across
nodes.

### Layer 1: Ray cluster bootstrap

Keep SSH only for first contact and cluster bring-up. A small launcher can
start Ray on the head node, join the worker nodes, and wait until the full set
of GPUs is visible before any serving process starts.

One reasonable shape is:

```text
hotloadctl cluster start
  -> ssh node0: ray start --head --port 26379 ...
  -> ssh node1: ray start --address node0:26379 ...
  -> ssh node2: ray start --address node0:26379 ...
  -> ray status until 3 nodes / 24 GPUs alive
```

Using a non-default Ray port is preferable when these hosts may also run other
Ray jobs. The exact port set can be adjusted to avoid collisions with existing
training clusters.

### Layer 2: Dummy vLLM service

Once the Ray cluster is healthy, run one hosted dummy vLLM server on the Ray
head node with managed weight-sync enabled. For a 3-node, 8-GPU-per-node
cluster, the likely starting point is TP=8 and PP=3 so tensor parallel stays
mostly within each node while pipeline parallel spans nodes.

The resulting service model is:

```text
SSH bootstraps Ray/processes
Ray owns distributed membership and resource scheduling
vLLM serves the dummy model first
managed endpoints pause / sleep / wake / resume generation
an external controller pushes real weights into the live service
```

### Important transport limitation

The current IPC-based hotload helper is a strong fit for single-node weight
pushes, but CUDA IPC handles are local-machine objects. That means the existing
`hf_push_ipc.py` path is not enough by itself for a true 3-node engine.

Two practical follow-on options are:

1. Keep the scope minimal and run 3 independent 8-GPU vLLM servers, one per
   node, then load-balance across them. This reuses the current IPC hotload
   path with almost no new transport work.
2. Build a true multi-node weight-push helper that coordinates one local
   transfer worker per node, then drives the managed prepare/update/finish
   flow across the whole 24-GPU engine.

For a first production pass, the minimal option is the safer choice unless the
target model genuinely needs all 24 GPUs as one serving engine.

## 1. Clone source and prepare branch

Work in the self-contained base workspace:

```text
/home/anhvth8/vllm_projects
```

The vLLM source tree is:

```text
/home/anhvth8/vllm_projects/vllm
```

The shared Python environment is:

```text
/home/anhvth8/vllm_projects/.venv
```

The PR/status docs live under:

```text
/home/anhvth8/vllm_projects/PR
```

Git tracking rules:

* `/home/anhvth8/vllm_projects/vllm` is the real implementation repository and
  must remain an ordinary git checkout of
  `https://github.com/vllm-project/vllm.git`.
* Do not make `/home/anhvth8/vllm_projects` itself the vLLM implementation git
  repo. It is the self-contained parent workspace that owns `.venv`,
  `pyproject.toml`, `PR/`, and the `vllm/` checkout.
* Before editing, confirm `/home/anhvth8/vllm_projects/vllm/.git` exists and `git status`
  is understood.
* Keep all implementation changes on `feature/managed-weight-sync-serve`.
* At the end, report `git status --short` and the current branch from
  `/home/anhvth8/vllm_projects/vllm`.

If `/home/anhvth8/vllm_projects/vllm` does not exist:

```bash
mkdir -p /home/anhvth8/vllm_projects
cd /home/anhvth8/vllm_projects
git clone https://github.com/vllm-project/vllm.git
cd /home/anhvth8/vllm_projects/vllm
git checkout main
git pull --ff-only origin main
```

If it already exists:

```bash
cd /home/anhvth8/vllm_projects/vllm
git fetch origin
git checkout main
git pull --ff-only origin main
```

Create branch:

```bash
cd /home/anhvth8/vllm_projects/vllm
git checkout -b feature/managed-weight-sync-serve
```

Use the shared base venv:

```bash
cd /home/anhvth8/vllm_projects
source .venv/bin/activate
```

Verify:

```bash
cd /home/anhvth8/vllm_projects/vllm
git status --short
git branch --show-current
python -c "import vllm; print(vllm.__version__)"
python -m vllm.entrypoints.openai.api_server --help || true
vllm serve --help | grep -E "weight-transfer|sleep|load-format|served-model-name" || true
```

## 2. Inspect existing implementation

Before editing, inspect current code paths:

```bash
cd /home/anhvth8/vllm_projects/vllm

rg "weight_transfer" vllm/entrypoints vllm/engine vllm/v1 vllm/config.py
rg "init_weight_transfer_engine|update_weights" vllm
rg "pause|resume|sleep|wake_up|is_sleeping" vllm/entrypoints vllm/engine vllm/v1
rg "VLLM_SERVER_DEV_MODE" vllm
rg "add_argument.*serve|cli_args" vllm/entrypoints
```

Important: prefer modifying the OpenAI-compatible server path, not the demo server.

Likely target files:

```text
vllm/entrypoints/openai/api_server.py
vllm/entrypoints/openai/cli_args.py
```

If the current codebase organizes routes differently, follow the current structure. Do not force these exact files if the repo has moved them.

## 3. Add CLI flags

Add these `vllm serve` flags:

```bash
--managed-weight-sync
--managed-weight-sync-prefix
```

Defaults:

```text
managed_weight_sync = False
managed_weight_sync_prefix = "/managed"
```

Optional nice-to-have:

```bash
--managed-weight-sync-require-dev-mode
```

Default:

```text
True
```

Behavior:

* If `--managed-weight-sync` is not passed, vLLM behavior must be unchanged.
* If `--managed-weight-sync` is passed and `VLLM_SERVER_DEV_MODE=1` is not set, fail startup with a clear error.
* If `--managed-weight-sync` is passed without `--weight-transfer-config`, allow status endpoints but return a clear error from weight-transfer endpoints.
* If sleep endpoints are called without `--enable-sleep-mode`, return a clear error.

## 4. Add managed route module

Create a new module if appropriate:

```text
vllm/entrypoints/openai/managed_weight_sync.py
```

This module should export a function similar to:

```python
def register_managed_weight_sync_routes(
    app,
    engine_client,
    args,
    prefix: str = "/managed",
) -> None:
    ...
```

Use FastAPI route registration consistent with the existing server.

Do not create a second FastAPI app. Register routes on the existing `vllm serve` app.

## 5. Managed endpoint list

Implement these endpoints:

```text
GET  /managed/status
GET  /managed/world_size

POST /managed/pause
POST /managed/resume

POST /managed/sleep
POST /managed/wake

POST /managed/init_weight_transfer

POST /managed/prepare_weight_update
POST /managed/finish_weight_update
```

All responses should be JSON.

Every successful response should include:

```json
{
  "ok": true
}
```

Every expected validation or runtime error should use the appropriate HTTP
status code and include:

```json
{
  "ok": false,
  "error": "...",
  "hint": "..."
}
```

Do not leak Python stack traces in normal HTTP responses.

Use `400` for invalid request payloads such as invalid sleep level or wake tags,
`403` for managed mode startup/security violations if they surface through an
HTTP path, `409` for endpoint calls that conflict with server state such as
sleep mode or weight transfer not being configured, and `500` only for
unexpected internal failures. Do not return `200` for known failed operations.

## 6. Endpoint behavior

### `GET /managed/status`

Return best-effort status:

```json
{
  "ok": true,
  "managed_weight_sync": true,
  "prefix": "/managed",
  "dev_mode": true,
  "served_model_names": ["qwen3-1.7b"],
  "load_format": "dummy",
  "weight_transfer_config": {
    "backend": "ipc"
  },
  "sleep_mode_enabled": true,
  "is_sleeping": false,
  "world_size": 1
}
```

If some values are not available, return `null`, not a crash.

### `GET /managed/world_size`

Wrap the existing world-size logic if available.

Response:

```json
{
  "ok": true,
  "world_size": 1
}
```

### `POST /managed/pause`

Pause generation using the same underlying mechanism as existing `/pause`, if present.

Response:

```json
{
  "ok": true,
  "paused": true
}
```

### `POST /managed/resume`

Resume generation using the same underlying mechanism as existing `/resume`, if present.

Response:

```json
{
  "ok": true,
  "resumed": true
}
```

### `POST /managed/sleep`

Input:

```json
{
  "level": 1
}
```

or:

```json
{
  "level": 2
}
```

Behavior:

* Validate level is `1` or `2`.
* Call existing sleep-mode logic.
* If sleep mode is disabled, return error with hint: start server with `--enable-sleep-mode`.

Response:

```json
{
  "ok": true,
  "level": 2,
  "sleeping": true
}
```

### `POST /managed/wake`

Input:

```json
{
  "tags": ["weights"]
}
```

or:

```json
{
  "tags": ["kv_cache"]
}
```

or:

```json
{
  "tags": null
}
```

Behavior:

* Call existing wake-up logic.
* Validate tags if needed.
* Support `weights` and `kv_cache`.

Response:

```json
{
  "ok": true,
  "woke": true,
  "tags": ["weights"]
}
```

### `POST /managed/init_weight_transfer`

Input for IPC:

```json
{
  "init_info": {}
}
```

Input for NCCL:

```json
{
  "init_info": {
    "master_address": "127.0.0.1",
    "master_port": 29500,
    "rank_offset": 1,
    "world_size": 2
  }
}
```

Behavior:

* Call existing `init_weight_transfer_engine`.
* Return clear error if weight transfer is not configured.

Response:

```json
{
  "ok": true,
  "initialized": true
}
```

### `POST /managed/prepare_weight_update`

This is the main convenience endpoint.

Input:

```json
{
  "sleep_level": 2,
  "wake_weights": true
}
```

Behavior:

1. Pause generation.
2. If `sleep_level` is `1` or `2`, call sleep.
3. If `wake_weights` is true, call wake with `tags=["weights"]`.

Response should include step-by-step status:

```json
{
  "ok": true,
  "steps": [
    {"step": "pause", "ok": true},
    {"step": "sleep", "level": 2, "ok": true},
    {"step": "wake_weights", "ok": true}
  ]
}
```

If a step fails, return:

```json
{
  "ok": false,
  "failed_step": "sleep",
  "steps": [
    {"step": "pause", "ok": true},
    {"step": "sleep", "ok": false, "error": "..."}
  ],
  "hint": "..."
}
```

If a step fails after an earlier step has already succeeded, stop immediately and
report the failed step. Do not attempt an automatic rollback or resume inside
`prepare_weight_update`; leave the server in its current state so the external
controller can inspect `/managed/status` and recover deliberately.

### `POST /managed/finish_weight_update`

Input:

```json
{
  "wake_kv_cache": true,
  "resume": true
}
```

Behavior:

1. If `wake_kv_cache` is true, call wake with `tags=["kv_cache"]`.
2. If `resume` is true, resume generation.

Response:

```json
{
  "ok": true,
  "steps": [
    {"step": "wake_kv_cache", "ok": true},
    {"step": "resume", "ok": true}
  ]
}
```

## 7. Security guard

This feature must be treated as unsafe for public exposure.

Rules:

* Managed endpoints are only registered when `--managed-weight-sync` is passed.
* Startup must fail if `--managed-weight-sync` is passed without `VLLM_SERVER_DEV_MODE=1`.
* Print a warning at startup:

```text
WARNING: managed weight-sync endpoints are enabled. Do not expose this server to an untrusted network.
```

* Keep default host behavior unchanged.
* Do not add auth in this PR.

## 8. Add example trainer client

Add:

```text
examples/managed_weight_sync/hf_push_ipc.py
```

Purpose: demonstrate pushing an in-memory Transformers model into the managed vLLM server.

The example should run from the LLaMAFactory/Transformers environment, but it must also import vLLM’s IPC transfer classes. If import fails, print setup instructions.

Default model:

```text
~/ckpt/hf_models/Qwen/Qwen3-1.7B/
```

Default server:

```text
http://127.0.0.1:8000
```

Implement:

```python
import os
import time
import argparse
import requests
import torch

from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerSendWeightsArgs,
    IPCWeightTransferEngine,
)
```

Script flow:

1. Load tokenizer.
2. Load HF model to `cuda:0`.
3. Build Qwen3 prompt with `enable_thinking=False`.
4. Generate before transfer using OpenAI client. This may be garbage because server has dummy weights.
5. Call `/managed/init_weight_transfer`.
6. Call `/managed/prepare_weight_update`.
7. Send weights:

```python
IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=IPCTrainerSendWeightsArgs(
        mode="http",
        url=base_url,
    ),
)
```

8. Call `/managed/finish_weight_update`.
9. Generate after transfer.
10. Support `--keep-alive` to keep source model process alive after IPC transfer.

CLI flags:

```bash
--model-path
--base-url
--served-model-name
--device
--dtype
--prompt
--keep-alive
--skip-before-generate
```

Default prompt:

```text
Explain SVD using the smallest possible 2D example. Keep it short.
```

Important env:

```python
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
```

If IPC import fails, print:

```text
Could not import vLLM IPC transfer classes from this Python environment.

Fix:
  cd /home/anhvth8/vllm_projects
  source .venv/bin/activate
  pip install -e /home/anhvth8/vllm_projects/vllm --no-build-isolation
```

## 9. Add example NCCL skeleton

Add:

```text
examples/managed_weight_sync/hf_push_nccl_skeleton.py
```

This does not need full multi-node support yet.

It should:

* parse `master_address`, `master_port`, `world_size`, `rank_offset`
* call `/managed/init_weight_transfer` with NCCL init info
* import NCCL transfer classes
* fail clearly if current vLLM version’s NCCL API signature differs

Keep this as a skeleton/example, not production code.

## 10. Add docs

Add:

```text
docs/source/serving/managed_weight_sync.md
```

If docs structure differs, put it in the closest appropriate docs location.

Document:

### What this feature is

A thin managed control-plane extension for `vllm serve` that helps external trainer/controller processes update vLLM weights without restarting the server or dumping HF checkpoints.

### What this feature is not

It does not pass a live Transformers Python object into vLLM.

It does not replace the OpenAI API server.

It does not implement a training loop.

### Quick start

Server:

```bash
VLLM_SERVER_DEV_MODE=1 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
uv run vllm serve ~/ckpt/hf_models/Qwen/Qwen3-1.7B/ \
  --served-model-name qwen3-1.7b \
  --trust-remote-code \
  --dtype bfloat16 \
  --load-format dummy \
  --weight-transfer-config '{"backend":"ipc"}' \
  --enable-sleep-mode \
  --managed-weight-sync \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.40 \
  --max-model-len 4096
```

Client:

```bash
cd /home/anhvth8/vllm_projects
source .venv/bin/activate

python vllm/examples/managed_weight_sync/hf_push_ipc.py \
  --model-path ~/ckpt/hf_models/Qwen/Qwen3-1.7B/ \
  --base-url http://127.0.0.1:8000 \
  --served-model-name qwen3-1.7b \
  --keep-alive
```

### Endpoint sequence

```text
POST /managed/init_weight_transfer
POST /managed/prepare_weight_update
trainer sends weights via IPC/NCCL engine
POST /managed/finish_weight_update
```

### IPC vs NCCL

```text
IPC:
  same node, same GPU, CUDA IPC handles

NCCL:
  separate GPUs or distributed weight broadcast
```

### Sleep levels

```text
level 1:
  offload model weights and discard KV cache

level 2:
  discard model weights and discard KV cache
  useful before external weight update
```

### Safety

These endpoints are dangerous and must not be exposed to untrusted networks.

Require:

```bash
VLLM_SERVER_DEV_MODE=1
```

## 11. Tests

Add lightweight tests where practical.

Search current test style first:

```bash
ls tests
rg "openai" tests
rg "api_server|serve" tests
```

Add tests for pure endpoint logic if possible.

Minimum expected test cases:

1. `--managed-weight-sync` flag exists in CLI parser.
2. Startup validation rejects managed mode without `VLLM_SERVER_DEV_MODE=1`.
3. Managed router is not registered when flag is absent.
4. `/managed/status` returns expected basic fields when enabled.
5. Request validation:

   * invalid sleep level fails
   * invalid wake tags fail
6. `prepare_weight_update` calls steps in order:

   * pause
   * sleep
   * wake weights
7. `finish_weight_update` calls steps in order:

   * wake KV cache
   * resume

Use mocks for engine client calls. Do not require GPU for unit tests.

## 12. End-to-end acceptance test

Stop criteria: do not consider the PR complete until this end-to-end test passes
with `Qwen3-1.7B`, tensor parallel size 2, and the managed IPC weight-update
flow. Unit tests are necessary but not sufficient.

Run server:

```bash
VLLM_SERVER_DEV_MODE=1 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
uv run vllm serve ~/ckpt/hf_models/Qwen/Qwen3-1.7B/ \
  --served-model-name qwen3-1.7b \
  --trust-remote-code \
  --dtype bfloat16 \
  --load-format dummy \
  --weight-transfer-config '{"backend":"ipc"}' \
  --enable-sleep-mode \
  --managed-weight-sync \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.40 \
  --max-model-len 4096
```

Check endpoints:

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/managed/status
curl http://127.0.0.1:8000/managed/world_size

curl -X POST http://127.0.0.1:8000/managed/pause
curl -X POST http://127.0.0.1:8000/managed/resume

curl -X POST http://127.0.0.1:8000/managed/sleep \
  -H 'Content-Type: application/json' \
  -d '{"level": 1}'

curl -X POST http://127.0.0.1:8000/managed/wake \
  -H 'Content-Type: application/json' \
  -d '{"tags": null}'

curl -X POST http://127.0.0.1:8000/managed/prepare_weight_update \
  -H 'Content-Type: application/json' \
  -d '{"sleep_level": 2, "wake_weights": true}'

curl -X POST http://127.0.0.1:8000/managed/finish_weight_update \
  -H 'Content-Type: application/json' \
  -d '{"wake_kv_cache": true, "resume": true}'
```

Run IPC push example:

```bash
cd /home/anhvth8/vllm_projects
source .venv/bin/activate

python vllm/examples/managed_weight_sync/hf_push_ipc.py \
  --model-path ~/ckpt/hf_models/Qwen/Qwen3-1.7B/ \
  --base-url http://127.0.0.1:8000 \
  --served-model-name qwen3-1.7b \
  --keep-alive
```

Chat completion test prompt:

```text
Explain SVD using the smallest possible 2D example. Keep it short.
```

Use the OpenAI-compatible API for each chat check:

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
    "max_tokens": 256
  }'
```

Then upload the base model into the same running server:

```bash
python vllm/examples/managed_weight_sync/hf_push_ipc.py \
  --model-path ~/ckpt/hf_models/Qwen/Qwen3-1.7B-Base/ \
  --base-url http://127.0.0.1:8000 \
  --served-model-name qwen3-1.7b \
  --prompt "Explain SVD using the smallest possible 2D example. Keep it short." \
  --keep-alive
```

Required pass sequence:

1. Start the server with dummy weights from `~/ckpt/hf_models/Qwen/Qwen3-1.7B/`
   and `--tensor-parallel-size 2`.
2. Confirm `/v1/models`, `/managed/status`, and `/managed/world_size` work.
   `/managed/status` should report managed mode enabled and `world_size` should
   be `2` if the current vLLM APIs expose it.
3. Call the OpenAI-compatible chat completions API before uploading real
   weights. Because the server is using `--load-format dummy`, this output may
   be garbage, repetitive, or incoherent. That is expected.
4. Upload `~/ckpt/hf_models/Qwen/Qwen3-1.7B/` with the IPC example into the
   running dummy server.
5. Call chat completions again with the same prompt. The response should be
   coherent, instruction-following, and chat-model-like: short explanation,
   recognizes SVD, and uses a small 2D matrix/vector example rather than random
   text.
6. Without restarting the server, upload
   `~/ckpt/hf_models/Qwen/Qwen3-1.7B-Base/` with the same managed flow.
7. Call chat completions again with the same prompt. The response should change
   in a way consistent with base-model weights: it may be less polished as an
   assistant, less chat-aligned, more continuation-like, or less strict about
   "keep it short", but it should not be dummy-weight garbage. This verifies
   that a second live weight update actually replaced the previous chat-tuned
   weights.
8. Confirm the normal OpenAI-compatible API still works after both uploads.

Expected overall:

* server starts with dummy weights
* tensor parallel size 2 is active
* `/v1/models` works
* `/managed/status` works
* pause/resume works
* sleep/wake works
* before weight transfer, generation may be garbage
* after uploading `Qwen3-1.7B`, generation is coherent and chat-aligned
* after uploading `Qwen3-1.7B-Base`, generation changes and remains non-garbage
* no `save_pretrained()`
* no checkpoint dump
* normal OpenAI-compatible API still works
* normal `vllm serve` behavior is unchanged when `--managed-weight-sync` is omitted

## 13. Non-goals

Do not implement a separate FastAPI/OpenAI server.

Do not rewrite vLLM scheduling, generation, streaming, batching, or tokenizer logic.

Do not implement full training loop.

Do not implement Ray.

Do not make NCCL fully production/multi-node in this PR. Add structure and a skeleton only.

Do not expose this on public interfaces by default.

Do not modify unrelated model loading behavior.

## 14. Definition of done

This PR is done when:

* `vllm serve --help` shows `--managed-weight-sync`
* managed endpoints are only registered when enabled
* managed mode refuses to start without `VLLM_SERVER_DEV_MODE=1`
* OpenAI API remains unchanged
* IPC example can push HF Qwen3-1.7B weights into dummy vLLM with tensor parallel size 2
* the end-to-end acceptance test passes for dummy -> Qwen3-1.7B chat -> Qwen3-1.7B-Base update
* docs include quick start and safety warning
* tests/mocks cover endpoint behavior
* final report includes current branch and `git status --short` from
  `/home/anhvth8/vllm_projects/vllm`

The most important implementation constraint: keep this as a **small `vllm serve` extension**, not a wrapper server. vLLM already has the OpenAI-compatible server, weight-transfer backends, and sleep-mode primitives; the PR should mainly add a safer, higher-level control route layer around those existing pieces.

[1]: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/?utm_source=chatgpt.com "OpenAI-Compatible Server - vLLM"
