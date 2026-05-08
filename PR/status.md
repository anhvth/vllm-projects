# Hotload vLLM Status

Last updated: 2026-05-08.

## Product decision

One served model replica must fit on one physical node. We do not support
splitting one model replica across nodes for managed hotload.

For a 3-node cluster with 8 GPUs per node, run up to three independent vLLM
replicas:

```text
node0: vLLM dummy -> hotload replica on local GPUs 0..7
node1: vLLM dummy -> hotload replica on local GPUs 0..7
node2: vLLM dummy -> hotload replica on local GPUs 0..7
```

## Endpoint contract

Inference clients get one public OpenAI-compatible endpoint:

```text
http://head-node:8000/v1
```

The public endpoint is a proxy/load balancer over private replica endpoints:

```text
http://node0:8100/v1
http://node1:8100/v1
http://node2:8100/v1
```

Managed hotload control is private and per-replica:

```text
http://node0:8100/managed
http://node1:8100/managed
http://node2:8100/managed
```

Regular application code should never need the private replica URLs.

## Suggested user UX

```bash
hotloadctl start --nodes node0,node1,node2 --gpus-per-replica 8
hotloadctl push /path/to/checkpoint
hotloadctl status
hotloadctl sleep
hotloadctl wake
hotloadctl stop
```

OpenAI client usage stays normal:

```python
from openai import OpenAI

client = OpenAI(base_url="http://head-node:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="qwen3-1.7b",
    messages=[{"role": "user", "content": "hello"}],
)
```

## Controller behavior

`hotloadctl start` should:

* start one dummy vLLM replica per selected node
* start or configure the public proxy/load balancer
* wait for `/managed/status` and `/v1/models` on every replica
* report the single public base URL

`hotloadctl push CHECKPOINT` should, for each replica:

* call `/managed/init_weight_transfer`
* call `/managed/prepare_weight_update`
* run local IPC weight push on that replica's node
* call `/managed/finish_weight_update`
* verify inference through the replica or public endpoint

`hotloadctl status` should show:

```text
public_base_url: http://head-node:8000/v1

replicas:
  node0: http://node0:8100/v1 healthy
  node1: http://node1:8100/v1 healthy
  node2: http://node2:8100/v1 healthy
```

## Current implementation posture

The managed endpoints and IPC hotload flow are designed around one local vLLM
server. The next orchestration step is to wrap that proven local flow in a
multi-replica controller, keeping the public inference endpoint singular and
the private managed endpoints per-node.

## Workspace implementation status

The workspace package now carries a workspace-scoped `hotloadctl` controller
plus a minimal public OpenAI proxy for the multi-replica design:

```bash
hotloadctl start \
  --nodes node0,node1,node2 \
  --gpus-per-replica 8 \
  --model-path ~/ckpt/hf_models/Qwen/Qwen3-1.7B \
  --served-model-name qwen3-1.7b \
  --base-port 8100 \
  --public-port 8000 \
  --public-host head-node \
  --ssh-user runner

hotloadctl start \
  --nodes node0,node1,node2 \
  --gpus-per-replica 8 \
  --model-path ~/ckpt/hf_models/Qwen/Qwen3-1.7B \
  --served-model-name qwen3-1.7b \
  --public-host head-node \
  --ssh-user runner \
  --dry-run

hotloadctl push ~/ckpt/hf_models/Qwen/Qwen3-1.7B-Base
hotloadctl status
hotloadctl sleep --level 1
hotloadctl wake
hotloadctl stop
```

The same Ray Serve controller is also exposed through a `vllm serve`-shaped
multi-node entrypoint:

```bash
ray-vllm serve ~/ckpt/hf_models/Qwen/Qwen3-1.7B \
  --nodes node0,node1,node2 \
  --tensor-parallel-size 8 \
  --served-model-name qwen3-1.7b \
  --trust-remote-code \
  --dtype bfloat16 \
  --fast-loading ram

ray-vllm push ~/ckpt/hf_models/Qwen/Qwen3-1.7B-Base
ray-vllm status
ray-vllm sleep --level 1
ray-vllm wake
ray-vllm stop
```

Behavior implemented in this workspace:

* `hotloadctl start` builds one local-GPU `vllm serve` replica per node as Ray
  Serve applications, plus a public proxy application.
* Replica startup enables `--load-format dummy`, `--managed-weight-sync`,
  `--enable-sleep-mode`, and
  `--weight-transfer-config '{"backend":"ipc"}'`.
* The public proxy is started under tmux on `--public-host`; when that host is
  remote, `hotloadctl` now launches it through SSH instead of incorrectly
  starting it only on the local controller machine.
* `hotloadctl push` fans out managed lifecycle calls and then runs the local
  IPC helper on each target node. The saved cluster state now keeps the
  configured `ssh_user` so later `push` and `stop` calls reuse the same remote
  identity by default.
* `hotloadctl status` reports the single public base URL plus each private
  replica `/v1` and `/managed` URL and a managed status summary.
* The public proxy only serves `/v1` traffic, rejects `/managed`, and
  round-robins across healthy replicas.
* `--dry-run` is available so the generated Ray Serve config and `serve run`
  command can be inspected before execution.
* `ray-vllm serve` maps familiar `vllm serve` flags such as model path,
  `--served-model-name`, `--tensor-parallel-size`, `--host`, `--port`,
  `--dtype`, `--max-model-len`, and `--trust-remote-code` onto the Ray Serve
  multi-replica controller.
* `ray-vllm serve --fast-loading ram` enables the RAM-stage safetensors loader
  for initial replica startup and persists that setting so later managed
  `ray-vllm push` reloads use the same RAM-stage loader options. Use
  `--fast-loading off` or `--no-fast-loading` to keep normal safetensors
  loading.

Verified locally in this workspace:

* `python -m unittest tests.test_hotloadctl -q` passes from the workspace.
* `uvx --from ruff ruff format --check` and `uvx --from ruff ruff check` pass
  for the touched workspace Python files:
  `src/vllm_hotload/hotloadctl.py`, `src/vllm_hotload/proxy.py`, and
  `tests/test_hotloadctl.py`.
* `uv run --group dev python tools/lint.py --file ...` is clean for the same
  touched Python files.
* Dry-run command generation for a 3-node start plan, including the generated
  Ray Serve config and `serve run` command.
* Dry-run push planning verifies per-node IPC helper commands and the public
  `/v1/models` verification target after the fanout.
* Dry-run stop planning verifies owned Ray Serve application shutdown.
* Mocked status aggregation covers healthy and unhealthy replicas.
* An in-process proxy test verifies that `/managed` is not exposed publicly and
  that healthy `/v1` requests rotate across replicas.
* `python -m unittest tests.test_hotloadctl -q` passes for the updated
  `ray-vllm` CLI and RAM-stage reload payload tests.
* `uvx --from ruff ruff format --check` and `uvx --from ruff ruff check` pass
  for `src/vllm_hotload/hotloadctl.py`, `src/vllm_hotload/ray_vllm.py`,
  `src/vllm_hotload/ray_serve_app.py`, `tests/test_hotloadctl.py`, and
  `tests/test_packaging_regression.py`.
* `PYTHONPATH=src python -m vllm_hotload.ray_vllm serve ... --fast-loading ram
  --dry-run` emits a Ray Serve config with one public `/v1` app and per-node
  managed replicas with `fast_loading_ram: true`.

Still requires real multi-node validation:

* Actual SSH reachability and tmux startup on remote worker nodes and the
  head-node proxy host.
* Live readiness checks against `/managed/status` and `/v1/models` for real
  `vllm serve` replicas on multiple nodes.
* Real checkpoint fanout over IPC on multiple physical nodes with GPUs.
* End-to-end public proxy forwarding from a normal OpenAI client against live
  replicas, including streaming behavior.
* This workspace currently has no `/home/anhvth8/vllm_projects/vllm` checkout,
  so verification here is intentionally limited to the workspace package,
  dry-run command generation, mocked HTTP behavior, and focused local tests.
* The first pass assumes the workspace path and checkpoint path are reachable at
  the same absolute path on every node because the SSH/tmux launch commands use
  the shared workspace state directly.
* `tests.test_packaging_regression` is currently out of sync with the checked-in
  `build.sh` / `pyproject.toml` bootstrap shape and fails before exercising the
  new `ray-vllm` assertion.
