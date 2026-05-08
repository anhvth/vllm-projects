# Agent Instructions for the PR Workspace

> This file describes the active PR: managed weight-sync control endpoints for `vllm serve`.

## Active PR

- **Title:** Extend `vllm serve` with managed weight-sync control endpoints
- **Feature branch:** `feature/managed-weight-sync-serve` in `vllm/`
- **PR docs:** [`PR/hotload_vllm.md`](PR/hotload_vllm.md)
- **E2E orchestration:** [`PR/run_hotload_vllm_e2e.sh`](PR/run_hotload_vllm_e2e.sh)

## Workspace layout

```
/home/anhvth8/vllm_projects/
├── .venv/                          # Shared Python venv (uv, Python 3.12)
├── pyproject.toml                  # Build/project metadata (not in vllm/)
├── vllm_patch/                     # Hotpatch overlay for managed weight-sync
│   ├── vllm/entrypoints/openai/
│   │   ├── managed_weight_sync.py  # Core managed endpoints (router)
│   │   └── cli_args.py             # CLI flags: --managed-weight-sync, etc.
│   └── examples/managed_weight_sync/
│       └── hf_push_ipc.py          # IPC weight-push example client
├── vllm/                           # Upstream vLLM git checkout (feature branch)
├── PR/
│   ├── hotload_vllm.md             # PR spec + implementation status
│   ├── run_hotload_vllm_e2e.sh     # Full end-to-end acceptance test
│   └── logs/                       # Artifacts from e2e runs
├── build.sh                        # Bootstrap: create venv, install vllm, enable overlay
├── CLAUDE.md                       # Workspace-level agent instructions
└── AGENTS.md                       # This file
```

## Hotpatch overlay mechanism

`build.sh` runs `uv sync` for this workspace, which installs the local project
defined by `pyproject.toml`. That local package exposes the files under
`vllm_patch/vllm/` as an installed overlay so they shadow the upstream `vllm`
package at runtime without requiring manual `PYTHONPATH` or custom `.pth`
injection.

Implementation changes live in `vllm_patch/`, **not** in `vllm/`.

## Key endpoints (managed weight-sync)

All under prefix `/managed` (configurable via `--managed-weight-sync-prefix`):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/managed/status` | Server status, config, sleep state |
| GET | `/managed/world_size` | DP/TP world size |
| POST | `/managed/pause` | Pause generation |
| POST | `/managed/resume` | Resume generation |
| POST | `/managed/sleep` | Offload weights (level 1) or discard (level 2) |
| POST | `/managed/wake` | Restore weights/KV cache |
| POST | `/managed/init_weight_transfer` | Init weight-transfer backend |
| POST | `/managed/prepare_weight_update` | Convenience: pause + sleep + wake_weights |
| POST | `/managed/finish_weight_update` | Convenience: wake_kv_cache + resume |

## Development workflow

- **Run the e2e test:** `bash PR/run_hotload_vllm_e2e.sh`
- **Activate venv:** `source .venv/bin/activate`
- **Runtime dependencies:** `build.sh` is the only place that may install or pin runtime dependencies for this workspace. Do not add Ray, vLLM, CUDA, `pyarrow`, `starlette`, or serving dependencies to `pyproject.toml`.
- **Ray Serve demos:** run serving demos from the active `build.sh` venv with `serve run` or `python`, not plain `uv run <demo>`. Plain `uv run` can resolve from `pyproject.toml`, mutate `.venv`, and drop packages installed by `build.sh`.
- **Ray Serve status:** `serve status` requires the dashboard HTTP address, for example `serve status -a http://100.96.5.35:8265`; `--address auto` is valid for `serve run`, not for status.
- **Run focused tests:**
  ```bash
  cd /home/anhvth8/vllm_projects
  source .venv/bin/activate
  cd vllm
  python -m pytest tests/entrypoints/openai/test_managed_weight_sync.py -q
  ```
- **Lint with ruff:** `ruff check vllm_patch/ && ruff format --check vllm_patch/`
- **Lint with pyright (filters overlay noise):** after `source .venv/bin/activate`, run `uv run --active --no-project python tools/lint.py` or `uv run --active --no-project python tools/lint.py --file <path>`
- **Start dev server manually:**
  ```bash
  VLLM_SERVER_DEV_MODE=1 uv run vllm serve ... --managed-weight-sync ...
  ```

## Ray Serve LLM recipes

- Recipe docs live under `docs/recipe/`; matching runnable demos live under `demo/`.
- Keep recipes focused on text-only LLM serving, especially Qwen/Qwen3-style models.
- For the hosted three-node Ray cluster, the known node resources are `node:100.96.5.35`, `node:100.96.34.48`, and `node:100.96.31.61`. Prefer explicit node-resource pinning when a recipe claims one replica per node.
- Ray Serve LLM injects placement groups for vLLM replicas. Do not set `max_replicas_per_node` on LLM deployments because it conflicts with those placement groups.
- Use replicated single-GPU deployments for data-parallel recipes. If multiple deployments need to expose one public model id, use a custom ingress/router; `build_openai_app` rejects duplicate model ids.
- Avoid pipeline-parallel Qwen3 smoke tests unless explicitly investigating pipeline parallelism. In this workspace, Qwen3-1.7B with `pipeline_parallel_size=3` failed during vLLM attention backend initialization; node-pinned replicated deployments were the working 3-node path.
- Keep vLLM pinned through `build.sh`. vLLM 0.20.x wheels require `libcudart.so.13` here, while the host/Torch runtime is CUDA 12.9.

## Safety

Managed endpoints require `VLLM_SERVER_DEV_MODE=1` and must not be exposed to untrusted networks.

## Environment

- Python venv at `.venv/` (Python 3.12, managed with `uv`)
- All paths are relative to the workspace root unless absolute
- For workspace-level bootstrap, see `build.sh`
