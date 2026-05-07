# Agent Instructions for the Workspace

> These instructions apply to the workspace root at `/home/anhvth8/vllm_projects`.
> For the nested vLLM repo, also follow [vllm/AGENTS.md](vllm/AGENTS.md).

## Scope

- The root contains the workspace wrapper files such as [build.sh](build.sh) and [pyproject.toml](pyproject.toml).
- The actual vLLM repository lives under [vllm/](vllm/).
- Keep root guidance short and link to the nested repo instructions instead of duplicating them.

## Environment

- Use `uv` for Python environment management.
- Prefer Python 3.12 when creating or refreshing the virtual environment.
- Use `.venv/bin/python` for checks and scripts instead of system `python3` or bare `pip`.

## Practical Workflow

- Use [build.sh](build.sh) when you need the scripted end-to-end workspace bootstrap.
- For day-to-day vLLM development, follow the workflow in [vllm/AGENTS.md](vllm/AGENTS.md#2-development-workflow).
- When changing `pyproject.toml`, treat it as the source of truth for Python version and build-tool constraints.

## Build Strategy

- `build.sh` uses `VLLM_USE_PRECOMPILED=1` to skip local CUDA kernel compilation.
- This is sufficient when working on **Python-only logic** (model code, inference paths, sampling, etc.).
- Only rebuild from source (remove `VLLM_USE_PRECOMPILED`) when touching C++/CUDA kernels, custom ops, or the CUDA extension layer.

## Reference Docs

- [vllm/README.md](vllm/README.md) for the project overview.
- [vllm/docs/contributing/README.md](vllm/docs/contributing/README.md) for contributor guidance.
- [vllm/docs/getting_started/installation/README.md](vllm/docs/getting_started/installation/README.md) for installation details.