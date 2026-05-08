"""Ray Serve LLM demo for Qwen3-1.7B on the existing Ray cluster.

Run from the workspace venv:

    serve run demo.ray_serve_qwen3_1p7b_2nodes:app --address auto --non-blocking

Then query the OpenAI-compatible endpoint:

    curl http://127.0.0.1:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -H 'Authorization: Bearer FAKE_KEY' \
      -d '{
        "model": "qwen3-1.7b",
        "messages": [{"role": "user", "content": "Say hello from Ray Serve."}],
        "max_tokens": 64
      }'

Defaults use two Ray nodes by spreading two one-GPU vLLM workers. Override with
environment variables such as RAY_SERVE_PIPELINE_PARALLEL_SIZE=1 for a cheaper
single-GPU smoke test.

If import fails with "No module named 'pyarrow'", install Ray Serve LLM extras in
the active venv, for example: uv pip install "ray[serve,llm]" pyarrow
"""

from __future__ import annotations

import os
from typing import Any

from ray.serve.llm import LLMConfig, build_openai_app


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def optional_env_vars(*names: str) -> dict[str, str]:
    return {name: value for name in names if (value := os.environ.get(name))}


MODEL_ID = os.environ.get("RAY_SERVE_MODEL_ID", "qwen3-1.7b")
MODEL_SOURCE = os.path.expanduser(
    os.environ.get("RAY_SERVE_MODEL_SOURCE", "~/ckpt/hf_models/Qwen/Qwen3-1.7B")
)
MAX_MODEL_LEN = env_int("RAY_SERVE_MAX_MODEL_LEN", 4096)
TENSOR_PARALLEL_SIZE = env_int("RAY_SERVE_TENSOR_PARALLEL_SIZE", 1)
PIPELINE_PARALLEL_SIZE = env_int("RAY_SERVE_PIPELINE_PARALLEL_SIZE", 2)
NUM_REPLICAS = env_int("RAY_SERVE_NUM_REPLICAS", 1)
GPUS_PER_REPLICA = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE


engine_kwargs: dict[str, Any] = {
    "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
    "pipeline_parallel_size": PIPELINE_PARALLEL_SIZE,
    "distributed_executor_backend": "ray",
    "max_model_len": MAX_MODEL_LEN,
    "trust_remote_code": True,
}

placement_group_config: dict[str, Any] | None = None
if GPUS_PER_REPLICA > 1:
    placement_group_config = {
        "bundles": [{"CPU": 1, "GPU": 1} for _ in range(GPUS_PER_REPLICA)],
        "strategy": os.environ.get("RAY_SERVE_PLACEMENT_STRATEGY", "STRICT_SPREAD"),
    }

accelerator_type = os.environ.get("RAY_SERVE_ACCELERATOR_TYPE")

llm_config_kwargs: dict[str, Any] = {
    "model_loading_config": {
        "model_id": MODEL_ID,
        "model_source": MODEL_SOURCE,
    },
    "deployment_config": {
        "num_replicas": NUM_REPLICAS,
        "max_ongoing_requests": env_int("RAY_SERVE_MAX_ONGOING_REQUESTS", 16),
    },
    "engine_kwargs": engine_kwargs,
    "runtime_env": {
        "env_vars": optional_env_vars("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    },
}

if accelerator_type:
    llm_config_kwargs["accelerator_type"] = accelerator_type

if placement_group_config is not None:
    llm_config_kwargs["placement_group_config"] = placement_group_config

print("[ray-serve-qwen3] Building Ray Serve LLM app", flush=True)
print(f"[ray-serve-qwen3] model_id={MODEL_ID}", flush=True)
print(f"[ray-serve-qwen3] model_source={MODEL_SOURCE}", flush=True)
print(f"[ray-serve-qwen3] num_replicas={NUM_REPLICAS}", flush=True)
print(f"[ray-serve-qwen3] gpus_per_replica={GPUS_PER_REPLICA}", flush=True)
print(f"[ray-serve-qwen3] engine_kwargs={engine_kwargs}", flush=True)
print(f"[ray-serve-qwen3] deployment_config={llm_config_kwargs['deployment_config']}", flush=True)
print(f"[ray-serve-qwen3] placement_group_config={placement_group_config}", flush=True)

llm_config = LLMConfig(**llm_config_kwargs)
app = build_openai_app({"llm_configs": [llm_config]})
