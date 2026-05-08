"""Ray Serve LLM demo for Qwen3-1.7B on the existing Ray cluster.

Run from the workspace venv built by ./build.sh:

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

Defaults use three Ray nodes by starting three single-GPU deployments pinned to
three node resources. Override RAY_SERVE_NODE_RESOURCES with comma-separated
Ray node resources for a different cluster.
"""

from __future__ import annotations

import os
from typing import Any

import ray
from ray import serve
from ray.llm._internal.common.utils.lora_utils import get_base_model_id
from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.llm._internal.serve.core.ingress.ingress import (
    OpenAiIngress,
    make_fastapi_ingress,
)
from ray.llm._internal.serve.core.server.builder import build_llm_deployment
from ray.serve.handle import DeploymentHandle


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def optional_env_vars(*names: str) -> dict[str, str]:
    return {name: value for name in names if (value := os.environ.get(name))}


def env_list(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


class RoundRobinOpenAiIngress(OpenAiIngress):
    def __init__(
        self,
        llm_deployments: list[DeploymentHandle],
        **kwargs: Any,
    ) -> None:
        self._round_robin_handles: dict[str, list[DeploymentHandle]] = {}
        self._round_robin_indexes: dict[str, int] = {}
        super().__init__(llm_deployments, **kwargs)

    async def _setup_handle_and_config_maps(
        self,
        llm_deployments: list[DeploymentHandle],
    ) -> None:
        for handle in llm_deployments:
            llm_config = await handle.llm_config.remote()
            self._default_serve_handles.setdefault(llm_config.model_id, handle)
            self._llm_configs.setdefault(llm_config.model_id, llm_config)
            self._round_robin_handles.setdefault(llm_config.model_id, []).append(handle)
        self._init_completed.set()

    def _get_configured_serve_handle(self, model_id: str) -> DeploymentHandle:
        base_model_id = get_base_model_id(model_id)
        handles = self._round_robin_handles.get(base_model_id)
        if not handles:
            return super()._get_configured_serve_handle(model_id)

        index = self._round_robin_indexes.get(model_id, 0)
        self._round_robin_indexes[model_id] = index + 1
        handle = handles[index % len(handles)]
        if model_id == base_model_id:
            return handle.options(stream=True)
        return handle.options(stream=True, multiplexed_model_id=model_id)


MODEL_ID = os.environ.get("RAY_SERVE_MODEL_ID", "qwen3-1.7b")
MODEL_SOURCE = os.path.expanduser(
    os.environ.get("RAY_SERVE_MODEL_SOURCE", "~/ckpt/hf_models/Qwen/Qwen3-1.7B")
)
MAX_MODEL_LEN = env_int("RAY_SERVE_MAX_MODEL_LEN", 4096)
TENSOR_PARALLEL_SIZE = env_int("RAY_SERVE_TENSOR_PARALLEL_SIZE", 1)
PIPELINE_PARALLEL_SIZE = env_int("RAY_SERVE_PIPELINE_PARALLEL_SIZE", 1)
NUM_REPLICAS = env_int("RAY_SERVE_NUM_REPLICAS", 3)
GPUS_PER_REPLICA = TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE
NODE_RESOURCE_KEYS = env_list(
    "RAY_SERVE_NODE_RESOURCES",
    (
        "node:100.96.5.35",
        "node:100.96.34.48",
        "node:100.96.31.61",
    ),
)

if len(NODE_RESOURCE_KEYS) < NUM_REPLICAS:
    raise ValueError(
        "RAY_SERVE_NODE_RESOURCES must provide at least one node resource per "
        f"replica. got={NODE_RESOURCE_KEYS!r} num_replicas={NUM_REPLICAS}"
    )


engine_kwargs: dict[str, Any] = {
    "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
    "pipeline_parallel_size": PIPELINE_PARALLEL_SIZE,
    "max_model_len": MAX_MODEL_LEN,
    "trust_remote_code": True,
}
if GPUS_PER_REPLICA > 1:
    engine_kwargs["distributed_executor_backend"] = "ray"

placement_group_config: dict[str, Any] | None = None
if GPUS_PER_REPLICA > 1:
    placement_group_config = {
        "bundles": [{"CPU": 1, "GPU": 1} for _ in range(GPUS_PER_REPLICA)],
        "strategy": os.environ.get("RAY_SERVE_PLACEMENT_STRATEGY", "STRICT_SPREAD"),
    }

accelerator_type = os.environ.get("RAY_SERVE_ACCELERATOR_TYPE")

max_ongoing_requests = env_int("RAY_SERVE_MAX_ONGOING_REQUESTS", 16)
llm_config_kwargs: dict[str, Any] = {
    "model_loading_config": {
        "model_id": MODEL_ID,
        "model_source": MODEL_SOURCE,
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
print(
    f"[ray-serve-qwen3] node_resources={NODE_RESOURCE_KEYS[:NUM_REPLICAS]}",
    flush=True,
)
print(f"[ray-serve-qwen3] engine_kwargs={engine_kwargs}", flush=True)
print(f"[ray-serve-qwen3] placement_group_config={placement_group_config}", flush=True)

llm_configs: list[LLMConfig] = []
for replica_index, node_resource_key in enumerate(NODE_RESOURCE_KEYS[:NUM_REPLICAS]):
    replica_kwargs = {
        **llm_config_kwargs,
        "deployment_config": {
            "num_replicas": 1,
            "max_ongoing_requests": max_ongoing_requests,
            "ray_actor_options": {
                "resources": {node_resource_key: 0.001},
            },
        },
    }
    if placement_group_config is not None:
        replica_kwargs["placement_group_config"] = placement_group_config
    if accelerator_type:
        replica_kwargs["accelerator_type"] = accelerator_type
    llm_configs.append(LLMConfig(**replica_kwargs))

llm_deployments = [
    build_llm_deployment(
        llm_config,
        name_prefix=f"LLMServerNode{replica_index}:",
    )
    for replica_index, llm_config in enumerate(llm_configs)
]

app = serve.deployment(
    make_fastapi_ingress(RoundRobinOpenAiIngress),
    **RoundRobinOpenAiIngress.get_deployment_options(llm_configs),
).bind(llm_deployments=llm_deployments)


def main() -> None:
    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))

    serve.run(app, blocking=True)


if __name__ == "__main__":
    main()
