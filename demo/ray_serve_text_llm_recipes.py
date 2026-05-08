"""Shared helpers for Ray Serve text-LLM recipes.

These recipes are intended to run from the workspace virtualenv created by
``./build.sh``. Use ``serve run`` from an activated venv; do not use plain
``uv run`` for serving demos.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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

DEFAULT_NODE_RESOURCES = (
    "node:100.96.5.35",
    "node:100.96.34.48",
    "node:100.96.31.61",
)


@dataclass(frozen=True)
class TextLLMRecipe:
    recipe_name: str
    model_id: str
    model_source: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    max_model_len: int = 4096
    max_ongoing_requests: int = 16
    trust_remote_code: bool = True
    placement_strategy: str = "PACK"
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    enforce_eager: bool | None = None


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def env_bool(name: str, default: bool | None) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def optional_env_vars(*names: str) -> dict[str, str]:
    return {name: value for name in names if (value := os.environ.get(name))}


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


def _engine_kwargs(recipe: TextLLMRecipe) -> tuple[dict[str, Any], int]:
    tensor_parallel_size = env_int(
        "RAY_SERVE_TENSOR_PARALLEL_SIZE", recipe.tensor_parallel_size
    )
    pipeline_parallel_size = env_int(
        "RAY_SERVE_PIPELINE_PARALLEL_SIZE", recipe.pipeline_parallel_size
    )
    max_model_len = env_int("RAY_SERVE_MAX_MODEL_LEN", recipe.max_model_len)
    trust_remote_code = env_bool(
        "RAY_SERVE_TRUST_REMOTE_CODE", recipe.trust_remote_code
    )
    gpu_memory_utilization = env_float(
        "RAY_SERVE_GPU_MEMORY_UTILIZATION", recipe.gpu_memory_utilization
    )
    enforce_eager = env_bool("RAY_SERVE_ENFORCE_EAGER", recipe.enforce_eager)
    dtype = os.environ.get("RAY_SERVE_DTYPE", recipe.dtype or "")

    engine_kwargs: dict[str, Any] = {
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "max_model_len": max_model_len,
        "trust_remote_code": trust_remote_code,
    }
    if gpu_memory_utilization is not None:
        engine_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    if enforce_eager is not None:
        engine_kwargs["enforce_eager"] = enforce_eager
    if dtype:
        engine_kwargs["dtype"] = dtype

    gpus_per_replica = tensor_parallel_size * pipeline_parallel_size
    if gpus_per_replica > 1:
        engine_kwargs["distributed_executor_backend"] = os.environ.get(
            "RAY_SERVE_DISTRIBUTED_EXECUTOR_BACKEND", "ray"
        )
    return engine_kwargs, gpus_per_replica


def _placement_group_config(
    *,
    gpus_per_replica: int,
    node_resource_key: str | None,
    placement_strategy: str,
) -> dict[str, Any] | None:
    if gpus_per_replica <= 1:
        return None

    bundles: list[dict[str, float]] = []
    node_resource_fraction = env_float("RAY_SERVE_NODE_RESOURCE_FRACTION", 0.001)
    for _ in range(gpus_per_replica):
        bundle: dict[str, float] = {"CPU": 1, "GPU": 1}
        if node_resource_key:
            bundle[node_resource_key] = node_resource_fraction or 0.001
        bundles.append(bundle)
    return {"bundles": bundles, "strategy": placement_strategy}


def _deployment_config(
    *,
    gpus_per_replica: int,
    node_resource_key: str | None,
    max_ongoing_requests: int,
) -> dict[str, Any]:
    deployment_config: dict[str, Any] = {
        "num_replicas": 1,
        "max_ongoing_requests": max_ongoing_requests,
    }
    if gpus_per_replica <= 1 and node_resource_key:
        node_resource_fraction = env_float("RAY_SERVE_NODE_RESOURCE_FRACTION", 0.001)
        deployment_config["ray_actor_options"] = {
            "resources": {node_resource_key: node_resource_fraction or 0.001},
        }
    return deployment_config


def build_text_llm_app(recipe: TextLLMRecipe):
    model_id = os.environ.get("RAY_SERVE_MODEL_ID", recipe.model_id)
    model_source = os.path.expanduser(
        os.environ.get("RAY_SERVE_MODEL_SOURCE", recipe.model_source)
    )
    data_parallel_size = env_int(
        "RAY_SERVE_DATA_PARALLEL_SIZE",
        env_int("RAY_SERVE_NUM_REPLICAS", recipe.data_parallel_size),
    )
    max_ongoing_requests = env_int(
        "RAY_SERVE_MAX_ONGOING_REQUESTS", recipe.max_ongoing_requests
    )
    placement_strategy = os.environ.get(
        "RAY_SERVE_PLACEMENT_STRATEGY", recipe.placement_strategy
    )
    node_resources = env_list("RAY_SERVE_NODE_RESOURCES", DEFAULT_NODE_RESOURCES)
    engine_kwargs, gpus_per_replica = _engine_kwargs(recipe)

    if data_parallel_size > len(node_resources):
        raise ValueError(
            "RAY_SERVE_NODE_RESOURCES must provide at least one node resource per "
            f"data-parallel replica. got={node_resources!r} "
            f"data_parallel_size={data_parallel_size}"
        )

    print(f"[ray-serve-recipe] recipe={recipe.recipe_name}", flush=True)
    print(f"[ray-serve-recipe] model_id={model_id}", flush=True)
    print(f"[ray-serve-recipe] model_source={model_source}", flush=True)
    print(f"[ray-serve-recipe] data_parallel_size={data_parallel_size}", flush=True)
    print(f"[ray-serve-recipe] gpus_per_replica={gpus_per_replica}", flush=True)
    print(
        f"[ray-serve-recipe] node_resources={node_resources[:data_parallel_size]}",
        flush=True,
    )
    print(f"[ray-serve-recipe] engine_kwargs={engine_kwargs}", flush=True)

    llm_deployments = []
    llm_configs: list[LLMConfig] = []
    for replica_index in range(data_parallel_size):
        node_resource_key = node_resources[replica_index]
        placement_group_config = _placement_group_config(
            gpus_per_replica=gpus_per_replica,
            node_resource_key=node_resource_key,
            placement_strategy=placement_strategy,
        )
        llm_config_kwargs: dict[str, Any] = {
            "model_loading_config": {
                "model_id": model_id,
                "model_source": model_source,
            },
            "deployment_config": _deployment_config(
                gpus_per_replica=gpus_per_replica,
                node_resource_key=node_resource_key,
                max_ongoing_requests=max_ongoing_requests,
            ),
            "engine_kwargs": engine_kwargs,
            "runtime_env": {
                "env_vars": optional_env_vars("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
            },
        }
        if placement_group_config is not None:
            llm_config_kwargs["placement_group_config"] = placement_group_config
        if accelerator_type := os.environ.get("RAY_SERVE_ACCELERATOR_TYPE"):
            llm_config_kwargs["accelerator_type"] = accelerator_type

        llm_config = LLMConfig(**llm_config_kwargs)
        llm_configs.append(llm_config)
        llm_deployments.append(
            build_llm_deployment(
                llm_config,
                name_prefix=f"{recipe.recipe_name}_dp{replica_index}:",
            )
        )

    ingress_cls = make_fastapi_ingress(RoundRobinOpenAiIngress)
    return serve.deployment(
        ingress_cls,
        **RoundRobinOpenAiIngress.get_deployment_options(llm_configs),
    ).bind(llm_deployments=llm_deployments)


def run_app(app) -> None:
    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    serve.run(app, blocking=True)
