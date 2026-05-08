from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI
from ray import serve
from ray.serve import Application
from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_hotload.proxy import create_app_from_replicas

DEFAULT_PROXY_DEPLOYMENT_NAME = "HotloadPublicProxy"
DEFAULT_REPLICA_DEPLOYMENT_NAME = "ManagedVLLMReplica"


def _set_managed_env() -> None:
    os.environ.setdefault("VLLM_SERVER_DEV_MODE", "1")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _parse_vllm_serve_args(argv: list[str]):
    _set_managed_env()
    parser = make_arg_parser(FlexibleArgumentParser(prog="vllm serve"))
    args = parser.parse_args(argv)
    validate_parsed_serve_args(args)
    return args


def _bind_fastapi_app(app: FastAPI, deployment_name: str) -> Application:
    @serve.deployment(name=deployment_name)
    @serve.ingress(app)
    class FastAPIIngress:
        pass

    return FastAPIIngress.bind()


def build_vllm_replica_app(
    *,
    model_path: str,
    served_model_name: str,
    gpus_per_replica: int,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.40,
    max_model_len: int = 4096,
    route_prefix: str = "/",
    trust_remote_code: bool = False,
    fast_loading_ram: bool = False,
    ram_stage_num_workers: int = 8,
    ram_stage_copy_delay: float = 0.0,
    ram_stage_small_file_threshold: int = 10_000_000,
) -> Application:
    argv = [
        model_path,
        "--served-model-name",
        served_model_name,
        "--dtype",
        dtype,
        "--load-format",
        "dummy",
        "--weight-transfer-config",
        json.dumps({"backend": "ipc"}),
        "--enable-sleep-mode",
        "--managed-weight-sync",
        "--tensor-parallel-size",
        str(gpus_per_replica),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--root-path",
        route_prefix,
    ]
    if trust_remote_code:
        argv.append("--trust-remote-code")
    if fast_loading_ram:
        argv.extend(
            [
                "--safetensors-load-strategy",
                "ram_stage",
                "--model-loader-extra-config",
                json.dumps(
                    {
                        "ram_stage_num_workers": ram_stage_num_workers,
                        "ram_stage_copy_delay": ram_stage_copy_delay,
                        "ram_stage_small_file_threshold": (
                            ram_stage_small_file_threshold
                        ),
                    }
                ),
            ]
        )

    app = api_server.build_app(_parse_vllm_serve_args(argv))
    return _bind_fastapi_app(app, DEFAULT_REPLICA_DEPLOYMENT_NAME)


def build_public_proxy_app(replicas: list[dict[str, Any]]) -> Application:
    return _bind_fastapi_app(
        create_app_from_replicas(replicas),
        DEFAULT_PROXY_DEPLOYMENT_NAME,
    )
