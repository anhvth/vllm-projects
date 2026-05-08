from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


def _requests() -> type:
    import requests as _r

    return _r


def _requests_exc() -> tuple:
    import requests as _r

    return _r.RequestException, _r.ConnectionError


DEFAULT_BASE_PORT = 8100
DEFAULT_PUBLIC_PORT = 8000
DEFAULT_SESSION_PREFIX = "vllm-hotload"
DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_DASHBOARD_ADDRESS = "http://localhost:8265"
DEFAULT_PRIVATE_ROUTE_ROOT = "/_hotloadctl/replicas"
DEFAULT_SERVE_CONFIG_NAME = "serve_config.yaml"
DEFAULT_PROXY_LOCATION = "EveryNode"
DEFAULT_REPLICA_IMPORT_PATH = "vllm_hotload.ray_serve_app:build_vllm_replica_app"
DEFAULT_PUBLIC_PROXY_IMPORT_PATH = "vllm_hotload.ray_serve_app:build_public_proxy_app"
DEFAULT_REPLICA_DEPLOYMENT_NAME = "ManagedVLLMReplica"
DEFAULT_PROXY_DEPLOYMENT_NAME = "HotloadPublicProxy"
DEFAULT_SERVER_LOAD_FORMAT = "safetensors"


@dataclass(frozen=True)
class ReplicaState:
    node: str
    route_prefix: str
    base_url: str
    v1_url: str
    managed_url: str
    app_name: str
    gpus_per_replica: int


@dataclass(frozen=True)
class ClusterState:
    workspace_dir: str
    ray_address: str
    dashboard_address: str
    public_base_url: str
    public_host: str
    public_port: int
    proxy_host: str
    public_app_name: str
    public_route_prefix: str
    serve_config_path: str
    served_model_name: str
    start_model_path: str
    gpus_per_replica: int
    nodes: list[str]
    replicas: list[ReplicaState]


def default_state_file(workspace_dir: Path) -> Path:
    return workspace_dir / ".hotloadctl" / "state.json"


def default_serve_config_file(workspace_dir: Path) -> Path:
    return workspace_dir / ".hotloadctl" / DEFAULT_SERVE_CONFIG_NAME


def parse_nodes(raw_nodes: str) -> list[str]:
    nodes = [node.strip() for node in raw_nodes.split(",") if node.strip()]
    if not nodes:
        raise ValueError("At least one node is required.")
    return nodes


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9-]+", "-", value).strip("-").lower()


def _build_replica_urls(host: str, port: int, route_prefix: str) -> tuple[str, str, str]:
    base_url = f"http://{host}:{port}{route_prefix}"
    return base_url, f"{base_url}/v1", f"{base_url}/managed"


def build_cluster_state(args: argparse.Namespace) -> ClusterState:
    workspace_dir = Path(args.workspace_dir).resolve()
    public_host = args.public_host or socket.gethostname()
    replicas: list[ReplicaState] = []

    for index, node in enumerate(parse_nodes(args.nodes), start=1):
        node_slug = _slugify(node) or f"node-{index}"
        route_prefix = f"{DEFAULT_PRIVATE_ROUTE_ROOT}/{index}-{node_slug}"
        base_url, v1_url, managed_url = _build_replica_urls(
            public_host,
            args.public_port,
            route_prefix,
        )
        replicas.append(
            ReplicaState(
                node=node,
                route_prefix=route_prefix,
                base_url=base_url,
                v1_url=v1_url,
                managed_url=managed_url,
                app_name=f"{args.session_prefix}-replica-{index}-{node_slug}",
                gpus_per_replica=args.gpus_per_replica,
            )
        )

    return ClusterState(
        workspace_dir=str(workspace_dir),
        ray_address=args.ray_address,
        dashboard_address=args.dashboard_address,
        public_base_url=f"http://{public_host}:{args.public_port}/v1",
        public_host=public_host,
        public_port=args.public_port,
        proxy_host=args.proxy_host,
        public_app_name=f"{args.session_prefix}-public",
        public_route_prefix="/",
        serve_config_path=str(default_serve_config_file(workspace_dir)),
        served_model_name=args.served_model_name,
        start_model_path=str(Path(args.model_path).expanduser()),
        gpus_per_replica=args.gpus_per_replica,
        nodes=[replica.node for replica in replicas],
        replicas=replicas,
    )


def save_state(state_file: Path, state: ClusterState) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n")


def _infer_route_prefix(base_url: str) -> str:
    path = urlparse(base_url).path
    return path or "/"


def load_state(state_file: Path) -> ClusterState:
    data = json.loads(state_file.read_text())
    workspace_dir = Path(data["workspace_dir"])
    return ClusterState(
        workspace_dir=data["workspace_dir"],
        ray_address=data.get("ray_address", "auto"),
        dashboard_address=data.get("dashboard_address", DEFAULT_DASHBOARD_ADDRESS),
        public_base_url=data["public_base_url"],
        public_host=data["public_host"],
        public_port=data["public_port"],
        proxy_host=data.get("proxy_host", "0.0.0.0"),
        public_app_name=data.get("public_app_name", f"{DEFAULT_SESSION_PREFIX}-public"),
        public_route_prefix=data.get("public_route_prefix", "/"),
        serve_config_path=data.get(
            "serve_config_path",
            str(default_serve_config_file(workspace_dir)),
        ),
        served_model_name=data["served_model_name"],
        start_model_path=data["start_model_path"],
        gpus_per_replica=data["gpus_per_replica"],
        nodes=data["nodes"],
        replicas=[
            ReplicaState(
                node=replica["node"],
                route_prefix=replica.get(
                    "route_prefix",
                    _infer_route_prefix(replica["base_url"]),
                ),
                base_url=replica["base_url"],
                v1_url=replica["v1_url"],
                managed_url=replica["managed_url"],
                app_name=replica.get(
                    "app_name",
                    replica.get("session_name", replica["node"]),
                ),
                gpus_per_replica=replica.get(
                    "gpus_per_replica",
                    data["gpus_per_replica"],
                ),
            )
            for replica in data["replicas"]
        ],
    )


def shell_join(parts: list[str]) -> str:
    return shlex.join(parts)


def resolve_node_ip(node: str) -> str:
    if node in {"127.0.0.1", "localhost"}:
        return "127.0.0.1"
    return socket.gethostbyname(node)


def _workspace_pythonpath(state: ClusterState) -> str:
    parts = [
        str(Path(state.workspace_dir) / "src"),
        str(Path(state.workspace_dir) / "vllm_patch"),
    ]
    existing = os.environ.get("PYTHONPATH", "").strip()
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _cli_env(state: ClusterState) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _workspace_pythonpath(state)
    return env


def _deployment_env_vars(state: ClusterState) -> dict[str, str]:
    return {
        "PYTHONPATH": _workspace_pythonpath(state),
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    }


def _workspace_bin(state: ClusterState, name: str) -> str:
    return str(Path(state.workspace_dir) / ".venv" / "bin" / name)


def _format_command_failure(context: str, exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return f"{context} failed because '{exc.filename}' was not found."

    if isinstance(exc, subprocess.TimeoutExpired):
        return f"{context} timed out after {exc.timeout} seconds."

    if isinstance(exc, subprocess.CalledProcessError):
        output = "\n".join(
            part.strip()
            for part in (exc.stdout or "", exc.stderr or "")
            if part and part.strip()
        )
        lowered = output.lower()
        if "version" in lowered and "mismatch" in lowered:
            return f"{context} failed due to a Ray version mismatch.\n{output}"
        if output:
            return f"{context} failed.\n{output}"
        return f"{context} failed with exit code {exc.returncode}."

    return f"{context} failed: {exc}"


def _run_cli(
    state: ClusterState,
    command: list[str],
    *,
    timeout: int = 30,
    echo_commands: bool = False,
) -> subprocess.CompletedProcess[str]:
    if echo_commands:
        print(shell_join(command))

    try:
        return subprocess.run(
            command,
            cwd=state.workspace_dir,
            env=_cli_env(state),
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise RuntimeError(_format_command_failure(shell_join(command), exc)) from exc


def build_ray_status_command(state: ClusterState) -> list[str]:
    return [_workspace_bin(state, "ray"), "status", "--address", state.ray_address]


def build_serve_status_command(state: ClusterState) -> list[str]:
    return [
        _workspace_bin(state, "serve"),
        "status",
        "--address",
        state.dashboard_address,
    ]


def build_serve_run_command(state: ClusterState) -> list[str]:
    return [
        _workspace_bin(state, "serve"),
        "run",
        "--app-dir",
        str(Path(state.workspace_dir) / "src"),
        "--address",
        state.ray_address,
        "--non-blocking",
        state.serve_config_path,
    ]


def build_serve_shutdown_command(state: ClusterState) -> list[str]:
    return [
        _workspace_bin(state, "serve"),
        "shutdown",
        "--yes",
        "--address",
        state.dashboard_address,
    ]


def preflight_start(state: ClusterState, echo_commands: bool) -> None:
    checks = [
        [_workspace_bin(state, "ray"), "--version"],
        build_ray_status_command(state),
        build_serve_status_command(state),
    ]
    for command in checks:
        _run_cli(state, command, timeout=30, echo_commands=echo_commands)


def _replica_internal_urls(
    public_port: int,
    route_prefix: str,
) -> tuple[str, str, str]:
    return _build_replica_urls("127.0.0.1", public_port, route_prefix)


def _owned_app_names(state: ClusterState) -> set[str]:
    return {state.public_app_name, *(replica.app_name for replica in state.replicas)}


def _node_resource_key(node: str, strict: bool) -> str:
    try:
        return f"node:{resolve_node_ip(node)}"
    except socket.gaierror as exc:
        if strict:
            raise ValueError(f"Unable to resolve node '{node}' to an IP address.") from exc
        return f"node:{node}"


def build_serve_config(
    state: ClusterState,
    args: argparse.Namespace,
    *,
    strict_node_resolution: bool,
) -> dict[str, Any]:
    deployment_runtime_env = {"env_vars": _deployment_env_vars(state)}
    proxy_replicas = []
    for replica in state.replicas:
        base_url, v1_url, managed_url = _replica_internal_urls(
            state.public_port,
            replica.route_prefix,
        )
        proxy_replicas.append(
            {
                "node": replica.node,
                "base_url": base_url,
                "v1_url": v1_url,
                "managed_url": managed_url,
            }
        )

    applications: list[dict[str, Any]] = [
        {
            "name": state.public_app_name,
            "route_prefix": state.public_route_prefix,
            "import_path": DEFAULT_PUBLIC_PROXY_IMPORT_PATH,
            "args": {"replicas": proxy_replicas},
            "deployments": [
                {
                    "name": DEFAULT_PROXY_DEPLOYMENT_NAME,
                    "num_replicas": 1,
                    "ray_actor_options": {
                        "num_cpus": 0.1,
                        "runtime_env": deployment_runtime_env,
                    },
                }
            ],
        }
    ]

    for replica in state.replicas:
        applications.append(
            {
                "name": replica.app_name,
                "route_prefix": replica.route_prefix,
                "import_path": DEFAULT_REPLICA_IMPORT_PATH,
                "args": {
                    "model_path": state.start_model_path,
                    "served_model_name": state.served_model_name,
                    "gpus_per_replica": replica.gpus_per_replica,
                    "dtype": args.dtype,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "max_model_len": args.max_model_len,
                    "route_prefix": replica.route_prefix,
                    "trust_remote_code": args.trust_remote_code,
                },
                "deployments": [
                    {
                        "name": DEFAULT_REPLICA_DEPLOYMENT_NAME,
                        "num_replicas": 1,
                        "max_replicas_per_node": 1,
                        "ray_actor_options": {
                            "num_gpus": replica.gpus_per_replica,
                            "resources": {
                                _node_resource_key(
                                    replica.node,
                                    strict_node_resolution,
                                ): 0.001,
                            },
                            "runtime_env": deployment_runtime_env,
                        },
                    }
                ],
            }
        )

    return {
        "proxy_location": DEFAULT_PROXY_LOCATION,
        "http_options": {
            "host": state.proxy_host,
            "port": state.public_port,
        },
        "applications": applications,
    }


def write_serve_config(serve_config_path: Path, serve_config: dict[str, Any]) -> None:
    serve_config_path.parent.mkdir(parents=True, exist_ok=True)
    serve_config_path.write_text(yaml.safe_dump(serve_config, sort_keys=False))


def _serve_status(state: ClusterState, echo_commands: bool = False) -> dict[str, Any]:
    completed = _run_cli(
        state,
        build_serve_status_command(state),
        timeout=30,
        echo_commands=echo_commands,
    )
    data = yaml.safe_load(completed.stdout) or {}
    if not isinstance(data, dict):
        return {}
    return data


def wait_for_serve_apps(
    state: ClusterState,
    timeout_secs: int,
    echo_commands: bool,
) -> None:
    owned_app_names = _owned_app_names(state)
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            status = _serve_status(state, echo_commands=echo_commands)
        except RuntimeError:
            time.sleep(2)
            continue

        applications = status.get("applications") or {}
        if all(applications.get(name, {}).get("status") == "RUNNING" for name in owned_app_names):
            return

        failed = {
            name: applications.get(name, {})
            for name in owned_app_names
            if applications.get(name, {}).get("status") == "DEPLOY_FAILED"
        }
        if failed:
            raise RuntimeError(
                "Serve reported a deployment failure for: "
                + ", ".join(sorted(failed))
            )

        time.sleep(2)

    raise TimeoutError(
        "Timed out waiting for Ray Serve applications to reach RUNNING state."
    )


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> Any:
    r = _requests()
    response = r.request(method, url, json=payload, timeout=timeout)
    response.raise_for_status()
    if response.content:
        return response.json()
    return None


def wait_for_replica(
    replica: ReplicaState, timeout_secs: int, request_timeout: float
) -> None:
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            request_json("GET", f"{replica.managed_url}/status", None, request_timeout)
            request_json("GET", f"{replica.v1_url}/models", None, request_timeout)
            return
        except _requests_exc()[0]:
            time.sleep(2)
    raise TimeoutError(
        f"Timed out waiting for replica {replica.node} at {replica.base_url}"
    )


def wait_for_public_proxy(
    state: ClusterState, timeout_secs: int, request_timeout: float
) -> None:
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            request_json("GET", f"{state.public_base_url}/models", None, request_timeout)
            return
        except _requests_exc()[0]:
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for public proxy {state.public_base_url}")


def collect_replica_status(
    replica: ReplicaState, request_timeout: float
) -> dict[str, Any]:
    try:
        managed = request_json(
            "GET", f"{replica.managed_url}/status", None, request_timeout
        )
        request_json("GET", f"{replica.v1_url}/models", None, request_timeout)
        return {
            "node": replica.node,
            "route_prefix": replica.route_prefix,
            "private_v1_url": replica.v1_url,
            "private_managed_url": replica.managed_url,
            "health": "healthy",
            "managed_status": managed,
        }
    except _requests_exc()[0] as exc:
        return {
            "node": replica.node,
            "route_prefix": replica.route_prefix,
            "private_v1_url": replica.v1_url,
            "private_managed_url": replica.managed_url,
            "health": "unhealthy",
            "error": str(exc),
            "managed_status": None,
        }


def collect_cluster_status(
    state: ClusterState, request_timeout: float
) -> dict[str, Any]:
    payload = {
        "public_base_url": state.public_base_url,
        "replicas": [
            collect_replica_status(replica, request_timeout)
            for replica in state.replicas
        ],
    }

    try:
        serve_status = _serve_status(state)
    except RuntimeError:
        return payload

    applications = serve_status.get("applications") or {}
    payload["serve_applications"] = {
        name: {
            "status": details.get("status"),
            "message": details.get("message"),
        }
        for name, details in applications.items()
        if name in _owned_app_names(state)
    }
    return payload


def emit_json(payload: Any) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def start_command(args: argparse.Namespace) -> None:
    state = build_cluster_state(args)
    state_file = Path(args.state_file).resolve()
    serve_config = build_serve_config(
        state,
        args,
        strict_node_resolution=not args.dry_run,
    )
    plan = {
        "public_base_url": state.public_base_url,
        "dashboard_address": state.dashboard_address,
        "serve_config_path": state.serve_config_path,
        "replicas": [asdict(replica) for replica in state.replicas],
        "serve_config": serve_config,
        "serve_run_command": shell_join(build_serve_run_command(state)),
    }
    if args.dry_run:
        emit_json(plan)
        return

    preflight_start(state, args.echo_commands)
    write_serve_config(Path(state.serve_config_path), serve_config)
    _run_cli(
        state,
        build_serve_run_command(state),
        timeout=60,
        echo_commands=args.echo_commands,
    )
    save_state(state_file, state)
    wait_for_serve_apps(state, args.ready_timeout_secs, args.echo_commands)
    for replica in state.replicas:
        wait_for_replica(replica, args.ready_timeout_secs, args.request_timeout)
    wait_for_public_proxy(state, args.ready_timeout_secs, args.request_timeout)
    emit_json(collect_cluster_status(state, args.request_timeout))


def push_command(args: argparse.Namespace) -> None:
    state = load_state(Path(args.state_file).resolve())
    checkpoint = str(Path(args.checkpoint).expanduser())
    operations: list[dict[str, Any]] = []
    for replica in state.replicas:
        operations.append(
            {
                "node": replica.node,
                "prepare_weight_update": {
                    "url": f"{replica.managed_url}/prepare_weight_update",
                    "payload": {"sleep_level": 2, "wake_weights": True},
                },
                "load_weights": {
                    "url": f"{replica.managed_url}/load_weights",
                    "payload": {
                        "model_path": checkpoint,
                        "load_format": DEFAULT_SERVER_LOAD_FORMAT,
                    },
                },
                "finish_weight_update": {
                    "url": f"{replica.managed_url}/finish_weight_update",
                    "payload": {"wake_kv_cache": True, "resume": True},
                },
                "verify_models": f"{replica.v1_url}/models",
            }
        )
    if args.dry_run:
        emit_json(
            {
                "checkpoint": checkpoint,
                "mode": "server-side-load",
                "operations": operations,
                "public_verify_models": f"{state.public_base_url}/models",
            }
        )
        return

    results = []
    for replica in state.replicas:
        prepare_result = request_json(
            "POST",
            f"{replica.managed_url}/prepare_weight_update",
            {"sleep_level": 2, "wake_weights": True},
            args.request_timeout,
        )
        load_result = request_json(
            "POST",
            f"{replica.managed_url}/load_weights",
            {
                "model_path": checkpoint,
                "load_format": DEFAULT_SERVER_LOAD_FORMAT,
            },
            args.request_timeout,
        )
        finish_result = request_json(
            "POST",
            f"{replica.managed_url}/finish_weight_update",
            {"wake_kv_cache": True, "resume": True},
            args.request_timeout,
        )
        verify_result = request_json(
            "GET", f"{replica.v1_url}/models", None, args.request_timeout
        )
        results.append(
            {
                "node": replica.node,
                "prepare_weight_update": prepare_result,
                "load_weights": load_result,
                "finish_weight_update": finish_result,
                "verify_models": verify_result,
            }
        )
    public_verify_result = request_json(
        "GET", f"{state.public_base_url}/models", None, args.request_timeout
    )
    emit_json(
        {
            "checkpoint": checkpoint,
            "mode": "server-side-load",
            "results": results,
            "public_verify_models": public_verify_result,
        }
    )


def status_command(args: argparse.Namespace) -> None:
    state = load_state(Path(args.state_file).resolve())
    emit_json(collect_cluster_status(state, args.request_timeout))


def fanout_command(
    args: argparse.Namespace, path: str, payload: dict[str, Any]
) -> None:
    state = load_state(Path(args.state_file).resolve())
    if args.dry_run:
        emit_json(
            {
                "operation": path,
                "requests": [
                    {
                        "node": replica.node,
                        "url": f"{replica.managed_url}/{path}",
                        "payload": payload,
                    }
                    for replica in state.replicas
                ],
            }
        )
        return

    results = []
    for replica in state.replicas:
        results.append(
            {
                "node": replica.node,
                "response": request_json(
                    "POST",
                    f"{replica.managed_url}/{path}",
                    payload,
                    args.request_timeout,
                ),
            }
        )
    emit_json({"operation": path, "results": results})


def stop_command(args: argparse.Namespace) -> None:
    state_file = Path(args.state_file).resolve()
    state = load_state(state_file)
    shutdown_command = build_serve_shutdown_command(state)
    payload = {
        "owned_app_names": sorted(_owned_app_names(state)),
        "shutdown_command": shell_join(shutdown_command),
    }
    if args.dry_run:
        emit_json(payload)
        return

    serve_status = _serve_status(state, echo_commands=args.echo_commands)
    applications = serve_status.get("applications") or {}
    foreign_app_names = sorted(
        name for name in applications if name not in _owned_app_names(state)
    )
    if foreign_app_names and not args.force:
        raise RuntimeError(
            "Refusing to shut down Ray Serve because non-hotload applications "
            f"are still running: {', '.join(foreign_app_names)}. Use --force to override."
        )

    if applications:
        _run_cli(
            state,
            shutdown_command,
            timeout=60,
            echo_commands=args.echo_commands,
        )
    state_file.unlink(missing_ok=True)
    emit_json(
        {
            "stopped": True,
            "public_base_url": state.public_base_url,
            "foreign_app_names": foreign_app_names,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hotloadctl",
        description=(
            "Manage per-node managed-hotload vLLM replicas and the public proxy "
            "through Ray Serve applications."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser(
        "start", help="Submit per-node replicas and the public proxy to Ray Serve."
    )
    start.add_argument("--nodes", required=True)
    start.add_argument("--gpus-per-replica", required=True, type=int)
    start.add_argument("--model-path", required=True)
    start.add_argument("--served-model-name", required=True)
    start.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    start.add_argument("--public-port", type=int, default=DEFAULT_PUBLIC_PORT)
    start.add_argument("--public-host")
    start.add_argument("--proxy-host", default="0.0.0.0")
    start.add_argument("--workspace-dir", default=str(Path.cwd()))
    start.add_argument("--state-file")
    start.add_argument("--session-prefix", default=DEFAULT_SESSION_PREFIX)
    start.add_argument("--dtype", default="bfloat16")
    start.add_argument("--gpu-memory-utilization", type=float, default=0.40)
    start.add_argument("--max-model-len", type=int, default=4096)
    start.add_argument("--trust-remote-code", action="store_true")
    start.add_argument("--ray-address", default="auto")
    start.add_argument("--dashboard-address", default=DEFAULT_DASHBOARD_ADDRESS)
    start.add_argument("--ready-timeout-secs", type=int, default=600)
    start.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    start.add_argument("--dry-run", action="store_true")
    start.add_argument("--echo-commands", action="store_true")
    start.set_defaults(handler=start_command)

    push = subparsers.add_parser(
        "push", help="Fan out a checkpoint push to every replica."
    )
    push.add_argument("checkpoint")
    push.add_argument("--state-file")
    push.add_argument("--dtype", default="bfloat16")
    push.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    push.add_argument("--dry-run", action="store_true")
    push.add_argument("--echo-commands", action="store_true")
    push.add_argument(
        "--server-side-load",
        action="store_true",
        help=(
            "Deprecated compatibility flag. hotloadctl push now uses the "
            "managed server-side load endpoint for Serve deployments."
        ),
    )
    push.set_defaults(handler=push_command)

    status = subparsers.add_parser("status", help="Show public and per-replica status.")
    status.add_argument("--state-file")
    status.add_argument(
        "--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT
    )
    status.set_defaults(handler=status_command)

    sleep_parser = subparsers.add_parser(
        "sleep", help="Fan out managed sleep to every replica."
    )
    sleep_parser.add_argument("--state-file")
    sleep_parser.add_argument("--level", type=int, default=1)
    sleep_parser.add_argument(
        "--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT
    )
    sleep_parser.add_argument("--dry-run", action="store_true")
    sleep_parser.set_defaults(
        handler=lambda args: fanout_command(args, "sleep", {"level": args.level})
    )

    wake = subparsers.add_parser("wake", help="Fan out managed wake to every replica.")
    wake.add_argument("--state-file")
    wake.add_argument("--tags")
    wake.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    wake.add_argument("--dry-run", action="store_true")
    wake.set_defaults(
        handler=lambda args: fanout_command(
            args,
            "wake",
            {
                "tags": None
                if not args.tags
                else [tag.strip() for tag in args.tags.split(",") if tag.strip()]
            },
        )
    )

    stop = subparsers.add_parser(
        "stop", help="Shutdown the owned Ray Serve applications for this hotload cluster."
    )
    stop.add_argument("--state-file")
    stop.add_argument("--dry-run", action="store_true")
    stop.add_argument("--echo-commands", action="store_true")
    stop.add_argument("--force", action="store_true")
    stop.set_defaults(handler=stop_command)

    return parser


def normalize_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    del parser
    if getattr(args, "state_file", None) is None:
        workspace_dir = Path(getattr(args, "workspace_dir", Path.cwd())).resolve()
        args.state_file = str(default_state_file(workspace_dir))
    return args


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args = normalize_args(parser, args)
    args.handler(args)


if __name__ == "__main__":
    main()
