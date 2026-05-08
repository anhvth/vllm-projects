from __future__ import annotations

import argparse
import json
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_PORT = 8100
DEFAULT_PUBLIC_PORT = 8000
DEFAULT_SESSION_PREFIX = "vllm-hotload"
DEFAULT_REQUEST_TIMEOUT = 30.0


@dataclass(frozen=True)
class ReplicaState:
    node: str
    base_url: str
    v1_url: str
    managed_url: str
    session_name: str
    gpus_per_replica: int
    gpu_devices: list[int]


@dataclass(frozen=True)
class ClusterState:
    workspace_dir: str
    public_base_url: str
    public_host: str
    public_port: int
    proxy_host: str
    proxy_session_name: str
    base_port: int
    served_model_name: str
    start_model_path: str
    gpus_per_replica: int
    nodes: list[str]
    replicas: list[ReplicaState]
    ssh_user: str | None = None


def default_state_file(workspace_dir: Path) -> Path:
    return workspace_dir / ".hotloadctl" / "state.json"


def parse_nodes(raw_nodes: str) -> list[str]:
    nodes = [node.strip() for node in raw_nodes.split(",") if node.strip()]
    if not nodes:
        raise ValueError("At least one node is required.")
    return nodes


def local_gpu_devices(gpus_per_replica: int) -> list[int]:
    if gpus_per_replica < 1:
        raise ValueError("gpus-per-replica must be at least 1.")
    return list(range(gpus_per_replica))


def build_cluster_state(args: argparse.Namespace) -> ClusterState:
    workspace_dir = Path(args.workspace_dir).resolve()
    public_host = args.public_host or socket.gethostname()
    public_base_url = f"http://{public_host}:{args.public_port}/v1"
    gpu_devices = local_gpu_devices(args.gpus_per_replica)
    replicas = [
        ReplicaState(
            node=node,
            base_url=f"http://{node}:{args.base_port}",
            v1_url=f"http://{node}:{args.base_port}/v1",
            managed_url=f"http://{node}:{args.base_port}/managed",
            session_name=f"{args.session_prefix}-{node}-serve",
            gpus_per_replica=args.gpus_per_replica,
            gpu_devices=gpu_devices,
        )
        for node in parse_nodes(args.nodes)
    ]
    return ClusterState(
        workspace_dir=str(workspace_dir),
        public_base_url=public_base_url,
        public_host=public_host,
        public_port=args.public_port,
        proxy_host=args.proxy_host,
        proxy_session_name=f"{args.session_prefix}-proxy",
        base_port=args.base_port,
        served_model_name=args.served_model_name,
        start_model_path=str(Path(args.model_path).expanduser()),
        gpus_per_replica=args.gpus_per_replica,
        ssh_user=args.ssh_user,
        nodes=[replica.node for replica in replicas],
        replicas=replicas,
    )


def save_state(state_file: Path, state: ClusterState) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n")


def load_state(state_file: Path) -> ClusterState:
    data = json.loads(state_file.read_text())
    return ClusterState(
        workspace_dir=data["workspace_dir"],
        public_base_url=data["public_base_url"],
        public_host=data["public_host"],
        public_port=data["public_port"],
        proxy_host=data["proxy_host"],
        proxy_session_name=data["proxy_session_name"],
        base_port=data["base_port"],
        served_model_name=data["served_model_name"],
        start_model_path=data["start_model_path"],
        gpus_per_replica=data["gpus_per_replica"],
        ssh_user=data.get("ssh_user"),
        nodes=data["nodes"],
        replicas=[ReplicaState(**replica) for replica in data["replicas"]],
    )


def shell_join(parts: list[str]) -> str:
    return shlex.join(parts)


def is_local_node(node: str) -> bool:
    local_names = {"127.0.0.1", "localhost", socket.gethostname(), socket.getfqdn()}
    return node in local_names


def ssh_target(node: str, ssh_user: str | None) -> str:
    return node if not ssh_user else f"{ssh_user}@{node}"


def build_replica_start_command(
    state: ClusterState,
    replica: ReplicaState,
    args: argparse.Namespace,
) -> str:
    workspace_dir = shlex.quote(state.workspace_dir)
    gpu_devices = ",".join(str(index) for index in replica.gpu_devices)
    serve_command = shell_join(
        [
            "uv",
            "run",
            "--directory",
            state.workspace_dir,
            "vllm",
            "serve",
            state.start_model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(state.base_port),
            "--served-model-name",
            state.served_model_name,
            "--dtype",
            args.dtype,
            "--load-format",
            "dummy",
            "--weight-transfer-config",
            '{"backend":"ipc"}',
            "--enable-sleep-mode",
            "--managed-weight-sync",
            "--tensor-parallel-size",
            str(replica.gpus_per_replica),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-model-len",
            str(args.max_model_len),
        ]
        + (["--trust-remote-code"] if args.trust_remote_code else [])
    )
    inner = (
        f"cd {workspace_dir} && "
        "unset VLLM_API_KEY && "
        "export VLLM_SERVER_DEV_MODE=1 && "
        "export VLLM_ALLOW_INSECURE_SERIALIZATION=1 && "
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(gpu_devices)} && "
        f"exec {serve_command}"
    )
    tmux_command = shell_join(
        ["tmux", "new-session", "-d", "-s", replica.session_name, inner]
    )
    if is_local_node(replica.node):
        return tmux_command

    return shell_join(["ssh", ssh_target(replica.node, args.ssh_user), tmux_command])


def build_proxy_start_command(state: ClusterState, state_file: Path) -> str:
    workspace_dir = shlex.quote(state.workspace_dir)
    proxy_command = shell_join(
        [
            "uv",
            "run",
            "--directory",
            state.workspace_dir,
            "hotloadctl",
            "_serve-proxy",
            "--state-file",
            str(state_file),
            "--host",
            state.proxy_host,
            "--port",
            str(state.public_port),
        ]
    )
    inner = f"cd {workspace_dir} && exec {proxy_command}"
    tmux_command = shell_join(
        ["tmux", "new-session", "-d", "-s", state.proxy_session_name, inner]
    )
    if is_local_node(state.public_host):
        return tmux_command
    return shell_join(
        ["ssh", ssh_target(state.public_host, state.ssh_user), tmux_command]
    )


def build_push_helper_command(
    state: ClusterState,
    replica: ReplicaState,
    checkpoint: str,
    args: argparse.Namespace,
) -> str:
    target_devices = ",".join(str(index) for index in replica.gpu_devices)
    helper_parts = [
        "uv",
        "run",
        "--directory",
        state.workspace_dir,
        "vllm-hotload-hf-push-ipc",
        "--model-path",
        checkpoint,
        "--base-url",
        f"http://127.0.0.1:{state.base_port}",
        "--served-model-name",
        state.served_model_name,
        "--dtype",
        args.dtype,
        "--skip-before-generate",
        "--skip-after-generate",
        "--skip-init-weight-transfer",
        "--skip-prepare-weight-update",
        "--skip-finish-weight-update",
    ]
    if args.server_side_load:
        helper_parts.append("--server-side-load")
    else:
        helper_parts.extend(["--target-devices", target_devices])

    helper_command = shell_join(helper_parts)
    inner = (
        f"cd {shlex.quote(state.workspace_dir)} && "
        "export VLLM_ALLOW_INSECURE_SERIALIZATION=1 && "
        f"exec {helper_command}"
    )
    if is_local_node(replica.node):
        return inner

    effective_ssh_user = args.ssh_user or state.ssh_user
    return shell_join(["ssh", ssh_target(replica.node, effective_ssh_user), inner])


def build_stop_command(node: str, session_name: str, ssh_user: str | None) -> str:
    tmux_command = shell_join(["tmux", "kill-session", "-t", session_name])
    if is_local_node(node):
        return tmux_command
    return shell_join(["ssh", ssh_target(node, ssh_user), tmux_command])


def run_shell_command(command: str, dry_run: bool, echo_commands: bool) -> None:
    if echo_commands or dry_run:
        print(command)
    if dry_run:
        return
    subprocess.run(command, shell=True, check=True)


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> Any:
    response = requests.request(method, url, json=payload, timeout=timeout)
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
        except requests.RequestException:
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
            request_json(
                "GET", f"{state.public_base_url}/models", None, request_timeout
            )
            return
        except requests.RequestException:
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
            "private_v1_url": replica.v1_url,
            "private_managed_url": replica.managed_url,
            "health": "healthy",
            "managed_status": managed,
        }
    except requests.RequestException as exc:
        return {
            "node": replica.node,
            "private_v1_url": replica.v1_url,
            "private_managed_url": replica.managed_url,
            "health": "unhealthy",
            "error": str(exc),
            "managed_status": None,
        }


def collect_cluster_status(
    state: ClusterState, request_timeout: float
) -> dict[str, Any]:
    return {
        "public_base_url": state.public_base_url,
        "replicas": [
            collect_replica_status(replica, request_timeout)
            for replica in state.replicas
        ],
    }


def emit_json(payload: Any) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def start_command(args: argparse.Namespace) -> None:
    state = build_cluster_state(args)
    state_file = Path(args.state_file).resolve()
    plan = {
        "public_base_url": state.public_base_url,
        "replicas": [asdict(replica) for replica in state.replicas],
        "commands": {
            replica.node: build_replica_start_command(state, replica, args)
            for replica in state.replicas
        },
        "proxy_command": build_proxy_start_command(state, state_file),
    }
    if args.dry_run:
        emit_json(plan)
        return

    save_state(state_file, state)
    for replica in state.replicas:
        run_shell_command(
            build_replica_start_command(state, replica, args),
            dry_run=False,
            echo_commands=args.echo_commands,
        )
    run_shell_command(
        build_proxy_start_command(state, state_file),
        dry_run=False,
        echo_commands=args.echo_commands,
    )
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
                "init_weight_transfer": f"{replica.managed_url}/init_weight_transfer",
                "prepare_weight_update": f"{replica.managed_url}/prepare_weight_update",
                "push_command": build_push_helper_command(
                    state, replica, checkpoint, args
                ),
                "finish_weight_update": f"{replica.managed_url}/finish_weight_update",
                "verify_models": f"{replica.v1_url}/models",
            }
        )
    if args.dry_run:
        emit_json(
            {
                "checkpoint": checkpoint,
                "operations": operations,
                "public_verify_models": f"{state.public_base_url}/models",
            }
        )
        return

    results = []
    for replica in state.replicas:
        init_result = request_json(
            "POST",
            f"{replica.managed_url}/init_weight_transfer",
            {"init_info": {}},
            args.request_timeout,
        )
        prepare_result = request_json(
            "POST",
            f"{replica.managed_url}/prepare_weight_update",
            {"sleep_level": 2, "wake_weights": True},
            args.request_timeout,
        )
        run_shell_command(
            build_push_helper_command(state, replica, checkpoint, args),
            dry_run=False,
            echo_commands=args.echo_commands,
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
                "init_weight_transfer": init_result,
                "prepare_weight_update": prepare_result,
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
    effective_ssh_user = args.ssh_user or state.ssh_user
    commands = {
        replica.node: build_stop_command(
            replica.node, replica.session_name, effective_ssh_user
        )
        for replica in state.replicas
    }
    proxy_command = build_stop_command(
        state.public_host, state.proxy_session_name, effective_ssh_user
    )
    if args.dry_run:
        emit_json({"commands": commands, "proxy_command": proxy_command})
        return

    for command in commands.values():
        run_shell_command(command, dry_run=False, echo_commands=args.echo_commands)
    run_shell_command(proxy_command, dry_run=False, echo_commands=args.echo_commands)
    state_file.unlink(missing_ok=True)
    emit_json({"stopped": True, "public_base_url": state.public_base_url})


def serve_proxy_command(args: argparse.Namespace) -> None:
    from vllm_hotload.proxy import run_proxy

    run_proxy(args.state_file, args.host, args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hotloadctl",
        description="Manage per-node managed-hotload vLLM replicas and the public proxy.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser(
        "start", help="Start per-node replicas and the public proxy."
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
    start.add_argument("--ssh-user")
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
    push.add_argument("--ssh-user")
    push.add_argument("--dtype", default="bfloat16")
    push.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    push.add_argument("--dry-run", action="store_true")
    push.add_argument("--echo-commands", action="store_true")
    push.add_argument(
        "--server-side-load",
        action="store_true",
        help=(
            "Delegate weight loading to the vLLM server. The server reads "
            "the checkpoint directly from disk using its RamStageManager."
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
        "stop", help="Stop per-node tmux sessions and the proxy."
    )
    stop.add_argument("--state-file")
    stop.add_argument("--ssh-user")
    stop.add_argument("--dry-run", action="store_true")
    stop.add_argument("--echo-commands", action="store_true")
    stop.set_defaults(handler=stop_command)

    proxy = subparsers.add_parser("_serve-proxy")
    proxy.add_argument("--state-file", required=True)
    proxy.add_argument("--host", required=True)
    proxy.add_argument("--port", required=True, type=int)
    proxy.set_defaults(handler=serve_proxy_command)

    return parser


def normalize_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
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
