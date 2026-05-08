from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vllm_hotload import hotloadctl


def _default_served_model_name(model_path: str) -> str:
    name = Path(model_path).expanduser().name
    return name or "model"


def _add_common_state_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-file")


def _serve_command(args: argparse.Namespace) -> None:
    fast_loading_ram = args.fast_loading == "ram"
    hotload_args = [
        "start",
        "--nodes",
        args.nodes,
        "--gpus-per-replica",
        str(args.gpus_per_replica),
        "--model-path",
        args.model,
        "--served-model-name",
        args.served_model_name or _default_served_model_name(args.model),
        "--public-port",
        str(args.port),
        "--proxy-host",
        args.host,
        "--workspace-dir",
        args.workspace_dir,
        "--session-prefix",
        args.session_prefix,
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--ray-address",
        args.ray_address,
        "--dashboard-address",
        args.dashboard_address,
        "--ready-timeout-secs",
        str(args.ready_timeout_secs),
        "--request-timeout",
        str(args.request_timeout),
        "--ram-stage-num-workers",
        str(args.ram_stage_num_workers),
        "--ram-stage-copy-delay",
        str(args.ram_stage_copy_delay),
        "--ram-stage-small-file-threshold",
        str(args.ram_stage_small_file_threshold),
    ]
    if args.public_host:
        hotload_args.extend(["--public-host", args.public_host])
    if args.state_file:
        hotload_args.extend(["--state-file", args.state_file])
    if args.trust_remote_code:
        hotload_args.append("--trust-remote-code")
    if fast_loading_ram:
        hotload_args.append("--fast-loading-ram")
    if args.dry_run:
        hotload_args.append("--dry-run")
    if args.echo_commands:
        hotload_args.append("--echo-commands")

    hotloadctl.main(hotload_args)


def _forward_command(command: str, args: argparse.Namespace) -> None:
    forwarded = [command]
    if command == "push":
        forwarded.append(args.checkpoint)
    if command == "sleep":
        forwarded.extend(["--level", str(args.level)])
    if command == "wake" and args.tags:
        forwarded.extend(["--tags", args.tags])
    if args.state_file:
        forwarded.extend(["--state-file", args.state_file])
    if getattr(args, "request_timeout", None) is not None:
        forwarded.extend(["--request-timeout", str(args.request_timeout)])
    if getattr(args, "dry_run", False):
        forwarded.append("--dry-run")
    if getattr(args, "force", False):
        forwarded.append("--force")
    hotloadctl.main(forwarded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ray-vllm",
        description=(
            "Run managed-hotload vLLM replicas through Ray Serve with a "
            "vllm-serve-like CLI."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser(
        "serve",
        help="Start Ray Serve backed vLLM replicas and a public /v1 endpoint.",
    )
    serve.add_argument("model")
    serve.add_argument(
        "--nodes",
        default="localhost",
        help="Comma-separated Ray node hostnames or IPs. Defaults to localhost.",
    )
    serve.add_argument("--gpus-per-replica", type=int)
    serve.add_argument("--tensor-parallel-size", type=int, default=1)
    serve.add_argument("--served-model-name")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=hotloadctl.DEFAULT_PUBLIC_PORT)
    serve.add_argument("--public-host")
    serve.add_argument("--workspace-dir", default=str(Path.cwd()))
    serve.add_argument("--state-file")
    serve.add_argument("--session-prefix", default=hotloadctl.DEFAULT_SESSION_PREFIX)
    serve.add_argument("--dtype", default="bfloat16")
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.40)
    serve.add_argument("--max-model-len", type=int, default=4096)
    serve.add_argument("--trust-remote-code", action="store_true")
    serve.add_argument(
        "--fast-loading",
        choices=("off", "ram"),
        default="off",
        help="Use 'ram' to enable safetensors RAM staging, or 'off' for parity with vLLM.",
    )
    serve.add_argument(
        "--fast-loading-ram",
        action="store_const",
        dest="fast_loading",
        const="ram",
        help="Alias for --fast-loading ram.",
    )
    serve.add_argument(
        "--no-fast-loading",
        action="store_const",
        dest="fast_loading",
        const="off",
        help="Disable fast-loading RAM staging.",
    )
    serve.add_argument("--ram-stage-num-workers", type=int, default=8)
    serve.add_argument("--ram-stage-copy-delay", type=float, default=0.0)
    serve.add_argument("--ram-stage-small-file-threshold", type=int, default=10_000_000)
    serve.add_argument("--ray-address", default="auto")
    serve.add_argument(
        "--dashboard-address",
        default=hotloadctl.DEFAULT_DASHBOARD_ADDRESS,
    )
    serve.add_argument("--ready-timeout-secs", type=int, default=600)
    serve.add_argument(
        "--request-timeout",
        type=float,
        default=hotloadctl.DEFAULT_REQUEST_TIMEOUT,
    )
    serve.add_argument("--dry-run", action="store_true")
    serve.add_argument("--echo-commands", action="store_true")
    serve.set_defaults(handler=_serve_command)

    push = subparsers.add_parser(
        "push", help="Hot-reload a checkpoint on every replica."
    )
    push.add_argument("checkpoint")
    _add_common_state_arg(push)
    push.add_argument(
        "--request-timeout", type=float, default=hotloadctl.DEFAULT_REQUEST_TIMEOUT
    )
    push.add_argument("--dry-run", action="store_true")
    push.set_defaults(handler=lambda args: _forward_command("push", args))

    status = subparsers.add_parser("status", help="Show public and per-replica status.")
    _add_common_state_arg(status)
    status.add_argument(
        "--request-timeout",
        type=float,
        default=hotloadctl.DEFAULT_REQUEST_TIMEOUT,
    )
    status.set_defaults(handler=lambda args: _forward_command("status", args))

    sleep = subparsers.add_parser("sleep", help="Sleep every managed replica.")
    _add_common_state_arg(sleep)
    sleep.add_argument("--level", type=int, default=1)
    sleep.add_argument(
        "--request-timeout", type=float, default=hotloadctl.DEFAULT_REQUEST_TIMEOUT
    )
    sleep.add_argument("--dry-run", action="store_true")
    sleep.set_defaults(handler=lambda args: _forward_command("sleep", args))

    wake = subparsers.add_parser("wake", help="Wake every managed replica.")
    _add_common_state_arg(wake)
    wake.add_argument("--tags")
    wake.add_argument(
        "--request-timeout", type=float, default=hotloadctl.DEFAULT_REQUEST_TIMEOUT
    )
    wake.add_argument("--dry-run", action="store_true")
    wake.set_defaults(handler=lambda args: _forward_command("wake", args))

    stop = subparsers.add_parser("stop", help="Shutdown the owned Ray Serve apps.")
    _add_common_state_arg(stop)
    stop.add_argument("--dry-run", action="store_true")
    stop.add_argument("--force", action="store_true")
    stop.set_defaults(handler=lambda args: _forward_command("stop", args))

    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "command", None) == "serve":
        if args.gpus_per_replica is None:
            args.gpus_per_replica = args.tensor_parallel_size
    return args


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = normalize_args(parser.parse_args(argv))
    args.handler(args)


if __name__ == "__main__":
    main(sys.argv[1:])
