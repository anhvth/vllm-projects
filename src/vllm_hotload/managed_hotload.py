from __future__ import annotations

import base64
import importlib
import json
import os
import pickle
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ManagedHotloadDemoConfig:
    workspace_dir: Path
    python_bin: Path
    vllm_patch_dir: Path
    push_script: Path
    server_log: Path
    host: str
    port: int
    served_model_name: str
    tp_size: int
    gpu_memory_utilization: str
    max_model_len: str
    chat_model_path: Path
    base_model_path: Path

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def dummy_load_format(self) -> str:
        return "dummy"

    def required_paths(self) -> tuple[Path, ...]:
        return (
            self.python_bin,
            self.vllm_patch_dir / "vllm",
            self.push_script,
            self.chat_model_path,
            self.base_model_path,
        )


def validate_demo_config(config: ManagedHotloadDemoConfig) -> None:
    for required_path in config.required_paths():
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    config.server_log.parent.mkdir(parents=True, exist_ok=True)


def describe_demo_config(
    config: ManagedHotloadDemoConfig,
    *,
    helper_file: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    validate_demo_config(config)
    description = {
        "base_url": config.base_url,
        "served_model_name": config.served_model_name,
        "dummy_load_format": config.dummy_load_format,
        "base_model_path": str(config.base_model_path),
        "chat_model_path": str(config.chat_model_path),
    }
    if helper_file is not None:
        description["helper_file"] = str(helper_file)
    return description


class ManagedHotloadClient:
    def __init__(
        self,
        *,
        base_url: str,
        served_model_name: str,
        tp_size: int,
        workspace_dir: Path,
        python_bin: Path,
        vllm_patch_dir: Path,
        push_script: Path,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.served_model_name = served_model_name
        self.tp_size = tp_size
        self.workspace_dir = Path(workspace_dir)
        self.python_bin = Path(python_bin)
        self.vllm_patch_dir = Path(vllm_patch_dir)
        self.push_script = Path(push_script)

    def get_json(self, path: str, timeout: int = 30) -> dict[str, Any]:
        request = Request(f"{self.base_url}{path}", method="GET")
        return self._load_json(request, timeout=timeout)

    def post_json(
        self,
        path: str,
        payload: dict[str, Any],
        timeout: int,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._load_json(request, timeout=timeout)

    def _load_json(self, request: Request, timeout: int) -> dict[str, Any]:
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(exc.read().decode("utf-8")) from exc
        except URLError as exc:
            raise RuntimeError(str(exc)) from exc

    def _managed_post(
        self,
        path: str,
        payload: dict[str, Any],
        timeout: int = 300,
    ) -> dict[str, Any]:
        return self.post_json(f"/managed/{path}", payload, timeout=timeout)

    def _target_devices(self) -> list[int]:
        return list(range(self.tp_size))

    def status(self) -> dict[str, Any]:
        return self.get_json("/managed/status")

    def world_size(self) -> dict[str, Any]:
        return self.get_json("/managed/world_size")

    def models(self) -> dict[str, Any]:
        return self.get_json("/v1/models")

    def pause(self) -> dict[str, Any]:
        return self._managed_post("pause", {})

    def resume(self) -> dict[str, Any]:
        return self._managed_post("resume", {})

    def sleep(self, level: int = 1) -> dict[str, Any]:
        return self._managed_post("sleep", {"level": level})

    def wake(self, tags: list[str] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if tags is not None:
            payload["tags"] = tags
        return self._managed_post("wake", payload)

    def init_weight_transfer(
        self,
        init_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._managed_post(
            "init_weight_transfer",
            {"init_info": init_info or {}},
        )

    def prepare_weight_update(
        self,
        *,
        sleep_level: int | None = 2,
        wake_weights: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"wake_weights": wake_weights}
        if sleep_level is not None:
            payload["sleep_level"] = sleep_level
        return self._managed_post("prepare_weight_update", payload)

    def finish_weight_update(
        self,
        *,
        wake_kv_cache: bool = True,
        resume: bool = True,
    ) -> dict[str, Any]:
        return self._managed_post(
            "finish_weight_update",
            {"wake_kv_cache": wake_kv_cache, "resume": resume},
        )

    def _push_model_module(self, model: Any) -> str:
        torch = importlib.import_module("torch")
        reductions = importlib.import_module("torch.multiprocessing.reductions")
        reduce_tensor = reductions.reduce_tensor

        named_parameters = getattr(model, "named_parameters", None)
        if not callable(named_parameters):
            raise TypeError(
                "push() expects either a model path or an in-memory module with "
                "named_parameters()."
            )

        named_parameters_fn = cast(
            Callable[[], Iterable[tuple[str, Any]]],
            named_parameters,
        )

        if hasattr(model, "eval"):
            model.eval()

        init_response = self.init_weight_transfer()
        prepare_response = self.prepare_weight_update(
            sleep_level=2,
            wake_weights=True,
        )

        target_devices = self._target_devices()
        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        ipc_handles: list[dict[str, Any]] = []
        retained_tensors: list[Any] = []

        for name, tensor in named_parameters_fn():
            names.append(name)
            dtype_names.append(str(tensor.dtype).split(".")[-1])
            shapes.append(list(tensor.shape))

            per_device_handles = {}
            for device_index in target_devices:
                target = torch.device("cuda", device_index)
                if tensor.device == target and tensor.is_contiguous():
                    weight = tensor.detach()
                else:
                    weight = tensor.detach().to(target, non_blocking=True).contiguous()
                retained_tensors.append(weight)
                gpu_uuid = str(torch.cuda.get_device_properties(device_index).uuid)
                per_device_handles[gpu_uuid] = reduce_tensor(weight)
            ipc_handles.append(per_device_handles)

        try:
            update_response = self.post_json(
                "/update_weights",
                {
                    "update_info": {
                        "names": names,
                        "dtype_names": dtype_names,
                        "shapes": shapes,
                        "ipc_handles_pickled": base64.b64encode(
                            pickle.dumps(ipc_handles)
                        ).decode("utf-8"),
                    }
                },
                timeout=300,
            )
        finally:
            retained_tensors.clear()

        finish_response = self.finish_weight_update(
            wake_kv_cache=True,
            resume=True,
        )

        return "\n".join(
            [
                "Transferred in-memory model weights from the notebook process.",
                f"Parameters sent: {len(names)}",
                f"Target devices: {target_devices}",
                f"init_weight_transfer: {json.dumps(init_response)}",
                f"prepare_weight_update: {json.dumps(prepare_response)}",
                f"update_weights: {json.dumps(update_response)}",
                f"finish_weight_update: {json.dumps(finish_response)}",
            ]
        )

    def push(self, model: Path | str | os.PathLike[str] | Any) -> str:
        if hasattr(model, "named_parameters"):
            return self._push_model_module(model)

        model_path = Path(model).expanduser()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.vllm_patch_dir)
        env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        command = [
            str(self.python_bin),
            str(self.push_script),
            "--model-path",
            str(model_path),
            "--base-url",
            self.base_url,
            "--served-model-name",
            self.served_model_name,
            "--target-devices",
            ",".join(str(index) for index in self._target_devices()),
            "--skip-before-generate",
        ]
        result = subprocess.run(
            command,
            cwd=self.workspace_dir,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout


class ManagedHotloadDemo(ManagedHotloadClient):
    def __init__(self, config: ManagedHotloadDemoConfig) -> None:
        super().__init__(
            base_url=config.base_url,
            served_model_name=config.served_model_name,
            tp_size=config.tp_size,
            workspace_dir=config.workspace_dir,
            python_bin=config.python_bin,
            vllm_patch_dir=config.vllm_patch_dir,
            push_script=config.push_script,
        )
        self.config = config
        self.server_process: subprocess.Popen[bytes] | None = None
        self.server_log_handle: BinaryIO | None = None

    def wait_for_ready(self, timeout_seconds: int = 600) -> None:
        deadline = time.time() + timeout_seconds
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                self.status()
                return
            except Exception as exc:  # pragma: no cover - readiness is environment-driven.
                last_error = exc
                time.sleep(2)
        raise TimeoutError(f"Server did not become ready: {last_error}")

    def start_dummy_service(self) -> dict[str, Any]:
        try:
            return self.status()
        except Exception:
            pass

        validate_demo_config(self.config)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.config.vllm_patch_dir)
        env["VLLM_SERVER_DEV_MODE"] = "1"
        env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        env.pop("VLLM_API_KEY", None)

        self.server_log_handle = self.config.server_log.open("ab")
        command = [
            str(self.config.python_bin),
            "-m",
            "vllm.entrypoints.openai.api_server",
            str(self.config.chat_model_path),
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--served-model-name",
            self.config.served_model_name,
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
            "--load-format",
            self.config.dummy_load_format,
            "--weight-transfer-config",
            '{"backend":"ipc"}',
            "--enable-sleep-mode",
            "--managed-weight-sync",
            "--tensor-parallel-size",
            str(self.config.tp_size),
            "--gpu-memory-utilization",
            self.config.gpu_memory_utilization,
            "--max-model-len",
            self.config.max_model_len,
        ]

        self.server_process = subprocess.Popen(
            command,
            cwd=self.config.workspace_dir,
            env=env,
            stdout=self.server_log_handle,
            stderr=subprocess.STDOUT,
        )
        self.wait_for_ready()
        return self.status()

    def stop(self) -> str:
        stopped = False
        if self.server_process is not None and self.server_process.poll() is None:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait(timeout=30)
            self.server_process = None
            stopped = True

        if self.server_log_handle is not None:
            self.server_log_handle.close()
            self.server_log_handle = None

        if stopped:
            return f"Stopped notebook-started server. Log: {self.config.server_log}"
        return f"No live notebook-started server to stop. Log: {self.config.server_log}"