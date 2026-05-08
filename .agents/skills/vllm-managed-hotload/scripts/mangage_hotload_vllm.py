from __future__ import annotations

from pathlib import Path

from vllm_hotload.managed_hotload import (  # pyright: ignore[reportMissingImports]
    ManagedHotloadClient as _ManagedHotloadClient,
    ManagedHotloadDemo as _ManagedHotloadDemo,
    ManagedHotloadDemoConfig,
    describe_demo_config as _describe_demo_config,
)

BASE_DIR = Path('/home/anhvth8/vllm_projects')
PYTHON_BIN = BASE_DIR / '.venv/bin/python'
VLLM_PATCH_DIR = BASE_DIR / 'vllm_patch'
PUSH_SCRIPT = VLLM_PATCH_DIR / 'examples/managed_weight_sync/hf_push_ipc.py'
SERVER_LOG = BASE_DIR / 'PR/logs/hotload_vllm_notebook_server.log'

HOST = '127.0.0.1'
PORT = 8000
SERVED_MODEL_NAME = 'qwen3-1.7b'
TP_SIZE = 2
GPU_MEMORY_UTILIZATION = '0.40'
MAX_MODEL_LEN = '4096'

CHAT_MODEL_PATH = Path.home() / 'ckpt/hf_models/Qwen/Qwen3-1.7B'
BASE_MODEL_PATH = Path.home() / 'ckpt/hf_models/Qwen/Qwen3-1.7B-Base'

DEMO_CONFIG = ManagedHotloadDemoConfig(
    workspace_dir=BASE_DIR,
    python_bin=PYTHON_BIN,
    vllm_patch_dir=VLLM_PATCH_DIR,
    push_script=PUSH_SCRIPT,
    server_log=SERVER_LOG,
    host=HOST,
    port=PORT,
    served_model_name=SERVED_MODEL_NAME,
    tp_size=TP_SIZE,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    max_model_len=MAX_MODEL_LEN,
    chat_model_path=CHAT_MODEL_PATH,
    base_model_path=BASE_MODEL_PATH,
)
BASE_URL = DEMO_CONFIG.base_url


class ManagedHotloadClient(_ManagedHotloadClient):
    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        served_model_name: str = SERVED_MODEL_NAME,
        tp_size: int = TP_SIZE,
    ) -> None:
        super().__init__(
            base_url=base_url,
            served_model_name=served_model_name,
            tp_size=tp_size,
            workspace_dir=BASE_DIR,
            python_bin=PYTHON_BIN,
            vllm_patch_dir=VLLM_PATCH_DIR,
            push_script=PUSH_SCRIPT,
        )


class ManagedHotloadDemo(_ManagedHotloadDemo):
    def __init__(self) -> None:
        super().__init__(DEMO_CONFIG)


def describe_demo_config() -> dict[str, str]:
    return _describe_demo_config(DEMO_CONFIG, helper_file=__file__)


demo = ManagedHotloadDemo()

__all__ = [
    'BASE_DIR',
    'BASE_MODEL_PATH',
    'BASE_URL',
    'CHAT_MODEL_PATH',
    'DEMO_CONFIG',
    'GPU_MEMORY_UTILIZATION',
    'HOST',
    'MAX_MODEL_LEN',
    'ManagedHotloadClient',
    'ManagedHotloadDemo',
    'ManagedHotloadDemoConfig',
    'PORT',
    'PUSH_SCRIPT',
    'PYTHON_BIN',
    'SERVER_LOG',
    'SERVED_MODEL_NAME',
    'TP_SIZE',
    'VLLM_PATCH_DIR',
    'demo',
    'describe_demo_config',
]