# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterator

import pybase64 as base64
import requests
import torch
from openai import OpenAI
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

DEFAULT_MODEL_PATH = "~/ckpt/hf_models/Qwen/Qwen3-1.7B/"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_PROMPT = "Explain SVD using the smallest possible 2D example. Keep it short."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push an in-memory Transformers model into a managed vLLM server."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--served-model-name", default="qwen3-1.7b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--target-devices",
        default="0",
        help=(
            "Comma-separated CUDA device indices used by the vLLM workers. "
            "For tensor parallel size 2 on local GPUs 0 and 1, use '0,1'."
        ),
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--keep-alive", action="store_true")
    parser.add_argument("--skip-before-generate", action="store_true")
    return parser.parse_args()


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def parse_target_devices(target_devices: str) -> list[int]:
    devices = [int(device.strip()) for device in target_devices.split(",")]
    if not devices:
        raise ValueError("At least one target device is required.")
    return devices


def managed_post(base_url: str, path: str, payload: dict) -> dict:
    response = requests.post(
        f"{base_url}/managed/{path}",
        json=payload,
        timeout=300,
    )
    if not response.ok:
        print(response.text)
    response.raise_for_status()
    return response.json()


def chat_completion(
    client: OpenAI,
    served_model_name: str,
    prompt: str,
) -> str:
    response = client.chat.completions.create(
        model=served_model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content or ""


def _gpu_uuid(device_index: int) -> str:
    props = torch.cuda.get_device_properties(device_index)
    return str(props.uuid)


def iter_ipc_payload(
    iterator: Iterator[tuple[str, torch.Tensor]],
    target_devices: list[int],
) -> tuple[list[str], list[str], list[list[int]], list[dict], list[torch.Tensor]]:
    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    ipc_handles: list[dict] = []
    retained_tensors: list[torch.Tensor] = []

    for name, tensor in iterator:
        names.append(name)
        dtype_names.append(str(tensor.dtype).split(".")[-1])
        shapes.append(list(tensor.shape))

        per_device_handles = {}
        for device_index in target_devices:
            target = torch.device("cuda", device_index)  # pyright: ignore[reportPrivateImportUsage]
            if tensor.device == target and tensor.is_contiguous():
                weight = tensor.detach()
            else:
                weight = tensor.detach().to(target, non_blocking=True).contiguous()
            retained_tensors.append(weight)
            per_device_handles[_gpu_uuid(device_index)] = reduce_tensor(weight)
        ipc_handles.append(per_device_handles)

    return names, dtype_names, shapes, ipc_handles, retained_tensors


def send_weights_http(
    base_url: str,
    iterator: Iterator[tuple[str, torch.Tensor]],
    target_devices: list[int],
) -> None:
    import pickle

    names, dtype_names, shapes, ipc_handles, retained_tensors = iter_ipc_payload(
        iterator, target_devices
    )
    try:
        pickled_handles = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")
        payload = {
            "update_info": {
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "ipc_handles_pickled": pickled_handles,
            }
        }
        response = requests.post(
            f"{base_url}/update_weights", json=payload, timeout=300
        )
        if not response.ok:
            print(response.text)
        response.raise_for_status()
    finally:
        retained_tensors.clear()


def main() -> None:
    args = parse_args()
    base_url = normalize_base_url(args.base_url)
    model_path = os.path.expanduser(args.model_path)
    target_devices = parse_target_devices(args.target_devices)

    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    messages = [{"role": "user", "content": args.prompt}]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print("Rendered chat prompt:")
    print(rendered_prompt)

    print(f"Loading model from {model_path} on {args.device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype(args.dtype),
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()

    client = OpenAI(base_url=f"{base_url}/v1", api_key="EMPTY")

    if not args.skip_before_generate:
        print("Generating before weight transfer:")
        print(chat_completion(client, args.served_model_name, args.prompt))

    print("Initializing managed IPC weight transfer")
    print(managed_post(base_url, "init_weight_transfer", {"init_info": {}}))

    print("Preparing server for weight update")
    print(
        managed_post(
            base_url,
            "prepare_weight_update",
            {"sleep_level": 2, "wake_weights": True},
        )
    )

    print(
        "Sending model weights via CUDA IPC to target devices "
        f"{','.join(map(str, target_devices))}"
    )
    send_weights_http(base_url, model.named_parameters(), target_devices)

    print("Finishing server weight update")
    print(
        managed_post(
            base_url,
            "finish_weight_update",
            {"wake_kv_cache": True, "resume": True},
        )
    )

    print("Generating after weight transfer:")
    print(chat_completion(client, args.served_model_name, args.prompt))

    if args.keep_alive:
        print("Keeping the source model process alive. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("Exiting.")


if __name__ == "__main__":
    main()