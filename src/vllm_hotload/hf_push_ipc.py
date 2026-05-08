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
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

DEFAULT_PROMPT = "Explain SVD using the smallest possible 2D example. Keep it short."


def default_target_devices() -> str:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        parsed_devices = [
            device.strip()
            for device in visible_devices.split(",")
            if device.strip() and device.strip() != "-1"
        ]
        if parsed_devices:
            return ",".join(str(index) for index in range(len(parsed_devices)))

    if not torch.cuda.is_available():
        return "0"

    device_count = torch.cuda.device_count()
    if device_count < 1:
        return "0"
    return ",".join(str(index) for index in range(device_count))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Push an in-memory Transformers model into a managed vLLM server. "
            "This helper is primarily for weight transfer; pre/post generation checks are optional diagnostics."
        )
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local Hugging Face model path to load and push into the managed vLLM server.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL for the managed vLLM server, for example http://127.0.0.1:8000.",
    )
    parser.add_argument(
        "--served-model-name",
        required=True,
        help="Served model name exposed by the target vLLM server.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--target-devices",
        default=default_target_devices(),
        help=(
            "Comma-separated CUDA device indices used by the vLLM workers. "
            "Defaults to all CUDA devices visible to this process. "
            "For multi-GPU or data-parallel workers, include every worker device."
        ),
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--keep-alive", action="store_true")
    parser.add_argument(
        "--before-generate",
        dest="run_before_generate",
        action="store_true",
        help="Run a chat completion before weight transfer. Disabled by default for managed dummy pushes.",
    )
    parser.add_argument(
        "--skip-before-generate",
        dest="run_before_generate",
        action="store_false",
        help="Backward-compatible alias that keeps pre-transfer generation disabled.",
    )
    parser.add_argument("--skip-init-weight-transfer", action="store_true")
    parser.add_argument("--skip-prepare-weight-update", action="store_true")
    parser.add_argument("--skip-finish-weight-update", action="store_true")
    parser.add_argument(
        "--after-generate",
        dest="run_after_generate",
        action="store_true",
        help="Run a chat completion after weight transfer. Disabled by default unless explicitly requested.",
    )
    parser.add_argument(
        "--skip-after-generate",
        dest="run_after_generate",
        action="store_false",
        help="Backward-compatible alias that keeps post-transfer generation disabled.",
    )
    parser.add_argument(
        "--server-side-load",
        action="store_true",
        help=(
            "Delegate weight loading to the vLLM server instead of sending "
            "weights via CUDA IPC. The server reads the checkpoint directly "
            "from disk using its RamStageManager. Skips loading the HF model "
            "in the push process."
        ),
    )
    parser.set_defaults(run_before_generate=False, run_after_generate=False)
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


def resolve_model_loader(model_path: str):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    architectures = tuple(getattr(config, "architectures", ()) or ())
    if any(arch.endswith("ForConditionalGeneration") for arch in architectures):
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


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

    if not args.server_side_load:
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

    if args.server_side_load:
        print("Server-side load mode: skip loading HF model in push process")
        model = None
        target_devices = []
    else:
        target_devices = parse_target_devices(args.target_devices)
        model_loader = resolve_model_loader(model_path)
        print(
            "Loading model from %s on %s with %s"
            % (model_path, args.device, model_loader.__name__)
        )
        model = model_loader.from_pretrained(
            model_path,
            dtype=torch_dtype(args.dtype),
            trust_remote_code=True,
        )
        model.to(args.device)
        model.eval()

    client = OpenAI(base_url=f"{base_url}/v1", api_key="EMPTY")

    if args.run_before_generate:
        print("Generating before weight transfer:")
        print(chat_completion(client, args.served_model_name, args.prompt))

    if not args.skip_init_weight_transfer and not args.server_side_load:
        print("Initializing managed IPC weight transfer")
        print(managed_post(base_url, "init_weight_transfer", {"init_info": {}}))

    if not args.skip_prepare_weight_update:
        print("Preparing server for weight update")
        print(
            managed_post(
                base_url,
                "prepare_weight_update",
                {"sleep_level": 2, "wake_weights": True},
            )
        )

    if args.server_side_load:
        print(
            "Sending load_weights request to server for path %s" % model_path
        )
        print(
            managed_post(
                base_url,
                "load_weights",
                {"model_path": model_path},
            )
        )
    else:
        print(
            "Sending model weights via CUDA IPC to target devices %s"
            % ",".join(map(str, target_devices))
        )
        send_weights_http(base_url, model.named_parameters(), target_devices)

    if not args.skip_finish_weight_update:
        print("Finishing server weight update")
        print(
            managed_post(
                base_url,
                "finish_weight_update",
                {"wake_kv_cache": True, "resume": True},
            )
        )

    if args.run_after_generate:
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
