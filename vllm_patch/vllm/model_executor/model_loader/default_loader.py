# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType

import torch

from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.ram_stage import (
    RamStageConfig,
    ram_staged_safetensors_weights_iterator,
)

logger = init_logger(__name__)


def _load_upstream_default_loader() -> ModuleType:
    current_file = Path(__file__).resolve()
    for root in sys.modules["vllm"].__path__:
        candidate = Path(root) / "model_executor" / "model_loader" / "default_loader.py"
        if candidate.resolve() == current_file or not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "_vllm_upstream_default_loader",
            candidate,
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ImportError("Unable to locate upstream vLLM default_loader.py")


_upstream = _load_upstream_default_loader()


def _get_int_config(extra_config: dict, key: str, default: int) -> int:
    value = extra_config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _get_float_config(extra_config: dict, key: str, default: float) -> float:
    value = extra_config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(f"{key} must be a non-negative number")
    return float(value)


class DefaultModelLoader(_upstream.DefaultModelLoader):  # type: ignore[misc]
    def __init__(self, load_config: LoadConfig):
        BaseModelLoader.__init__(self, load_config)
        self.local_expert_ids: set[int] | None = None

        extra_config = load_config.model_loader_extra_config
        allowed_keys = {
            "enable_multithread_load",
            "num_threads",
            "enable_weights_track",
            "ram_stage_num_workers",
            "ram_stage_copy_delay",
            "ram_stage_small_file_threshold",
        }
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )

        self.enable_weights_track: bool | None = extra_config.get(
            "enable_weights_track", None
        )

    def _get_weights_iterator(
        self, source: "_upstream.DefaultModelLoader.Source"
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        extra_config = self.load_config.model_loader_extra_config
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if self.load_config.load_format == "npcache":
            assert use_safetensors is False
            weights_iterator = _upstream.np_cache_weights_iterator(
                source.model_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
            )
        elif use_safetensors:
            if self.load_config.load_format == "fastsafetensors":
                weights_iterator = _upstream.fastsafetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            elif self.load_config.load_format == "instanttensor":
                weights_iterator = _upstream.instanttensor_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            elif self.load_config.safetensors_load_strategy == "ram_stage":
                if extra_config.get("enable_multithread_load"):
                    logger.info_once(
                        "Using safetensors_load_strategy='ram_stage'; "
                        "bypassing enable_multithread_load to avoid loading "
                        "whole shards into CPU memory."
                    )
                stage_config = RamStageConfig(
                    num_workers=_get_int_config(
                        extra_config,
                        "ram_stage_num_workers",
                        8,
                    ),
                    copy_delay=_get_float_config(
                        extra_config,
                        "ram_stage_copy_delay",
                        0.0,
                    ),
                    small_file_threshold=_get_int_config(
                        extra_config,
                        "ram_stage_small_file_threshold",
                        10_000_000,
                    ),
                )
                weights_iterator = ram_staged_safetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    stage_config,
                    local_expert_ids=self.local_expert_ids,
                )
            else:
                if extra_config.get("enable_multithread_load"):
                    weights_iterator = (
                        _upstream.multi_thread_safetensors_weights_iterator(
                            hf_weights_files,
                            self.load_config.use_tqdm_on_load,
                            max_workers=extra_config.get(
                                "num_threads", self.DEFAULT_NUM_THREADS
                            ),
                        )
                    )
                else:
                    weights_iterator = _upstream.safetensors_weights_iterator(
                        hf_weights_files,
                        self.load_config.use_tqdm_on_load,
                        self.load_config.safetensors_load_strategy,
                        local_expert_ids=self.local_expert_ids,
                    )
        else:
            if extra_config.get("enable_multithread_load"):
                weights_iterator = _upstream.multi_thread_pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                    max_workers=extra_config.get(
                        "num_threads", self.DEFAULT_NUM_THREADS
                    ),
                )
            else:
                weights_iterator = _upstream.pt_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.pt_load_map_location,
                )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = _upstream.time.perf_counter()
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)


Source = DefaultModelLoader.Source
