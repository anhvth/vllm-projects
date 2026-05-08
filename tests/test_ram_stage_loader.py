from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.ram_stage import (
    RamStageConfig,
    RamStageManager,
    ram_staged_safetensors_weights_iterator,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _write_shard(path: Path, name: str, value: float) -> None:
    save_file({name: torch.tensor([value])}, path)


def test_ram_stage_iterator_yields_tensors_and_cleans_tmp(tmp_path: Path) -> None:
    shard_1 = tmp_path / "model-00001-of-00002.safetensors"
    shard_2 = tmp_path / "model-00002-of-00002.safetensors"
    _write_shard(shard_1, "first", 1.0)
    _write_shard(shard_2, "second", 2.0)

    manager: RamStageManager | None = None
    real_manager = RamStageManager

    class TrackingRamStageManager(real_manager):
        def __enter__(self) -> "TrackingRamStageManager":
            nonlocal manager
            manager = self
            return super().__enter__()

    with mock.patch(
        "vllm.model_executor.model_loader.ram_stage.RamStageManager",
        TrackingRamStageManager,
    ):
        weights = list(
            ram_staged_safetensors_weights_iterator(
                [str(shard_2), str(shard_1)],
                use_tqdm_on_load=False,
                config=RamStageConfig(num_workers=2),
            )
        )

    assert [name for name, _ in weights] == ["first", "second"]
    assert [tensor.item() for _, tensor in weights] == [1.0, 2.0]
    assert manager is not None
    manager.wait_for_cleanup(timeout=5)
    assert not Path(manager.staged_files[0]).parent.exists()


def test_ram_stage_copy_failure_keeps_symlink_fallback(tmp_path: Path) -> None:
    shard = tmp_path / "model-00001-of-00001.safetensors"
    _write_shard(shard, "weight", 3.0)

    manager: RamStageManager | None = None
    real_copy2 = "vllm.model_executor.model_loader.ram_stage.shutil.copy2"

    class TrackingRamStageManager(RamStageManager):
        def __enter__(self) -> "TrackingRamStageManager":
            nonlocal manager
            manager = self
            return super().__enter__()

    with (
        mock.patch(
            "vllm.model_executor.model_loader.ram_stage.RamStageManager",
            TrackingRamStageManager,
        ),
        mock.patch(real_copy2, side_effect=OSError("no space left")),
    ):
        weights = list(
            ram_staged_safetensors_weights_iterator(
                [str(shard)],
                use_tqdm_on_load=False,
                config=RamStageConfig(num_workers=1),
            )
        )

    assert weights[0][0] == "weight"
    assert weights[0][1].item() == 3.0
    assert manager is not None
    manager.wait_for_cleanup(timeout=5)
    stage_dir = Path(manager.staged_files[0]).parent
    assert not list(stage_dir.glob("*.partial"))


def test_ram_stage_cli_and_loader_extra_config_are_accepted() -> None:
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "facebook/opt-125m",
            "--safetensors-load-strategy",
            "ram_stage",
        ]
    )

    assert args.safetensors_load_strategy == "ram_stage"

    loader = DefaultModelLoader(
        LoadConfig(
            load_format="safetensors",
            safetensors_load_strategy="ram_stage",
            model_loader_extra_config={
                "ram_stage_num_workers": 2,
                "ram_stage_copy_delay": 0.0,
                "ram_stage_small_file_threshold": 16,
            },
        )
    )
    assert loader.enable_weights_track is None


def test_ram_stage_loader_rejects_unknown_extra_config() -> None:
    with pytest.raises(ValueError, match="Unexpected extra config keys"):
        DefaultModelLoader(
            LoadConfig(
                load_format="safetensors",
                safetensors_load_strategy="ram_stage",
                model_loader_extra_config={"surprise": True},
            )
        )


def test_ram_stage_bypasses_multithread_full_shard_loader(tmp_path: Path) -> None:
    shard = tmp_path / "model-00001-of-00001.safetensors"
    _write_shard(shard, "weight", 4.0)
    loader = DefaultModelLoader(
        LoadConfig(
            load_format="safetensors",
            safetensors_load_strategy="ram_stage",
            model_loader_extra_config={
                "enable_multithread_load": True,
                "ram_stage_num_workers": 1,
            },
            use_tqdm_on_load=False,
        )
    )
    source = DefaultModelLoader.Source(
        model_or_path=str(tmp_path),
        revision=None,
        fall_back_to_pt=False,
    )

    with (
        mock.patch.object(
            loader,
            "_prepare_weights",
            return_value=(str(tmp_path), [str(shard)], True),
        ),
        mock.patch(
            "vllm.model_executor.model_loader.default_loader."
            "_upstream.multi_thread_safetensors_weights_iterator",
            side_effect=AssertionError("should not use full-shard loader"),
        ),
    ):
        weights = list(loader._get_weights_iterator(source))

    assert weights[0][0] == "weight"
    assert weights[0][1].item() == 4.0


def test_ram_stage_copy_progress_bar_has_clear_name(tmp_path: Path) -> None:
    shard = tmp_path / "model-00001-of-00001.safetensors"
    _write_shard(shard, "weight", 5.0)
    progress_calls: list[dict[str, object]] = []

    class FakeTqdm:
        def __init__(self, **kwargs: object) -> None:
            progress_calls.append(kwargs)
            self.n = 0
            self.closed = False

        def update(self, value: int) -> None:
            self.n += value

        def close(self) -> None:
            self.closed = True

    with (
        mock.patch(
            "vllm.model_executor.model_loader.ram_stage.enable_tqdm",
            return_value=True,
        ),
        mock.patch(
            "vllm.model_executor.model_loader.ram_stage.tqdm",
            FakeTqdm,
        ),
    ):
        manager = RamStageManager(
            [str(shard)],
            RamStageConfig(num_workers=1),
            use_tqdm_on_load=True,
        )
        manager.__enter__()
        assert manager._thread is not None
        manager._thread.join(timeout=5)
        manager.close()
        manager.wait_for_cleanup(timeout=5)

    assert progress_calls[0]["desc"] == "Ram-stage realcopy safetensors"
    assert progress_calls[0]["total"] == 1
    assert progress_calls[0]["disable"] is False
