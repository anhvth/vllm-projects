# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import (
    _BAR_FORMAT,
    _natural_sort_key,
    enable_tqdm,
    safetensors_weights_iterator,
)
from tqdm.auto import tqdm

logger = init_logger(__name__)


@dataclass(frozen=True)
class RamStageConfig:
    num_workers: int = 8
    copy_delay: float = 0.0
    small_file_threshold: int = 10_000_000


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def _is_real_ok(target: Path, src: Path) -> bool:
    src_size = _file_size(src)
    return src_size >= 0 and not target.is_symlink() and _file_size(target) == src_size


def _format_bytes(size: int | float) -> str:
    if size >= 1 << 30:
        return f"{size / (1 << 30):.1f} GiB"
    if size >= 1 << 20:
        return f"{size / (1 << 20):.0f} MiB"
    return f"{size / (1 << 10):.0f} KiB"


class RamStageManager:
    def __init__(
        self,
        source_files: list[str],
        config: RamStageConfig,
        use_tqdm_on_load: bool,
    ) -> None:
        self._source_files = [Path(path).resolve() for path in source_files]
        self._config = config
        self._use_tqdm_on_load = use_tqdm_on_load
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cleanup_thread: threading.Thread | None = None
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self.staged_files: list[str] = []

    def __enter__(self) -> "RamStageManager":
        self._tempdir = tempfile.TemporaryDirectory(prefix="vllm-ram-stage-")
        stage_dir = Path(self._tempdir.name)
        pairs: list[tuple[Path, Path]] = []

        for index, src in enumerate(self._source_files):
            target = stage_dir / f"{index:05d}-{src.name}"
            target.symlink_to(src)
            self.staged_files.append(str(target))
            pairs.append((src, target))

        total_bytes = sum(max(_file_size(path), 0) for path in self._source_files)
        logger.info(
            "Ram-staging %d safetensors shards (%s) in %s",
            len(self._source_files),
            _format_bytes(total_bytes),
            stage_dir,
        )

        self._thread = threading.Thread(
            target=self._copy_files,
            args=(pairs,),
            name="vllm-ram-stage-copy",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._stop.set()
        tempdir = self._tempdir
        copy_thread = self._thread
        if tempdir is None:
            return

        self._tempdir = None

        def _cleanup_when_idle() -> None:
            if copy_thread is not None:
                copy_thread.join()
            tempdir.cleanup()

        self._cleanup_thread = threading.Thread(
            target=_cleanup_when_idle,
            name="vllm-ram-stage-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

    def wait_for_cleanup(self, timeout: float | None = None) -> None:
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=timeout)

    def _copy_files(self, pairs: list[tuple[Path, Path]]) -> None:
        if not pairs:
            return
        if self._config.copy_delay > 0:
            deadline = time.monotonic() + self._config.copy_delay
            while time.monotonic() < deadline:
                if self._stop.is_set():
                    return
                time.sleep(0.1)

        ordered = sorted(pairs, key=self._copy_priority_key)
        queue: Queue[tuple[Path, Path]] = Queue()
        for pair in ordered:
            queue.put(pair)

        stats = {"copied": 0, "skipped": 0, "bytes": 0, "errors": 0}
        lock = threading.Lock()
        started = time.monotonic()
        progress = tqdm(
            total=len(ordered),
            desc="Ram-stage realcopy safetensors",
            disable=not enable_tqdm(self._use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
        )

        def _worker() -> None:
            while not self._stop.is_set():
                try:
                    src, target = queue.get(timeout=0.2)
                except Empty:
                    return
                try:
                    copied_bytes = self._copy_one(src, target)
                    with lock:
                        if copied_bytes is None:
                            stats["skipped"] += 1
                        else:
                            stats["copied"] += 1
                            stats["bytes"] += copied_bytes
                except Exception as err:
                    partial = target.with_suffix(target.suffix + ".partial")
                    partial.unlink(missing_ok=True)
                    with lock:
                        stats["errors"] += 1
                        if not self._stop.is_set() and stats["errors"] <= 3:
                            logger.warning(
                                "Ram-stage copy failed for %s: %s. "
                                "Keeping symlink fallback.",
                                src,
                                err,
                            )
                finally:
                    with lock:
                        progress.update(1)
                    queue.task_done()

        try:
            worker_count = max(1, self._config.num_workers)
            threads = [
                threading.Thread(
                    target=_worker,
                    name="vllm-ram-stage-worker",
                    daemon=True,
                )
                for _ in range(min(worker_count, len(ordered)))
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            progress.close()

        elapsed = time.monotonic() - started
        rate = stats["bytes"] / elapsed if elapsed > 0 else 0
        logger.info(
            "Ram-stage background copy finished: copied=%d (%s), skipped=%d, "
            "errors=%d, time=%.2fs (%s/s)",
            stats["copied"],
            _format_bytes(stats["bytes"]),
            stats["skipped"],
            stats["errors"],
            elapsed,
            _format_bytes(rate),
        )

    def _copy_one(self, src: Path, target: Path) -> int | None:
        if self._stop.is_set() or _is_real_ok(target, src):
            return None

        partial = target.with_suffix(target.suffix + ".partial")
        partial.unlink(missing_ok=True)
        shutil.copy2(src, partial)
        if self._stop.is_set():
            partial.unlink(missing_ok=True)
            return None
        os.replace(partial, target)
        return max(_file_size(target), 0)

    def _copy_priority_key(self, pair: tuple[Path, Path]) -> tuple[int, str]:
        src, _ = pair
        size = _file_size(src)
        name = src.name.lower()
        if size >= 0 and size < self._config.small_file_threshold:
            return (0, name)
        return (1, name)


def ram_staged_safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    config: RamStageConfig,
    local_expert_ids: set[int] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)
    with RamStageManager(sorted_files, config, use_tqdm_on_load) as stage:
        yield from safetensors_weights_iterator(
            stage.staged_files,
            use_tqdm_on_load,
            "lazy",
            local_expert_ids=local_expert_ids,
        )
