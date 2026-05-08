# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility aliases for Ray versions expecting pooling.score."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_native_score_protocol():
    this_file = Path(__file__).resolve()
    for entry in sys.path:
        candidate = Path(entry) / "vllm/entrypoints/pooling/score/protocol.py"
        if candidate.exists() and candidate.resolve() != this_file:
            spec = importlib.util.spec_from_file_location(
                "_vllm_native_pooling_score_protocol",
                candidate,
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    from vllm.entrypoints.pooling.scoring import protocol  # type: ignore

    return protocol


_protocol = _load_native_score_protocol()
ScoreResponse = _protocol.ScoreResponse
ScoreTextRequest = _protocol.ScoreTextRequest

__all__ = ["ScoreResponse", "ScoreTextRequest"]
