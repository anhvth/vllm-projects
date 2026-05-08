# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility aliases for Ray versions expecting pooling.score."""

from vllm.entrypoints.pooling.scoring.protocol import (  # type: ignore
    ScoreResponse,
    ScoreTextRequest,
)

__all__ = ["ScoreResponse", "ScoreTextRequest"]
