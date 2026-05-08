# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .protocol import ScoreResponse, ScoreTextRequest

__all__ = ["ScoreResponse", "ScoreTextRequest"]
