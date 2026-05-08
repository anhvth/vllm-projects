"""Compatibility entrypoint for the Qwen3 Ray Serve demo.

The maintained demo lives in ray_serve_qwen3_1p7b_dp3_node_pinned.py.
"""

from demo.ray_serve_qwen3_1p7b_dp3_node_pinned import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
