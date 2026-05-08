"""Qwen3-1.7B recipe: three node-pinned single-GPU replicas.

Run from the build.sh-managed venv:

    serve run --app-dir /home/anhvth8/vllm_projects --address auto --non-blocking demo.ray_serve_qwen3_1p7b_dp3_node_pinned:app
"""

from __future__ import annotations

from vllm_hotload.ray_serve_text_llm_recipes import (
    TextLLMRecipe,
    build_text_llm_app,
    run_app,
)


RECIPE = TextLLMRecipe(
    recipe_name="qwen3_1p7b_dp3_node_pinned",
    model_id="qwen3-1.7b",
    model_source="~/ckpt/hf_models/Qwen/Qwen3-1.7B",
    tensor_parallel_size=1,
    data_parallel_size=3,
    max_model_len=4096,
)

app = build_text_llm_app(RECIPE)


if __name__ == "__main__":
    run_app(app)
