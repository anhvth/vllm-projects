"""Qwen3-8B recipe: TP=2 on each node, DP=3 across node-pinned replicas.

Run from the build.sh-managed venv:

    serve run --app-dir /home/anhvth8/vllm_projects --address auto --non-blocking demo.ray_serve_qwen3_8b_tp2_dp3_node_pinned:app
"""

from __future__ import annotations

from vllm_hotload.ray_serve_text_llm_recipes import (
    TextLLMRecipe,
    build_text_llm_app,
    run_app,
)


RECIPE = TextLLMRecipe(
    recipe_name="qwen3_8b_tp2_dp3_node_pinned",
    model_id="qwen3-8b",
    model_source="~/ckpt/hf_models/Qwen/Qwen3-8B",
    tensor_parallel_size=2,
    data_parallel_size=3,
    max_model_len=8192,
)

app = build_text_llm_app(RECIPE)


if __name__ == "__main__":
    run_app(app)
