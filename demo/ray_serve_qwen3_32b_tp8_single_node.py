"""Qwen3-32B recipe: tensor-parallel eight-GPU deployment on one node.

Run from the build.sh-managed venv:

    serve run --app-dir /home/anhvth8/vllm_projects --address auto --non-blocking demo.ray_serve_qwen3_32b_tp8_single_node:app
"""

from __future__ import annotations

from demo.ray_serve_text_llm_recipes import TextLLMRecipe, build_text_llm_app, run_app


RECIPE = TextLLMRecipe(
    recipe_name="qwen3_32b_tp8_single_node",
    model_id="qwen3-32b",
    model_source="~/ckpt/hf_models/Qwen/Qwen3-32B",
    tensor_parallel_size=8,
    data_parallel_size=1,
    max_model_len=8192,
)

app = build_text_llm_app(RECIPE)


if __name__ == "__main__":
    run_app(app)
