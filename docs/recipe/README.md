# Ray Serve Text LLM Recipes

These recipes serve OpenAI-compatible text-generation endpoints for Qwen-style LLMs on the hosted Ray cluster.

Always build the venv with `./build.sh` and run demos from the active venv. Do not use plain `uv run` for these serving demos.

## Recipe Matrix

| Recipe | Demo | Model size | TP | DP | GPUs |
|---|---|---:|---:|---:|---:|
| [Qwen3 1.7B Single GPU](qwen3-1p7b-single-gpu.md) | `demo/ray_serve_qwen3_1p7b_single_gpu.py` | 1.7B | 1 | 1 | 1 |
| [Qwen3 1.7B DP3 Node Pinned](qwen3-1p7b-dp3-node-pinned.md) | `demo/ray_serve_qwen3_1p7b_dp3_node_pinned.py` | 1.7B | 1 | 3 | 3 |
| [Qwen3 4B Single GPU](qwen3-4b-single-gpu.md) | `demo/ray_serve_qwen3_4b_single_gpu.py` | 4B | 1 | 1 | 1 |
| [Qwen3 8B TP2 Single Node](qwen3-8b-tp2-single-node.md) | `demo/ray_serve_qwen3_8b_tp2_single_node.py` | 8B | 2 | 1 | 2 |
| [Qwen3 8B TP2 DP3 Node Pinned](qwen3-8b-tp2-dp3-node-pinned.md) | `demo/ray_serve_qwen3_8b_tp2_dp3_node_pinned.py` | 8B | 2 | 3 | 6 |
| [Qwen3.6 27B TP4 Single Node](qwen3p6-27b-tp4-single-node.md) | `demo/ray_serve_qwen3p6_27b_tp4_single_node.py` | 27B | 4 | 1 | 4 |
| [Qwen3 32B TP8 Single Node](qwen3-32b-tp8-single-node.md) | `demo/ray_serve_qwen3_32b_tp8_single_node.py` | 32B | 8 | 1 | 8 |

## Common Commands

```bash
cd /home/anhvth8/vllm_projects
source .venv/bin/activate
serve status -a http://100.96.5.35:8265
```

```bash
curl -sS --max-time 30 http://127.0.0.1:8000/v1/models \
  -H 'Authorization: Bearer FAKE_KEY'
```

```bash
curl -sS --max-time 180 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer FAKE_KEY' \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"Say hello from Ray Serve in one short sentence."}],"max_tokens":64,"temperature":0}'
```

## Common Overrides

- `RAY_SERVE_MODEL_SOURCE=/path/to/checkpoint`
- `RAY_SERVE_MODEL_ID=name-seen-by-clients`
- `RAY_SERVE_NODE_RESOURCES=node:100.96.5.35,node:100.96.34.48,node:100.96.31.61`
- `RAY_SERVE_MAX_MODEL_LEN=4096`
- `RAY_SERVE_GPU_MEMORY_UTILIZATION=0.90`
- `RAY_SERVE_DTYPE=bfloat16`
