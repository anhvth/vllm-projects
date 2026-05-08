# Qwen3.6 27B TP4 Single Node

Use this to test a larger local Qwen text model with four-way tensor parallelism on one H200 node.

- Demo: `demo/ray_serve_qwen3p6_27b_tp4_single_node.py`
- Default model source: `~/ckpt/hf_models/Qwen/Qwen3.6-27B`
- Client model id: `qwen3.6-27b`
- Tensor parallelism: `4`
- Data parallel replicas: `1`
- GPU footprint: `4` GPUs on one node

## Run

```bash
cd /home/anhvth8/vllm_projects
source .venv/bin/activate
serve run --app-dir /home/anhvth8/vllm_projects --address auto --non-blocking demo.ray_serve_qwen3p6_27b_tp4_single_node:app
```

## Validate

```bash
serve status -a http://100.96.5.35:8265
ray status
```

```bash
curl -sS --max-time 240 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer FAKE_KEY' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"Give one practical reason to use Ray Serve."}],"max_tokens":64,"temperature":0}'
```

## Useful Overrides

```bash
RAY_SERVE_MAX_MODEL_LEN=4096 RAY_SERVE_GPU_MEMORY_UTILIZATION=0.85 \
serve run --app-dir /home/anhvth8/vllm_projects --address auto --non-blocking demo.ray_serve_qwen3p6_27b_tp4_single_node:app
```
