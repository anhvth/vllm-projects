# Qwen3 8B TP2 Single Node

Use this to test tensor parallelism on one Ray node with a larger Qwen text model.

- Demo: `demo/ray_serve_qwen3_8b_tp2_single_node.py`
- Default model source: `~/ckpt/hf_models/Qwen/Qwen3-8B`
- Client model id: `qwen3-8b`
- Tensor parallelism: `2`
- Data parallel replicas: `1`
- GPU footprint: `2` GPUs on one node

## Run

```bash
cd /home/anhvth8/vllm_projects
source .venv/bin/activate
serve run --app-dir /home/anhvth8/vllm_projects --address auto --non-blocking demo.ray_serve_qwen3_8b_tp2_single_node:app
```

By default, the helper pins the placement group to the first configured node resource. Override if needed:

```bash
RAY_SERVE_NODE_RESOURCES=node:100.96.34.48 \
serve run --app-dir /home/anhvth8/vllm_projects --address auto --non-blocking demo.ray_serve_qwen3_8b_tp2_single_node:app
```

## Validate

```bash
serve status -a http://100.96.5.35:8265
ray status
```

```bash
curl -sS --max-time 180 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer FAKE_KEY' \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Answer in one short sentence: what is tensor parallelism?"}],"max_tokens":64,"temperature":0}'
```
