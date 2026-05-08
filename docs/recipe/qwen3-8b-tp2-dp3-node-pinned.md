# Qwen3 8B TP2 DP3 Node Pinned

Use this to test both tensor parallelism and data-parallel serving. It starts three node-pinned deployments. Each deployment uses two GPUs for tensor parallelism, so the public endpoint load-balances across three TP2 replicas.

- Demo: `demo/ray_serve_qwen3_8b_tp2_dp3_node_pinned.py`
- Default model source: `~/ckpt/hf_models/Qwen/Qwen3-8B`
- Client model id: `qwen3-8b`
- Tensor parallelism: `2`
- Data parallel replicas: `3`
- GPU footprint: `6` GPUs total, two per node

## Run

```bash
cd /home/anhvth8/vllm_projects
source .venv/bin/activate
serve run --app-dir /home/anhvth8/vllm_projects --address auto --non-blocking demo.ray_serve_qwen3_8b_tp2_dp3_node_pinned:app
```

## Validate

```bash
serve status -a http://100.96.5.35:8265
ray status
```

Expected shape: three healthy LLM deployments, one healthy ingress, and `6.0` GPUs reserved in placement groups.

```bash
curl -sS --max-time 180 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer FAKE_KEY' \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Explain TP2 DP3 in one sentence."}],"max_tokens":80,"temperature":0}'
```
