# Qwen3 1.7B Single GPU

Use this as the cheapest smoke test for Ray Serve LLM, vLLM imports, model loading, and the OpenAI-compatible API.

- Demo: `demo/ray_serve_qwen3_1p7b_single_gpu.py`
- Default model source: `~/ckpt/hf_models/Qwen/Qwen3-1.7B`
- Client model id: `qwen3-1.7b`
- Tensor parallelism: `1`
- Data parallel replicas: `1`
- GPU footprint: `1` GPU

## Run

```bash
cd /home/anhvth8/vllm_projects
source .venv/bin/activate
serve run --app-dir /home/anhvth8/vllm_projects --address auto --non-blocking demo.ray_serve_qwen3_1p7b_single_gpu:app
```

## Validate

```bash
serve status -a http://100.96.5.35:8265
```

```bash
curl -sS --max-time 180 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer FAKE_KEY' \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"Say hello from Ray Serve in one short sentence."}],"max_tokens":64,"temperature":0}'
```

## Useful Overrides

```bash
RAY_SERVE_NODE_RESOURCES=node:100.96.34.48 \
serve run --app-dir /home/anhvth8/vllm_projects --address auto --non-blocking demo.ray_serve_qwen3_1p7b_single_gpu:app
```
