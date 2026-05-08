# Qwen3 32B TP8 Single Node

Use this to test a larger Qwen text model on one full eight-GPU node with tensor parallelism.

- Demo: `demo/ray_serve_qwen3_32b_tp8_single_node.py`
- Default model source: `~/ckpt/hf_models/Qwen/Qwen3-32B`
- Client model id: `qwen3-32b`
- Tensor parallelism: `8`
- Data parallel replicas: `1`
- GPU footprint: `8` GPUs on one node

## Run

```bash
cd /home/anhvth8/vllm_projects
source .venv/bin/activate
serve run --app-dir /home/anhvth8/vllm_projects --address auto --non-blocking demo.ray_serve_qwen3_32b_tp8_single_node:app
```

If the model needs a shorter context for the first smoke test:

```bash
RAY_SERVE_MAX_MODEL_LEN=4096 \
serve run --app-dir /home/anhvth8/vllm_projects --address auto --non-blocking demo.ray_serve_qwen3_32b_tp8_single_node:app
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
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Say hello from the 32B recipe in one sentence."}],"max_tokens":64,"temperature":0}'
```
