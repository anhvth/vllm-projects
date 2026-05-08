# Qwen3 4B Single GPU

Use this to try the next Qwen text-model size while keeping the serving topology simple.

- Demo: `demo/ray_serve_qwen3_4b_single_gpu.py`
- Default model source: `~/ckpt/hf_models/Qwen/Qwen3-4B`
- Client model id: `qwen3-4b`
- Tensor parallelism: `1`
- Data parallel replicas: `1`
- GPU footprint: `1` GPU

## Run

```bash
cd /home/anhvth8/vllm_projects
source .venv/bin/activate
serve run --app-dir /home/anhvth8/vllm_projects --address auto --non-blocking demo.ray_serve_qwen3_4b_single_gpu:app
```

If the local checkpoint path differs, override it:

```bash
RAY_SERVE_MODEL_SOURCE=/abs/path/to/Qwen3-4B \
serve run --app-dir /home/anhvth8/vllm_projects --address auto --non-blocking demo.ray_serve_qwen3_4b_single_gpu:app
```

## Validate

```bash
curl -sS --max-time 180 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer FAKE_KEY' \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"Answer in one short sentence: what is Ray Serve?"}],"max_tokens":64,"temperature":0}'
```
