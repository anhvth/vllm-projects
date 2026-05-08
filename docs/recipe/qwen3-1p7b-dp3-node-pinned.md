# Qwen3 1.7B DP3 Node Pinned

This is the known-good three-node recipe from the hosted cluster smoke test. It runs three independent single-GPU model deployments, pins one deployment to each Ray node resource, and exposes one public OpenAI model id through a round-robin ingress.

- Demo: `demo/ray_serve_qwen3_1p7b_dp3_node_pinned.py`
- Default model source: `~/ckpt/hf_models/Qwen/Qwen3-1.7B`
- Client model id: `qwen3-1.7b`
- Tensor parallelism: `1`
- Data parallel replicas: `3`
- GPU footprint: `3` GPUs total, one per node

## Run

```bash
cd /home/anhvth8/vllm_projects
source .venv/bin/activate
serve run --app-dir /home/anhvth8/vllm_projects --address auto --non-blocking demo.ray_serve_qwen3_1p7b_dp3_node_pinned:app
```

## Validate

```bash
serve status -a http://100.96.5.35:8265
ray status
```

Expected shape: three healthy LLM deployments, one healthy ingress, and `3.0` GPUs reserved in placement groups.

```bash
curl -sS --max-time 180 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer FAKE_KEY' \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"Say hello from Ray Serve in one short sentence."}],"max_tokens":64,"temperature":0}'
```

## Notes

Do not use `max_replicas_per_node` for this recipe. Ray Serve LLM injects placement groups and rejects that option when placement groups are present.
