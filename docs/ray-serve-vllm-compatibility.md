# Ray Serve vLLM Compatibility

Ray Serve LLM provides an OpenAI-compatible API that aligns with vLLM's
OpenAI-compatible server. Most of the `engine_kwargs` that work with
`vllm serve` work with Ray Serve LLM, giving you access to vLLM's feature
set through Ray Serve's distributed deployment capabilities.

## Benefits

- Use the same model configurations and engine arguments as vLLM.
- Leverage vLLM's latest features (multimodal, structured output, reasoning
  models).
- Switch between `vllm serve` and Ray Serve LLM with no code changes and
  scale.
- Take advantage of Ray Serve's production features (autoscaling,
  multi-model serving, advanced routing).

## Embeddings

Generate embeddings by setting `task="embed"` in `engine_kwargs`.

```python
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen-0.5b",
        model_source="Qwen/Qwen2.5-0.5B-Instruct",
    ),
    engine_kwargs=dict(task="embed"),
)

app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app, blocking=True)
```

See the [vLLM embedding models docs](https://docs.vllm.ai/en/latest/models/supported_models/embedding.html)
for supported models.

## Transcriptions

Deploy Speech-to-Text (STT) models for Automatic Speech Recognition (ASR).

```python
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config={
        "model_id": "whisper-small",
        "model_source": "openai/whisper-small",
    },
    deployment_config={
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": 4,
        }
    },
    accelerator_type="A10G",
    log_engine_metrics=True,
)

app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app, blocking=True)
```

See the [vLLM transcription models docs](https://docs.vllm.ai/en/latest/models/supported_models/audio/)
for supported models.

## Structured Output

### JSON Mode

```python
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen-0.5b",
        model_source="Qwen/Qwen2.5-0.5B-Instruct",
    ),
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=2),
    ),
    accelerator_type="A10G",
)

app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app, blocking=True)
```

Client request:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="fake-key")

response = client.chat.completions.create(
    model="qwen-0.5b",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
        {"role": "user", "content": "List three colors in JSON format"},
    ],
)
```

### JSON Schema with Pydantic

```python
from openai import OpenAI
from typing import List, Literal
from pydantic import BaseModel

client = OpenAI(base_url="http://localhost:8000/v1", api_key="fake-key")


class Color(BaseModel):
    colors: List[Literal["cyan", "magenta", "yellow"]]


response = client.chat.completions.create(
    model="qwen-0.5b",
    response_format={"type": "json_schema", "json_schema": Color.model_json_schema()},
    messages=[
        {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
        {"role": "user", "content": "List three colors in JSON format"},
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Vision Language Models

Deploy multimodal models that process both text and images.

```python
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="pixtral-12b",
        model_source="mistral-community/pixtral-12b",
    ),
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=2),
    ),
    accelerator_type="L40S",
    engine_kwargs=dict(tensor_parallel_size=1, max_model_len=8192),
)

app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app, blocking=True)
```

See the [vLLM multimodal models docs](https://docs.vllm.ai/en/latest/models/supported_models/multimodal/)
for a complete list of supported vision models.

## Reasoning Models

Ray Serve LLM supports reasoning models such as DeepSeek-R1 and QwQ through
vLLM. These models use extended thinking processes before generating final
responses. See the [vLLM reasoning models
docs](https://docs.vllm.ai/en/latest/features/reasoning.html) for configuration
details.

## References

- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/)
- [vLLM OpenAI compatibility](https://docs.vllm.ai/en/latest/servers/openai_compatible_server.html)
- [Ray Serve LLM Quickstart](https://docs.ray.io/en/latest/serve/llm/quickstart.html)
