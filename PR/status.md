# Hotload vLLM Status

Last updated: 2026-05-08.

## Product decision

One served model replica must fit on one physical node. We do not support
splitting one model replica across nodes for managed hotload.

For a 3-node cluster with 8 GPUs per node, run up to three independent vLLM
replicas:

```text
node0: vLLM dummy -> hotload replica on local GPUs 0..7
node1: vLLM dummy -> hotload replica on local GPUs 0..7
node2: vLLM dummy -> hotload replica on local GPUs 0..7
```

## Endpoint contract

Inference clients get one public OpenAI-compatible endpoint:

```text
http://head-node:8000/v1
```

The public endpoint is a proxy/load balancer over private replica endpoints:

```text
http://node0:8100/v1
http://node1:8100/v1
http://node2:8100/v1
```

Managed hotload control is private and per-replica:

```text
http://node0:8100/managed
http://node1:8100/managed
http://node2:8100/managed
```

Regular application code should never need the private replica URLs.

## Suggested user UX

```bash
hotloadctl start --nodes node0,node1,node2 --gpus-per-replica 8
hotloadctl push /path/to/checkpoint
hotloadctl status
hotloadctl sleep
hotloadctl wake
hotloadctl stop
```

OpenAI client usage stays normal:

```python
from openai import OpenAI

client = OpenAI(base_url="http://head-node:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="qwen3-1.7b",
    messages=[{"role": "user", "content": "hello"}],
)
```

## Controller behavior

`hotloadctl start` should:

* start one dummy vLLM replica per selected node
* start or configure the public proxy/load balancer
* wait for `/managed/status` and `/v1/models` on every replica
* report the single public base URL

`hotloadctl push CHECKPOINT` should, for each replica:

* call `/managed/init_weight_transfer`
* call `/managed/prepare_weight_update`
* run local IPC weight push on that replica's node
* call `/managed/finish_weight_update`
* verify inference through the replica or public endpoint

`hotloadctl status` should show:

```text
public_base_url: http://head-node:8000/v1

replicas:
  node0: http://node0:8100/v1 healthy
  node1: http://node1:8100/v1 healthy
  node2: http://node2:8100/v1 healthy
```

## Current implementation posture

The managed endpoints and IPC hotload flow are designed around one local vLLM
server. The next orchestration step is to wrap that proven local flow in a
multi-replica controller, keeping the public inference endpoint singular and
the private managed endpoints per-node.
