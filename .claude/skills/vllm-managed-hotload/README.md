# vllm-managed-hotload

A Claude Code skill for hosting a long-lived vLLM serve process with managed weight-sync — hot-load or swap Hugging Face model weights into a live server without restarting.

## Install

```bash
npx skills add anhvth8/vllm-managed-hotload-skill
```

## Usage

Invoke in Claude Code with:

```
/vllm-managed-hotload <optional model path or task summary>
```

Or manually trigger the skill prompt.

## What it does

- Start `vllm serve` with dummy weights in a tmux session
- Push real model weights into the live server via CUDA IPC
- Swap checkpoints without restarting the server
- Pause, sleep, wake, or stop the server to manage GPU usage
- Smoke-test `/v1` and `/managed` endpoints

## Requirements

- vLLM with the `--managed-weight-sync` flag
- CUDA-capable GPUs
- `tmux`, `curl`, and Python 3.12+
