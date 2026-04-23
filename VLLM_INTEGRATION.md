# vLLM Integration for RedBlackBench

This document describes the vLLM provider integration for RedBlackBench, enabling high-throughput local inference with Qwen3-14B and other models.

## Overview

The vLLM provider allows RedBlackBench to use locally-hosted models through vLLM's OpenAI-compatible API. This provides:

- **High throughput**: PagedAttention, continuous batching, chunked prefill
- **Cost savings**: No API costs for local inference
- **Privacy**: Data never leaves your infrastructure
- **Flexibility**: Support for base models and LoRA adapters

## Setup

### 1. Install vLLM

vLLM is already installed in this environment. If you need to install it elsewhere:

```bash
pip install vllm
```

### 2. Start vLLM Server

Start the optimized vLLM server for Qwen3-14B:

```bash
cd /workspace
bash start_qwen3_base_server.sh
```

This starts vLLM with the following optimizations:
- **Model**: Qwen3-14B base (28GB, bfloat16)
- **Server**: 0.0.0.0:8000
- **GPU Memory**: 95% utilization (~93GB for KV cache)
- **Features**: Prefix caching, chunked prefill, continuous batching
- **Max sequences**: 64 concurrent requests
- **Context length**: 4096 tokens

The server will be available at `http://localhost:8000/v1`

### 3. Verify Installation

Test the vLLM provider integration:

```bash
cd /workspace/RedBlackBench
python test_vllm_provider.py
```

## Usage

### Configuration Files

Three example configurations are provided:

#### 1. Quick Test (2 agents, 3 rounds)
```yaml
# experiments/configs/vllm_qwen3_quick_test.yaml
default_provider:
  type: vllm
  model: /workspace/models/Qwen3-14B
  base_url: http://localhost:8000/v1
  temperature: 0.7
```

#### 2. Full Experiment (5 agents, 10 rounds)
```yaml
# experiments/configs/vllm_qwen3_14b.yaml
default_provider:
  type: vllm
  model: /workspace/models/Qwen3-14B
  base_url: http://localhost:8000/v1
  temperature: 0.7
```

#### 3. Base vs LoRA Comparison
```yaml
# experiments/configs/vllm_base_vs_lora.yaml
team_a:
  provider:
    type: vllm
    model: /workspace/models/Qwen3-14B  # Untrained base

team_b:
  provider:
    type: vllm
    model: qwen3-14b-v2  # Trained LoRA adapter
```

### Running Experiments

Run a quick test:
```bash
python -m redblackbench experiments/configs/vllm_qwen3_quick_test.yaml
```

Run a full experiment:
```bash
python -m redblackbench experiments/configs/vllm_qwen3_14b.yaml
```

Compare base model vs LoRA:
```bash
# First, start vLLM with LoRA support
bash /workspace/start_vllm_table3.sh

# Then run experiment
python -m redblackbench experiments/configs/vllm_base_vs_lora.yaml
```

### Programmatic Usage

You can also use the vLLM provider directly in Python:

```python
from redblackbench.providers import VLLMProvider
import asyncio

async def main():
    # Create provider
    provider = VLLMProvider(
        model="/workspace/models/Qwen3-14B",
        base_url="http://localhost:8000/v1",
        temperature=0.7,
    )

    # Generate response
    system_prompt = "You are a strategic game player."
    messages = [{"role": "user", "content": "What should I do?"}]
    response = await provider.generate(system_prompt, messages)
    print(response)

asyncio.run(main())
```

## Provider Configuration Options

The vLLM provider supports the following configuration options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `type` | str | - | Must be `"vllm"` |
| `model` | str | Required | Model identifier (path or LoRA adapter name) |
| `base_url` | str | `http://localhost:8000/v1` | vLLM server URL |
| `temperature` | float | `0.7` | Sampling temperature (0.0-2.0) |
| `max_tokens` | int | `1024` | Maximum tokens in response |
| `api_key` | str | `"EMPTY"` | API key (vLLM doesn't require one) |

## Architecture

### Provider Hierarchy

```
BaseLLMProvider (abstract)
├── OpenAIProvider
├── AnthropicProvider
├── OpenRouterProvider
└── VLLMProvider (new)
```

### File Structure

```
redblackbench/
├── providers/
│   ├── base.py              # Base provider interface
│   ├── vllm_provider.py     # vLLM provider implementation (new)
│   └── __init__.py          # Updated to export VLLMProvider
├── cli.py                   # Updated to support vllm provider type
experiments/configs/
├── vllm_qwen3_quick_test.yaml  # Quick test config
├── vllm_qwen3_14b.yaml         # Full experiment config
└── vllm_base_vs_lora.yaml      # Base vs LoRA comparison
```

## Performance

### Expected Throughput

With the current setup (RTX PRO 6000 Blackwell, 98GB VRAM):

- **Single request latency**: ~50-200ms (first token)
- **Throughput**: 1000+ tokens/second
- **Concurrent requests**: Up to 64 sequences
- **Games/hour**: 50-100+ (depending on configuration)

### Optimization Tips

1. **Batch size**: Increase `--max-num-seqs` for higher throughput
2. **Context length**: Reduce `--max-model-len` if you don't need long contexts
3. **GPU memory**: Adjust `--gpu-memory-utilization` based on VRAM
4. **Prefix caching**: Enabled by default, reuses system prompts
5. **Chunked prefill**: Enabled by default, reduces first-token latency

## Troubleshooting

### Server won't start

**Error**: `libcusparseLt.so.0: cannot open shared object file`

**Solution**: The startup script already sets `LD_LIBRARY_PATH`. If issues persist:
```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
```

### Connection refused

**Error**: `Connection refused` when running experiments

**Solution**: Verify vLLM server is running:
```bash
curl http://localhost:8000/v1/models
```

### Out of memory

**Error**: GPU OOM during inference

**Solutions**:
1. Reduce `--gpu-memory-utilization` (default: 0.95)
2. Reduce `--max-num-seqs` (default: 64)
3. Reduce `--max-model-len` (default: 4096)

### Slow inference

**Issue**: Lower than expected throughput

**Solutions**:
1. Check GPU utilization: `nvidia-smi`
2. Increase batch size: `--max-num-seqs 128`
3. Verify prefix caching is working (repeated prompts should be faster)
4. Check network latency if accessing remotely

## Comparison with Other Providers

| Feature | OpenAI | Anthropic | vLLM |
|---------|--------|-----------|------|
| Cost | High (per-token) | High (per-token) | Free (local) |
| Latency | 200-1000ms | 200-1000ms | 50-200ms |
| Throughput | Limited by API | Limited by API | 1000+ tok/s |
| Privacy | External API | External API | Local |
| Customization | Limited | Limited | Full control |
| Setup | Easy | Easy | Moderate |

## Advanced Usage

### Using LoRA Adapters

To use LoRA adapters with vLLM:

1. Start server with LoRA support:
```bash
bash /workspace/start_vllm_table3.sh
```

2. Reference adapter by name in config:
```yaml
provider:
  type: vllm
  model: qwen3-14b-v2  # LoRA adapter name
  base_url: http://localhost:8000/v1
```

### Multiple Models

vLLM can serve multiple models/adapters simultaneously. Configure different providers for different teams:

```yaml
team_a:
  provider:
    type: vllm
    model: /workspace/models/Qwen3-14B

team_b:
  provider:
    type: vllm
    model: qwen3-14b-v2
```

## Contributing

When extending the vLLM provider:

1. Follow the `BaseLLMProvider` interface
2. Add tests in `test_vllm_provider.py`
3. Update this documentation
4. Submit PR to `asuka/dev` branch

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-14B)
