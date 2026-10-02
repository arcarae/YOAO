# Serving the models with vLLM

Every experiment talks to an OpenAI-compatible endpoint. The paper's runs use a local vLLM
server that hosts the unmodified Qwen3-14B and the cooperative LoRA seed adapter side by side.

## 1. Get the weights

```bash
pip install -U "huggingface_hub[cli]" vllm
hf download Qwen/Qwen3-14B --local-dir models/Qwen3-14B
hf download Arcarae/redblackbench-qwen3-14b-sft-v2 --local-dir models/Qwen3-14B-LoRA
```

The adapter is 2 GB (r=128 LoRA on all attention and MLP projections). vLLM 0.6+ with
`--enable-lora` serves it without merging.

## 2. Start the server

```bash
vllm serve models/Qwen3-14B \
  --served-model-name Qwen/Qwen3-14B \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-lora --max-lora-rank 128 \
  --lora-modules redblackbench-qwen3-14b-sft-v2=models/Qwen3-14B-LoRA \
  --host 0.0.0.0 --port 8000
```

Notes:

- **Context length.** Agents clear their deliberation history each round, so prompts stay at a
  few thousand tokens; 16k is comfortable and 32k leaves room for `--enable-thinking` or
  `LLMAgent(clear_history_each_round=False)`, which keeps a 30-message window across rounds.
- The control and uncooperative adapters load the same way as extra `--lora-modules` entries
  (`docs/ablations.md`).
- One H100/A100 80 GB handles 6 concurrent games comfortably. Lower `--max-num-seqs` on
  smaller cards.
- The two served names above are the repo defaults (`redblackbench/defaults.py`). If you
  choose different ones, export `YOAO_BASE_MODEL` / `YOAO_SFT_MODEL`, and `YOAO_VLLM_URL` if the
  server is not on `localhost:8000`.

Check it:

```bash
curl -s localhost:8000/v1/models | python -m json.tool | grep '"id"'
# "Qwen/Qwen3-14B"
# "redblackbench-qwen3-14b-sft-v2"
```

## 3. Use it from the repo

YAML configs (`redblackbench run --config ...`):

```yaml
default_provider:
  type: vllm
  model: Qwen/Qwen3-14B                 # or redblackbench-qwen3-14b-sft-v2
  base_url: "${YOAO_VLLM_URL}"          # unset -> http://localhost:8000/v1
  temperature: 0.7
```

Scripts (`scripts/eval_*.py`, `scripts/run_goal_experiment.py`) default to the vLLM provider and
the names above. Programmatic use:

```python
from redblackbench.providers import VLLMProvider
from redblackbench import defaults

base = VLLMProvider(model=defaults.BASE_MODEL, base_url=defaults.VLLM_URL, temperature=0.7)
seed = VLLMProvider(model=defaults.SFT_MODEL, base_url=defaults.VLLM_URL, temperature=0.7)
```

Qwen3's thinking mode is disabled for all Red-Black calls on vLLM (`enable_thinking=False` is
sent through `chat_template_kwargs`); pass `--enable-thinking` to the eval scripts to turn it on.

## Provider options

| option | default | meaning |
|---|---|---|
| `type` | | `vllm` |
| `model` | `$YOAO_BASE_MODEL` | served model or adapter name |
| `base_url` | `$YOAO_VLLM_URL` | server URL |
| `temperature` | 0.7 | sampling temperature; every paper evaluation uses 0.7 |
| `max_tokens` | 1024 | response cap |

## Troubleshooting

- `Connection refused`: the server is not up yet; `curl localhost:8000/v1/models`.
- `maximum context length is 4096 tokens`: restart with a larger `--max-model-len`.
- `OPENROUTER_API_KEY is not set`: a script is routing some model through OpenRouter (cross-model
  teammates, the Kimi-K2 teacher, or an evaluator). Pass `--provider vllm`, `--evaluator-provider
  vllm` and `--moral-evaluator-provider vllm`, or export the key.
