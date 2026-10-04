# Training the seed agent

LoRA supervised fine-tuning of Qwen3-14B on cooperative Red-Black Game deliberations. This is
the code that produced the released adapter
[`Arcarae/redblackbench-qwen3-14b-sft-v2`](https://huggingface.co/Arcarae/redblackbench-qwen3-14b-sft-v2)
(a mirror of Asuka's original upload, `spacezenmasterr/redblackbench-qwen3-14b-sft-v2`)
from the dataset [`Arcarae/redblackbench-sft-batch-1`](https://huggingface.co/datasets/Arcarae/redblackbench-sft-batch-1).

## Data

The dataset is Kimi-K2 (`moonshotai/kimi-k2-thinking`) ideal responses generated with the
meta-prompt in `redblackbench/training/sft_generator.py`, for Red-Black games played by
unmodified Qwen3-14B teams against the scripted opponents in `redblackbench/strategies`,
across the five training scenarios (Climate, Pandemic, AGI Safety, Election, Standards).

`sft_final_all.json` holds 12,613 examples from 345 games; every target ends in `VOTE: A`.
`prepare_data.py` splits it by game into train / val / test (85 / 10 / 5 %), validates that each
example's scenario prompt matches the scenario registry, and writes chat-format JSONL.

```bash
pip install -r training/requirements.txt
mkdir -p training/hf_dataset
hf download Arcarae/redblackbench-sft-batch-1 --repo-type dataset --local-dir training/hf_dataset
cd training && ./run.sh prepare          # -> training/data/{train,val,test}.jsonl + stats.json
```

To regenerate the raw data instead, see `scripts/generate_sft_streaming.py` (needs
`OPENROUTER_API_KEY`; the teacher is served through OpenRouter).

## Train

```bash
cd training
./run.sh estimate                        # GPU memory estimate, no training
./run.sh train                           # defaults below; one H100/H200 80 GB is enough
```

Defaults (the paper's SFT configuration table):

| setting | value |
|---|---|
| base model | `Qwen/Qwen3-14B` |
| LoRA rank / alpha / dropout | 128 / 256 / 0.05 |
| target modules | q, k, v, o, gate, up, down projections |
| learning rate | 1e-5, cosine schedule, 200 warmup steps, weight decay 0.01 |
| effective batch | 16 (2 per device x 8 accumulation) |
| epochs | 3 |
| max sequence length | 4096 |
| precision | FP16 (base weights FP16, LoRA parameters FP32 under AMP) |

Override with environment variables (`MODEL`, `EPOCHS`, `LORA_R`, `LR`, `BATCH_SIZE`,
`GRAD_ACCUM`, `MAX_LENGTH`, `OUTPUT_DIR`) or pass flags straight to `train.py`.
The adapter lands in `outputs/final_model/`; serve it with vLLM as described in
[`docs/vllm.md`](../docs/vllm.md).

## Provenance

`train.py`, `prepare_data.py`, `run.sh` and `REDBLACK_TRAINING_PLAN.md` are the training files
from Asuka Yuxi Zheng's `sft` repository, with defaults set to the paper's configuration. The
plan document describes the earlier r=64 run and is kept for history.
