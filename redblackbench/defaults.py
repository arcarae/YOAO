"""Shared defaults for model names and endpoints.

Every experiment script reads these instead of hardcoding a machine-specific path, so one
set of environment variables points the whole repo at your servers:

    YOAO_VLLM_URL   OpenAI-compatible base URL of the local vLLM server
                    (default http://localhost:8000/v1)
    YOAO_BASE_MODEL Name the base model is served under on that server
                    (default Qwen/Qwen3-14B; pass --served-model-name Qwen/Qwen3-14B to vLLM)
    YOAO_SFT_MODEL  Name of the cooperative LoRA adapter on that server
                    (default redblackbench-qwen3-14b-sft-v2, the Hugging Face adapter id)
    YOAO_OPENROUTER_BASE_MODEL  OpenRouter id of the unmodified model (default qwen/qwen3-14b)

See README.md "Serving the models" for the matching ``vllm serve`` command.
"""

import os

VLLM_URL: str = os.environ.get("YOAO_VLLM_URL", "http://localhost:8000/v1")
BASE_MODEL: str = os.environ.get("YOAO_BASE_MODEL", "Qwen/Qwen3-14B")
SFT_MODEL: str = os.environ.get("YOAO_SFT_MODEL", "redblackbench-qwen3-14b-sft-v2")
OPENROUTER_BASE_MODEL: str = os.environ.get("YOAO_OPENROUTER_BASE_MODEL", "qwen/qwen3-14b")

# Hugging Face locations of the released artifacts (the adapter is mirrored from
# spacezenmasterr/redblackbench-qwen3-14b-sft-v2, identical weights).
HF_SFT_ADAPTER: str = "Arcarae/redblackbench-qwen3-14b-sft-v2"
HF_SFT_ADAPTER_V1: str = "Arcarae/redblackbench-qwen3-14b-sft-v1"
HF_SFT_DATASET: str = "Arcarae/redblackbench-sft-batch-1"
HF_BASE_MODEL: str = "Qwen/Qwen3-14B"
# Rebuttal / ablation artifacts (paper RQ2 cross-architecture seed, "Propagation is bidirectional",
# "Non-Cooperative SFT Training"); same LoRA shape as the seed, see each model card.
HF_SEED_ADAPTER_LLAMA: str = "Arcarae/redblackbench-llama-3.1-8b-coop-seed"
HF_UNCOOP_ADAPTER: str = "Arcarae/redblackbench-qwen3-14b-uncoop"
HF_CONTROL_ADAPTER: str = "Arcarae/redblackbench-qwen3-14b-control"
HF_SFT_DATASET_BATCHES_2_3: str = "Arcarae/redblackbench-sft-batches-2-3-10k"
HF_UNCOOP_DATASET: str = "Arcarae/redblackbench-sft-uncoop-v1"
HF_CONTROL_DATASET: str = "Arcarae/redblackbench-control-openthoughts-10k"

# Game settings used throughout the paper.
PAPER_MULTIPLIERS = {5: 3, 8: 5, 10: 10}
PAPER_NUM_ROUNDS = 10
PAPER_TEAM_SIZE = 5
PAPER_OPPONENT = "always_defect"
TRAIN_SCENARIOS = [
    "climate_cooperation",
    "pandemic_vaccines",
    "agi_safety",
    "election_crisis",
    "standards_coordination",
]
HELD_OUT_SCENARIOS = ["baseline", "trade_war", "gpu_contention"]


# Sampling temperature used for every paper evaluation (inference table in the appendix).
PAPER_TEMPERATURE = 0.7


def vllm_provider_config(model: str, temperature: float = PAPER_TEMPERATURE, max_tokens: int = 512,
                         base_url: str | None = None) -> dict:
    """Provider config dict for :func:`redblackbench.cli.create_provider` (vLLM)."""
    return {
        "type": "vllm",
        "model": model,
        "base_url": base_url or VLLM_URL,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def openrouter_provider_config(model: str, temperature: float = PAPER_TEMPERATURE,
                               include_reasoning: bool = False) -> dict:
    """Provider config dict for OpenRouter-hosted models. Requires OPENROUTER_API_KEY."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set (needed for OpenRouter-hosted models)")
    return {
        "type": "openrouter",
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "include_reasoning": include_reasoning,
    }
