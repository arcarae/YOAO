"""Configuration presets for Sugarscape experiments.

``paper_config()`` returns the environment used in the paper's Sugarscape experiments:
a 20x20 torus, 100 agents, 100 ticks, sugar + spice, endowments in [45, 85], vision in
[1, 6], old age in [60, 100], metabolism specialisation (half the agents need more sugar,
half more spice), the five-phase encounter protocol with two small-talk and two
negotiation turns, reflection, event-triggered identity review and end-of-life reports.
Model, provider and identity mix are left for the caller.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from sugarscape.config import SugarscapeConfig

PAPER_WIDTH = 20
PAPER_HEIGHT = 20
PAPER_POPULATION = 100
PAPER_TICKS = 100
PAPER_ENDOWMENT = (45, 85)
PAPER_VISION = (1, 6)
PAPER_MAX_AGE = (60, 100)

IDENTITY_ALL_NORMIE: Dict[str, float] = {"altruist": 0.0, "exploiter": 0.0, "survivor": 1.0}
IDENTITY_ALL_EXPLOITER: Dict[str, float] = {"altruist": 0.0, "exploiter": 1.0, "survivor": 0.0}


def seeded_normie_identity(altruist_fraction: float) -> Dict[str, float]:
    """Normie population seeded with a fraction of Altruists (the paper's 20/40/50 % runs)."""
    if not 0.0 <= altruist_fraction <= 1.0:
        raise ValueError("altruist_fraction must be in [0, 1]")
    return {"altruist": altruist_fraction, "exploiter": 0.0, "survivor": 1.0 - altruist_fraction}


def paper_config(
    *,
    provider: str = "vllm",
    model: str = "Qwen/Qwen3-14B",
    vllm_url: Optional[str] = None,
    goal_preset: str = "survival",
    identity_distribution: Optional[Dict[str, float]] = None,
    ticks: int = PAPER_TICKS,
    population: int = PAPER_POPULATION,
    seed: int = 42,
    enable_llm_evaluation: bool = True,
    evaluator_provider: str = "vllm",
    evaluator_model: Optional[str] = None,
    enable_external_moral_evaluation: bool = True,
    moral_evaluator_provider: str = "vllm",
    moral_evaluator_model: Optional[str] = None,
    resource_specialization: bool = True,
    checkpoint_interval: int = 50,
) -> SugarscapeConfig:
    """Build the paper's Sugarscape configuration.

    Args:
        provider: "vllm" (local server) or "openrouter" for the agents' model.
        model: Served model name (vLLM) or OpenRouter id.
        vllm_url: vLLM base URL; defaults to $YOAO_VLLM_URL or http://localhost:8000/v1.
        goal_preset: Goal prompt preset for every agent ("survival" = the paper's Normie goal).
        identity_distribution: Origin identity mix, e.g. ``seeded_normie_identity(0.2)``.
        ticks, population, seed: Run length, agent count and RNG seed.
        enable_llm_evaluation / evaluator_*: End-of-run behavioural evaluator.
        enable_external_moral_evaluation / moral_evaluator_*: Per-reflection moral scorer.
        resource_specialization: Half the agents need more sugar, half more spice.
        checkpoint_interval: Ticks between checkpoints (0 disables).
    """
    vllm_url = vllm_url or os.environ.get("YOAO_VLLM_URL", "http://localhost:8000/v1")
    evaluator_model = evaluator_model or model
    moral_evaluator_model = moral_evaluator_model or model
    identity_distribution = identity_distribution or IDENTITY_ALL_NORMIE

    return SugarscapeConfig(
        # Grid and resources
        width=PAPER_WIDTH,
        height=PAPER_HEIGHT,
        max_sugar_capacity=4,
        sugar_growback_rate=1,
        enable_spice=True,
        max_spice_capacity=4,
        spice_growback_rate=1,
        enable_resource_specialization=resource_specialization,
        # Trade
        enable_trade=True,
        trade_mode="dialogue",
        trade_dialogue_rounds=4,
        trade_allow_fraud=True,
        trade_memory_maxlen=50,
        trade_history_in_prompt=True,
        trade_history_prompt_limit=10,
        trade_dialogue_repair_json=True,
        trade_dialogue_repair_attempts=2,
        trade_dialogue_coerce_protocol=True,
        trade_dialogue_two_stage=True,
        trade_dialogue_thinking_tokens=128,
        trade_dialogue_json_tokens=128,
        # Encounter protocol: small talk -> intent -> negotiation -> execution -> reflection
        enable_new_encounter_protocol=True,
        small_talk_rounds=2,
        dialogue_thinking_tokens=128,
        dialogue_response_tokens=200,
        small_talk_allow_thinking=False,
        negotiation_rounds=2,
        encounter_protocol_mode="full",
        enable_broker=False,
        # Reflection and identity: periodic Identity Review every 10 ticks (paper), event-triggered
        # reviews as well, and an End-of-Life Report on death
        enable_reflection=True,
        reflection_max_tokens=256,
        identity_edit_interval=10,
        enable_identity_review=True,
        identity_review_interval=10,
        enable_event_triggered_identity_review=True,
        enable_end_of_life_report=True,
        # Evaluators
        enable_llm_evaluation=enable_llm_evaluation,
        llm_evaluator_model=evaluator_model,
        llm_evaluator_provider=evaluator_provider,
        enable_external_moral_evaluation=enable_external_moral_evaluation,
        external_moral_evaluator_model=moral_evaluator_model,
        external_moral_evaluator_provider=moral_evaluator_provider,
        moral_overall_transform="tanh",
        moral_overall_tanh_k=2.2,
        moral_self_tanh_k=4.0,
        # Population
        initial_population=population,
        enable_rebirth=False,
        initial_wealth_range=PAPER_ENDOWMENT,
        initial_spice_range=PAPER_ENDOWMENT,
        metabolism_range=(1, 4),
        metabolism_spice_range=(1, 4),
        vision_range=PAPER_VISION,
        max_age_range=PAPER_MAX_AGE,
        # Agents
        enable_llm_agents=True,
        llm_agent_ratio=1.0,
        rule_based_movement=False,
        llm_provider_type=provider,
        llm_provider_model=model,
        llm_vllm_base_url=vllm_url,
        llm_history_limit=15,
        llm_goal_preset=goal_preset,
        enable_origin_identity=True,
        origin_identity_distribution=dict(identity_distribution),
        enable_survival_pressure=True,
        social_memory_visible=True,
        # Run
        max_ticks=ticks,
        seed=seed,
        checkpoint_interval=checkpoint_interval,
        enable_debug_logging=True,
        debug_log_decisions=True,
        debug_log_llm=True,
        debug_log_trades=True,
        debug_log_deaths=True,
        debug_log_efficiency=True,
    )
