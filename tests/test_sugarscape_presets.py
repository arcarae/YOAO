"""The paper preset and the goal-experiment CLI helpers."""

import importlib.util
from pathlib import Path

import pytest

from sugarscape import presets


def test_paper_config_matches_paper_environment():
    cfg = presets.paper_config(identity_distribution=presets.seeded_normie_identity(0.2))
    assert (cfg.width, cfg.height) == (20, 20)
    assert cfg.initial_population == 100 and cfg.max_ticks == 100
    assert cfg.enable_spice and cfg.enable_trade and cfg.trade_mode == "dialogue"
    assert cfg.initial_wealth_range == (45, 85) and cfg.initial_spice_range == (45, 85)
    assert cfg.vision_range == (1, 6) and cfg.max_age_range == (60, 100)
    assert cfg.enable_resource_specialization
    assert cfg.small_talk_rounds == 2 and cfg.negotiation_rounds == 2
    assert cfg.enable_event_triggered_identity_review and cfg.enable_end_of_life_report
    assert cfg.enable_origin_identity
    assert cfg.origin_identity_distribution == {"altruist": 0.2, "exploiter": 0.0, "survivor": 0.8}
    assert cfg.llm_provider_type == "vllm" and cfg.llm_vllm_base_url.startswith("http")
    assert cfg.llm_evaluator_provider == "vllm" and cfg.external_moral_evaluator_provider == "vllm"


def test_paper_config_openrouter_and_overrides():
    cfg = presets.paper_config(provider="openrouter", model="meta-llama/llama-3.3-70b-instruct",
                               ticks=200, evaluator_provider="openrouter",
                               moral_evaluator_provider="openrouter")
    assert cfg.llm_provider_type == "openrouter"
    assert cfg.llm_provider_model == "meta-llama/llama-3.3-70b-instruct"
    assert cfg.llm_evaluator_model == cfg.llm_provider_model
    assert cfg.max_ticks == 200


def test_seeded_identity_validates():
    with pytest.raises(ValueError):
        presets.seeded_normie_identity(1.5)
    assert presets.IDENTITY_ALL_EXPLOITER["exploiter"] == 1.0


def _load_goal_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_goal_experiment.py"
    spec = importlib.util.spec_from_file_location("run_goal_experiment", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_identity_distribution_accepts_percent_fraction_and_alias():
    mod = _load_goal_script()
    parse = mod.parse_identity_distribution
    assert parse("altruist:20,survivor:80") == {"altruist": 0.2, "survivor": 0.8, "exploiter": 0.0}
    assert parse("altruist:0.2,survivor:0.8") == {"altruist": 0.2, "survivor": 0.8, "exploiter": 0.0}
    assert parse("normie:100") == {"survivor": 1.0, "altruist": 0.0, "exploiter": 0.0}
    with pytest.raises(ValueError):
        parse("altruist:30,survivor:30")
    with pytest.raises(ValueError):
        parse("wizard:100")
