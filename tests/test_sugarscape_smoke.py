"""Smoke tests for sugarscape features added in the broker / moral-eval bundle.

These are not exhaustive — they verify that each feature's public surface
constructs, initializes, and runs without crashing on minimally-valid
input. They exist to catch regressions in the three subsystems landed in
commit 11d7ecc (broker, resource specialization, moral evaluator) and the
basic simulation loop, which previously had no dedicated tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python3 tests/test_sugarscape_smoke.py` direct invocation
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sugarscape.agent import SugarAgent
from sugarscape.config import SugarscapeConfig
from sugarscape.environment import SugarEnvironment
from sugarscape.moral_evaluator import MoralEvaluator, MoralRubric
from sugarscape.prompts import build_broker_introduction_prompt
from sugarscape.simulation import SugarSimulation


class _NullProvider:
    """Minimal mock provider for tests that must not hit any network."""

    provider_name = "null"

    async def generate(self, system_prompt, messages, **kwargs):  # noqa: D401
        return "{}"


def _make_agent(agent_id: int, pos=(0, 0), wealth: int = 50, spice: int = 50) -> SugarAgent:
    return SugarAgent(
        agent_id=agent_id,
        pos=pos,
        vision=3,
        metabolism=2,
        max_age=80,
        wealth=wealth,
        spice=spice,
        metabolism_spice=2,
    )


def test_build_broker_introduction_prompt_returns_nonempty_pair():
    """The broker prompt builder should return a (system, user) tuple of non-empty
    strings that mention both agents' names."""
    a = _make_agent(1, pos=(2, 3), wealth=80, spice=30)
    a.name = "Alice"
    b = _make_agent(2, pos=(4, 5), wealth=30, spice=80)
    b.name = "Bob"

    system_prompt, user_prompt = build_broker_introduction_prompt(a, b, env=None)

    assert isinstance(system_prompt, str) and system_prompt.strip()
    assert isinstance(user_prompt, str) and user_prompt.strip()
    assert "Alice" in user_prompt
    assert "Bob" in user_prompt
    # Broker framing should surface in the system prompt
    assert "broker" in system_prompt.lower() or "altruistic" in system_prompt.lower()


def test_moral_evaluator_constructs_with_default_rubric():
    """MoralEvaluator should accept a provider (no rubric) and initialize without
    making any network calls."""
    evaluator = MoralEvaluator(provider=_NullProvider())
    assert evaluator.provider is not None
    assert isinstance(evaluator.rubric, MoralRubric)


def test_resource_specialization_produces_heterogeneous_metabolism():
    """With enable_resource_specialization=True the simulation should split
    agents into sugar-heavy and spice-heavy specialists rather than
    homogeneous metabolism ranges."""
    config = SugarscapeConfig(
        width=10,
        height=10,
        enable_spice=True,
        initial_population=40,
        max_ticks=1,
        enable_llm_agents=False,
        enable_resource_specialization=True,
        seed=42,
    )
    sim = SugarSimulation(config=config, experiment_name="smoke_resource_spec")
    metabolisms = [(a.metabolism, a.metabolism_spice) for a in sim.agents]

    # Specialists: one stat high (>=3), the other low (<=2). With 40 agents at
    # seed 42 we should see both camps populated.
    sugar_heavy = [(m, ms) for (m, ms) in metabolisms if m >= 3 and ms <= 2]
    spice_heavy = [(m, ms) for (m, ms) in metabolisms if m <= 2 and ms >= 3]

    assert len(sugar_heavy) > 0, "no sugar specialists produced"
    assert len(spice_heavy) > 0, "no spice specialists produced"
    assert len(sugar_heavy) + len(spice_heavy) == len(metabolisms), (
        "resource specialization left some agents in neither camp"
    )


def test_sugar_simulation_basic_loop_three_ticks():
    """Rule-based SugarSimulation should complete a 3-tick loop without LLM
    calls and produce non-empty metric output."""
    config = SugarscapeConfig(
        width=10,
        height=10,
        enable_spice=True,
        initial_population=8,
        max_ticks=3,
        enable_llm_agents=False,
        enable_origin_identity=False,
        enable_identity_review=False,
        enable_reflection=False,
        enable_llm_evaluation=False,
        enable_external_moral_evaluation=False,
        checkpoint_interval=0,
        seed=7,
    )
    sim = SugarSimulation(config=config, experiment_name="smoke_basic_loop")

    initial_pop = len(sim.agents)
    assert initial_pop == 8

    for _ in range(3):
        sim.step()

    stats = sim.get_stats()
    assert stats["tick"] == 3
    assert "mean_wealth" in stats
    assert stats["population"] >= 0  # agents may die off, but the loop ran
