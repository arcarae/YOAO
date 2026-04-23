import asyncio
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running directly via `python3 tests/test_e2e_mock_llm.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import pytest  # type: ignore
except Exception:  # pragma: no cover
    pytest = None  # type: ignore

from sugarscape.config import SugarscapeConfig
from sugarscape.environment import SugarEnvironment
from sugarscape.llm_agent import LLMSugarAgent
from sugarscape.prompts import build_sugarscape_reflection_prompt
from sugarscape.trade import DialogueTradeSystem


@dataclass
class _LLMCall:
    system_prompt: str
    user_prompt: str
    kwargs: Dict[str, Any]


class _SequenceProvider:
    """Deterministic async provider for tests (records calls, replays a script)."""

    def __init__(
        self,
        scripted_outputs: List[object],
        *,
        reject_chat_template_kwargs: bool = False,
    ):
        self._scripted_outputs = list(scripted_outputs)
        self.reject_chat_template_kwargs = reject_chat_template_kwargs
        self.calls: List[_LLMCall] = []

    async def generate(self, system_prompt: str, messages: List[Dict[str, str]], **kwargs) -> str:
        user_prompt = messages[-1]["content"]
        self.calls.append(_LLMCall(system_prompt=system_prompt, user_prompt=user_prompt, kwargs=dict(kwargs)))

        if self.reject_chat_template_kwargs and "chat_template_kwargs" in kwargs:
            raise TypeError("unexpected keyword argument 'chat_template_kwargs'")

        if not self._scripted_outputs:
            return "{}"

        out = self._scripted_outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return str(out)


def _make_env(*, enable_reflection: bool = True) -> SugarEnvironment:
    cfg = SugarscapeConfig(
        width=5,
        height=5,
        enable_spice=True,
        enable_trade=True,
        trade_mode="dialogue",
        enable_reflection=enable_reflection,
        enable_origin_identity=True,
        seed=123,
    )
    return SugarEnvironment(cfg)


def _make_llm_agent(*, agent_id: int, provider: Any, origin_type: str = "altruist") -> LLMSugarAgent:
    agent = LLMSugarAgent(
        provider=provider,
        goal_prompt=SugarscapeConfig.get_goal_prompt("survival"),
        agent_id=agent_id,
        pos=(0, 0),
        vision=2,
        metabolism=2,
        max_age=20,
        wealth=20,
        age=0,
        spice=20,
        metabolism_spice=2,
        name=f"MockAgent_{agent_id}",
    )

    agent.origin_identity = origin_type
    agent.origin_identity_prompt = SugarscapeConfig.get_origin_identity_prompt(origin_type)
    agent.policy_list = SugarscapeConfig.get_default_policies(origin_type)
    agent.belief_ledger = SugarscapeConfig.get_default_beliefs(origin_type)
    agent.self_identity_leaning = 0.8 if origin_type == "altruist" else -0.8
    return agent


def test_reflection_prompt_includes_immutable_origin_identity_in_user_prompt():
    env = _make_env(enable_reflection=True)
    provider = _SequenceProvider([])
    a = _make_llm_agent(agent_id=1, provider=provider, origin_type="altruist")
    b = _make_llm_agent(agent_id=2, provider=provider, origin_type="exploiter")

    user_prompt = build_sugarscape_reflection_prompt(
        self_agent=a,
        partner_agent=b,
        encounter_outcome="completed",
        encounter_summary="Trade completed.",
    )

    assert "# POST-ENCOUNTER REFLECTION" in user_prompt
    assert "### Your Core Identity" in user_prompt
    assert "## YOUR CORE IDENTITY" in user_prompt


def test_generate_json_with_retry_preserves_identity_context_and_falls_back_when_kwargs_unsupported():
    env = _make_env(enable_reflection=True)

    # Script:
    # - initial call with chat_template_kwargs => TypeError (forces fallback call without kwargs)
    # - fallback call returns non-JSON => triggers retry loop
    # - retry call with chat_template_kwargs => TypeError (forces fallback again)
    # - fallback retry returns thinking + JSON => must parse successfully
    provider = _SequenceProvider(
        scripted_outputs=[
            "NOT JSON",
            "__THINKING_START__ignore__THINKING_END__ {\"belief_updates\": {\"world\": {\"x\": \"y\"}}}",
        ],
        reject_chat_template_kwargs=True,
    )
    a = _make_llm_agent(agent_id=1, provider=provider, origin_type="altruist")
    b = _make_llm_agent(agent_id=2, provider=provider, origin_type="exploiter")
    ts = DialogueTradeSystem(env)

    # Call the same reflection entrypoint used after encounters.
    asyncio.run(
        ts._reflect_agent(  # noqa: SLF001 (intentional E2E test of reflection pipeline)
            self_agent=a,
            partner_agent=b,
            tick=1,
            outcome="timeout",
            encounter_summary="No trade occurred (outcome: timeout)",
            conversation_highlights="",
            conversation=[],
        )
    )

    # We expect at least 4 provider calls (2 TypeErrors via kwarg rejection + 2 actual returns).
    assert len(provider.calls) >= 2
    assert any("chat_template_kwargs" in c.kwargs for c in provider.calls)
    assert any("chat_template_kwargs" not in c.kwargs for c in provider.calls)

    # Identity context must be preserved in system prompt even on retry attempts.
    assert all("## YOUR CORE IDENTITY" in c.system_prompt for c in provider.calls)
    assert any("IMPORTANT: Output VALID JSON only." in c.system_prompt for c in provider.calls)


def test_dialogue_trade_reflection_applies_policy_and_belief_updates():
    env = _make_env(enable_reflection=True)

    provider = _SequenceProvider(
        scripted_outputs=[
            "{\"belief_updates\": {\"world\": {\"resource_scarcity\": \"high\"}, \"partner_2\": {\"trustworthy\": \"mixed\"}},"
            " \"policy_updates\": {\"add\": [{\"rule\": \"Avoid trading with agents who have low spice\", \"influenced_by_partner\": true}]},"
            " \"identity_shift\": -0.1}"
        ],
        reject_chat_template_kwargs=False,
    )
    a = _make_llm_agent(agent_id=1, provider=provider, origin_type="altruist")
    b = _make_llm_agent(agent_id=2, provider=provider, origin_type="exploiter")
    ts = DialogueTradeSystem(env)

    asyncio.run(
        ts._run_reflection_for_pair(  # noqa: SLF001 (intentional E2E test of reflection pipeline)
            agent_a=a,
            agent_b=b,
            tick=1,
            outcome="both_declined",
            conversation=[{"type": "trade_intent", "speaker": a.name, "intent": "DECLINE", "message": "No"}],
        )
    )

    assert any("resource_scarcity" in k for k in a.belief_ledger.get("world", {}).keys())
    assert a.belief_ledger["world"]["resource_scarcity"] == "high"
    assert any("Avoid trading with agents who have low spice" in p for p in a.policy_list)
    assert any("# POST-ENCOUNTER REFLECTION" in c.user_prompt for c in provider.calls)


def test_identity_review_parsing_and_updates_applied():
    env = _make_env(enable_reflection=True)

    provider = _SequenceProvider(
        scripted_outputs=[
            "REFLECTION: I should be slightly kinder.\n"
            "IDENTITY_ASSESSMENT: leaning_altruist\n"
            "JSON: {\"identity_shift\": 0.1, \"policy_updates\": {\"add\": [\"Share sugar with starving neighbors\"]}}"
        ]
    )
    a = _make_llm_agent(agent_id=1, provider=provider, origin_type="altruist")
    before = a.self_identity_leaning

    result = asyncio.run(a.async_identity_review(env, tick=2))

    assert result["parsed"] is not None
    assert a.self_identity_leaning > before
    assert any("Share sugar with starving neighbors" in p for p in a.policy_list)


def test_end_of_life_report_parsing():
    env = _make_env(enable_reflection=True)

    provider = _SequenceProvider(
        scripted_outputs=[
            "FINAL_REFLECTION: It was a hard life.\n"
            "LIFE_ASSESSMENT: stayed_mixed\n"
            "REGRETS: I should have trusted more.\n"
            "ADVICE: Be pragmatic, but don't lose your humanity."
        ]
    )
    a = _make_llm_agent(agent_id=1, provider=provider, origin_type="altruist")

    result = asyncio.run(a.async_end_of_life_report(env, tick=10, death_cause="simulation_end"))
    parsed = result["parsed"]

    assert parsed is not None
    assert "hard life" in parsed["final_reflection"].lower()
    assert parsed["life_assessment"] == "stayed_mixed"
    assert "trusted" in parsed["regrets"].lower()


if __name__ == "__main__":
    if pytest is None:
        # Allow `python3 tests/test_e2e_mock_llm.py` without requiring pytest installed.
        test_reflection_prompt_includes_immutable_origin_identity_in_user_prompt()
        test_generate_json_with_retry_preserves_identity_context_and_falls_back_when_kwargs_unsupported()
        test_dialogue_trade_reflection_applies_policy_and_belief_updates()
        test_identity_review_parsing_and_updates_applied()
        test_end_of_life_report_parsing()
        print("OK")
    else:
        raise SystemExit(pytest.main([__file__]))
