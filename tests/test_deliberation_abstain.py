"""Failed agent calls must abstain, never count as cooperative votes."""

import pytest

from redblackbench.agents.base import AgentResponse, BaseAgent
from redblackbench.game.scoring import Choice
from redblackbench.teams.deliberation import Deliberation, NoValidVotesError, count_abstentions


class FixedAgent(BaseAgent):
    """Votes a fixed choice; optionally raises to simulate a provider failure."""

    def __init__(self, agent_id, choice=Choice.RED, fail=False):
        super().__init__(agent_id, "Team A")
        self._choice = choice
        self._fail = fail

    async def get_willingness_to_speak(self, round_context, team_identifier, seen_messages):
        return 1

    async def get_initial_opinion(self, round_context, team_identifier, prior_messages=None):
        if self._fail:
            raise RuntimeError("provider down")
        return AgentResponse(choice=self._choice, reasoning=f"{self.agent_id} says {self._choice}")

    async def get_final_vote(self, round_context, team_identifier, teammate_opinions):
        if self._fail:
            raise RuntimeError("provider down")
        return AgentResponse(choice=self._choice, reasoning="final")

    async def get_followup_response(self, round_context, team_identifier, discussion_history):
        return await self.get_initial_opinion(round_context, team_identifier)


def test_abstain_response_has_no_choice():
    r = AgentResponse.abstain("[Failed to get vote]")
    assert r.choice is None and r.failed
    assert r.to_dict()["choice"] is None
    assert count_abstentions([r, AgentResponse(Choice.BLACK, "x")]) == 1


def test_majority_ignores_abstentions():
    d = Deliberation([])
    votes = [
        AgentResponse(Choice.RED, "r"),
        AgentResponse(Choice.RED, "r"),
        AgentResponse.abstain("[Failed to get vote]"),
        AgentResponse.abstain("[Failed to get vote]"),
        AgentResponse.abstain("[Failed to get vote]"),
    ]
    choice, counts, unanimous = d._determine_majority(votes)
    assert choice == Choice.RED
    assert counts == {Choice.RED: 2}
    assert unanimous  # unanimous among the agents that voted


def test_all_votes_failed_raises_instead_of_cooperating():
    d = Deliberation([])
    with pytest.raises(NoValidVotesError):
        d._determine_majority([AgentResponse.abstain("x")] * 5)


@pytest.mark.asyncio
async def test_failed_final_vote_abstains(monkeypatch):
    import asyncio

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    agents = [FixedAgent("a", Choice.RED), FixedAgent("b", Choice.RED), FixedAgent("c", fail=True)]
    d = Deliberation(agents)
    result = await d.deliberate({"current_round": 1, "history": []}, "A")
    assert result.final_choice == Choice.RED
    assert result.abstentions == 1
    assert [v.failed for v in result.final_votes].count(True) == 1


@pytest.mark.asyncio
async def test_failed_opinion_is_not_broadcast(monkeypatch):
    import asyncio

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    agents = [FixedAgent("a", Choice.BLACK), FixedAgent("b", fail=True)]
    d = Deliberation(agents)
    pairs = await d._gather_initial_opinions({"current_round": 1, "history": []}, "A")
    failed = [resp for _, resp in pairs if resp.failed]
    assert len(failed) == 1 and failed[0].choice is None


async def _no_sleep(*_args, **_kwargs):
    return None
