"""A wrapper that lets an agent vote but not argue (the paper's "mute test").

The wrapped agent (normally an SFT seed) still reasons and votes through its own model, but
everything it broadcasts to teammates is replaced by a bare recommendation such as
``"I vote BLACK."`` This isolates the effect of persuasive dialogue: if cooperation collapses
when seeds are muted, the propagation vector is the argument content, not the vote itself.
"""

from typing import List, Optional

from redblackbench.agents.base import AgentResponse, BaseAgent
from redblackbench.game.scoring import Choice


class MutedAgent(BaseAgent):
    """Delegates every decision to ``inner`` but strips the reasoning from what it says."""

    def __init__(
        self,
        inner: BaseAgent,
        choice_name_black: str = "BLACK",
        choice_name_red: str = "RED",
    ):
        super().__init__(agent_id=inner.agent_id, team_name=inner.team_name)
        self.inner = inner
        self.choice_name_black = choice_name_black
        self.choice_name_red = choice_name_red
        # Mirror attributes some scripts read for bookkeeping.
        self._is_trained = getattr(inner, "_is_trained", True)
        self._model_name = f"muted:{getattr(inner, '_model_name', type(inner).__name__)}"

    @property
    def conversation_history(self) -> List[dict]:  # type: ignore[override]
        return self.inner.conversation_history

    @conversation_history.setter
    def conversation_history(self, value: List[dict]) -> None:
        # BaseAgent.__init__ assigns this before ``inner`` exists; ignore that first write.
        if hasattr(self, "inner"):
            self.inner.conversation_history = value

    def _bare(self, response: AgentResponse) -> AgentResponse:
        if response.choice is None:
            text = "[abstained]"
        else:
            name = self.choice_name_black if response.choice == Choice.BLACK else self.choice_name_red
            text = f"I vote {name}."
        return AgentResponse(
            choice=response.choice,
            reasoning=text,
            confidence=response.confidence,
            raw_response=response.raw_response,
            private_thought=response.reasoning,  # keep the real argument for the trajectory only
            failed=response.failed,
        )

    async def get_initial_opinion(
        self,
        round_context: dict,
        team_identifier: str,
        prior_messages: Optional[List[dict]] = None,
    ) -> AgentResponse:
        return self._bare(await self.inner.get_initial_opinion(round_context, team_identifier, prior_messages))

    async def get_final_vote(
        self,
        round_context: dict,
        team_identifier: str,
        teammate_opinions: List[AgentResponse],
    ) -> AgentResponse:
        return self._bare(await self.inner.get_final_vote(round_context, team_identifier, teammate_opinions))

    async def get_willingness_to_speak(
        self,
        round_context: dict,
        team_identifier: str,
        seen_messages: list,
    ) -> int:
        return await self.inner.get_willingness_to_speak(round_context, team_identifier, seen_messages)

    async def get_followup_response(
        self,
        round_context: dict,
        team_identifier: str,
        discussion_history: List[dict],
    ) -> AgentResponse:
        return self._bare(await self.inner.get_followup_response(round_context, team_identifier, discussion_history))

    def reset(self) -> None:
        self.inner.reset()
