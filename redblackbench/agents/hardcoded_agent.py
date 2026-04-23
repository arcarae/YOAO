"""
Hardcoded agent that always votes for the cooperative (BLACK) choice.
Used for ablation studies to isolate the effect of reasoning quality on cooperation rates.
"""

from typing import List, Optional
from redblackbench.agents.base import BaseAgent, AgentResponse
from redblackbench.game.scoring import Choice


class HardcodedBlackAgent(BaseAgent):
    """
    A hardcoded agent that always votes BLACK (cooperative choice) without reasoning.

    This agent is used for ablation experiments to measure the impact of reasoning
    quality on cooperation rates by comparing against trained/untrained LLM agents.
    """

    def __init__(
        self,
        agent_id: str,
        team_name: str,
        choice_name_black: str = "BLACK",
        choice_name_red: str = "RED",
    ):
        """
        Initialize a hardcoded agent.

        Args:
            agent_id: Unique identifier for the agent
            team_name: Name of the team this agent belongs to
            choice_name_black: Scenario-specific name for BLACK choice (e.g., "SHARE", "HUMANITY")
            choice_name_red: Scenario-specific name for RED choice (e.g., "HOARD", "TRIBE")
        """
        super().__init__(agent_id=agent_id, team_name=team_name)
        self.choice_name_black = choice_name_black
        self.choice_name_red = choice_name_red
        self._is_trained = False
        self._model_name = "hardcoded_black"
        self.conversation_history = []

    async def get_initial_opinion(
        self,
        round_context: dict,
        team_identifier: str,
    ) -> AgentResponse:
        """Return a hardcoded opinion to vote BLACK."""
        response_text = f"I vote {self.choice_name_black}."

        # Add to conversation history in format expected by trajectory collector
        self.conversation_history.append({
            "role": "user",
            "content": f"Round {round_context.get('round_num', '?')}: Provide your initial opinion.",
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
        })

        return AgentResponse(
            choice=Choice.BLACK,
            reasoning=response_text,
            confidence=1.0,
            raw_response=response_text,
            private_thought=None
        )

    async def get_final_vote(
        self,
        round_context: dict,
        team_identifier: str,
        teammate_opinions: List[AgentResponse],
    ) -> AgentResponse:
        """Return a hardcoded final vote for BLACK."""
        response_text = f"I vote {self.choice_name_black}."

        # Add to conversation history in format expected by trajectory collector
        self.conversation_history.append({
            "role": "user",
            "content": f"Round {round_context.get('round_num', '?')}: Provide your final vote.",
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
        })

        return AgentResponse(
            choice=Choice.BLACK,
            reasoning=response_text,
            confidence=1.0,
            raw_response=response_text,
            private_thought=None
        )

    async def get_willingness_to_speak(
        self,
        round_context: dict,
        team_identifier: str,
        seen_messages: list,
    ) -> int:
        """Return neutral willingness score."""
        return 1

    def reset(self) -> None:
        """Reset the agent's conversation history."""
        self.conversation_history = []
