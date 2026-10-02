"""Team class managing multiple agents for RedBlackBench."""

from typing import List, Optional, TYPE_CHECKING

from redblackbench.game.scoring import Choice
from redblackbench.teams.deliberation import Deliberation, DeliberationResult

if TYPE_CHECKING:
    from redblackbench.agents.base import BaseAgent
    from redblackbench.game.coordinator import GameState


class Team:
    """A team of agents that deliberate and vote together.
    
    Manages a group of agents, coordinates their deliberation process,
    and produces a single team choice each round.
    """
    
    def __init__(
        self,
        name: str,
        agents: List["BaseAgent"],
        discussion_rounds: int = 1,
    ):
        """Initialize the team.

        Args:
            name: Team name/identifier
            agents: List of agents on this team
            discussion_rounds: Number of discussion rounds (1 = initial opinions only)
        """
        self._name = name
        self.agents = agents
        self.deliberation = Deliberation(agents, discussion_rounds=discussion_rounds)
        self.deliberation_history: List[DeliberationResult] = []
    
    @property
    def name(self) -> str:
        """Team name."""
        return self._name
    
    @property
    def size(self) -> int:
        """Number of agents on the team."""
        return len(self.agents)
    
    async def make_choice(
        self,
        game_state: "GameState",
        team_identifier: str,
    ) -> Choice:
        """Make a choice for the current round through deliberation.
        
        Runs the full deliberation process:
        1. Each agent shares their initial opinion
        2. All opinions are shared with all agents
        3. Each agent casts their final vote
        4. Majority vote determines team choice
        
        Args:
            game_state: Current game state
            team_identifier: 'A' or 'B' indicating which team this is
            
        Returns:
            The team's final choice (RED or BLACK)
        """
        round_context = game_state.get_round_context()
        
        # Run deliberation
        result = await self.deliberation.deliberate(round_context, team_identifier)
        
        # Store result in history
        self.deliberation_history.append(result)
        
        return result.final_choice
    
    def get_last_deliberation(self) -> Optional[DeliberationResult]:
        """Get the most recent deliberation result.
        
        Returns:
            The last deliberation result, or None if no deliberations yet
        """
        if self.deliberation_history:
            return self.deliberation_history[-1]
        return None
    
    def get_deliberation_for_round(self, round_num: int) -> Optional[DeliberationResult]:
        """Get the deliberation result for a specific round.
        
        Args:
            round_num: Round number (1-indexed)
            
        Returns:
            Deliberation result for that round, or None if not found
        """
        if 0 < round_num <= len(self.deliberation_history):
            return self.deliberation_history[round_num - 1]
        return None
    
    def reset(self) -> None:
        """Reset the team for a new game."""
        self.deliberation_history = []
        for agent in self.agents:
            agent.reset()
    
    def get_consensus_rate(self) -> float:
        """Calculate how often the team reached unanimous consensus.
        
        Returns:
            Percentage of rounds with unanimous votes (0.0 to 1.0)
        """
        if not self.deliberation_history:
            return 0.0
        unanimous_count = sum(1 for d in self.deliberation_history if d.was_unanimous)
        return unanimous_count / len(self.deliberation_history)
    
    def get_cooperation_rate(self) -> float:
        """Calculate how often the team chose BLACK (cooperation).
        
        Returns:
            Percentage of rounds where team chose BLACK (0.0 to 1.0)
        """
        if not self.deliberation_history:
            return 0.0
        black_count = sum(
            1 for d in self.deliberation_history 
            if d.final_choice == Choice.BLACK
        )
        return black_count / len(self.deliberation_history)
    
    def __repr__(self) -> str:
        return f"Team(name={self._name}, agents={len(self.agents)})"

