"""Scripted Team implementation for training data generation.

A team that follows pre-defined strategies instead of using an LLM.
Used to create diverse opponent behaviors for training data.
"""

from typing import Optional, List
from redblackbench.game.scoring import Choice
from redblackbench.game.coordinator import GameState
from redblackbench.strategies.scripted import (
    ScriptedStrategy,
    StrategyConfig,
    get_strategy,
    STRATEGY_REGISTRY,
)


class ScriptedTeam:
    """A team that follows a scripted strategy.

    Implements TeamProtocol for use with GameCoordinator.
    Does NOT use LLM agents - follows deterministic strategies instead.
    """

    def __init__(
        self,
        strategy_id: str,
        team_name: str = "Scripted Team B",
    ):
        """Initialize with a strategy.

        Args:
            strategy_id: ID of the strategy to use
            team_name: Display name for the team
        """
        self._name = team_name
        self.strategy_id = strategy_id

        strategy = get_strategy(strategy_id)
        if strategy is None:
            raise ValueError(f"Unknown strategy: {strategy_id}")
        self.strategy = strategy

        # Track history for strategies that need it
        self._team_a_history: List[str] = []
        self._team_b_history: List[str] = []

        # Empty list for compatibility with TrajectoryCollector
        # ScriptedTeam doesn't have real agents - it's deterministic
        self.agents: List = []
        self.deliberation_history: List = []

    @property
    def name(self) -> str:
        return self._name

    async def make_choice(self, game_state: GameState, team_identifier: str) -> Choice:
        """Make a choice based on the scripted strategy.

        Args:
            game_state: Current game state
            team_identifier: 'A' or 'B' indicating which team this is

        Returns:
            The team's choice (RED or BLACK)
        """
        # Extract history from game_state (uses 'history' not 'round_history')
        team_a_history = []
        team_b_history = []

        for result in game_state.history:
            # A = cooperate = BLACK, B = defect = RED
            team_a_history.append("A" if result.team_a_choice == Choice.BLACK else "B")
            team_b_history.append("A" if result.team_b_choice == Choice.BLACK else "B")

        # Get round info from config
        current_round = game_state.current_round
        total_rounds = game_state.config.num_rounds
        multiplier = game_state.config.get_multiplier(current_round)

        # Critical rounds are those with multiplier > 1
        is_critical = multiplier > 1

        # Get scripted move
        move = self.strategy.get_move(
            round_number=current_round,
            total_rounds=total_rounds,
            team_a_history=team_a_history,
            team_b_history=team_b_history,
            is_critical=is_critical,
            multiplier=multiplier,
        )

        # Convert to Choice (A = cooperate = BLACK, B = defect = RED)
        return Choice.BLACK if move == "A" else Choice.RED

    def reset(self):
        """Reset for a new game."""
        self.strategy.reset()
        self._team_a_history = []
        self._team_b_history = []


def create_scripted_team(
    strategy_id: str,
    team_name: Optional[str] = None,
) -> ScriptedTeam:
    """Create a scripted team with the given strategy.

    Args:
        strategy_id: ID of the strategy to use
        team_name: Optional display name (defaults to strategy name)

    Returns:
        ScriptedTeam instance
    """
    if strategy_id not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy '{strategy_id}'. Available: {available}")

    config = STRATEGY_REGISTRY[strategy_id]
    name = team_name or f"Scripted: {config.name}"

    return ScriptedTeam(strategy_id=strategy_id, team_name=name)
