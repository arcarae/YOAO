"""
RedBlackBench: the Red-Black Game, a team-based iterated social dilemma for measuring how
cooperative behaviour propagates through multi-agent LLM systems.

Accompanies "You Only Align Once: Propagating Cooperative Behaviors in Multi-Agent Systems
through Seed Agents" (EMNLP 2026; arXiv:2605.27586).
"""

__version__ = "0.2.0"

from redblackbench.game.config import GameConfig
from redblackbench.game.coordinator import GameCoordinator, GameState
from redblackbench.game.scoring import Choice, ScoringMatrix

__all__ = [
    "GameConfig",
    "GameCoordinator",
    "GameState",
    "Choice",
    "ScoringMatrix",
]
