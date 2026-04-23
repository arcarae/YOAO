"""
RedBlackBench: Team-based social dilemma for evaluating cooperative alignment in multi-agent LLM systems.

Accompanies the YOAO paper (ICLR 2026 LLA Workshop), Tables 2, 3, and B.
"""

__version__ = "0.1.0"

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
