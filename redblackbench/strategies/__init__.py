"""Scripted opponent strategies for training data generation.

These strategies create diverse opponent behaviors to train robust cooperation:
- Different base strategies (always cooperate, always defect, tit-for-tat, etc.)
- Different exploitation timing (early, mid, late)
- Different recovery patterns (returns to cooperation or not)
"""

from redblackbench.strategies.scripted import (
    ScriptedStrategy,
    StrategyConfig,
    get_strategy,
    list_strategies,
    STRATEGY_REGISTRY,
    TRAINING_DISTRIBUTION,
    sample_strategy_for_training,
)
from redblackbench.strategies.scripted_team import (
    ScriptedTeam,
    create_scripted_team,
)

__all__ = [
    "ScriptedStrategy",
    "StrategyConfig",
    "get_strategy",
    "list_strategies",
    "STRATEGY_REGISTRY",
    "TRAINING_DISTRIBUTION",
    "sample_strategy_for_training",
    "ScriptedTeam",
    "create_scripted_team",
]
