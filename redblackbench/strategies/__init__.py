"""Scripted opponent strategies for the Red-Black Game.

Deterministic (or seeded) opponents used as Team B during SFT data generation and
evaluation: cooperative, exploitative, tit-for-tat, and timed-betrayal patterns.
See ``scripted.py`` for the round-by-round patterns.
"""

from redblackbench.strategies.scripted import (
    ScriptedStrategy,
    StrategyConfig,
    STRATEGY_ALIASES,
    STRATEGY_REGISTRY,
    TRAINING_DISTRIBUTION,
    get_strategy,
    list_strategies,
    resolve_strategy_id,
    sample_strategy_for_training,
    strategy_patterns,
)
from redblackbench.strategies.scripted_team import (
    ScriptedTeam,
    create_scripted_team,
)

__all__ = [
    "ScriptedStrategy",
    "StrategyConfig",
    "STRATEGY_ALIASES",
    "STRATEGY_REGISTRY",
    "TRAINING_DISTRIBUTION",
    "get_strategy",
    "list_strategies",
    "resolve_strategy_id",
    "sample_strategy_for_training",
    "strategy_patterns",
    "ScriptedTeam",
    "create_scripted_team",
]
