"""Trajectory tracking for the Red-Black game's per-round team state.

Captures opinion/vote/round-outcome timesteps during Red-Black team
deliberation. Coupled to `redblackbench.game.coordinator`.

Note: Sugarscape has its own trajectory recorder at
`redblackbench.sugarscape.trajectory` with different timestep semantics
(movement + trade + reflection per tick). The two are intentionally kept
separate because they record different domains.
"""

from redblackbench.trajectory.trajectory import (
    GameTrajectory,
    TrajectoryTimestep,
    TeamSnapshot,
    AgentSnapshot,
    ActionRecord,
    DialogueExchange,
    Outcome,
    TimestepType,
)
from redblackbench.trajectory.collector import TrajectoryCollector

__all__ = [
    "GameTrajectory",
    "TrajectoryTimestep",
    "TeamSnapshot",
    "AgentSnapshot",
    "ActionRecord",
    "DialogueExchange",
    "Outcome",
    "TimestepType",
    "TrajectoryCollector",
]

