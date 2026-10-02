"""GameCoordinator result accessors used by the evaluation scripts."""

from typing import List

import pytest

from redblackbench.game.config import GameConfig
from redblackbench.game.coordinator import GameCoordinator, GameState
from redblackbench.game.scoring import Choice


class FixedTeam:
    def __init__(self, name: str, choices: List[Choice]):
        self._name = name
        self._choices = choices
        self._i = 0
        # Attributes the trajectory collector snapshots (a scripted team has no agents)
        self.agents: list = []
        self.deliberation_history: list = []

    @property
    def name(self) -> str:
        return self._name

    async def make_choice(self, game_state: GameState, team_identifier: str) -> Choice:
        c = self._choices[self._i % len(self._choices)]
        self._i += 1
        return c


@pytest.mark.asyncio
async def test_final_scores_and_team_history():
    config = GameConfig(num_rounds=4, multipliers={})
    team_a = FixedTeam("A", [Choice.BLACK, Choice.RED, Choice.BLACK, Choice.BLACK])
    team_b = FixedTeam("B", [Choice.RED])
    coord = GameCoordinator(team_a=team_a, team_b=team_b, config=config)
    await coord.play_game()

    assert coord.get_team_history("A") == ["A", "B", "A", "A"]
    assert coord.get_team_history("B") == ["B", "B", "B", "B"]
    scores = coord.get_final_scores()
    # A: -6 -3 -6 -6 = -21 ; B: +6 -3 +6 +6 = +15
    assert scores == {"Team A": -21, "Team B": 15}
    with pytest.raises(ValueError):
        coord.get_team_history("C")


@pytest.mark.asyncio
async def test_collector_save_trajectory(tmp_path):
    import json

    from redblackbench.trajectory import TrajectoryCollector

    collector = TrajectoryCollector()
    with pytest.raises(RuntimeError):
        collector.save_trajectory(str(tmp_path / "never.json"))

    config = GameConfig(num_rounds=2, multipliers={})
    coord = GameCoordinator(
        team_a=FixedTeam("A", [Choice.BLACK]),
        team_b=FixedTeam("B", [Choice.BLACK]),
        config=config,
        trajectory_collector=collector,
    )
    await coord.play_game()
    out = tmp_path / "game.json"
    collector.save_trajectory(str(out))
    data = json.loads(out.read_text())
    assert data["trajectory_id"] == collector.trajectory_id
    rounds = [t for t in data["timesteps"] if t["timestep_type"] == "round_end"]
    assert len(rounds) == 2
