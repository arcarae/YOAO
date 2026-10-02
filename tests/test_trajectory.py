"""Tests for the trajectory tracking system."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from typing import List

from redblackbench.trajectory import (
    GameTrajectory,
    TrajectoryTimestep,
    TeamSnapshot,
    AgentSnapshot,
    ActionRecord,
    DialogueExchange,
    Outcome,
    TimestepType,
    TrajectoryCollector,
)
from redblackbench.game.scoring import Choice, RoundResult
from redblackbench.game.config import GameConfig
from redblackbench.agents.base import AgentResponse


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, agent_id: str, team_name: str):
        self.agent_id = agent_id
        self.team_name = team_name
        self.conversation_history = []
    
    def reset(self):
        self.conversation_history = []


class MockTeam:
    """Mock team for testing."""
    
    def __init__(self, name: str, num_agents: int = 3):
        self._name = name
        self.agents = [MockAgent(f"{name}_agent_{i}", name) for i in range(num_agents)]
        self.deliberation_history = []
    
    @property
    def name(self) -> str:
        return self._name


class TestDialogueExchange:
    """Tests for DialogueExchange."""
    
    def test_to_dict(self):
        exchange = DialogueExchange(
            role="user",
            content="What should we choose?",
            exchange_type="initial_opinion",
        )
        
        d = exchange.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "What should we choose?"
        assert d["exchange_type"] == "initial_opinion"
        assert "timestamp" in d


class TestAgentSnapshot:
    """Tests for AgentSnapshot."""
    
    def test_to_dict(self):
        snapshot = AgentSnapshot(
            agent_id="agent_1",
            team_name="Team A",
            current_opinion="BLACK",
            current_reasoning="Cooperation is best",
        )
        
        d = snapshot.to_dict()
        assert d["agent_id"] == "agent_1"
        assert d["current_opinion"] == "BLACK"


class TestActionRecord:
    """Tests for ActionRecord."""
    
    def test_individual_vote(self):
        action = ActionRecord(
            action_type="individual_vote",
            actor="agent_1",
            choice="BLACK",
            reasoning="Trust builds cooperation",
            round_num=1,
            phase="initial_opinion",
        )
        
        d = action.to_dict()
        assert d["action_type"] == "individual_vote"
        assert d["phase"] == "initial_opinion"
    
    def test_team_choice(self):
        action = ActionRecord(
            action_type="team_choice",
            actor="Team A",
            choice="BLACK",
            round_num=1,
        )
        
        d = action.to_dict()
        assert d["action_type"] == "team_choice"


class TestOutcome:
    """Tests for Outcome."""
    
    def test_round_outcome(self):
        outcome = Outcome(
            outcome_type="round",
            round_num=1,
            team_a_score=3,
            team_b_score=3,
            team_a_choice="BLACK",
            team_b_choice="BLACK",
            both_cooperated=True,
            multiplier=1,
        )
        
        d = outcome.to_dict()
        assert d["outcome_type"] == "round"
        assert d["both_cooperated"] is True
    
    def test_game_outcome(self):
        outcome = Outcome(
            outcome_type="game",
            team_a_score=30,
            team_b_score=30,
            total_score=60,
            max_possible_score=150,
            efficiency=0.7,
            cooperation_rate=0.8,
        )
        
        d = outcome.to_dict()
        assert d["outcome_type"] == "game"
        assert d["efficiency"] == 0.7


class TestGameTrajectory:
    """Tests for GameTrajectory."""
    
    def test_create_trajectory(self):
        trajectory = GameTrajectory(
            trajectory_id="test_1",
            team_a_name="Team A",
            team_b_name="Team B",
        )
        
        assert trajectory.trajectory_id == "test_1"
        assert len(trajectory.timesteps) == 0
    
    def test_add_timestep(self):
        trajectory = GameTrajectory(trajectory_id="test_1")
        
        timestep = trajectory.add_timestep(
            timestep_type=TimestepType.GAME_START,
            round_num=0,
            metadata={"test": True},
        )
        
        assert len(trajectory.timesteps) == 1
        assert timestep.timestep_type == TimestepType.GAME_START
    
    def test_get_action_sequence(self):
        trajectory = GameTrajectory(trajectory_id="test_1")
        
        trajectory.add_timestep(
            timestep_type=TimestepType.FINAL_VOTES,
            round_num=1,
            actions=[
                ActionRecord("team_choice", "Team A", "BLACK", round_num=1),
                ActionRecord("team_choice", "Team B", "BLACK", round_num=1),
            ],
        )
        
        actions = trajectory.get_action_sequence()
        assert len(actions) == 2
        assert actions[0].choice == "BLACK"
    
    def test_save_and_load(self):
        trajectory = GameTrajectory(
            trajectory_id="test_save",
            team_a_name="Alpha",
            team_b_name="Beta",
            game_config={"num_rounds": 10},
        )
        
        trajectory.add_timestep(
            timestep_type=TimestepType.GAME_START,
            round_num=0,
        )
        
        trajectory.final_outcome = Outcome(
            outcome_type="game",
            total_score=60,
            max_possible_score=150,
        )
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            trajectory.save(f.name)
            
            loaded = GameTrajectory.load(f.name)
            
            assert loaded.trajectory_id == "test_save"
            assert loaded.team_a_name == "Alpha"
            assert loaded.final_outcome.total_score == 60


class TestTrajectoryCollector:
    """Tests for TrajectoryCollector."""
    
    def test_start_game(self):
        collector = TrajectoryCollector(trajectory_id="test_collector")
        
        config = GameConfig()
        team_a = MockTeam("Team A")
        team_b = MockTeam("Team B")
        
        trajectory = collector.start_game(config, team_a, team_b)
        
        assert trajectory is not None
        assert trajectory.trajectory_id == "test_collector"
        assert len(trajectory.timesteps) == 1  # Game start
        assert trajectory.timesteps[0].timestep_type == TimestepType.GAME_START
    
    def test_record_round_start(self):
        collector = TrajectoryCollector()
        
        config = GameConfig()
        team_a = MockTeam("Team A")
        team_b = MockTeam("Team B")
        
        collector.start_game(config, team_a, team_b)
        collector.record_round_start(1, 1)
        
        trajectory = collector.get_trajectory()
        assert len(trajectory.timesteps) == 2
        assert trajectory.timesteps[1].timestep_type == TimestepType.ROUND_START
    
    def test_full_flow(self):
        collector = TrajectoryCollector()
        
        config = GameConfig(num_rounds=2, team_size=2, multipliers={})
        team_a = MockTeam("Team A", num_agents=2)
        team_b = MockTeam("Team B", num_agents=2)
        
        # Start game
        collector.start_game(config, team_a, team_b)
        
        # Round 1
        collector.record_round_start(1, 1)
        
        # Create mock game state
        class MockGameState:
            team_a_total = 0
            team_b_total = 0
            cooperation_rate = 0.0
            efficiency = 0.0
            total_score = 0
            max_possible_score = 12
            history = []
        
        game_state = MockGameState()
        
        # Record deliberation for team A
        collector.record_deliberation_start(1, team_a, "A")
        
        mock_opinion_pairs = [
            (team_a.agents[0], AgentResponse(Choice.BLACK, "Cooperate!", raw_response="RECOMMENDATION: BLACK")),
            (team_a.agents[1], AgentResponse(Choice.BLACK, "Trust!", raw_response="RECOMMENDATION: BLACK")),
        ]
        collector.record_initial_opinions(1, team_a, "A", mock_opinion_pairs, game_state)
        
        # Create mock deliberation result
        from redblackbench.teams.deliberation import DeliberationResult
        delib_result = DeliberationResult(
            final_choice=Choice.BLACK,
            initial_opinions=[resp for _, resp in mock_opinion_pairs],
            final_votes=[resp for _, resp in mock_opinion_pairs],
            vote_counts={Choice.BLACK: 2},
            was_unanimous=True,
        )
        collector.record_final_votes(1, team_a, "A", delib_result, game_state)
        
        # Record round end
        round_result = RoundResult(
            round_num=1,
            team_a_choice=Choice.BLACK,
            team_b_choice=Choice.BLACK,
            team_a_score=3,
            team_b_score=3,
            multiplier=1,
        )
        game_state.team_a_total = 3
        game_state.team_b_total = 3
        game_state.total_score = 6
        
        collector.record_round_end(round_result, game_state)
        
        # End game
        trajectory = collector.end_game(game_state, team_a, team_b)
        
        # Verify trajectory
        assert trajectory.final_outcome is not None
        assert trajectory.final_outcome.outcome_type == "game"
        
        # Check action sequence includes team choices
        actions = trajectory.get_action_sequence()
        team_choices = [a for a in actions if a.action_type == "team_choice"]
        assert len(team_choices) >= 1
        
        # Check outcomes
        outcomes = trajectory.get_outcomes()
        round_outcomes = [o for o in outcomes if o.outcome_type == "round"]
        assert len(round_outcomes) == 1
