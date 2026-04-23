"""Tests for the game coordinator."""

import pytest
import asyncio
from typing import List
from unittest.mock import AsyncMock

from redblackbench.game.config import GameConfig
from redblackbench.game.coordinator import GameCoordinator, GameState
from redblackbench.game.scoring import Choice, RoundResult


class MockTeam:
    """Mock team for testing."""
    
    def __init__(self, name: str, choices: List[Choice]):
        """Initialize mock team with predetermined choices."""
        self._name = name
        self.choices = choices
        self.choice_index = 0
    
    @property
    def name(self) -> str:
        return self._name
    
    async def make_choice(self, game_state: GameState, team_identifier: str) -> Choice:
        """Return the next predetermined choice."""
        choice = self.choices[self.choice_index % len(self.choices)]
        self.choice_index += 1
        return choice


class TestGameState:
    """Tests for GameState."""
    
    def test_initial_state(self):
        """Test initial game state values."""
        config = GameConfig()
        state = GameState(config=config)
        
        assert state.current_round == 1
        assert state.team_a_total == 0
        assert state.team_b_total == 0
        assert state.history == []
        assert state.is_complete is False
    
    def test_cooperation_rate_empty(self):
        """Test cooperation rate with no history."""
        config = GameConfig()
        state = GameState(config=config)
        
        assert state.cooperation_rate == 0.0
    
    def test_cooperation_rate_all_black(self):
        """Test cooperation rate when all choices are BLACK."""
        config = GameConfig()
        state = GameState(config=config)
        
        # Add some history
        state.history = [
            RoundResult(1, Choice.BLACK, Choice.BLACK, 3, 3, 1),
            RoundResult(2, Choice.BLACK, Choice.BLACK, 3, 3, 1),
        ]
        
        assert state.cooperation_rate == 1.0
    
    def test_cooperation_rate_mixed(self):
        """Test cooperation rate with mixed choices."""
        config = GameConfig()
        state = GameState(config=config)
        
        # 2 BLACK, 2 RED total
        state.history = [
            RoundResult(1, Choice.BLACK, Choice.RED, -6, 6, 1),
        ]
        
        # 1 BLACK out of 2 total choices = 50%
        assert state.cooperation_rate == 0.5
    
    def test_get_round_context(self):
        """Test round context generation."""
        config = GameConfig()
        state = GameState(config=config)
        state.team_a_total = 10
        state.team_b_total = 5
        
        context = state.get_round_context()
        
        assert context["current_round"] == 1
        assert context["total_rounds"] == 10
        assert context["team_a_score"] == 10
        assert context["team_b_score"] == 5
        assert context["total_score"] == 15


class TestGameCoordinator:
    """Tests for GameCoordinator."""
    
    @pytest.mark.asyncio
    async def test_play_round_both_black(self):
        """Test playing a round where both teams choose BLACK."""
        team_a = MockTeam("Team A", [Choice.BLACK])
        team_b = MockTeam("Team B", [Choice.BLACK])
        
        coordinator = GameCoordinator(team_a, team_b)
        result = await coordinator.play_round()
        
        assert result.team_a_choice == Choice.BLACK
        assert result.team_b_choice == Choice.BLACK
        assert result.team_a_score == 3
        assert result.team_b_score == 3
        assert coordinator.state.team_a_total == 3
        assert coordinator.state.team_b_total == 3
    
    @pytest.mark.asyncio
    async def test_play_round_both_red(self):
        """Test playing a round where both teams choose RED."""
        team_a = MockTeam("Team A", [Choice.RED])
        team_b = MockTeam("Team B", [Choice.RED])
        
        coordinator = GameCoordinator(team_a, team_b)
        result = await coordinator.play_round()
        
        assert result.team_a_choice == Choice.RED
        assert result.team_b_choice == Choice.RED
        assert result.team_a_score == -3
        assert result.team_b_score == -3
    
    @pytest.mark.asyncio
    async def test_play_full_game(self):
        """Test playing a complete game."""
        # Both teams always cooperate
        team_a = MockTeam("Team A", [Choice.BLACK])
        team_b = MockTeam("Team B", [Choice.BLACK])
        
        config = GameConfig(num_rounds=3, multipliers={})
        coordinator = GameCoordinator(team_a, team_b, config=config)
        
        final_state = await coordinator.play_game()
        
        assert final_state.is_complete is True
        assert len(final_state.history) == 3
        assert final_state.team_a_total == 9  # 3 rounds * 3 points
        assert final_state.team_b_total == 9
    
    @pytest.mark.asyncio
    async def test_cannot_play_after_complete(self):
        """Test that playing after game is complete raises error."""
        team_a = MockTeam("Team A", [Choice.BLACK])
        team_b = MockTeam("Team B", [Choice.BLACK])
        
        config = GameConfig(num_rounds=1, multipliers={})
        coordinator = GameCoordinator(team_a, team_b, config=config)
        
        await coordinator.play_game()
        
        with pytest.raises(RuntimeError):
            await coordinator.play_round()
    
    @pytest.mark.asyncio
    async def test_reset(self):
        """Test resetting the game."""
        team_a = MockTeam("Team A", [Choice.BLACK])
        team_b = MockTeam("Team B", [Choice.BLACK])
        
        config = GameConfig(num_rounds=2, multipliers={})
        coordinator = GameCoordinator(team_a, team_b, config=config)
        
        await coordinator.play_game()
        assert coordinator.state.is_complete is True
        
        coordinator.reset()
        
        assert coordinator.state.is_complete is False
        assert coordinator.state.current_round == 1
        assert coordinator.state.team_a_total == 0
    
    @pytest.mark.asyncio
    async def test_get_summary(self):
        """Test summary generation."""
        team_a = MockTeam("Team A", [Choice.BLACK])
        team_b = MockTeam("Team B", [Choice.BLACK])
        
        config = GameConfig(num_rounds=2, multipliers={})
        coordinator = GameCoordinator(team_a, team_b, config=config)
        
        await coordinator.play_game()
        summary = coordinator.get_summary()
        
        assert summary["is_complete"] is True
        assert summary["rounds_played"] == 2
        assert summary["team_a"]["name"] == "Team A"
        assert summary["team_a"]["score"] == 6
        assert summary["cooperation_rate"] == 1.0

