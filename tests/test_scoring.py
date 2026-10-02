"""Tests for the scoring system."""

import pytest
from redblackbench.game.config import GameConfig
from redblackbench.game.scoring import Choice, ScoringMatrix, RoundResult


class TestChoice:
    """Tests for the Choice enum."""
    
    def test_choice_values(self):
        """Test that choices have correct values."""
        assert Choice.RED.value == "RED"
        assert Choice.BLACK.value == "BLACK"
    
    def test_choice_str(self):
        """Test string representation."""
        assert str(Choice.RED) == "RED"
        assert str(Choice.BLACK) == "BLACK"


class TestGameConfig:
    """Tests for game configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GameConfig()
        assert config.num_rounds == 10
        assert config.team_size == 5
        assert config.both_black_score == 3
        assert config.both_red_score == -3
        assert config.red_wins_score == 6
        assert config.black_loses_score == -6
    
    def test_multipliers(self):
        """Test multiplier retrieval."""
        config = GameConfig()
        assert config.get_multiplier(1) == 1
        assert config.get_multiplier(5) == 3
        assert config.get_multiplier(8) == 5
        assert config.get_multiplier(10) == 10
    
    def test_max_possible_score(self):
        """Test maximum possible score calculation."""
        config = GameConfig()
        # 7 normal rounds * 6 + round 5 * 18 + round 8 * 30 + round 10 * 60
        # = 42 + 18 + 30 + 60 = 150
        # Wait, let's recalculate:
        # Each round with both BLACK: 2 teams * 3 points = 6 total
        # Normal rounds (1,2,3,4,6,7,9): 7 * 6 = 42
        # Round 5 (3x): 6 * 3 = 18
        # Round 8 (5x): 6 * 5 = 30
        # Round 10 (10x): 6 * 10 = 60
        # Total = 42 + 18 + 30 + 60 = 150
        expected = 42 + 18 + 30 + 60
        assert config.calculate_max_possible_score() == expected
    
    def test_invalid_num_rounds(self):
        """Test validation of num_rounds."""
        with pytest.raises(ValueError):
            GameConfig(num_rounds=0)
    
    def test_invalid_team_size(self):
        """Test validation of team_size."""
        with pytest.raises(ValueError):
            GameConfig(team_size=0)


class TestScoringMatrix:
    """Tests for the scoring matrix."""
    
    def test_both_black(self):
        """Test scoring when both teams choose BLACK."""
        config = GameConfig()
        scoring = ScoringMatrix(config)
        
        score_a, score_b = scoring.calculate_scores(Choice.BLACK, Choice.BLACK, 1)
        assert score_a == 3
        assert score_b == 3
    
    def test_both_red(self):
        """Test scoring when both teams choose RED."""
        config = GameConfig()
        scoring = ScoringMatrix(config)
        
        score_a, score_b = scoring.calculate_scores(Choice.RED, Choice.RED, 1)
        assert score_a == -3
        assert score_b == -3
    
    def test_red_vs_black(self):
        """Test scoring when choices differ."""
        config = GameConfig()
        scoring = ScoringMatrix(config)
        
        # Team A red, Team B black
        score_a, score_b = scoring.calculate_scores(Choice.RED, Choice.BLACK, 1)
        assert score_a == 6
        assert score_b == -6
        
        # Team A black, Team B red
        score_a, score_b = scoring.calculate_scores(Choice.BLACK, Choice.RED, 1)
        assert score_a == -6
        assert score_b == 6
    
    def test_multiplier_applied(self):
        """Test that multipliers are correctly applied."""
        config = GameConfig()
        scoring = ScoringMatrix(config)
        
        # Round 5 has 3x multiplier
        score_a, score_b = scoring.calculate_scores(Choice.BLACK, Choice.BLACK, 5)
        assert score_a == 9  # 3 * 3
        assert score_b == 9
        
        # Round 10 has 10x multiplier
        score_a, score_b = scoring.calculate_scores(Choice.BLACK, Choice.BLACK, 10)
        assert score_a == 30  # 3 * 10
        assert score_b == 30
    
    def test_create_round_result(self):
        """Test RoundResult creation."""
        config = GameConfig()
        scoring = ScoringMatrix(config)
        
        result = scoring.create_round_result(5, Choice.BLACK, Choice.BLACK)
        
        assert result.round_num == 5
        assert result.team_a_choice == Choice.BLACK
        assert result.team_b_choice == Choice.BLACK
        assert result.team_a_score == 9
        assert result.team_b_score == 9
        assert result.multiplier == 3
        assert result.both_cooperated is True
        assert result.both_defected is False


class TestRoundResult:
    """Tests for RoundResult dataclass."""
    
    def test_total_score(self):
        """Test total_score property."""
        result = RoundResult(
            round_num=1,
            team_a_choice=Choice.BLACK,
            team_b_choice=Choice.BLACK,
            team_a_score=3,
            team_b_score=3,
            multiplier=1,
        )
        assert result.total_score == 6
    
    def test_both_cooperated(self):
        """Test both_cooperated property."""
        result = RoundResult(
            round_num=1,
            team_a_choice=Choice.BLACK,
            team_b_choice=Choice.BLACK,
            team_a_score=3,
            team_b_score=3,
            multiplier=1,
        )
        assert result.both_cooperated is True
        assert result.both_defected is False
    
    def test_both_defected(self):
        """Test both_defected property."""
        result = RoundResult(
            round_num=1,
            team_a_choice=Choice.RED,
            team_b_choice=Choice.RED,
            team_a_score=-3,
            team_b_score=-3,
            multiplier=1,
        )
        assert result.both_cooperated is False
        assert result.both_defected is True

