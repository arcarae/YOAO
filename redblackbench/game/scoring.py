"""Scoring logic for the Red-Black game."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from redblackbench.game.config import GameConfig


class Choice(Enum):
    """Represents a team's choice in a round."""
    RED = "RED"
    BLACK = "BLACK"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class RoundResult:
    """Result of a single round.
    
    Attributes:
        round_num: The round number (1-indexed)
        team_a_choice: Team A's choice
        team_b_choice: Team B's choice
        team_a_score: Points earned by Team A this round
        team_b_score: Points earned by Team B this round
        multiplier: The multiplier applied to this round
    """
    round_num: int
    team_a_choice: Choice
    team_b_choice: Choice
    team_a_score: int
    team_b_score: int
    multiplier: int
    
    @property
    def total_score(self) -> int:
        """Combined score for both teams this round."""
        return self.team_a_score + self.team_b_score
    
    @property
    def both_cooperated(self) -> bool:
        """Whether both teams chose Black (cooperation)."""
        return self.team_a_choice == Choice.BLACK and self.team_b_choice == Choice.BLACK
    
    @property
    def both_defected(self) -> bool:
        """Whether both teams chose Red (defection)."""
        return self.team_a_choice == Choice.RED and self.team_b_choice == Choice.RED


class ScoringMatrix:
    """Handles score calculations for the Red-Black game.
    
    Scoring Matrix:
    - Both BLACK: +3 / +3 (cooperation - best collective outcome)
    - Both RED: -3 / -3 (mutual defection - worst collective outcome)  
    - RED vs BLACK: +6 / -6 (exploitation)
    """
    
    def __init__(self, config: GameConfig):
        """Initialize scoring matrix with game configuration.
        
        Args:
            config: Game configuration containing scoring values
        """
        self.config = config
    
    def calculate_scores(
        self, 
        team_a_choice: Choice, 
        team_b_choice: Choice,
        round_num: int
    ) -> Tuple[int, int]:
        """Calculate scores for both teams based on their choices.
        
        Args:
            team_a_choice: Team A's choice (RED or BLACK)
            team_b_choice: Team B's choice (RED or BLACK)
            round_num: Current round number for multiplier lookup
            
        Returns:
            Tuple of (team_a_score, team_b_score)
        """
        multiplier = self.config.get_multiplier(round_num)
        
        if team_a_choice == Choice.BLACK and team_b_choice == Choice.BLACK:
            # Both cooperate - best collective outcome
            base_a = self.config.both_black_score
            base_b = self.config.both_black_score
        elif team_a_choice == Choice.RED and team_b_choice == Choice.RED:
            # Both defect - worst collective outcome
            base_a = self.config.both_red_score
            base_b = self.config.both_red_score
        elif team_a_choice == Choice.RED and team_b_choice == Choice.BLACK:
            # Team A exploits Team B
            base_a = self.config.red_wins_score
            base_b = self.config.black_loses_score
        else:  # team_a_choice == Choice.BLACK and team_b_choice == Choice.RED
            # Team B exploits Team A
            base_a = self.config.black_loses_score
            base_b = self.config.red_wins_score
        
        return base_a * multiplier, base_b * multiplier
    
    def create_round_result(
        self,
        round_num: int,
        team_a_choice: Choice,
        team_b_choice: Choice
    ) -> RoundResult:
        """Create a RoundResult with calculated scores.
        
        Args:
            round_num: Current round number
            team_a_choice: Team A's choice
            team_b_choice: Team B's choice
            
        Returns:
            RoundResult with all scores calculated
        """
        team_a_score, team_b_score = self.calculate_scores(
            team_a_choice, team_b_choice, round_num
        )
        
        return RoundResult(
            round_num=round_num,
            team_a_choice=team_a_choice,
            team_b_choice=team_b_choice,
            team_a_score=team_a_score,
            team_b_score=team_b_score,
            multiplier=self.config.get_multiplier(round_num)
        )
    
    def get_scoring_explanation(self) -> str:
        """Get a human-readable explanation of the scoring rules.
        
        Returns:
            String explaining the scoring matrix
        """
        return f"""Scoring Matrix:
- Both teams choose BLACK: Each team gets +{self.config.both_black_score} points
- Both teams choose RED: Each team gets {self.config.both_red_score} points  
- One RED, one BLACK: RED gets +{self.config.red_wins_score}, BLACK gets {self.config.black_loses_score}

Multiplier Rounds: {', '.join(f'Round {r}: {m}x' for r, m in sorted(self.config.multipliers.items()))}

Maximum possible total score (if both teams always choose BLACK): {self.config.calculate_max_possible_score()}
"""

