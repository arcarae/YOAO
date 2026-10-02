"""Game configuration for RedBlackBench."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class GameConfig:
    """Configuration for a Red-Black game session.
    
    Attributes:
        num_rounds: Total number of rounds in the game (default: 10)
        team_size: Number of agents per team (default: 5)
        multipliers: Dictionary mapping round numbers to score multipliers
        base_scores: Base scoring matrix values
    """
    
    num_rounds: int = 10
    team_size: int = 5
    
    # Round multipliers (1-indexed round numbers)
    # Default: Round 5 = 3x, Round 8 = 5x, Round 10 = 10x
    multipliers: Dict[int, int] = field(default_factory=lambda: {
        5: 3,
        8: 5,
        10: 10,
    })
    
    # Base scoring values
    both_black_score: int = 3      # Both cooperate: +3 each
    both_red_score: int = -3       # Both defect: -3 each
    red_wins_score: int = 6        # Red gets +6 when opponent plays Black
    black_loses_score: int = -6    # Black gets -6 when opponent plays Red
    seed: int | None = None        # Optional randomness seed for reproducibility

    # Deliberation settings
    discussion_rounds: int = 1     # Number of discussion rounds before final vote (1 = original behavior)
    
    def get_multiplier(self, round_num: int) -> int:
        """Get the score multiplier for a given round.
        
        Args:
            round_num: The round number (1-indexed)
            
        Returns:
            The multiplier for that round (1 if no special multiplier)
        """
        return self.multipliers.get(round_num, 1)
    
    def calculate_max_possible_score(self) -> int:
        """Calculate the theoretical maximum total score if both teams always cooperate.
        
        Returns:
            Maximum possible combined score for both teams
        """
        total = 0
        for round_num in range(1, self.num_rounds + 1):
            multiplier = self.get_multiplier(round_num)
            # Both teams get +3 each when both play Black
            round_score = 2 * self.both_black_score * multiplier
            total += round_score
        return total
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_rounds < 1:
            raise ValueError("num_rounds must be at least 1")
        if self.team_size < 1:
            raise ValueError("team_size must be at least 1")
        for round_num, mult in self.multipliers.items():
            if round_num < 1 or round_num > self.num_rounds:
                raise ValueError(f"Multiplier round {round_num} is out of range [1, {self.num_rounds}]")
            if mult < 1:
                raise ValueError(f"Multiplier must be positive, got {mult} for round {round_num}")


# Default configuration matching the classic Red-Black game
DEFAULT_CONFIG = GameConfig()
