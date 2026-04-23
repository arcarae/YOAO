"""Game event logging for RedBlackBench."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from redblackbench.game.config import GameConfig
    from redblackbench.game.coordinator import GameState
    from redblackbench.game.scoring import RoundResult
    from redblackbench.teams.deliberation import DeliberationResult


class GameLogger:
    """Logs game events to JSON files for later analysis.
    
    Creates structured logs of all game events including:
    - Game start/end
    - Round results
    - Team deliberations
    - Final metrics
    """
    
    def __init__(
        self,
        output_dir: str = "results",
        experiment_name: Optional[str] = None,
    ):
        """Initialize the game logger.
        
        Args:
            output_dir: Directory to save log files
            experiment_name: Optional name for this experiment
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.events: List[dict] = []
        self.game_id: Optional[str] = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_event(self, event_type: str, data: dict) -> dict:
        """Create a structured event record.
        
        Args:
            event_type: Type of event (e.g., 'game_start', 'round_result')
            data: Event-specific data
            
        Returns:
            Structured event dictionary
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "game_id": self.game_id,
            "data": data,
        }
    
    async def log_game_start(
        self,
        config: "GameConfig",
        team_a_name: str,
        team_b_name: str,
        team_a_model: Optional[str] = None,
        team_b_model: Optional[str] = None,
    ) -> None:
        """Log the start of a new game.

        Args:
            config: Game configuration
            team_a_name: Name of Team A
            team_b_name: Name of Team B
            team_a_model: Model name/ID used by Team A
            team_b_model: Model name/ID used by Team B
        """
        self.game_id = f"{self.experiment_name}_{datetime.now().strftime('%H%M%S%f')}"
        self.events = []
        self.team_a_model = team_a_model
        self.team_b_model = team_b_model

        event = self._create_event("game_start", {
            "config": {
                "num_rounds": config.num_rounds,
                "team_size": config.team_size,
                "multipliers": config.multipliers,
                "scoring": {
                    "both_black": config.both_black_score,
                    "both_red": config.both_red_score,
                    "red_wins": config.red_wins_score,
                    "black_loses": config.black_loses_score,
                },
                "max_possible_score": config.calculate_max_possible_score(),
            },
            "team_a": team_a_name,
            "team_b": team_b_name,
            "team_a_model": team_a_model,
            "team_b_model": team_b_model,
        })
        self.events.append(event)
    
    async def log_round_start(self, round_num: int, multiplier: int) -> None:
        """Log the start of a new round.
        
        Args:
            round_num: Round number (1-indexed)
            multiplier: Score multiplier for this round
        """
        event = self._create_event("round_start", {
            "round_num": round_num,
            "multiplier": multiplier,
        })
        self.events.append(event)
    
    async def log_deliberation(
        self,
        team_name: str,
        team_identifier: str,
        result: "DeliberationResult",
    ) -> None:
        """Log a team's deliberation process.
        
        Args:
            team_name: Name of the team
            team_identifier: 'A' or 'B'
            result: Deliberation result with all opinions and votes
        """
        event = self._create_event("deliberation", {
            "team_name": team_name,
            "team_identifier": team_identifier,
            "result": result.to_dict(),
        })
        self.events.append(event)
    
    async def log_round_result(self, result: "RoundResult") -> None:
        """Log the result of a completed round.
        
        Args:
            result: Round result with choices and scores
        """
        event = self._create_event("round_result", {
            "round_num": result.round_num,
            "multiplier": result.multiplier,
            "team_a_choice": str(result.team_a_choice),
            "team_b_choice": str(result.team_b_choice),
            "team_a_score": result.team_a_score,
            "team_b_score": result.team_b_score,
            "total_score": result.total_score,
            "both_cooperated": result.both_cooperated,
            "both_defected": result.both_defected,
        })
        self.events.append(event)
    
    async def log_game_end(self, state: "GameState") -> None:
        """Log the end of a game with final statistics.
        
        Args:
            state: Final game state
        """
        event = self._create_event("game_end", {
            "rounds_played": len(state.history),
            "team_a_total": state.team_a_total,
            "team_b_total": state.team_b_total,
            "combined_total": state.total_score,
            "max_possible": state.max_possible_score,
            "efficiency": state.efficiency,
            "cooperation_rate": state.cooperation_rate,
        })
        self.events.append(event)
        
        # Save to file
        await self.save()
    
    async def save(self) -> str:
        """Save all events to a JSON file.

        Returns:
            Path to the saved file
        """
        filename = f"{self.game_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump({
                "game_id": self.game_id,
                "experiment_name": self.experiment_name,
                "team_a_model": getattr(self, 'team_a_model', None),
                "team_b_model": getattr(self, 'team_b_model', None),
                "events": self.events,
            }, f, indent=2)

        return str(filepath)
    
    def get_events(self) -> List[dict]:
        """Get all logged events.
        
        Returns:
            List of all event records
        """
        return self.events.copy()
    
    @staticmethod
    def load_game(filepath: str) -> dict:
        """Load a game log from file.
        
        Args:
            filepath: Path to the JSON log file
            
        Returns:
            Dictionary with game_id, experiment_name, and events
        """
        with open(filepath, 'r') as f:
            return json.load(f)

