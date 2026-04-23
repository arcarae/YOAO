"""Metrics collection and analysis for RedBlackBench."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import json

from redblackbench.game.scoring import Choice


@dataclass
class GameMetrics:
    """Metrics for a single game.
    
    Attributes:
        game_id: Unique identifier for the game
        total_rounds: Number of rounds played
        team_a_score: Team A's final score
        team_b_score: Team B's final score
        combined_score: Combined total score
        max_possible: Maximum possible combined score
        efficiency: Score achieved as percentage of maximum
        cooperation_rate: Percentage of BLACK choices
        rounds_both_cooperated: Number of rounds both teams chose BLACK
        rounds_both_defected: Number of rounds both teams chose RED
        rounds_exploited: Number of rounds with mixed choices
    """
    game_id: str
    total_rounds: int
    team_a_score: int
    team_b_score: int
    combined_score: int
    max_possible: int
    efficiency: float
    cooperation_rate: float
    rounds_both_cooperated: int = 0
    rounds_both_defected: int = 0
    rounds_exploited: int = 0
    round_choices: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "total_rounds": self.total_rounds,
            "team_a_score": self.team_a_score,
            "team_b_score": self.team_b_score,
            "combined_score": self.combined_score,
            "max_possible": self.max_possible,
            "efficiency": self.efficiency,
            "cooperation_rate": self.cooperation_rate,
            "rounds_both_cooperated": self.rounds_both_cooperated,
            "rounds_both_defected": self.rounds_both_defected,
            "rounds_exploited": self.rounds_exploited,
            "round_choices": self.round_choices,
        }


@dataclass 
class AggregateMetrics:
    """Aggregated metrics across multiple games.
    
    Attributes:
        num_games: Number of games analyzed
        avg_efficiency: Average efficiency across games
        avg_cooperation_rate: Average cooperation rate
        avg_combined_score: Average combined score
        total_cooperation_rounds: Total rounds where both cooperated
        total_defection_rounds: Total rounds where both defected
        total_exploitation_rounds: Total rounds with mixed choices
    """
    num_games: int = 0
    avg_efficiency: float = 0.0
    avg_cooperation_rate: float = 0.0
    avg_combined_score: float = 0.0
    total_cooperation_rounds: int = 0
    total_defection_rounds: int = 0
    total_exploitation_rounds: int = 0
    min_efficiency: float = 1.0
    max_efficiency: float = 0.0
    games: List[GameMetrics] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "num_games": self.num_games,
            "avg_efficiency": self.avg_efficiency,
            "avg_cooperation_rate": self.avg_cooperation_rate,
            "avg_combined_score": self.avg_combined_score,
            "total_cooperation_rounds": self.total_cooperation_rounds,
            "total_defection_rounds": self.total_defection_rounds,
            "total_exploitation_rounds": self.total_exploitation_rounds,
            "min_efficiency": self.min_efficiency,
            "max_efficiency": self.max_efficiency,
        }


class MetricsCollector:
    """Collects and aggregates metrics from game logs."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.games: List[GameMetrics] = []
    
    def extract_metrics_from_log(self, log_data: dict) -> GameMetrics:
        """Extract metrics from a game log.
        
        Args:
            log_data: Parsed game log dictionary
            
        Returns:
            GameMetrics for the game
        """
        events = log_data.get("events", [])
        game_id = log_data.get("game_id", "unknown")
        
        # Find game_end event for final stats
        game_end = None
        round_results = []
        
        for event in events:
            if event["event_type"] == "game_end":
                game_end = event["data"]
            elif event["event_type"] == "round_result":
                round_results.append(event["data"])
        
        if not game_end:
            raise ValueError(f"No game_end event found in log {game_id}")
        
        # Count round outcomes
        both_cooperated = sum(1 for r in round_results if r.get("both_cooperated", False))
        both_defected = sum(1 for r in round_results if r.get("both_defected", False))
        exploited = len(round_results) - both_cooperated - both_defected
        
        # Extract round choices
        round_choices = [
            {
                "round": r["round_num"],
                "team_a": r["team_a_choice"],
                "team_b": r["team_b_choice"],
            }
            for r in round_results
        ]
        
        return GameMetrics(
            game_id=game_id,
            total_rounds=game_end["rounds_played"],
            team_a_score=game_end["team_a_total"],
            team_b_score=game_end["team_b_total"],
            combined_score=game_end["combined_total"],
            max_possible=game_end["max_possible"],
            efficiency=game_end["efficiency"],
            cooperation_rate=game_end["cooperation_rate"],
            rounds_both_cooperated=both_cooperated,
            rounds_both_defected=both_defected,
            rounds_exploited=exploited,
            round_choices=round_choices,
        )
    
    def add_game(self, metrics: GameMetrics) -> None:
        """Add a game's metrics to the collection.
        
        Args:
            metrics: Game metrics to add
        """
        self.games.append(metrics)
    
    def load_from_file(self, filepath: str) -> GameMetrics:
        """Load and extract metrics from a log file.
        
        Args:
            filepath: Path to the JSON log file
            
        Returns:
            Extracted GameMetrics
        """
        with open(filepath, 'r') as f:
            log_data = json.load(f)
        
        metrics = self.extract_metrics_from_log(log_data)
        self.add_game(metrics)
        return metrics
    
    def load_from_directory(self, directory: str) -> List[GameMetrics]:
        """Load all game logs from a directory.
        
        Args:
            directory: Path to directory containing JSON log files
            
        Returns:
            List of extracted GameMetrics
        """
        dir_path = Path(directory)
        loaded = []
        
        for filepath in dir_path.glob("*.json"):
            try:
                metrics = self.load_from_file(str(filepath))
                loaded.append(metrics)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Warning: Could not load {filepath}: {e}")
        
        return loaded
    
    def get_aggregate_metrics(self) -> AggregateMetrics:
        """Calculate aggregate metrics across all games.
        
        Returns:
            Aggregated metrics
        """
        if not self.games:
            return AggregateMetrics()
        
        num_games = len(self.games)
        total_efficiency = sum(g.efficiency for g in self.games)
        total_coop_rate = sum(g.cooperation_rate for g in self.games)
        total_combined = sum(g.combined_score for g in self.games)
        
        return AggregateMetrics(
            num_games=num_games,
            avg_efficiency=total_efficiency / num_games,
            avg_cooperation_rate=total_coop_rate / num_games,
            avg_combined_score=total_combined / num_games,
            total_cooperation_rounds=sum(g.rounds_both_cooperated for g in self.games),
            total_defection_rounds=sum(g.rounds_both_defected for g in self.games),
            total_exploitation_rounds=sum(g.rounds_exploited for g in self.games),
            min_efficiency=min(g.efficiency for g in self.games),
            max_efficiency=max(g.efficiency for g in self.games),
            games=self.games,
        )
    
    def generate_summary(self) -> str:
        """Generate a human-readable summary of metrics.
        
        Returns:
            Formatted summary string
        """
        agg = self.get_aggregate_metrics()
        
        if agg.num_games == 0:
            return "No games recorded."
        
        total_rounds = agg.total_cooperation_rounds + agg.total_defection_rounds + agg.total_exploitation_rounds
        
        lines = [
            "=" * 60,
            "RedBlackBench Metrics Summary",
            "=" * 60,
            f"Games Analyzed: {agg.num_games}",
            "",
            "Performance Metrics:",
            f"  Average Efficiency: {agg.avg_efficiency:.1%}",
            f"  Efficiency Range: {agg.min_efficiency:.1%} - {agg.max_efficiency:.1%}",
            f"  Average Combined Score: {agg.avg_combined_score:.1f}",
            "",
            "Cooperation Metrics:",
            f"  Average Cooperation Rate: {agg.avg_cooperation_rate:.1%}",
            f"  Total Rounds Both Cooperated: {agg.total_cooperation_rounds} ({agg.total_cooperation_rounds/total_rounds:.1%})" if total_rounds > 0 else "  Total Rounds Both Cooperated: 0",
            f"  Total Rounds Both Defected: {agg.total_defection_rounds} ({agg.total_defection_rounds/total_rounds:.1%})" if total_rounds > 0 else "  Total Rounds Both Defected: 0",
            f"  Total Rounds Exploited: {agg.total_exploitation_rounds} ({agg.total_exploitation_rounds/total_rounds:.1%})" if total_rounds > 0 else "  Total Rounds Exploited: 0",
            "=" * 60,
        ]
        
        return "\n".join(lines)

