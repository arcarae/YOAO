"""Legacy sugarscape experiment logger (`ExperimentLogger`) and metrics
calculator (`MetricsCalculator`).

Historical note: this module predates `debug_logger.py` and remains alive
because `simulation.py` still imports `ExperimentLogger` and
`MetricsCalculator` from here for run-directory layout (config.json,
metrics.csv, initial_state.json, final_state.json, checkpoints/). The
richer per-agent debug streams (agent_decisions.csv, trade_dialogues.jsonl,
reflections.jsonl, moral_scores.csv, etc.) live in `debug_logger.py` and
are written in parallel, not through this module.

If you are adding a new metric or output artifact:
- **Aggregate, per-tick time series**: add it to `MetricsCalculator` in
  this file and let `ExperimentLogger.save_metrics()` write it into
  `metrics.csv`.
- **Per-agent / per-event detail stream**: add it to `DebugLogger` in
  `debug_logger.py` so it lands under `debug/` alongside the other CSV /
  JSONL streams.

The two loggers are not unified because they have different scopes
(run-level artifacts vs. per-event debug trace) and a merge would touch
every simulation call site — left as future refactoring.
"""

import os
import json
import csv
import time
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from sugarscape.config import SugarscapeConfig

class MetricsCalculator:
    """Calculates advanced metrics for Sugarscape."""
    
    @staticmethod
    def calculate_moran_i(width: int, height: int, agent_positions: List[tuple], agent_wealths: List[int]) -> float:
        """
        Calculate Moran's I for spatial autocorrelation of wealth.
        
        Args:
            width, height: Grid dimensions
            agent_positions: List of (x, y) tuples
            agent_wealths: List of wealth values corresponding to positions
            
        Returns:
            float: Moran's I index (-1 to 1)
        """
        if not agent_wealths or len(agent_wealths) < 2:
            return 0.0
            
        n = len(agent_wealths)
        mean_wealth = np.mean(agent_wealths)
        
        # Create a grid representation for easier neighbor lookup
        # Only occupied cells matter for this implementation
        grid = {}
        for pos, wealth in zip(agent_positions, agent_wealths):
            grid[pos] = wealth
            
        numerator = 0.0
        denominator = 0.0
        
        # Weights: 1 if adjacent (Moore neighborhood), 0 otherwise
        # We only sum over existing agents pairs
        
        total_weight = 0
        
        for i in range(n):
            pos_i = agent_positions[i]
            val_i = agent_wealths[i] - mean_wealth
            denominator += val_i ** 2
            
            # Check neighbors
            x, y = pos_i
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    nx = (x + dx) % width
                    ny = (y + dy) % height
                    if (nx, ny) in grid:
                        neighbors.append(grid[(nx, ny)])
            
            for val_j_raw in neighbors:
                val_j = val_j_raw - mean_wealth
                numerator += val_i * val_j
                total_weight += 1
                
        if denominator == 0 or total_weight == 0:
            return 0.0
            
        return (n / total_weight) * (numerator / denominator)

    @staticmethod
    def calculate_mobility_stats(agents: List[Any]) -> Dict[str, float]:
        """Calculate mobility statistics."""
        if not agents:
            return {"avg_displacement": 0.0, "avg_exploration": 0.0}
            
        displacements = []
        explorations = []
        
        for agent in agents:
            if hasattr(agent, 'metrics'):
                displacements.append(agent.metrics.get('displacement', 0))
                explorations.append(agent.metrics.get('unique_visited', 1))
                
        return {
            "avg_displacement": float(np.mean(displacements)) if displacements else 0.0,
            "avg_exploration": float(np.mean(explorations)) if explorations else 0.0
        }

class ExperimentLogger:
    """Manages experiment data storage and logging."""
    
    def __init__(self, base_dir: str = "results/sugarscape", experiment_type: str = "baseline", config: Optional[SugarscapeConfig] = None):
        self.base_dir = Path(base_dir)
        self.experiment_type = experiment_type
        
        # Create timestamped run directory with microseconds + random suffix for uniqueness
        import random as rand_mod
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Include microseconds
        random_suffix = rand_mod.randint(1000, 9999)  # Random 4-digit suffix
        unique_id = f"{timestamp}_{random_suffix}"
        self.run_dir = self.base_dir / experiment_type / f"experiment_{unique_id}"
        self.experiment_id = f"experiment_{unique_id}" # Store ID for trajectory naming
        self.plots_dir = self.run_dir / "plots"
        
        self._init_directories()
        
        if config:
            self.save_config(config)
            
        # Initialize CSV writer for time series metrics
        self.csv_file = self.run_dir / "metrics.csv"
        self._csv_headers = [
            "tick", "population", "mean_wealth", "gini", "moran_i", 
            "mean_age", "avg_displacement", "avg_exploration",
            # Welfare metrics
            "utilitarian_welfare", "average_welfare", "rawlsian_welfare", "nash_welfare",
            "welfare_gini", "gini_adjusted_welfare", "atkinson_index_05", "atkinson_adjusted_05",
            "survival_rate", "mean_lifespan_utilization",
            "welfare_std", "welfare_median", "welfare_max", "welfare_min"
        ]
        self._init_csv()

    def _init_directories(self):
        """Create necessary directories."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        print(f"Experiment initialized at: {self.run_dir}")

    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self._csv_headers)

    def save_config(self, config: SugarscapeConfig):
        """Save configuration to JSON."""
        config_dict = {k: str(v) for k, v in config.__dict__.items()}
        with open(self.run_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

    def log_step(self, metrics: Dict[str, Any]):
        """Log metrics for a single simulation step."""
        row = [metrics.get(h, 0) for h in self._csv_headers]
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def save_snapshot(self, data: Dict[str, Any], filename: str = "detailed_data.json"):
        """Save a detailed snapshot of the simulation state."""
        # Convert numpy types to python types for JSON serialization
        def default_converter(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            if isinstance(o, (np.float64, np.float32)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)
            
        with open(self.run_dir / filename, 'w') as f:
            json.dump(data, f, default=default_converter, indent=2)

    def get_plots_dir(self) -> str:
        return str(self.plots_dir)

    def get_log_path(self, filename: str) -> str:
        """Get full path for a log file in the run directory."""
        return str(self.run_dir / filename)

    def save_checkpoint(self, data: Dict[str, Any], tick: int) -> Path:
        """Save checkpoint using pickle for complex nested structures.

        Args:
            data: Complete simulation state dictionary
            tick: Current simulation tick (for filename)

        Returns:
            Path to the saved checkpoint file
        """
        checkpoint_path = self.checkpoints_dir / f"checkpoint_tick_{tick}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return checkpoint_path

    def get_checkpoint_path(self, tick: int) -> Path:
        """Get path for a checkpoint file at given tick."""
        return self.checkpoints_dir / f"checkpoint_tick_{tick}.pkl"

    def list_checkpoints(self) -> List[Path]:
        """List all checkpoint files in order."""
        checkpoints = list(self.checkpoints_dir.glob("checkpoint_tick_*.pkl"))
        # Sort by tick number
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))
        return checkpoints
