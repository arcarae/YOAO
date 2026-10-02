"""Trajectory tracking for Sugarscape RL post-training.

Captures the exact inputs (prompts) and outputs (reasoning + moves) of LLM
agents, along with the environment state and rewards, to facilitate
Reinforcement Learning.

Note: this is intentionally a separate module from `redblackbench.trajectory`
(at the package root). That package provides
`GameTrajectory`/`TrajectoryCollector` for the Red-Black game's team
deliberation flow (opinions → votes → round outcomes) and is coupled to
`redblackbench.game.coordinator`. This file serves the Sugarscape tick
loop (movement → trade dialogues → reflections) with completely different
timestep semantics. The two are not merged because they record different
domains and unifying them would add a layer of abstraction without
eliminating any duplication.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
import time
import numpy as np
from datetime import datetime

class SugarJSONEncoder(json.JSONEncoder):
    """Custom encoder for Sugarscape types."""
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@dataclass
class SugarActionRecord:
    """Record of a single agent's action in a timestep."""
    agent_id: int
    
    # RL Input (State/Context)
    # We store the full prompt parts so they can be reconstructed exactly
    system_prompt: str
    user_prompt: str
    
    # RL Output (Action)
    raw_response: str  # The full generated text (Reasoning + Move)
    parsed_move: Optional[Tuple[int, int]]  # The extracted action
    
    # Reward Signals
    reward_sugar: int = 0
    reward_spice: int = 0
    metabolic_cost_sugar: int = 0
    metabolic_cost_spice: int = 0
    
    # Derived net reward can be calculated during post-processing
    
    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "raw_response": self.raw_response,
            "parsed_move": self.parsed_move,
            "rewards": {
                "sugar_harvested": self.reward_sugar,
                "spice_harvested": self.reward_spice,
                "sugar_metabolism": self.metabolic_cost_sugar,
                "spice_metabolism": self.metabolic_cost_spice
            }
        }

@dataclass
class SugarTimestep:
    """Snapshot of the simulation at a single tick."""
    tick: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Global State (for context/rendering)
    # We store a sparse representation or summary to save space
    # For full replay, we might need more, but for RL (S,A,R), 
    # the agent-centric prompts in ActionRecord are the primary 'State'.
    population_count: int = 0
    
    # Actions taken this tick
    actions: List[SugarActionRecord] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "timestamp": self.timestamp,
            "population_count": self.population_count,
            "actions": [a.to_dict() for a in self.actions]
        }

@dataclass
class SugarTrajectory:
    """Complete trajectory of a Sugarscape simulation run."""
    run_id: str
    config: Dict[str, Any]
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    timesteps: List[SugarTimestep] = field(default_factory=list)
    
    def add_timestep(self, tick: int, population: int) -> SugarTimestep:
        ts = SugarTimestep(tick=tick, population_count=population)
        self.timesteps.append(ts)
        return ts
        
    def save(self, filepath: str):
        """Save trajectory to JSON."""
        data = {
            "run_id": self.run_id,
            "config": self.config,
            "start_time": self.start_time,
            "timesteps": [t.to_dict() for t in self.timesteps]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=SugarJSONEncoder)
