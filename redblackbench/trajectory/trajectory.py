"""Full game trajectory tracking for RedBlackBench.

Captures complete game trajectories including:
- Game states (full snapshots at each timestep)
- Action sequences (team choices and individual agent votes)
- Dialogue exchanges (full LLM conversation histories)
- Outcomes (scores, cooperation metrics, final results)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import json
import copy

from redblackbench.game.scoring import Choice


class TimestepType(Enum):
    """Type of event that occurred at a timestep."""
    GAME_START = "game_start"
    ROUND_START = "round_start"
    DELIBERATION_START = "deliberation_start"
    INITIAL_OPINIONS = "initial_opinions"
    FINAL_VOTES = "final_votes"
    ROUND_END = "round_end"
    GAME_END = "game_end"


@dataclass
class DialogueExchange:
    """A single dialogue exchange in an agent's conversation history.
    
    Attributes:
        role: 'system', 'user', or 'assistant'
        content: The message content
        timestamp: When this exchange occurred
        exchange_type: Type of exchange (e.g., 'initial_opinion', 'final_vote')
    """
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    exchange_type: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "exchange_type": self.exchange_type,
        }


@dataclass
class AgentSnapshot:
    """Full state snapshot of an agent at a point in time.
    
    Attributes:
        agent_id: Unique identifier for this agent
        team_name: Name of the agent's team
        conversation_history: Full LLM conversation history up to this point
        current_opinion: Agent's current opinion/recommendation (if any)
        current_reasoning: Agent's current reasoning (if any)
        vote_history: List of all votes this agent has cast
    """
    agent_id: str
    team_name: str
    conversation_history: List[DialogueExchange] = field(default_factory=list)
    current_opinion: Optional[str] = None  # "RED" or "BLACK"
    current_reasoning: Optional[str] = None
    vote_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "team_name": self.team_name,
            "conversation_history": [d.to_dict() for d in self.conversation_history],
            "current_opinion": self.current_opinion,
            "current_reasoning": self.current_reasoning,
            "vote_history": self.vote_history,
        }


@dataclass
class ActionRecord:
    """Record of an action taken (vote or team choice).
    
    Attributes:
        action_type: 'individual_vote' or 'team_choice'
        actor: Agent ID or team name
        choice: The choice made (RED or BLACK)
        reasoning: The reasoning provided
        round_num: Which round this action was for
        phase: 'initial_opinion' or 'final_vote' for individual votes
        timestamp: When this action occurred
        private_thought: Hidden thinking process (if any)
    """
    action_type: str  # 'individual_vote' or 'team_choice'
    actor: str  # agent_id or team_name
    choice: str  # "RED" or "BLACK"
    reasoning: Optional[str] = None
    round_num: int = 0
    phase: Optional[str] = None  # 'initial_opinion' or 'final_vote'
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    private_thought: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "actor": self.actor,
            "choice": self.choice,
            "reasoning": self.reasoning,
            "round_num": self.round_num,
            "phase": self.phase,
            "timestamp": self.timestamp,
            "private_thought": self.private_thought,
        }


@dataclass
class TeamSnapshot:
    """Full state snapshot of a team at a point in time.
    
    Attributes:
        team_name: Name of the team
        team_identifier: 'A' or 'B'
        agents: Snapshots of all agents on this team
        current_score: Team's current cumulative score
        choices_made: List of choices made so far (per round)
        consensus_history: Whether team reached consensus each round
    """
    team_name: str
    team_identifier: str  # 'A' or 'B'
    agents: List[AgentSnapshot] = field(default_factory=list)
    current_score: int = 0
    choices_made: List[str] = field(default_factory=list)  # ["BLACK", "RED", ...]
    consensus_history: List[bool] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "team_name": self.team_name,
            "team_identifier": self.team_identifier,
            "agents": [a.to_dict() for a in self.agents],
            "current_score": self.current_score,
            "choices_made": self.choices_made,
            "consensus_history": self.consensus_history,
        }
    
    def get_team_dialogue(self) -> List[Dict[str, Any]]:
        """Get all dialogue exchanges for this team's agents.
        
        Returns:
            List of dialogue exchanges with agent attribution
        """
        all_dialogue = []
        for agent in self.agents:
            for exchange in agent.conversation_history:
                all_dialogue.append({
                    "agent_id": agent.agent_id,
                    **exchange.to_dict()
                })
        # Sort by timestamp
        all_dialogue.sort(key=lambda x: x["timestamp"])
        return all_dialogue


@dataclass
class Outcome:
    """Outcome information for a round or the full game.
    
    Attributes:
        outcome_type: 'round' or 'game'
        round_num: Round number (for round outcomes)
        team_a_score: Team A's score this round/game
        team_b_score: Team B's score this round/game
        team_a_choice: Team A's choice (for round outcomes)
        team_b_choice: Team B's choice (for round outcomes)
        both_cooperated: Whether both teams chose BLACK
        both_defected: Whether both teams chose RED
        multiplier: Score multiplier for this round
        cooperation_rate: Running cooperation rate
        efficiency: Score efficiency vs maximum possible
    """
    outcome_type: str  # 'round' or 'game'
    round_num: Optional[int] = None
    team_a_score: int = 0
    team_b_score: int = 0
    team_a_choice: Optional[str] = None
    team_b_choice: Optional[str] = None
    both_cooperated: bool = False
    both_defected: bool = False
    multiplier: int = 1
    cooperation_rate: float = 0.0
    efficiency: float = 0.0
    total_score: int = 0
    max_possible_score: int = 0
    
    def to_dict(self) -> dict:
        return {
            "outcome_type": self.outcome_type,
            "round_num": self.round_num,
            "team_a_score": self.team_a_score,
            "team_b_score": self.team_b_score,
            "team_a_choice": self.team_a_choice,
            "team_b_choice": self.team_b_choice,
            "both_cooperated": self.both_cooperated,
            "both_defected": self.both_defected,
            "multiplier": self.multiplier,
            "cooperation_rate": self.cooperation_rate,
            "efficiency": self.efficiency,
            "total_score": self.total_score,
            "max_possible_score": self.max_possible_score,
        }


@dataclass
class TrajectoryTimestep:
    """A single timestep in the game trajectory.
    
    Captures the complete game state at a specific moment in time.
    
    Attributes:
        timestep_id: Unique identifier for this timestep
        timestep_type: Type of event at this timestep
        round_num: Current round number
        timestamp: When this timestep occurred
        team_a_snapshot: Full state of Team A
        team_b_snapshot: Full state of Team B
        actions: Actions taken at this timestep
        outcome: Outcome of this timestep (if applicable)
        metadata: Additional metadata
    """
    timestep_id: int
    timestep_type: TimestepType
    round_num: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    team_a_snapshot: Optional[TeamSnapshot] = None
    team_b_snapshot: Optional[TeamSnapshot] = None
    actions: List[ActionRecord] = field(default_factory=list)
    outcome: Optional[Outcome] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "timestep_id": self.timestep_id,
            "timestep_type": self.timestep_type.value,
            "round_num": self.round_num,
            "timestamp": self.timestamp,
            "team_a_snapshot": self.team_a_snapshot.to_dict() if self.team_a_snapshot else None,
            "team_b_snapshot": self.team_b_snapshot.to_dict() if self.team_b_snapshot else None,
            "actions": [a.to_dict() for a in self.actions],
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "metadata": self.metadata,
        }


@dataclass
class GameTrajectory:
    """Complete trajectory of a Red-Black game.

    Captures the full history of a game including all states, actions,
    dialogue, and outcomes for later analysis and replay.

    Attributes:
        trajectory_id: Unique identifier for this trajectory
        game_config: Configuration used for this game
        timesteps: List of all timesteps in chronological order
        team_a_name: Name of Team A
        team_b_name: Name of Team B
        team_a_model: Model name/ID used by Team A
        team_b_model: Model name/ID used by Team B
        start_time: When the game started
        end_time: When the game ended (if complete)
        final_outcome: Final game outcome (if complete)
    """
    trajectory_id: str
    game_config: Dict[str, Any] = field(default_factory=dict)
    timesteps: List[TrajectoryTimestep] = field(default_factory=list)
    team_a_name: str = "Team A"
    team_b_name: str = "Team B"
    team_a_model: Optional[str] = None
    team_b_model: Optional[str] = None
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    final_outcome: Optional[Outcome] = None

    # Internal state tracking
    _current_timestep_id: int = field(default=0, repr=False)
    _team_a_state: Optional[TeamSnapshot] = field(default=None, repr=False)
    _team_b_state: Optional[TeamSnapshot] = field(default=None, repr=False)
    
    def add_timestep(
        self,
        timestep_type: TimestepType,
        round_num: int,
        actions: Optional[List[ActionRecord]] = None,
        outcome: Optional[Outcome] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryTimestep:
        """Add a new timestep to the trajectory.
        
        Args:
            timestep_type: Type of event at this timestep
            round_num: Current round number
            actions: Actions taken at this timestep
            outcome: Outcome of this timestep
            metadata: Additional metadata
            
        Returns:
            The created TrajectoryTimestep
        """
        timestep = TrajectoryTimestep(
            timestep_id=self._current_timestep_id,
            timestep_type=timestep_type,
            round_num=round_num,
            team_a_snapshot=copy.deepcopy(self._team_a_state),
            team_b_snapshot=copy.deepcopy(self._team_b_state),
            actions=actions or [],
            outcome=outcome,
            metadata=metadata or {},
        )
        self.timesteps.append(timestep)
        self._current_timestep_id += 1
        return timestep
    
    def update_team_state(
        self,
        team_identifier: str,
        snapshot: TeamSnapshot,
    ) -> None:
        """Update the current state of a team.
        
        Args:
            team_identifier: 'A' or 'B'
            snapshot: New team snapshot
        """
        if team_identifier == "A":
            self._team_a_state = snapshot
        else:
            self._team_b_state = snapshot
    
    def get_team_dialogue_history(self, team_identifier: str) -> List[Dict[str, Any]]:
        """Get the complete dialogue history for a team.
        
        This is useful for providing context to an agent about their
        team's discussions (but not the other team's).
        
        Args:
            team_identifier: 'A' or 'B'
            
        Returns:
            List of all dialogue exchanges for that team
        """
        state = self._team_a_state if team_identifier == "A" else self._team_b_state
        if state:
            return state.get_team_dialogue()
        return []
    
    def get_full_trajectory(self) -> Dict[str, Any]:
        """Get the complete trajectory as a dictionary.

        Returns:
            Dictionary containing all trajectory data
        """
        return {
            "trajectory_id": self.trajectory_id,
            "game_config": self.game_config,
            "team_a_name": self.team_a_name,
            "team_b_name": self.team_b_name,
            "team_a_model": self.team_a_model,
            "team_b_model": self.team_b_model,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "timesteps": [t.to_dict() for t in self.timesteps],
            "final_outcome": self.final_outcome.to_dict() if self.final_outcome else None,
            "summary": self.get_summary(),
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the trajectory.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.final_outcome:
            return {"status": "incomplete"}
        
        # Count action types
        total_actions = sum(len(t.actions) for t in self.timesteps)
        individual_votes = sum(
            1 for t in self.timesteps 
            for a in t.actions 
            if a.action_type == "individual_vote"
        )
        team_choices = sum(
            1 for t in self.timesteps
            for a in t.actions
            if a.action_type == "team_choice"
        )
        
        # Count dialogue exchanges
        total_dialogue = 0
        if self._team_a_state:
            for agent in self._team_a_state.agents:
                total_dialogue += len(agent.conversation_history)
        if self._team_b_state:
            for agent in self._team_b_state.agents:
                total_dialogue += len(agent.conversation_history)
        
        return {
            "status": "complete",
            "total_timesteps": len(self.timesteps),
            "total_actions": total_actions,
            "individual_votes": individual_votes,
            "team_choices": team_choices,
            "total_dialogue_exchanges": total_dialogue,
            "final_total_score": self.final_outcome.total_score,
            "max_possible_score": self.final_outcome.max_possible_score,
            "efficiency": self.final_outcome.efficiency,
            "cooperation_rate": self.final_outcome.cooperation_rate,
        }
    
    def get_action_sequence(self) -> List[ActionRecord]:
        """Get the complete sequence of all actions.
        
        Returns:
            List of all ActionRecords in chronological order
        """
        all_actions = []
        for timestep in self.timesteps:
            all_actions.extend(timestep.actions)
        return all_actions
    
    def get_outcomes(self) -> List[Outcome]:
        """Get all outcomes (round and game).
        
        Returns:
            List of all Outcome objects
        """
        outcomes = []
        for timestep in self.timesteps:
            if timestep.outcome:
                outcomes.append(timestep.outcome)
        return outcomes
    
    def save(self, filepath: str) -> None:
        """Save the trajectory to a JSON file.
        
        Args:
            filepath: Path to save the trajectory
        """
        with open(filepath, 'w') as f:
            json.dump(self.get_full_trajectory(), f, indent=2)
    
    @staticmethod
    def _reconstruct_team_snapshot(data: dict) -> TeamSnapshot:
        agents = []
        for a_data in data.get("agents", []):
            history = []
            for h_data in a_data.get("conversation_history", []):
                history.append(DialogueExchange(
                    role=h_data["role"],
                    content=h_data["content"],
                    timestamp=h_data.get("timestamp", ""),
                    exchange_type=h_data.get("exchange_type")
                ))
            
            agents.append(AgentSnapshot(
                agent_id=a_data["agent_id"],
                team_name=a_data["team_name"],
                conversation_history=history,
                current_opinion=a_data.get("current_opinion"),
                current_reasoning=a_data.get("current_reasoning"),
                vote_history=a_data.get("vote_history", [])
            ))
            
        return TeamSnapshot(
            team_name=data["team_name"],
            team_identifier=data["team_identifier"],
            agents=agents,
            current_score=data.get("current_score", 0),
            choices_made=data.get("choices_made", []),
            consensus_history=data.get("consensus_history", [])
        )

    @classmethod
    def load(cls, filepath: str) -> "GameTrajectory":
        """Load a trajectory from a JSON file.
        
        Args:
            filepath: Path to the trajectory file
            
        Returns:
            Loaded GameTrajectory
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        trajectory = cls(
            trajectory_id=data["trajectory_id"],
            game_config=data["game_config"],
            team_a_name=data["team_a_name"],
            team_b_name=data["team_b_name"],
            team_a_model=data.get("team_a_model"),
            team_b_model=data.get("team_b_model"),
            start_time=data["start_time"],
            end_time=data.get("end_time"),
        )
        
        # Reconstruct timesteps
        for t_data in data["timesteps"]:
            timestep = TrajectoryTimestep(
                timestep_id=t_data["timestep_id"],
                timestep_type=TimestepType(t_data["timestep_type"]),
                round_num=t_data["round_num"],
                timestamp=t_data["timestamp"],
                metadata=t_data.get("metadata", {}),
            )
            
            # Reconstruct snapshots
            if t_data.get("team_a_snapshot"):
                timestep.team_a_snapshot = cls._reconstruct_team_snapshot(t_data["team_a_snapshot"])
            if t_data.get("team_b_snapshot"):
                timestep.team_b_snapshot = cls._reconstruct_team_snapshot(t_data["team_b_snapshot"])
            
            # Reconstruct actions
            for a_data in t_data.get("actions", []):
                timestep.actions.append(ActionRecord(
                    action_type=a_data["action_type"],
                    actor=a_data["actor"],
                    choice=a_data["choice"],
                    reasoning=a_data.get("reasoning"),
                    round_num=a_data.get("round_num", 0),
                    phase=a_data.get("phase"),
                    timestamp=a_data.get("timestamp", ""),
                    private_thought=a_data.get("private_thought"),
                ))
            
            # Reconstruct outcome if present
            if t_data.get("outcome"):
                o_data = t_data["outcome"]
                timestep.outcome = Outcome(
                    outcome_type=o_data["outcome_type"],
                    round_num=o_data.get("round_num"),
                    team_a_score=o_data.get("team_a_score", 0),
                    team_b_score=o_data.get("team_b_score", 0),
                    team_a_choice=o_data.get("team_a_choice"),
                    team_b_choice=o_data.get("team_b_choice"),
                    both_cooperated=o_data.get("both_cooperated", False),
                    both_defected=o_data.get("both_defected", False),
                    multiplier=o_data.get("multiplier", 1),
                    cooperation_rate=o_data.get("cooperation_rate", 0.0),
                    efficiency=o_data.get("efficiency", 0.0),
                    total_score=o_data.get("total_score", 0),
                    max_possible_score=o_data.get("max_possible_score", 0),
                )
            
            trajectory.timesteps.append(timestep)
        
        # Reconstruct final outcome
        if data.get("final_outcome"):
            fo = data["final_outcome"]
            trajectory.final_outcome = Outcome(
                outcome_type=fo["outcome_type"],
                team_a_score=fo.get("team_a_score", 0),
                team_b_score=fo.get("team_b_score", 0),
                cooperation_rate=fo.get("cooperation_rate", 0.0),
                efficiency=fo.get("efficiency", 0.0),
                total_score=fo.get("total_score", 0),
                max_possible_score=fo.get("max_possible_score", 0),
            )
            
        # Restore internal state snapshots to the latest available
        last_a = None
        last_b = None
        for t in trajectory.timesteps:
            if t.team_a_snapshot:
                last_a = t.team_a_snapshot
            if t.team_b_snapshot:
                last_b = t.team_b_snapshot
        
        if last_a:
            trajectory._team_a_state = last_a
        if last_b:
            trajectory._team_b_state = last_b
            
        # Restore _current_timestep_id
        if trajectory.timesteps:
            trajectory._current_timestep_id = trajectory.timesteps[-1].timestep_id + 1
        
        return trajectory
