"""Trajectory collector for capturing full game trajectories."""

from datetime import datetime
from typing import List, Optional, TYPE_CHECKING
import uuid

from redblackbench.trajectory.trajectory import (
    GameTrajectory,
    TrajectoryTimestep,
    TeamSnapshot,
    AgentSnapshot,
    ActionRecord,
    DialogueExchange,
    Outcome,
    TimestepType,
)
from redblackbench.game.scoring import Choice

if TYPE_CHECKING:
    from redblackbench.game.config import GameConfig
    from redblackbench.game.coordinator import GameState
    from redblackbench.game.scoring import RoundResult
    from redblackbench.teams.team import Team
    from redblackbench.teams.deliberation import DeliberationResult
    from redblackbench.agents.base import BaseAgent, AgentResponse


class TrajectoryCollector:
    """Collects and builds game trajectories during play.
    
    Integrates with the game coordinator to capture full state snapshots,
    action sequences, dialogue exchanges, and outcomes.
    """
    
    def __init__(self, trajectory_id: Optional[str] = None):
        """Initialize the collector.
        
        Args:
            trajectory_id: Optional custom ID (generates UUID if not provided)
        """
        self.trajectory_id = trajectory_id or str(uuid.uuid4())
        self.trajectory: Optional[GameTrajectory] = None
    
    def start_game(
        self,
        config: "GameConfig",
        team_a: "Team",
        team_b: "Team",
        existing_trajectory: Optional[GameTrajectory] = None,
        team_a_model: Optional[str] = None,
        team_b_model: Optional[str] = None,
    ) -> GameTrajectory:
        """Initialize trajectory collection for a new game or resume existing.

        Args:
            config: Game configuration
            team_a: Team A
            team_b: Team B
            existing_trajectory: Optional trajectory to resume from
            team_a_model: Model name/ID used by Team A
            team_b_model: Model name/ID used by Team B

        Returns:
            The initialized GameTrajectory
        """
        if existing_trajectory:
            self.trajectory = existing_trajectory
            print(f"Resuming trajectory {self.trajectory.trajectory_id}")
            return self.trajectory

        self.trajectory = GameTrajectory(
            trajectory_id=self.trajectory_id,
            game_config={
                "num_rounds": config.num_rounds,
                "team_size": config.team_size,
                "multipliers": config.multipliers,
                "both_black_score": config.both_black_score,
                "both_red_score": config.both_red_score,
                "red_wins_score": config.red_wins_score,
                "black_loses_score": config.black_loses_score,
                "max_possible_score": config.calculate_max_possible_score(),
            },
            team_a_name=team_a.name,
            team_b_name=team_b.name,
            team_a_model=team_a_model,
            team_b_model=team_b_model,
        )
        
        # Initialize team state snapshots
        self.trajectory.update_team_state("A", self._create_team_snapshot(team_a, "A"))
        self.trajectory.update_team_state("B", self._create_team_snapshot(team_b, "B"))
        
        # Add game start timestep
        self.trajectory.add_timestep(
            timestep_type=TimestepType.GAME_START,
            round_num=0,
            metadata={
                "config": self.trajectory.game_config,
                "team_a": team_a.name,
                "team_b": team_b.name,
            },
        )
        
        return self.trajectory
    
    def _create_team_snapshot(
        self,
        team: "Team",
        team_identifier: str,
        score: int = 0,
    ) -> TeamSnapshot:
        """Create a snapshot of a team's current state.
        
        Args:
            team: The team to snapshot
            team_identifier: 'A' or 'B'
            score: Current team score
            
        Returns:
            TeamSnapshot with full agent states
        """
        agent_snapshots = []
        for agent in team.agents:
            agent_snapshots.append(self._create_agent_snapshot(agent))
        
        # Get choices from deliberation history
        choices = [
            str(d.final_choice) for d in team.deliberation_history
        ]
        
        # Get consensus history
        consensus = [d.was_unanimous for d in team.deliberation_history]
        
        return TeamSnapshot(
            team_name=team.name,
            team_identifier=team_identifier,
            agents=agent_snapshots,
            current_score=score,
            choices_made=choices,
            consensus_history=consensus,
        )
    
    def _create_agent_snapshot(self, agent: "BaseAgent") -> AgentSnapshot:
        """Create a snapshot of an agent's current state.
        
        Args:
            agent: The agent to snapshot
            
        Returns:
            AgentSnapshot with full conversation history
        """
        # Convert conversation history to DialogueExchange objects
        dialogue_history = []
        for msg in agent.conversation_history:
            dialogue_history.append(DialogueExchange(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg.get("timestamp", datetime.now().isoformat()),
                exchange_type=msg.get("exchange_type"),
            ))
        
        return AgentSnapshot(
            agent_id=agent.agent_id,
            team_name=agent.team_name,
            conversation_history=dialogue_history,
        )
    
    def record_round_start(self, round_num: int, multiplier: int) -> None:
        """Record the start of a new round.
        
        Args:
            round_num: Round number (1-indexed)
            multiplier: Score multiplier for this round
        """
        if not self.trajectory:
            return
        
        self.trajectory.add_timestep(
            timestep_type=TimestepType.ROUND_START,
            round_num=round_num,
            metadata={"multiplier": multiplier},
        )
    
    def record_deliberation_start(
        self,
        round_num: int,
        team: "Team",
        team_identifier: str,
    ) -> None:
        """Record the start of a team's deliberation.
        
        Args:
            round_num: Current round number
            team: The team starting deliberation
            team_identifier: 'A' or 'B'
        """
        if not self.trajectory:
            return
        
        self.trajectory.add_timestep(
            timestep_type=TimestepType.DELIBERATION_START,
            round_num=round_num,
            metadata={
                "team": team.name,
                "team_identifier": team_identifier,
            },
        )
    
    def record_initial_opinions(
        self,
        round_num: int,
        team: "Team",
        team_identifier: str,
        opinions_with_agents: List[tuple["BaseAgent", "AgentResponse"]],
        game_state: "GameState",
    ) -> None:
        """Record initial opinions from a team's agents.
        
        Args:
            round_num: Current round number
            team: The team
            team_identifier: 'A' or 'B'
            opinions: List of agent responses
            game_state: Current game state
        """
        if not self.trajectory:
            return
        
        actions = []
        for i, (agent, opinion) in enumerate(opinions_with_agents):
            actions.append(ActionRecord(
                action_type="individual_opinion",
                actor=agent.agent_id,
                choice=str(opinion.choice),
                reasoning=opinion.reasoning,
                private_thought=opinion.private_thought,
                round_num=round_num,
                phase="opinion_turn",
            ))
        
        # Update team snapshot
        score = game_state.team_a_total if team_identifier == "A" else game_state.team_b_total
        self.trajectory.update_team_state(
            team_identifier,
            self._create_team_snapshot(team, team_identifier, score),
        )
        
        self.trajectory.add_timestep(
            timestep_type=TimestepType.INITIAL_OPINIONS,
            round_num=round_num,
            actions=actions,
            metadata={"team": team.name, "team_identifier": team_identifier, "turns": len(actions)},
        )
    
    def record_final_votes(
        self,
        round_num: int,
        team: "Team",
        team_identifier: str,
        deliberation_result: "DeliberationResult",
        game_state: "GameState",
    ) -> None:
        """Record final votes and team choice.
        
        Args:
            round_num: Current round number
            team: The team
            team_identifier: 'A' or 'B'
            deliberation_result: Result of deliberation
            game_state: Current game state
        """
        if not self.trajectory:
            return
        
        actions = []
        
        # Record individual final votes
        for i, (agent, vote) in enumerate(zip(team.agents, deliberation_result.final_votes)):
            actions.append(ActionRecord(
                action_type="individual_vote",
                actor=agent.agent_id,
                choice=str(vote.choice),
                reasoning=vote.reasoning,
                private_thought=vote.private_thought,
                round_num=round_num,
                phase="final_vote",
            ))
        
        # Record team choice
        actions.append(ActionRecord(
            action_type="team_choice",
            actor=team.name,
            choice=str(deliberation_result.final_choice),
            round_num=round_num,
            reasoning=f"Vote counts: {deliberation_result.vote_counts}, Unanimous: {deliberation_result.was_unanimous}",
        ))
        
        # Update team snapshot
        score = game_state.team_a_total if team_identifier == "A" else game_state.team_b_total
        self.trajectory.update_team_state(
            team_identifier,
            self._create_team_snapshot(team, team_identifier, score),
        )
        
        self.trajectory.add_timestep(
            timestep_type=TimestepType.FINAL_VOTES,
            round_num=round_num,
            actions=actions,
            metadata={
                "team": team.name,
                "team_identifier": team_identifier,
                "was_unanimous": deliberation_result.was_unanimous,
                "vote_counts": {str(k): v for k, v in deliberation_result.vote_counts.items()},
            },
        )
    
    def record_round_end(
        self,
        round_result: "RoundResult",
        game_state: "GameState",
    ) -> None:
        """Record the end of a round with outcomes.
        
        Args:
            round_result: Result of the round
            game_state: Current game state after round
        """
        if not self.trajectory:
            return
        
        outcome = Outcome(
            outcome_type="round",
            round_num=round_result.round_num,
            team_a_score=round_result.team_a_score,
            team_b_score=round_result.team_b_score,
            team_a_choice=str(round_result.team_a_choice),
            team_b_choice=str(round_result.team_b_choice),
            both_cooperated=round_result.both_cooperated,
            both_defected=round_result.both_defected,
            multiplier=round_result.multiplier,
            cooperation_rate=game_state.cooperation_rate,
            total_score=game_state.total_score,
            max_possible_score=game_state.max_possible_score,
        )
        
        self.trajectory.add_timestep(
            timestep_type=TimestepType.ROUND_END,
            round_num=round_result.round_num,
            outcome=outcome,
        )
    
    def end_game(
        self,
        game_state: "GameState",
        team_a: "Team",
        team_b: "Team",
    ) -> GameTrajectory:
        """Finalize the trajectory at game end.
        
        Args:
            game_state: Final game state
            team_a: Team A
            team_b: Team B
            
        Returns:
            The complete GameTrajectory
        """
        if not self.trajectory:
            raise RuntimeError("No trajectory initialized")
        
        # Final team snapshots
        self.trajectory.update_team_state(
            "A",
            self._create_team_snapshot(team_a, "A", game_state.team_a_total),
        )
        self.trajectory.update_team_state(
            "B",
            self._create_team_snapshot(team_b, "B", game_state.team_b_total),
        )
        
        # Create final outcome
        final_outcome = Outcome(
            outcome_type="game",
            team_a_score=game_state.team_a_total,
            team_b_score=game_state.team_b_total,
            cooperation_rate=game_state.cooperation_rate,
            efficiency=game_state.efficiency,
            total_score=game_state.total_score,
            max_possible_score=game_state.max_possible_score,
        )
        
        self.trajectory.final_outcome = final_outcome
        self.trajectory.end_time = datetime.now().isoformat()
        
        self.trajectory.add_timestep(
            timestep_type=TimestepType.GAME_END,
            round_num=len(game_state.history),
            outcome=final_outcome,
        )
        
        return self.trajectory
    
    def get_trajectory(self) -> Optional[GameTrajectory]:
        """Get the current trajectory.
        
        Returns:
            The GameTrajectory, or None if not initialized
        """
        return self.trajectory
