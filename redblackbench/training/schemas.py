"""Training data schemas for RedBlackBench (rbbench.v1).

Complete schema for SFT and preference learning training data.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import json


@dataclass
class AgentMessage:
    """A single message from an agent during deliberation.

    Attributes:
        agent_uid: Unique identifier for the agent (e.g., "A_00", "A_01")
        agent_name: Human-readable name (e.g., "Dr. Sarah Chen")
        turn: Speaking turn within the round (1-indexed)
        public_message: What the agent said publicly to teammates
        private_reasoning: Hidden reasoning/thinking (if captured)
        recommendation: The agent's recommended choice ("A" or "B")
    """
    agent_uid: str
    agent_name: str
    turn: int
    public_message: str
    private_reasoning: Optional[str] = None
    recommendation: str = ""  # "A" or "B"

    def to_dict(self) -> dict:
        return {
            "agent_uid": self.agent_uid,
            "agent_name": self.agent_name,
            "turn": self.turn,
            "public_message": self.public_message,
            "private_reasoning": self.private_reasoning,
            "recommendation": self.recommendation,
        }


@dataclass
class TeamVotes:
    """Voting results for a team.

    Attributes:
        votes: Dict mapping agent_uid to their vote ("A" or "B")
        tally: Count of votes for each option
        team_decision: Final team decision after voting
        was_unanimous: Whether all agents voted the same
    """
    votes: Dict[str, str]  # agent_uid -> "A" or "B"
    tally: Dict[str, int]  # {"A": 3, "B": 2}
    team_decision: str  # "A" or "B"
    was_unanimous: bool = False

    def to_dict(self) -> dict:
        return {
            "votes": self.votes,
            "tally": self.tally,
            "team_decision": self.team_decision,
            "was_unanimous": self.was_unanimous,
        }


@dataclass
class DiplomacyExchange:
    """Diplomatic message exchange between teams.

    Attributes:
        team_a_message: Message sent by Team A
        team_b_message: Message sent by Team B
        team_a_sender: Agent who sent Team A's message
        team_b_sender: Agent who sent Team B's message
    """
    team_a_message: Optional[str] = None
    team_b_message: Optional[str] = None
    team_a_sender: Optional[str] = None
    team_b_sender: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "team_a_message": self.team_a_message,
            "team_b_message": self.team_b_message,
            "team_a_sender": self.team_a_sender,
            "team_b_sender": self.team_b_sender,
        }


@dataclass
class RoundState:
    """Game state at the start of a round.

    Attributes:
        history: List of previous round outcomes
        scores: Current cumulative scores
        rounds_remaining: How many rounds left
        max_possible_from_here: Maximum achievable score from this point
    """
    history: List[Dict[str, Any]]
    scores: Dict[str, int]  # {"team_a": X, "team_b": Y, "sum": Z}
    rounds_remaining: int
    max_possible_from_here: int

    def to_dict(self) -> dict:
        return {
            "history": self.history,
            "scores": self.scores,
            "rounds_remaining": self.rounds_remaining,
            "max_possible_from_here": self.max_possible_from_here,
        }


@dataclass
class RoundOutcome:
    """Outcome of a single round.

    Attributes:
        joint_action: Combined action (e.g., "AA", "AB", "BA", "BB")
        base_payoff: Payoffs before multiplier
        applied_multiplier: Multiplier for this round
        round_delta: Score changes this round
        new_scores: Cumulative scores after this round
    """
    joint_action: str  # "AA", "AB", "BA", "BB"
    base_payoff: Dict[str, int]  # {"team_a": X, "team_b": Y, "sum": Z}
    applied_multiplier: int
    round_delta: Dict[str, int]
    new_scores: Dict[str, int]

    def to_dict(self) -> dict:
        return {
            "joint_action": self.joint_action,
            "base_payoff": self.base_payoff,
            "applied_multiplier": self.applied_multiplier,
            "round_delta": self.round_delta,
            "new_scores": self.new_scores,
        }


@dataclass
class TrainingRound:
    """A single round in the training trajectory.

    Contains full Team A deliberation, and Team B info based on logging mode.
    """
    round_index: int
    multiplier: int
    is_critical: bool
    has_diplomacy: bool

    # State at round start
    state: RoundState

    # Diplomacy (if enabled for this round)
    diplomacy: Optional[DiplomacyExchange] = None

    # Team A: Full deliberation
    team_a_deliberation: List[AgentMessage] = field(default_factory=list)
    team_a_votes: Optional[TeamVotes] = None

    # Team B: Always include decision + tally, full deliberation in FULL mode
    team_b_decision: str = ""  # "A" or "B"
    team_b_tally: Dict[str, int] = field(default_factory=dict)
    team_b_deliberation: Optional[List[AgentMessage]] = None  # Only in FULL mode
    team_b_votes: Optional[TeamVotes] = None  # Only in FULL mode

    # Round outcome
    outcome: Optional[RoundOutcome] = None

    def to_dict(self, include_full_team_b: bool = False) -> dict:
        result = {
            "round_index": self.round_index,
            "multiplier": self.multiplier,
            "is_critical": self.is_critical,
            "has_diplomacy": self.has_diplomacy,
            "state": self.state.to_dict(),
            "diplomacy": self.diplomacy.to_dict() if self.diplomacy else None,
            "team_a": {
                "deliberation": [m.to_dict() for m in self.team_a_deliberation],
                "votes": self.team_a_votes.to_dict() if self.team_a_votes else None,
            },
            "team_b": {
                "decision": self.team_b_decision,
                "tally": self.team_b_tally,
            },
            "outcome": self.outcome.to_dict() if self.outcome else None,
        }

        # Include full Team B logs in FULL mode
        if include_full_team_b:
            if self.team_b_deliberation:
                result["team_b"]["deliberation"] = [m.to_dict() for m in self.team_b_deliberation]
            if self.team_b_votes:
                result["team_b"]["votes"] = self.team_b_votes.to_dict()

        return result


@dataclass
class TaskDefinition:
    """Task/scenario definition for the trajectory.

    Attributes:
        benchmark: Always "RedBlackBench"
        scenario_id: Scenario identifier (e.g., "pandemic_vaccines")
        scenario_variant_id: Specific variant for paraphrasing
        difficulty: Difficulty level
        options: Available choices
        rounds_total: Total number of rounds
        critical_rounds: Rounds with multipliers > 1
        critical_multipliers: Mapping of round -> multiplier
        diplomacy_rounds: Rounds where diplomacy is enabled
        payoff_matrix_id: Identifier for the payoff matrix
        payoff_matrix: The actual payoff values
        objective: The stated objective for agents
    """
    benchmark: str = "RedBlackBench"
    scenario_id: str = ""
    scenario_variant_id: str = ""
    difficulty: str = "neutral"  # "neutral", "hard_adversarial", "baseline"
    options: List[str] = field(default_factory=lambda: ["A", "B"])
    rounds_total: int = 10
    critical_rounds: List[int] = field(default_factory=list)
    critical_multipliers: Dict[int, int] = field(default_factory=dict)
    diplomacy_rounds: List[int] = field(default_factory=list)
    payoff_matrix_id: str = "rbclassic.v1"
    payoff_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    objective: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "scenario_id": self.scenario_id,
            "scenario_variant_id": self.scenario_variant_id,
            "difficulty": self.difficulty,
            "options": self.options,
            "rounds_total": self.rounds_total,
            "critical_rounds": self.critical_rounds,
            "critical_multipliers": {str(k): v for k, v in self.critical_multipliers.items()},
            "diplomacy_rounds": self.diplomacy_rounds,
            "payoff_matrix_id": self.payoff_matrix_id,
            "payoff_matrix": self.payoff_matrix,
            "objective": self.objective,
        }


@dataclass
class Participant:
    """A participant (agent) in the game.

    Attributes:
        agent_uid: Unique identifier (e.g., "A_00")
        agent_name: Human-readable name
        team: "team_a" or "team_b"
        role: Agent's role (e.g., "advisor")
        is_trained: Whether this agent uses a trained model
        model: Model information
        prompt_ids: References to prompts in prompts.json
    """
    agent_uid: str
    agent_name: str
    team: str  # "team_a" or "team_b"
    role: str = "advisor"
    is_trained: bool = False
    trained_checkpoint: Optional[str] = None
    model: Dict[str, Any] = field(default_factory=dict)
    prompt_ids: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "agent_uid": self.agent_uid,
            "agent_name": self.agent_name,
            "team": self.team,
            "role": self.role,
            "is_trained": self.is_trained,
            "model": self.model,
            "prompt_ids": self.prompt_ids,
        }
        if self.trained_checkpoint:
            result["trained_checkpoint"] = self.trained_checkpoint
        return result


@dataclass
class FinalSummary:
    """Final game summary metrics.

    Attributes:
        scores: Final scores for each team and combined
        cooperation_rates: Cooperation rate per team
        outcome_category: Classification of the outcome
    """
    scores: Dict[str, Any]  # team_a, team_b, sum, max_possible
    cooperation_rates: Dict[str, float]  # team_a, team_b
    outcome_category: str  # "full_cooperation", "partial_cooperation", "mutual_defection", etc.

    def to_dict(self) -> dict:
        return {
            "scores": self.scores,
            "cooperation_rates": self.cooperation_rates,
            "outcome_category": self.outcome_category,
        }


@dataclass
class TrajectoryQuality:
    """Quality metrics for the trajectory.

    Attributes:
        scalar_reward: Overall quality score (0-1)
        components: Breakdown of quality components
    """
    scalar_reward: float
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scalar_reward": self.scalar_reward,
            "components": self.components,
        }


@dataclass
class AgentAdherence:
    """Per-agent principle adherence metrics.

    Attributes:
        cooperation_rate: Fraction of rounds agent voted to cooperate
        influenced_by_defection: Whether agent changed behavior after opponent defection
        always_cooperative: Whether agent always voted to cooperate
        defection_rounds: List of rounds where agent voted to defect
    """
    cooperation_rate: float
    influenced_by_defection: bool = False
    always_cooperative: bool = False
    defection_rounds: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cooperation_rate": self.cooperation_rate,
            "influenced_by_defection": self.influenced_by_defection,
            "always_cooperative": self.always_cooperative,
            "defection_rounds": self.defection_rounds,
        }


@dataclass
class SFTTargets:
    """Targets for supervised fine-tuning.

    Attributes:
        ideal_votes_by_round: What the ideal agent should vote each round
        self_critique_prompts: Questions agents should ask themselves (Stage 1 method)
    """
    ideal_votes_by_round: Dict[str, str] = field(default_factory=dict)  # round_num -> "A" or "B"
    self_critique_prompts: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ideal_votes_by_round": self.ideal_votes_by_round,
            "self_critique_prompts": self.self_critique_prompts,
        }


@dataclass
class TrainingLabels:
    """Labels for training (SFT targets and reward signals).

    Attributes:
        trajectory_quality: Scalar reward and components
        agent_principle_adherence: Per-agent cooperation metrics
        sft_targets: Ideal behaviors for SFT training
    """
    trajectory_quality: Optional[TrajectoryQuality] = None
    agent_principle_adherence: Dict[str, AgentAdherence] = field(default_factory=dict)
    sft_targets: Optional[SFTTargets] = None

    def to_dict(self) -> dict:
        return {
            "trajectory_quality": self.trajectory_quality.to_dict() if self.trajectory_quality else None,
            "agent_principle_adherence": {k: v.to_dict() for k, v in self.agent_principle_adherence.items()},
            "sft_targets": self.sft_targets.to_dict() if self.sft_targets else None,
        }


@dataclass
class EnvironmentConfig:
    """Environment configuration for the game.

    Attributes:
        seed: Random seed for reproducibility
        team_size: Number of agents per team
        voting_rule: How team decisions are made
        tie_break: How ties are broken
        deliberation_turns_per_round: Number of discussion turns
        max_message_length: Maximum length of agent messages
        diplomacy_message_length: Maximum length of diplomacy messages
    """
    seed: Optional[int] = None
    team_size: Dict[str, int] = field(default_factory=lambda: {"team_a": 5, "team_b": 5})
    voting_rule: str = "majority"
    tie_break: str = "random"
    deliberation_turns_per_round: int = 1
    max_message_length: int = 500
    diplomacy_message_length: int = 200

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "team_size": self.team_size,
            "voting_rule": self.voting_rule,
            "tie_break": self.tie_break,
            "deliberation_turns_per_round": self.deliberation_turns_per_round,
            "max_message_length": self.max_message_length,
            "diplomacy_message_length": self.diplomacy_message_length,
        }


@dataclass
class TrainingTrajectory:
    """Complete training trajectory in rbbench.v1 format.

    This is the main output format for training data generation.
    """
    schema_version: str = "rbbench.v1"
    trajectory_id: str = ""
    logging_mode: str = "lite"  # "lite" or "full"

    # Task definition
    task: TaskDefinition = field(default_factory=TaskDefinition)

    # Environment configuration
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    # Participants (Team A fully, Team B based on logging mode)
    participants: List[Participant] = field(default_factory=list)

    # Round-by-round data
    rounds: List[TrainingRound] = field(default_factory=list)

    # Final summary
    final: Optional[FinalSummary] = None

    # Training labels
    labels: Optional[TrainingLabels] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        include_full_team_b = self.logging_mode == "full"
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "task": self.task.to_dict(),
            "environment": self.environment.to_dict(),
            "participants": [p.to_dict() for p in self.participants],
            "rounds": [r.to_dict(include_full_team_b=include_full_team_b) for r in self.rounds],
            "final": self.final.to_dict() if self.final else None,
            "labels": self.labels.to_dict() if self.labels else None,
            "metadata": self.metadata,
        }

    def save(self, filepath: str) -> None:
        """Save trajectory to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "TrainingTrajectory":
        """Load trajectory from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingTrajectory":
        """Reconstruct trajectory from dictionary."""
        trajectory = cls(
            schema_version=data.get("schema_version", "rbbench.v1"),
            trajectory_id=data.get("trajectory_id", ""),
            logging_mode=data.get("logging_mode", "lite"),
        )

        # Task
        task_data = data.get("task", {})
        trajectory.task = TaskDefinition(
            benchmark=task_data.get("benchmark", "RedBlackBench"),
            scenario_id=task_data.get("scenario_id", ""),
            scenario_variant_id=task_data.get("scenario_variant_id", ""),
            difficulty=task_data.get("difficulty", "neutral"),
            options=task_data.get("options", ["A", "B"]),
            rounds_total=task_data.get("rounds_total", 10),
            critical_rounds=task_data.get("critical_rounds", []),
            critical_multipliers={int(k): v for k, v in task_data.get("critical_multipliers", {}).items()},
            diplomacy_rounds=task_data.get("diplomacy_rounds", []),
            payoff_matrix_id=task_data.get("payoff_matrix_id", ""),
            payoff_matrix=task_data.get("payoff_matrix", {}),
            objective=task_data.get("objective", {}),
        )

        # Environment
        env_data = data.get("environment", {})
        trajectory.environment = EnvironmentConfig(
            seed=env_data.get("seed"),
            team_size=env_data.get("team_size", {"team_a": 5, "team_b": 5}),
            voting_rule=env_data.get("voting_rule", "majority"),
            tie_break=env_data.get("tie_break", "random"),
            deliberation_turns_per_round=env_data.get("deliberation_turns_per_round", 1),
            max_message_length=env_data.get("max_message_length", 500),
            diplomacy_message_length=env_data.get("diplomacy_message_length", 200),
        )

        # Participants
        for p_data in data.get("participants", []):
            trajectory.participants.append(Participant(
                agent_uid=p_data["agent_uid"],
                agent_name=p_data["agent_name"],
                team=p_data["team"],
                role=p_data.get("role", "advisor"),
                is_trained=p_data.get("is_trained", False),
                trained_checkpoint=p_data.get("trained_checkpoint"),
                model=p_data.get("model", {}),
                prompt_ids=p_data.get("prompt_ids", {}),
            ))

        # Rounds
        for r_data in data.get("rounds", []):
            state_data = r_data.get("state", {})
            round_obj = TrainingRound(
                round_index=r_data["round_index"],
                multiplier=r_data["multiplier"],
                is_critical=r_data["is_critical"],
                has_diplomacy=r_data["has_diplomacy"],
                state=RoundState(
                    history=state_data.get("history", []),
                    scores=state_data.get("scores", {}),
                    rounds_remaining=state_data.get("rounds_remaining", 0),
                    max_possible_from_here=state_data.get("max_possible_from_here", 0),
                ),
                team_b_decision=r_data.get("team_b", {}).get("decision", ""),
                team_b_tally=r_data.get("team_b", {}).get("tally", {}),
            )

            # Team A deliberation
            team_a_data = r_data.get("team_a", {})
            for msg_data in team_a_data.get("deliberation", []):
                round_obj.team_a_deliberation.append(AgentMessage(
                    agent_uid=msg_data["agent_uid"],
                    agent_name=msg_data["agent_name"],
                    turn=msg_data["turn"],
                    public_message=msg_data["public_message"],
                    private_reasoning=msg_data.get("private_reasoning"),
                    recommendation=msg_data.get("recommendation", ""),
                ))

            # Team A votes
            if team_a_data.get("votes"):
                votes_data = team_a_data["votes"]
                round_obj.team_a_votes = TeamVotes(
                    votes=votes_data["votes"],
                    tally=votes_data["tally"],
                    team_decision=votes_data["team_decision"],
                    was_unanimous=votes_data.get("was_unanimous", False),
                )

            # Team B deliberation (FULL mode)
            team_b_data = r_data.get("team_b", {})
            if team_b_data.get("deliberation"):
                round_obj.team_b_deliberation = []
                for msg_data in team_b_data["deliberation"]:
                    round_obj.team_b_deliberation.append(AgentMessage(
                        agent_uid=msg_data["agent_uid"],
                        agent_name=msg_data["agent_name"],
                        turn=msg_data["turn"],
                        public_message=msg_data["public_message"],
                        private_reasoning=msg_data.get("private_reasoning"),
                        recommendation=msg_data.get("recommendation", ""),
                    ))

            # Team B votes (FULL mode)
            if team_b_data.get("votes"):
                votes_data = team_b_data["votes"]
                round_obj.team_b_votes = TeamVotes(
                    votes=votes_data["votes"],
                    tally=votes_data["tally"],
                    team_decision=votes_data["team_decision"],
                    was_unanimous=votes_data.get("was_unanimous", False),
                )

            # Diplomacy
            if r_data.get("diplomacy"):
                dip_data = r_data["diplomacy"]
                round_obj.diplomacy = DiplomacyExchange(
                    team_a_message=dip_data.get("team_a_message"),
                    team_b_message=dip_data.get("team_b_message"),
                    team_a_sender=dip_data.get("team_a_sender"),
                    team_b_sender=dip_data.get("team_b_sender"),
                )

            # Outcome
            if r_data.get("outcome"):
                out_data = r_data["outcome"]
                round_obj.outcome = RoundOutcome(
                    joint_action=out_data["joint_action"],
                    base_payoff=out_data["base_payoff"],
                    applied_multiplier=out_data["applied_multiplier"],
                    round_delta=out_data["round_delta"],
                    new_scores=out_data["new_scores"],
                )

            trajectory.rounds.append(round_obj)

        # Final
        if data.get("final"):
            final_data = data["final"]
            trajectory.final = FinalSummary(
                scores=final_data["scores"],
                cooperation_rates=final_data["cooperation_rates"],
                outcome_category=final_data["outcome_category"],
            )

        # Labels
        if data.get("labels"):
            labels_data = data["labels"]
            trajectory.labels = TrainingLabels()

            if labels_data.get("trajectory_quality"):
                tq_data = labels_data["trajectory_quality"]
                trajectory.labels.trajectory_quality = TrajectoryQuality(
                    scalar_reward=tq_data["scalar_reward"],
                    components=tq_data.get("components", {}),
                )

            if labels_data.get("agent_principle_adherence"):
                for agent_uid, adh_data in labels_data["agent_principle_adherence"].items():
                    trajectory.labels.agent_principle_adherence[agent_uid] = AgentAdherence(
                        cooperation_rate=adh_data["cooperation_rate"],
                        influenced_by_defection=adh_data.get("influenced_by_defection", False),
                        always_cooperative=adh_data.get("always_cooperative", False),
                        defection_rounds=adh_data.get("defection_rounds", []),
                    )

            if labels_data.get("sft_targets"):
                sft_data = labels_data["sft_targets"]
                trajectory.labels.sft_targets = SFTTargets(
                    ideal_votes_by_round=sft_data.get("ideal_votes_by_round", {}),
                    self_critique_prompts=sft_data.get("self_critique_prompts", []),
                )

        # Metadata
        trajectory.metadata = data.get("metadata", {})

        return trajectory
