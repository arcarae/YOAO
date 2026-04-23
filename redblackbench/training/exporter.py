"""Training data exporter for RedBlackBench.

Converts raw game trajectories into the rbbench.v1 training format.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from redblackbench.trajectory.trajectory import GameTrajectory, TimestepType
from redblackbench.training.schemas import (
    TrainingTrajectory,
    TrainingRound,
    TaskDefinition,
    EnvironmentConfig,
    Participant,
    AgentMessage,
    TeamVotes,
    DiplomacyExchange,
    RoundState,
    RoundOutcome,
    FinalSummary,
)
from redblackbench.training.prompts import PromptRegistry


class TrainingDataExporter:
    """Exports raw trajectories to rbbench.v1 training format.

    Handles conversion from the full trajectory format to the streamlined
    training format, with options for LITE or FULL logging modes.
    """

    def __init__(
        self,
        prompt_registry: Optional[PromptRegistry] = None,
        logging_mode: str = "lite",
    ):
        """Initialize the exporter.

        Args:
            prompt_registry: Registry for externalizing prompts
            logging_mode: "lite" (Team A full, Team B summary) or "full" (both teams full)
        """
        self.prompt_registry = prompt_registry
        self.logging_mode = logging_mode

    def export(
        self,
        trajectory: GameTrajectory,
        scenario_id: Optional[str] = None,
        scenario_variant_id: Optional[str] = None,
        difficulty: str = "neutral",
    ) -> TrainingTrajectory:
        """Export a raw trajectory to training format.

        Args:
            trajectory: Raw GameTrajectory to convert
            scenario_id: Scenario identifier
            scenario_variant_id: Specific scenario variant
            difficulty: Difficulty classification

        Returns:
            TrainingTrajectory in rbbench.v1 format
        """
        training_traj = TrainingTrajectory(
            trajectory_id=trajectory.trajectory_id,
            logging_mode=self.logging_mode,
        )

        # Build task definition
        training_traj.task = self._build_task_definition(
            trajectory, scenario_id, scenario_variant_id, difficulty
        )

        # Build environment config
        training_traj.environment = self._build_environment(trajectory)

        # Build participants
        training_traj.participants = self._build_participants(trajectory)

        # Build rounds
        training_traj.rounds = self._build_rounds(trajectory)

        # Build final summary
        training_traj.final = self._build_final_summary(trajectory)

        # Metadata
        training_traj.metadata = {
            "source_trajectory_id": trajectory.trajectory_id,
            "exported_at": datetime.now().isoformat(),
            "exporter_version": "rbbench.exporter.v1",
            "team_a_model": trajectory.team_a_model,
            "team_b_model": trajectory.team_b_model,
            "start_time": trajectory.start_time,
            "end_time": trajectory.end_time,
        }

        return training_traj

    def _build_task_definition(
        self,
        trajectory: GameTrajectory,
        scenario_id: Optional[str],
        scenario_variant_id: Optional[str],
        difficulty: str,
    ) -> TaskDefinition:
        """Build task definition from trajectory."""
        config = trajectory.game_config

        # Identify critical rounds (multiplier > 1)
        multipliers = config.get("multipliers", {})
        critical_rounds = [int(r) for r, m in multipliers.items() if m > 1]
        critical_multipliers = {int(r): m for r, m in multipliers.items() if m > 1}

        # Build payoff matrix from config
        payoff_matrix = {
            "AA": {
                "team_a": config.get("both_black_score", 3),
                "team_b": config.get("both_black_score", 3),
                "sum": config.get("both_black_score", 3) * 2,
            },
            "BB": {
                "team_a": config.get("both_red_score", -3),
                "team_b": config.get("both_red_score", -3),
                "sum": config.get("both_red_score", -3) * 2,
            },
            "AB": {
                "team_a": config.get("black_loses_score", -6),
                "team_b": config.get("red_wins_score", 6),
                "sum": 0,
            },
            "BA": {
                "team_a": config.get("red_wins_score", 6),
                "team_b": config.get("black_loses_score", -6),
                "sum": 0,
            },
        }

        return TaskDefinition(
            benchmark="RedBlackBench",
            scenario_id=scenario_id or "unknown",
            scenario_variant_id=scenario_variant_id or "",
            difficulty=difficulty,
            options=["A", "B"],
            rounds_total=config.get("num_rounds", 10),
            critical_rounds=sorted(critical_rounds),
            critical_multipliers=critical_multipliers,
            diplomacy_rounds=[],  # TODO: Extract from config if available
            payoff_matrix_id="rbclassic.v1",
            payoff_matrix=payoff_matrix,
            objective={
                "type": "maximize_combined_total",
                "text": "Success is scored by the combined total outcome across both parties over all cycles.",
            },
        )

    def _build_environment(self, trajectory: GameTrajectory) -> EnvironmentConfig:
        """Build environment config from trajectory."""
        config = trajectory.game_config

        return EnvironmentConfig(
            seed=config.get("seed"),
            team_size={
                "team_a": config.get("team_size", 5),
                "team_b": config.get("team_size", 5),
            },
            voting_rule="majority",
            tie_break="random",
            deliberation_turns_per_round=config.get("discussion_rounds", 1),
            max_message_length=500,
            diplomacy_message_length=200,
        )

    def _build_participants(self, trajectory: GameTrajectory) -> List[Participant]:
        """Build participant list from trajectory."""
        participants = []

        # Get team snapshots from first timestep
        for timestep in trajectory.timesteps:
            if timestep.team_a_snapshot:
                for i, agent in enumerate(timestep.team_a_snapshot.agents):
                    participants.append(Participant(
                        agent_uid=f"A_{i:02d}",
                        agent_name=agent.agent_id,
                        team="team_a",
                        role="advisor",
                        is_trained=False,
                        model={
                            "provider": "unknown",
                            "name": trajectory.team_a_model or "unknown",
                        },
                        prompt_ids={},  # Will be filled if prompt_registry is set
                    ))
                break

        for timestep in trajectory.timesteps:
            if timestep.team_b_snapshot:
                for i, agent in enumerate(timestep.team_b_snapshot.agents):
                    participants.append(Participant(
                        agent_uid=f"B_{i:02d}",
                        agent_name=agent.agent_id,
                        team="team_b",
                        role="advisor",
                        is_trained=False,
                        model={
                            "provider": "unknown",
                            "name": trajectory.team_b_model or "unknown",
                        },
                        prompt_ids={},
                    ))
                break

        return participants

    def _build_rounds(self, trajectory: GameTrajectory) -> List[TrainingRound]:
        """Build round data from trajectory timesteps."""
        rounds = []
        config = trajectory.game_config
        multipliers = config.get("multipliers", {})

        # Group timesteps by round
        round_data: Dict[int, Dict[str, Any]] = {}

        for timestep in trajectory.timesteps:
            round_num = timestep.round_num
            if round_num == 0:
                continue  # Skip game_start

            if round_num not in round_data:
                round_data[round_num] = {
                    "team_a_opinions": [],
                    "team_a_votes": None,
                    "team_b_opinions": [],
                    "team_b_votes": None,
                    "outcome": None,
                    "state": None,
                }

            rd = round_data[round_num]

            # Extract data based on timestep type
            if timestep.timestep_type == TimestepType.INITIAL_OPINIONS:
                team_id = timestep.metadata.get("team_identifier", "")
                opinions = self._extract_opinions(timestep)
                if team_id == "A":
                    rd["team_a_opinions"] = opinions
                else:
                    rd["team_b_opinions"] = opinions

            elif timestep.timestep_type == TimestepType.FINAL_VOTES:
                team_id = timestep.metadata.get("team_identifier", "")
                votes = self._extract_votes(timestep)
                if team_id == "A":
                    rd["team_a_votes"] = votes
                else:
                    rd["team_b_votes"] = votes

            elif timestep.timestep_type == TimestepType.ROUND_END:
                rd["outcome"] = timestep.outcome

            elif timestep.timestep_type == TimestepType.ROUND_START:
                # Extract state from snapshot
                if timestep.team_a_snapshot:
                    rd["state"] = self._extract_state(timestep, round_num, config)

        # Convert to TrainingRound objects
        for round_num in sorted(round_data.keys()):
            rd = round_data[round_num]
            multiplier = multipliers.get(str(round_num), 1)
            is_critical = multiplier > 1

            # Build state
            state = rd.get("state") or RoundState(
                history=[],
                scores={"team_a": 0, "team_b": 0, "sum": 0},
                rounds_remaining=config.get("num_rounds", 10) - round_num,
                max_possible_from_here=0,
            )

            # Build outcome
            outcome = None
            if rd["outcome"]:
                o = rd["outcome"]
                joint_action = self._choice_to_option(o.team_a_choice) + self._choice_to_option(o.team_b_choice)
                outcome = RoundOutcome(
                    joint_action=joint_action,
                    base_payoff={
                        "team_a": o.team_a_score // o.multiplier if o.multiplier else o.team_a_score,
                        "team_b": o.team_b_score // o.multiplier if o.multiplier else o.team_b_score,
                        "sum": (o.team_a_score + o.team_b_score) // o.multiplier if o.multiplier else (o.team_a_score + o.team_b_score),
                    },
                    applied_multiplier=o.multiplier,
                    round_delta={
                        "team_a": o.team_a_score,
                        "team_b": o.team_b_score,
                        "sum": o.team_a_score + o.team_b_score,
                    },
                    new_scores={
                        "team_a": o.total_score - o.team_b_score if hasattr(o, 'total_score') else 0,
                        "team_b": o.team_b_score,
                        "sum": o.total_score if hasattr(o, 'total_score') else 0,
                    },
                )

            # Build round
            training_round = TrainingRound(
                round_index=round_num,
                multiplier=multiplier,
                is_critical=is_critical,
                has_diplomacy=False,  # TODO: Detect diplomacy rounds
                state=state,
                team_a_deliberation=rd["team_a_opinions"],
                team_a_votes=rd["team_a_votes"],
                team_b_decision=self._get_team_decision(rd["team_b_votes"]),
                team_b_tally=self._get_team_tally(rd["team_b_votes"]),
                outcome=outcome,
            )

            # Include full Team B in FULL mode
            if self.logging_mode == "full":
                training_round.team_b_deliberation = rd["team_b_opinions"]
                training_round.team_b_votes = rd["team_b_votes"]

            rounds.append(training_round)

        return rounds

    def _extract_opinions(self, timestep) -> List[AgentMessage]:
        """Extract agent opinions from a timestep."""
        messages = []
        for i, action in enumerate(timestep.actions):
            if action.action_type == "individual_opinion":
                # Map actor name to UID
                team_prefix = "A" if "team_a" in str(timestep.metadata.get("team", "")).lower() or timestep.metadata.get("team_identifier") == "A" else "B"

                messages.append(AgentMessage(
                    agent_uid=f"{team_prefix}_{i:02d}",
                    agent_name=action.actor,
                    turn=i + 1,
                    public_message=action.reasoning or "",
                    private_reasoning=action.private_thought,
                    recommendation=self._choice_to_option(action.choice),
                ))
        return messages

    def _extract_votes(self, timestep) -> Optional[TeamVotes]:
        """Extract voting results from a timestep."""
        votes = {}
        tally = {"A": 0, "B": 0}
        team_decision = ""

        for i, action in enumerate(timestep.actions):
            if action.action_type == "individual_vote":
                team_prefix = "A" if timestep.metadata.get("team_identifier") == "A" else "B"
                agent_uid = f"{team_prefix}_{i:02d}"
                option = self._choice_to_option(action.choice)
                votes[agent_uid] = option
                tally[option] = tally.get(option, 0) + 1
            elif action.action_type == "team_choice":
                team_decision = self._choice_to_option(action.choice)

        if not votes:
            return None

        was_unanimous = timestep.metadata.get("was_unanimous", False)
        return TeamVotes(
            votes=votes,
            tally=tally,
            team_decision=team_decision,
            was_unanimous=was_unanimous,
        )

    def _extract_state(self, timestep, round_num: int, config: dict) -> RoundState:
        """Extract game state from a timestep."""
        history = []

        # Build history from previous rounds
        snapshot = timestep.team_a_snapshot
        if snapshot:
            for i, choice in enumerate(snapshot.choices_made):
                history.append({
                    "round": i + 1,
                    "team_a": self._choice_to_option(choice),
                    "team_b": "?",  # Would need team_b snapshot
                })

        scores = {
            "team_a": snapshot.current_score if snapshot else 0,
            "team_b": 0,  # Would need team_b snapshot
            "sum": snapshot.current_score if snapshot else 0,
        }

        return RoundState(
            history=history,
            scores=scores,
            rounds_remaining=config.get("num_rounds", 10) - round_num + 1,
            max_possible_from_here=config.get("max_possible_score", 150),
        )

    def _choice_to_option(self, choice: str) -> str:
        """Convert Choice enum string to option (A/B)."""
        if not choice:
            return ""
        choice_str = str(choice).upper()
        if "BLACK" in choice_str:
            return "A"  # BLACK = cooperate = A
        elif "RED" in choice_str:
            return "B"  # RED = defect = B
        elif choice_str in ["A", "B"]:
            return choice_str
        return ""

    def _get_team_decision(self, votes: Optional[TeamVotes]) -> str:
        """Get team decision from votes."""
        return votes.team_decision if votes else ""

    def _get_team_tally(self, votes: Optional[TeamVotes]) -> Dict[str, int]:
        """Get vote tally from votes."""
        return votes.tally if votes else {}

    def _build_final_summary(self, trajectory: GameTrajectory) -> Optional[FinalSummary]:
        """Build final summary from trajectory."""
        if not trajectory.final_outcome:
            return None

        fo = trajectory.final_outcome

        # Determine outcome category
        coop_rate = fo.cooperation_rate
        if coop_rate >= 0.9:
            outcome_category = "full_cooperation"
        elif coop_rate >= 0.6:
            outcome_category = "partial_cooperation"
        elif coop_rate >= 0.3:
            outcome_category = "mixed"
        else:
            outcome_category = "mutual_defection"

        return FinalSummary(
            scores={
                "team_a": fo.team_a_score,
                "team_b": fo.team_b_score,
                "sum": fo.total_score,
                "max_possible": fo.max_possible_score,
            },
            cooperation_rates={
                "team_a": coop_rate,  # Would need per-team rates
                "team_b": coop_rate,
            },
            outcome_category=outcome_category,
        )

    def export_batch(
        self,
        trajectories: List[GameTrajectory],
        output_dir: str,
        scenario_id: Optional[str] = None,
    ) -> List[str]:
        """Export multiple trajectories to a directory.

        Args:
            trajectories: List of trajectories to export
            output_dir: Directory to save exported files
            scenario_id: Scenario identifier for all trajectories

        Returns:
            List of paths to exported files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_paths = []
        for traj in trajectories:
            training_traj = self.export(traj, scenario_id=scenario_id)
            filepath = output_path / f"{training_traj.trajectory_id}.json"
            training_traj.save(str(filepath))
            exported_paths.append(str(filepath))

        return exported_paths


def export_trajectory_file(
    input_path: str,
    output_path: str,
    scenario_id: Optional[str] = None,
    logging_mode: str = "lite",
) -> TrainingTrajectory:
    """Export a single trajectory file to training format.

    Args:
        input_path: Path to raw trajectory JSON
        output_path: Path to save training format JSON
        scenario_id: Scenario identifier
        logging_mode: "lite" or "full"

    Returns:
        The exported TrainingTrajectory
    """
    trajectory = GameTrajectory.load(input_path)
    exporter = TrainingDataExporter(logging_mode=logging_mode)
    training_traj = exporter.export(trajectory, scenario_id=scenario_id)
    training_traj.save(output_path)
    return training_traj
