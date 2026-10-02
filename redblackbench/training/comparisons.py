"""Comparison generation for preference learning (DPO/RLHF).

Generates comparisons.jsonl files for training preference/reward models.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import random

from redblackbench.training.schemas import TrainingTrajectory


@dataclass
class Comparison:
    """A single pairwise comparison for preference learning.

    Attributes:
        comparison_id: Unique identifier
        better_trajectory_id: ID of the preferred trajectory
        worse_trajectory_id: ID of the less preferred trajectory
        confidence: Confidence in this comparison (0-1)
        margin: Score difference (better - worse)
        reason: Why one is better
        comparison_type: Type of comparison (trajectory, round, response)
        metadata: Additional context
    """
    comparison_id: str
    better_trajectory_id: str
    worse_trajectory_id: str
    confidence: float
    margin: float
    reason: str
    comparison_type: str = "trajectory"  # trajectory, round, response
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "comparison_id": self.comparison_id,
            "better": self.better_trajectory_id,
            "worse": self.worse_trajectory_id,
            "confidence": self.confidence,
            "margin": self.margin,
            "reason": self.reason,
            "comparison_type": self.comparison_type,
            "metadata": self.metadata,
        }

    def to_jsonl(self) -> str:
        """Convert to JSONL line."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "Comparison":
        return cls(
            comparison_id=data["comparison_id"],
            better_trajectory_id=data["better"],
            worse_trajectory_id=data["worse"],
            confidence=data["confidence"],
            margin=data.get("margin", 0.0),
            reason=data["reason"],
            comparison_type=data.get("comparison_type", "trajectory"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RoundComparison:
    """Comparison between agent responses in a single round.

    For fine-grained preference learning at the response level.
    """
    comparison_id: str
    trajectory_id: str
    round_index: int
    better_agent_uid: str
    worse_agent_uid: str
    better_response: str
    worse_response: str
    confidence: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "comparison_id": self.comparison_id,
            "trajectory_id": self.trajectory_id,
            "round_index": self.round_index,
            "better_agent": self.better_agent_uid,
            "worse_agent": self.worse_agent_uid,
            "better_response": self.better_response,
            "worse_response": self.worse_response,
            "confidence": self.confidence,
            "reason": self.reason,
            "metadata": self.metadata,
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict())


class ComparisonGenerator:
    """Generates pairwise comparisons from trajectories.

    Supports multiple comparison strategies:
    - Trajectory-level: Compare overall game outcomes
    - Round-level: Compare decisions within specific rounds
    - Response-level: Compare individual agent responses
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        min_margin: float = 0.1,
    ):
        """Initialize the generator.

        Args:
            min_confidence: Minimum confidence to include a comparison
            min_margin: Minimum score margin to include a comparison
        """
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self._comparison_count = 0

    def _generate_id(self) -> str:
        """Generate unique comparison ID."""
        self._comparison_count += 1
        return f"cmp_{datetime.now().strftime('%Y%m%d')}_{self._comparison_count:06d}"

    def compare_trajectories(
        self,
        trajectories: List[TrainingTrajectory],
        comparison_metric: str = "collective_welfare",
    ) -> List[Comparison]:
        """Generate pairwise comparisons between trajectories.

        Args:
            trajectories: List of trajectories to compare
            comparison_metric: Metric to use for comparison
                - "collective_welfare": Total combined score
                - "cooperation_rate": Overall cooperation rate
                - "efficiency": Score as fraction of maximum
                - "principle_adherence": Team A's cooperation rate

        Returns:
            List of Comparison objects
        """
        comparisons = []

        # Score each trajectory
        scored = []
        for traj in trajectories:
            score = self._score_trajectory(traj, comparison_metric)
            scored.append((traj, score))

        # Generate pairwise comparisons
        for i, (traj_a, score_a) in enumerate(scored):
            for j, (traj_b, score_b) in enumerate(scored):
                if i >= j:
                    continue

                margin = abs(score_a - score_b)
                if margin < self.min_margin:
                    continue

                # Determine which is better
                if score_a > score_b:
                    better, worse = traj_a, traj_b
                    better_score, worse_score = score_a, score_b
                else:
                    better, worse = traj_b, traj_a
                    better_score, worse_score = score_b, score_a

                # Calculate confidence based on margin
                confidence = min(0.5 + margin, 1.0)
                if confidence < self.min_confidence:
                    continue

                comparison = Comparison(
                    comparison_id=self._generate_id(),
                    better_trajectory_id=better.trajectory_id,
                    worse_trajectory_id=worse.trajectory_id,
                    confidence=confidence,
                    margin=better_score - worse_score,
                    reason=f"higher_{comparison_metric}",
                    comparison_type="trajectory",
                    metadata={
                        "metric": comparison_metric,
                        "better_score": better_score,
                        "worse_score": worse_score,
                    },
                )
                comparisons.append(comparison)

        return comparisons

    def _score_trajectory(
        self,
        trajectory: TrainingTrajectory,
        metric: str,
    ) -> float:
        """Score a trajectory by the given metric."""
        if not trajectory.final:
            return 0.0

        if metric == "collective_welfare":
            max_possible = trajectory.final.scores.get("max_possible", 150)
            total = trajectory.final.scores.get("sum", 0)
            # Normalize to 0-1
            return (total + max_possible) / (2 * max_possible) if max_possible else 0.0

        elif metric == "cooperation_rate":
            return trajectory.final.cooperation_rates.get("team_a", 0.0)

        elif metric == "efficiency":
            max_possible = trajectory.final.scores.get("max_possible", 150)
            total = trajectory.final.scores.get("sum", 0)
            return total / max_possible if max_possible else 0.0

        elif metric == "principle_adherence":
            # Team A's cooperation rate as a proxy for principle adherence
            return trajectory.final.cooperation_rates.get("team_a", 0.0)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def compare_rounds(
        self,
        trajectory: TrainingTrajectory,
        focus_on_critical: bool = True,
    ) -> List[RoundComparison]:
        """Generate comparisons between agent responses within rounds.

        Useful for fine-grained preference learning on individual responses.

        Args:
            trajectory: Trajectory to analyze
            focus_on_critical: Only compare critical rounds (higher multiplier)

        Returns:
            List of RoundComparison objects
        """
        comparisons = []

        for round_data in trajectory.rounds:
            if focus_on_critical and not round_data.is_critical:
                continue

            # Compare agent responses within Team A
            messages = round_data.team_a_deliberation
            if len(messages) < 2:
                continue

            # Find agents who recommended cooperation (A) vs defection (B)
            cooperators = [m for m in messages if m.recommendation == "A"]
            defectors = [m for m in messages if m.recommendation == "B"]

            if not cooperators or not defectors:
                continue

            # Create comparisons: cooperators are "better" (principle-aligned)
            for coop in cooperators:
                for defect in defectors:
                    comparison = RoundComparison(
                        comparison_id=self._generate_id(),
                        trajectory_id=trajectory.trajectory_id,
                        round_index=round_data.round_index,
                        better_agent_uid=coop.agent_uid,
                        worse_agent_uid=defect.agent_uid,
                        better_response=coop.public_message,
                        worse_response=defect.public_message,
                        confidence=0.8 if round_data.is_critical else 0.7,
                        reason="cooperation_recommendation",
                        metadata={
                            "multiplier": round_data.multiplier,
                            "is_critical": round_data.is_critical,
                            "better_recommendation": coop.recommendation,
                            "worse_recommendation": defect.recommendation,
                        },
                    )
                    comparisons.append(comparison)

        return comparisons

    def compare_by_outcome(
        self,
        trajectories: List[TrainingTrajectory],
    ) -> List[Comparison]:
        """Generate comparisons based on game outcomes.

        Groups trajectories by outcome category and compares between groups.

        Args:
            trajectories: List of trajectories to compare

        Returns:
            List of Comparison objects
        """
        # Group by outcome category
        by_category: Dict[str, List[TrainingTrajectory]] = {}
        for traj in trajectories:
            if not traj.final:
                continue
            category = traj.final.outcome_category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(traj)

        # Define category ordering (best to worst)
        category_order = [
            "full_cooperation",
            "partial_cooperation",
            "mixed",
            "mutual_defection",
        ]

        comparisons = []

        # Compare trajectories from different categories
        for i, better_cat in enumerate(category_order):
            if better_cat not in by_category:
                continue

            for worse_cat in category_order[i + 1:]:
                if worse_cat not in by_category:
                    continue

                # Sample pairs to avoid combinatorial explosion
                better_trajs = by_category[better_cat]
                worse_trajs = by_category[worse_cat]

                max_pairs = min(len(better_trajs) * len(worse_trajs), 100)
                pairs = []
                for b in better_trajs:
                    for w in worse_trajs:
                        pairs.append((b, w))

                if len(pairs) > max_pairs:
                    pairs = random.sample(pairs, max_pairs)

                for better, worse in pairs:
                    confidence = 0.7 + 0.1 * (category_order.index(worse_cat) - category_order.index(better_cat))
                    confidence = min(confidence, 0.95)

                    comparison = Comparison(
                        comparison_id=self._generate_id(),
                        better_trajectory_id=better.trajectory_id,
                        worse_trajectory_id=worse.trajectory_id,
                        confidence=confidence,
                        margin=0.0,  # Categorical comparison
                        reason=f"outcome_category_{better_cat}_vs_{worse_cat}",
                        comparison_type="trajectory",
                        metadata={
                            "better_category": better_cat,
                            "worse_category": worse_cat,
                        },
                    )
                    comparisons.append(comparison)

        return comparisons


class ComparisonWriter:
    """Writes comparisons to JSONL files."""

    def __init__(self, output_path: str):
        """Initialize the writer.

        Args:
            output_path: Path to the output .jsonl file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        comparisons: List[Comparison],
        append: bool = False,
    ) -> int:
        """Write comparisons to file.

        Args:
            comparisons: List of comparisons to write
            append: If True, append to existing file

        Returns:
            Number of comparisons written
        """
        mode = 'a' if append else 'w'
        with open(self.output_path, mode) as f:
            for comp in comparisons:
                f.write(comp.to_jsonl() + '\n')

        return len(comparisons)

    def write_round_comparisons(
        self,
        comparisons: List[RoundComparison],
        append: bool = False,
    ) -> int:
        """Write round-level comparisons to file.

        Args:
            comparisons: List of round comparisons to write
            append: If True, append to existing file

        Returns:
            Number of comparisons written
        """
        mode = 'a' if append else 'w'
        with open(self.output_path, mode) as f:
            for comp in comparisons:
                f.write(comp.to_jsonl() + '\n')

        return len(comparisons)

    @staticmethod
    def load(filepath: str) -> List[Comparison]:
        """Load comparisons from a JSONL file.

        Args:
            filepath: Path to the .jsonl file

        Returns:
            List of Comparison objects
        """
        comparisons = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    comparisons.append(Comparison.from_dict(data))
        return comparisons


def generate_comparisons_from_trajectories(
    trajectory_dir: str,
    output_path: str,
    metrics: List[str] = None,
    include_round_comparisons: bool = True,
) -> Dict[str, int]:
    """Generate all comparisons from a directory of trajectories.

    Args:
        trajectory_dir: Directory containing training trajectory JSON files
        output_path: Path to save comparisons.jsonl
        metrics: List of metrics to compare by (default: all)
        include_round_comparisons: Include round-level comparisons

    Returns:
        Dict with counts of comparisons generated by type
    """
    if metrics is None:
        metrics = ["collective_welfare", "cooperation_rate", "principle_adherence"]

    # Load all trajectories
    traj_path = Path(trajectory_dir)
    trajectories = []
    for f in traj_path.glob("*.json"):
        try:
            traj = TrainingTrajectory.load(str(f))
            trajectories.append(traj)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    if not trajectories:
        print("No trajectories found")
        return {"total": 0}

    generator = ComparisonGenerator()
    writer = ComparisonWriter(output_path)

    counts = {}
    all_comparisons = []

    # Trajectory-level comparisons
    for metric in metrics:
        comps = generator.compare_trajectories(trajectories, metric)
        counts[f"trajectory_{metric}"] = len(comps)
        all_comparisons.extend(comps)

    # Outcome-based comparisons
    outcome_comps = generator.compare_by_outcome(trajectories)
    counts["trajectory_outcome"] = len(outcome_comps)
    all_comparisons.extend(outcome_comps)

    # Write trajectory comparisons
    writer.write(all_comparisons)

    # Round-level comparisons (separate file)
    if include_round_comparisons:
        round_comps = []
        for traj in trajectories:
            round_comps.extend(generator.compare_rounds(traj))

        if round_comps:
            round_output = output_path.replace(".jsonl", "_rounds.jsonl")
            round_writer = ComparisonWriter(round_output)
            round_writer.write_round_comparisons(round_comps)
            counts["round_comparisons"] = len(round_comps)

    counts["total"] = sum(counts.values())
    return counts
