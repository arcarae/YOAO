"""Training data generation for RedBlackBench.

This module provides tools to convert raw game trajectories into
the rbbench.v1 training format suitable for SFT and preference learning.

Key components:
- schemas: Data structures for training trajectories (rbbench.v1)
- prompts: Prompt registry for externalizing prompts
- exporter: Converts raw trajectories to training format
- comparisons: Generates comparisons.jsonl for preference learning
- labeler: Auto-generates rewards and adherence scores
"""

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
    TrainingLabels,
    TrajectoryQuality,
    AgentAdherence,
    SFTTargets,
)
from redblackbench.training.prompts import PromptRegistry, PromptEntry
from redblackbench.training.exporter import TrainingDataExporter, export_trajectory_file
from redblackbench.training.comparisons import (
    Comparison,
    RoundComparison,
    ComparisonGenerator,
    ComparisonWriter,
    generate_comparisons_from_trajectories,
)
from redblackbench.training.labeler import (
    TrajectoryLabeler,
    LabelingConfig,
    label_trajectory,
    label_trajectories,
)
from redblackbench.training.sft_generator import (
    SFTExample,
    SFTDataset,
    SFTContextBuilder,
    SFTGenerator,
    generate_sft_data,
)

__all__ = [
    # Schemas
    "TrainingTrajectory",
    "TrainingRound",
    "TaskDefinition",
    "EnvironmentConfig",
    "Participant",
    "AgentMessage",
    "TeamVotes",
    "DiplomacyExchange",
    "RoundState",
    "RoundOutcome",
    "FinalSummary",
    "TrainingLabels",
    "TrajectoryQuality",
    "AgentAdherence",
    "SFTTargets",
    # Prompts
    "PromptRegistry",
    "PromptEntry",
    # Exporter
    "TrainingDataExporter",
    "export_trajectory_file",
    # Comparisons
    "Comparison",
    "RoundComparison",
    "ComparisonGenerator",
    "ComparisonWriter",
    "generate_comparisons_from_trajectories",
    # Labeler
    "TrajectoryLabeler",
    "LabelingConfig",
    "label_trajectory",
    "label_trajectories",
    # SFT Generator
    "SFTExample",
    "SFTDataset",
    "SFTContextBuilder",
    "SFTGenerator",
    "generate_sft_data",
]
