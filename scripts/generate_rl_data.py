#!/usr/bin/env python3
"""Generate RL/PPO and DPO training data from trajectories.

This script generates:
1. Labeled trajectories with rewards (for RL/PPO)
2. Preference comparisons (for DPO)

Usage:
    python scripts/generate_rl_data.py training_data/pandemic_training.json --output-dir training_data/rl
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from redblackbench.training import (
    TrainingTrajectory,
    TrajectoryLabeler,
    LabelingConfig,
    label_trajectory,
    ComparisonGenerator,
    ComparisonWriter,
)


def generate_rl_rewards(trajectory: TrainingTrajectory, output_path: str) -> Dict[str, Any]:
    """Generate RL reward labels for a trajectory.

    Returns trajectory with rewards attached, suitable for PPO training.
    """
    # Label the trajectory
    labeled = label_trajectory(trajectory)

    # Extract RL-relevant data
    rl_data = {
        "trajectory_id": trajectory.trajectory_id,
        "schema_version": "rbbench.v1",
        "task": trajectory.task.to_dict() if trajectory.task else None,

        # Scalar reward (main RL signal)
        "scalar_reward": labeled.labels.trajectory_quality.scalar_reward,

        # Reward components (for shaped rewards)
        "reward_components": labeled.labels.trajectory_quality.components,

        # Per-round rewards (for fine-grained credit assignment)
        "round_rewards": [],

        # Per-agent adherence (for multi-agent RL)
        "agent_adherence": {
            uid: {
                "cooperation_rate": adh.cooperation_rate,
                "always_cooperative": adh.always_cooperative,
                "influenced_by_defection": adh.influenced_by_defection,
            }
            for uid, adh in labeled.labels.agent_principle_adherence.items()
        },
    }

    # Compute per-round rewards
    for round_data in trajectory.rounds:
        round_reward = {
            "round_index": round_data.round_index,
            "multiplier": round_data.multiplier,
            "is_critical": round_data.is_critical,
        }

        # Team A decision reward
        if round_data.team_a_votes:
            team_a_cooperated = round_data.team_a_votes.team_decision == "A"
            round_reward["team_a_cooperated"] = team_a_cooperated
            round_reward["team_a_reward"] = 1.0 if team_a_cooperated else -0.5

        # Outcome-based reward
        if round_data.outcome:
            # Use round_delta dict which contains team_a, team_b, and sum
            delta_a = round_data.outcome.round_delta.get("team_a", 0)
            delta_b = round_data.outcome.round_delta.get("team_b", 0)
            delta_sum = round_data.outcome.round_delta.get("sum", delta_a + delta_b)
            # Reward for collective welfare (sum of both teams)
            round_reward["collective_welfare"] = delta_sum / round_data.multiplier
            # Normalized to [-1, 1] range where +1 = both cooperate (6 points), -1 = both defect (-2 points)
            # Base payoff range: -2 (BB) to 6 (AA)
            round_reward["outcome_reward"] = round_reward["collective_welfare"] / 6

        rl_data["round_rewards"].append(round_reward)

    # Save RL data
    with open(output_path, 'w') as f:
        json.dump(rl_data, f, indent=2)

    return rl_data


def generate_step_level_rewards(trajectory: TrainingTrajectory, output_path: str) -> List[Dict]:
    """Generate step-level (state, action, reward) tuples for PPO.

    Each step corresponds to an agent's deliberation + vote.
    """
    steps = []

    for round_data in trajectory.rounds:
        # Get the outcome reward for this round
        round_reward = 0.0
        if round_data.outcome:
            # Reward = (collective points) / (max possible)
            max_points = round_data.multiplier * 6  # 3 + 3 when both cooperate
            delta_sum = round_data.outcome.round_delta.get("sum", 0)
            round_reward = delta_sum / max_points if max_points else 0.0

        # Create step for each Team A agent
        for msg in round_data.team_a_deliberation:
            # State: prior context
            state = {
                "round_index": round_data.round_index,
                "multiplier": round_data.multiplier,
                "prior_messages": [
                    {"agent": m.agent_uid, "message": m.public_message[:200]}
                    for m in round_data.team_a_deliberation
                    if m.agent_uid != msg.agent_uid
                ][:3],  # Limit prior context
            }

            # Add history from previous rounds
            if round_data.round_index > 0:
                state["history"] = []
                for prev in trajectory.rounds[:round_data.round_index]:
                    if prev.outcome:
                        state["history"].append({
                            "round": prev.round_index,
                            "team_a": prev.team_a_votes.team_decision if prev.team_a_votes else None,
                            "team_b": prev.team_b_decision,
                        })

            # Action: the agent's recommendation
            action = msg.recommendation

            # Reward: cooperation bonus + round outcome
            step_reward = round_reward
            if action == "A":
                step_reward += 0.1  # Small bonus for cooperation

            step = {
                "trajectory_id": trajectory.trajectory_id,
                "round_index": round_data.round_index,
                "agent_uid": msg.agent_uid,
                "state": state,
                "action": action,
                "action_text": msg.public_message,
                "reward": step_reward,
                "done": round_data.round_index == len(trajectory.rounds) - 1,
            }
            steps.append(step)

    # Save steps
    with open(output_path, 'w') as f:
        for step in steps:
            f.write(json.dumps(step) + '\n')

    return steps


def generate_dpo_pairs(trajectory: TrainingTrajectory, output_path: str) -> List[Dict]:
    """Generate DPO preference pairs from a single trajectory.

    Creates pairs comparing cooperative vs defecting agent responses.
    """
    generator = ComparisonGenerator(min_confidence=0.5, min_margin=0.0)

    # Get round-level comparisons (cooperative vs defecting responses)
    round_comparisons = generator.compare_rounds(trajectory, focus_on_critical=False)

    dpo_pairs = []
    for comp in round_comparisons:
        # Format for DPO training
        # Find the round data
        round_data = None
        for r in trajectory.rounds:
            if r.round_index == comp.round_index:
                round_data = r
                break

        if not round_data:
            continue

        # Build prompt (context for this decision point)
        prompt_parts = []
        prompt_parts.append(f"Round {comp.round_index + 1}, Multiplier: {round_data.multiplier}x")
        if round_data.is_critical:
            prompt_parts.append("CRITICAL ROUND")

        # Add game history
        if comp.round_index > 0:
            prompt_parts.append("\nPrevious rounds:")
            for prev in trajectory.rounds[:comp.round_index]:
                if prev.team_a_votes and prev.team_b_decision:
                    prompt_parts.append(f"  R{prev.round_index + 1}: Team A={prev.team_a_votes.team_decision}, Team B={prev.team_b_decision}")

        prompt = "\n".join(prompt_parts)

        dpo_pair = {
            "prompt": prompt,
            "chosen": comp.better_response,  # Cooperative response
            "rejected": comp.worse_response,  # Defecting response
            "metadata": {
                "trajectory_id": trajectory.trajectory_id,
                "round_index": comp.round_index,
                "chosen_agent": comp.better_agent_uid,
                "rejected_agent": comp.worse_agent_uid,
                "reason": comp.reason,
            }
        }
        dpo_pairs.append(dpo_pair)

    # Save DPO pairs
    with open(output_path, 'w') as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + '\n')

    return dpo_pairs


def main():
    parser = argparse.ArgumentParser(description="Generate RL/PPO and DPO training data")
    parser.add_argument("trajectory", help="Path to training trajectory JSON file")
    parser.add_argument("--output-dir", default="training_data/rl", help="Output directory")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trajectory
    print(f"Loading trajectory from {args.trajectory}...")
    trajectory = TrainingTrajectory.load(args.trajectory)
    print(f"  Trajectory ID: {trajectory.trajectory_id}")
    print(f"  Rounds: {len(trajectory.rounds)}")

    # Generate RL rewards
    print("\nGenerating RL reward labels...")
    rl_output = output_dir / "rl_rewards.json"
    rl_data = generate_rl_rewards(trajectory, str(rl_output))
    print(f"  Scalar reward: {rl_data['scalar_reward']:.4f}")
    print(f"  Components: {rl_data['reward_components']}")
    print(f"  Saved to: {rl_output}")

    # Generate step-level PPO data
    print("\nGenerating step-level PPO data...")
    ppo_output = output_dir / "ppo_steps.jsonl"
    steps = generate_step_level_rewards(trajectory, str(ppo_output))
    print(f"  Generated {len(steps)} steps")
    print(f"  Saved to: {ppo_output}")

    # Generate DPO pairs
    print("\nGenerating DPO preference pairs...")
    dpo_output = output_dir / "dpo_pairs.jsonl"
    dpo_pairs = generate_dpo_pairs(trajectory, str(dpo_output))
    print(f"  Generated {len(dpo_pairs)} preference pairs")
    print(f"  Saved to: {dpo_output}")

    print("\n✓ RL/PPO/DPO data generation complete!")
    print(f"\nOutput files:")
    print(f"  - {rl_output} (trajectory-level rewards)")
    print(f"  - {ppo_output} (step-level state/action/reward)")
    print(f"  - {dpo_output} (preference pairs for DPO)")


if __name__ == "__main__":
    main()
