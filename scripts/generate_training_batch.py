#!/usr/bin/env python3
"""Generate diverse training trajectories and SFT data at scale.

This script:
1. Runs games across multiple scenarios with diverse opponent strategies
2. Exports trajectories to rbbench.v1 format
3. Generates SFT data from all trajectories

Diversity axes:
- 5 scenarios (pandemic, climate, AGI safety, standards, election crisis)
- 10+ opponent strategies (cooperative, exploitative, tit-for-tat, etc.)
- Exploitation timing (early, mid, late, critical rounds)
- Recovery patterns (never, quick, gradual, conditional)

Usage:
    # Generate 50 games (pilot run) with scripted opponents
    python scripts/generate_training_batch.py --num-games 50 --output-dir training_data/batch_01

    # Generate 200 games (full scale) - ~10K SFT examples
    python scripts/generate_training_batch.py --num-games 200 --output-dir training_data/batch_02

    # Skip game generation, just generate SFT from existing trajectories
    python scripts/generate_training_batch.py --sft-only --trajectory-dir training_data/batch_01/trajectories

    # Use LLM opponents instead of scripted (less diverse but more realistic)
    python scripts/generate_training_batch.py --num-games 50 --llm-opponents
"""

import argparse
import asyncio
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Available scenarios with their difficulty levels
SCENARIOS = [
    {"id": "pandemic_vaccines", "difficulty": "neutral", "weight": 1.0},
    {"id": "climate_cooperation", "difficulty": "neutral", "weight": 1.0},
    {"id": "agi_safety", "difficulty": "neutral", "weight": 1.0},
    {"id": "standards_coordination", "difficulty": "neutral", "weight": 1.0},
    {"id": "election_crisis", "difficulty": "hard_adversarial", "weight": 1.0},
]

# Scripted opponent strategies for diverse training data
# Distribution: 20% easy, 30% medium, 30% hard, 20% very hard
OPPONENT_STRATEGIES = {
    # Easy (cooperative) - 20%
    "always_cooperate": 0.10,
    "mostly_cooperate": 0.10,

    # Medium (mixed) - 30%
    "tit_for_tat": 0.10,
    "early_exploiter": 0.10,
    "mid_exploiter": 0.10,

    # Hard (exploitation) - 30%
    "late_exploiter": 0.10,
    "critical_exploiter": 0.10,
    "defect_critical": 0.10,

    # Very hard (sustained exploitation) - 20%
    "always_defect": 0.10,
    "early_exploiter_no_recovery": 0.10,
}

# Models to use for generating games
DEFAULT_MODEL = "qwen/qwen3-14b"  # Cheap model for trajectory generation
THINKING_MODEL = "moonshotai/kimi-k2-thinking"  # For SFT ideal response generation


def sample_strategy() -> str:
    """Sample a strategy according to the training distribution."""
    strategies = list(OPPONENT_STRATEGIES.keys())
    weights = list(OPPONENT_STRATEGIES.values())
    return random.choices(strategies, weights=weights, k=1)[0]


def distribute_games(num_games: int) -> List[Tuple[str, str]]:
    """Distribute games across scenarios and strategies.

    Returns:
        List of (scenario_id, strategy_id) tuples
    """
    # Distribute evenly across scenarios
    scenarios_per_game = []
    scenario_ids = [s["id"] for s in SCENARIOS]

    for i in range(num_games):
        scenario = scenario_ids[i % len(scenario_ids)]
        strategy = sample_strategy()
        scenarios_per_game.append((scenario, strategy))

    random.shuffle(scenarios_per_game)
    return scenarios_per_game


async def run_single_game(
    scenario_id: str,
    strategy_id: str,
    game_index: int,
    output_dir: Path,
    model: str,
    team_size: int,
) -> Optional[Path]:
    """Run a single game with scripted opponent.

    Returns:
        Path to saved trajectory, or None if failed
    """
    try:
        from redblackbench.game.coordinator import GameCoordinator
        from redblackbench.game.config import GameConfig
        from redblackbench.teams.team import Team
        from redblackbench.strategies import create_scripted_team
        from redblackbench.scenarios import get_scenario
        from redblackbench.trajectory import TrajectoryCollector
        from redblackbench.agents.llm_agent import LLMAgent
        from redblackbench.cli import create_provider

        # Get scenario
        scenario = get_scenario(scenario_id)
        if not scenario:
            print(f"    Unknown scenario: {scenario_id}")
            return None

        # Create game config (default already has multipliers for rounds 5, 8, 10)
        config = GameConfig(
            num_rounds=10,
            multipliers={5: 3, 8: 5, 10: 10},
        )

        # Create Team A agents (LLM-based)
        prompt_template = scenario.to_prompt_template()
        team_a_names = ["Dr. Sarah Chen", "Marcus Webb", "Dr. Priya Sharma", "James O'Connor", "Dr. Elena Vasquez"]

        # Get API key from environment
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        agents = []
        for i in range(team_size):
            agent_id = team_a_names[i % len(team_a_names)]
            provider = create_provider({
                "type": "openrouter",
                "model": model,
                "temperature": 1.0,
                "api_key": api_key,
            })
            agent = LLMAgent(
                agent_id=agent_id,
                team_name="Team A",
                provider=provider,
                prompt_template=prompt_template,
            )
            agents.append(agent)

        team_a = Team(name="Team A", agents=agents)

        # Create Team B (scripted opponent)
        team_b = create_scripted_team(
            strategy_id=strategy_id,
            team_name=f"Team B ({strategy_id})",
        )

        # Setup trajectory collection
        results_dir = output_dir / "results" / scenario_id
        results_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = results_dir / f"game_{game_index}_{strategy_id}.json"

        collector = TrajectoryCollector()

        # Create coordinator
        coordinator = GameCoordinator(
            team_a=team_a,
            team_b=team_b,
            config=config,
            trajectory_collector=collector,
            trajectory_save_path=str(trajectory_path),
            team_a_model=model,
            team_b_model=f"scripted:{strategy_id}",
        )

        # Run the game with timeout protection (same as CLI)
        try:
            await asyncio.wait_for(coordinator.play_game(), timeout=1800)  # 30 min timeout
        except asyncio.TimeoutError:
            print(f"    Game timed out after 30 minutes")
            return None

        return trajectory_path

    except Exception as e:
        print(f"    Error running game: {e}")
        import traceback
        traceback.print_exc()
        return None


async def run_single_game_llm(
    scenario_id: str,
    game_index: int,
    output_dir: Path,
    model: str,
    team_size: int,
) -> Optional[Path]:
    """Run a single game with LLM opponent (both teams use LLM)."""
    try:
        from redblackbench.game.coordinator import GameCoordinator
        from redblackbench.game.config import GameConfig
        from redblackbench.teams.team import Team
        from redblackbench.scenarios import get_scenario
        from redblackbench.trajectory import TrajectoryCollector
        from redblackbench.agents.llm_agent import LLMAgent
        from redblackbench.cli import create_provider

        # Get scenario
        scenario = get_scenario(scenario_id)
        if not scenario:
            print(f"    Unknown scenario: {scenario_id}")
            return None

        # Create game config (default already has multipliers for rounds 5, 8, 10)
        config = GameConfig(
            num_rounds=10,
            multipliers={5: 3, 8: 5, 10: 10},
        )

        # Create both teams as LLM
        prompt_template = scenario.to_prompt_template()
        team_a_names = ["Dr. Sarah Chen", "Marcus Webb", "Dr. Priya Sharma", "James O'Connor", "Dr. Elena Vasquez"]
        team_b_names = ["Dr. Thomas Berg", "Linda Okonkwo", "Dr. Raj Mehta", "Susan Clarke", "Dr. Antonio Silva"]

        # Get API key from environment
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        # Create Team A agents
        agents_a = []
        for i in range(team_size):
            agent_id = team_a_names[i % len(team_a_names)]
            provider = create_provider({
                "type": "openrouter",
                "model": model,
                "temperature": 1.0,
                "api_key": api_key,
            })
            agent = LLMAgent(
                agent_id=agent_id,
                team_name="Team A",
                provider=provider,
                prompt_template=prompt_template,
            )
            agents_a.append(agent)
        team_a = Team(name="Team A", agents=agents_a)

        # Create Team B agents
        agents_b = []
        for i in range(team_size):
            agent_id = team_b_names[i % len(team_b_names)]
            provider = create_provider({
                "type": "openrouter",
                "model": model,
                "temperature": 1.0,
                "api_key": api_key,
            })
            agent = LLMAgent(
                agent_id=agent_id,
                team_name="Team B",
                provider=provider,
                prompt_template=prompt_template,
            )
            agents_b.append(agent)
        team_b = Team(name="Team B", agents=agents_b)

        # Setup trajectory collection
        results_dir = output_dir / "results" / scenario_id
        results_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = results_dir / f"game_{game_index}_llm.json"

        collector = TrajectoryCollector()

        # Create coordinator
        coordinator = GameCoordinator(
            team_a=team_a,
            team_b=team_b,
            config=config,
            trajectory_collector=collector,
            trajectory_save_path=str(trajectory_path),
            team_a_model=model,
            team_b_model=model,
        )

        # Run the game with timeout protection (same as CLI)
        try:
            await asyncio.wait_for(coordinator.play_game(), timeout=1800)  # 30 min timeout
        except asyncio.TimeoutError:
            print(f"    Game timed out after 30 minutes")
            return None

        return trajectory_path

    except Exception as e:
        print(f"    Error running game: {e}")
        import traceback
        traceback.print_exc()
        return None


async def generate_games(
    num_games: int,
    output_dir: Path,
    model: str,
    team_size: int,
    use_llm_opponents: bool = False,
) -> List[Path]:
    """Generate diverse game trajectories."""
    print(f"\n{'='*60}")
    print(f"PHASE 1: Generating {num_games} games")
    print(f"Opponent type: {'LLM' if use_llm_opponents else 'Scripted strategies'}")
    print(f"{'='*60}")

    if use_llm_opponents:
        # Simple scenario distribution for LLM opponents
        scenario_ids = [s["id"] for s in SCENARIOS]
        distribution = [(scenario_ids[i % len(scenario_ids)], "llm") for i in range(num_games)]
    else:
        # Diverse distribution with scripted opponents
        distribution = distribute_games(num_games)

    # Count distributions
    scenario_counts = Counter(s for s, _ in distribution)
    strategy_counts = Counter(st for _, st in distribution)

    print("\nScenario distribution:")
    for scenario_id, count in sorted(scenario_counts.items()):
        print(f"  {scenario_id}: {count} games")

    if not use_llm_opponents:
        print("\nStrategy distribution:")
        for strategy_id, count in sorted(strategy_counts.items()):
            print(f"  {strategy_id}: {count} games")

    # Run games
    successful = []
    failed = []

    for i, (scenario_id, strategy_id) in enumerate(distribution):
        print(f"\n[{i+1}/{num_games}] {scenario_id} vs {strategy_id}...")

        if use_llm_opponents:
            result = await run_single_game_llm(
                scenario_id=scenario_id,
                game_index=i,
                output_dir=output_dir,
                model=model,
                team_size=team_size,
            )
        else:
            result = await run_single_game(
                scenario_id=scenario_id,
                strategy_id=strategy_id,
                game_index=i,
                output_dir=output_dir,
                model=model,
                team_size=team_size,
            )

        if result:
            successful.append(result)
            print(f"  ✓ Saved: {result.name}")
        else:
            failed.append((scenario_id, strategy_id))
            print(f"  ✗ Failed")

    print(f"\n{len(successful)}/{num_games} games completed successfully")
    if failed:
        print(f"Failed: {len(failed)}")

    return successful


def export_trajectories(results_dir: Path, output_dir: Path) -> List[Path]:
    """Export game results to rbbench.v1 training format."""
    print(f"\n{'='*60}")
    print("PHASE 2: Exporting trajectories to training format")
    print(f"{'='*60}")

    from redblackbench.training import export_trajectory_file

    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    exported = []

    # Find all result JSON files
    for result_file in results_dir.rglob("*.json"):
        if "config" in result_file.name:
            continue

        try:
            output_path = trajectories_dir / f"{result_file.stem}_training.json"
            export_trajectory_file(str(result_file), str(output_path))
            exported.append(output_path)
            print(f"  ✓ Exported: {result_file.name}")
        except Exception as e:
            print(f"  ✗ Failed to export {result_file.name}: {e}")

    print(f"\n{len(exported)} trajectories exported")
    return exported


def save_sft_checkpoint(
    output_path: Path,
    all_examples: List[Dict],
    num_trajectories_processed: int,
    total_trajectories: int,
):
    """Save incremental checkpoint of SFT data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined = {
        "schema_version": "sft.v1",
        "generated_at": datetime.now().isoformat(),
        "num_examples": len(all_examples),
        "num_trajectories_processed": num_trajectories_processed,
        "total_trajectories": total_trajectories,
        "is_complete": num_trajectories_processed >= total_trajectories,
        "examples": all_examples,
    }

    with open(output_path, 'w') as f:
        json.dump(combined, f, indent=2)


def load_sft_checkpoint(output_path: Path) -> Tuple[List[Dict], int]:
    """Load existing checkpoint if available.

    Returns:
        Tuple of (examples_list, num_trajectories_already_processed)
    """
    if not output_path.exists():
        return [], 0

    try:
        with open(output_path, 'r') as f:
            data = json.load(f)

        examples = data.get("examples", [])
        processed = data.get("num_trajectories_processed", 0)
        is_complete = data.get("is_complete", False)

        if is_complete:
            print(f"  Found complete checkpoint with {len(examples)} examples")
            return examples, processed

        print(f"  Resuming from checkpoint: {processed} trajectories, {len(examples)} examples")
        return examples, processed

    except Exception as e:
        print(f"  Could not load checkpoint: {e}")
        return [], 0


async def generate_sft_data(
    trajectory_paths: List[Path],
    output_path: Path,
    max_per_round: int = 3,
) -> int:
    """Generate SFT data from trajectories with incremental saving."""
    print(f"\n{'='*60}")
    print("PHASE 3: Generating SFT data")
    print(f"{'='*60}")

    from redblackbench.training import TrainingTrajectory, generate_sft_data as gen_sft

    # Load checkpoint if exists
    all_examples, start_index = load_sft_checkpoint(output_path)

    if start_index >= len(trajectory_paths):
        print(f"All {len(trajectory_paths)} trajectories already processed!")
        return len(all_examples)

    if start_index > 0:
        print(f"Resuming from trajectory {start_index + 1}/{len(trajectory_paths)}")

    for i, traj_path in enumerate(trajectory_paths):
        # Skip already processed
        if i < start_index:
            continue

        print(f"\n[{i+1}/{len(trajectory_paths)}] {traj_path.name}")

        try:
            # Generate SFT examples
            dataset = await gen_sft(
                trajectory_path=str(traj_path),
                output_path=None,  # Don't save individual files
                max_per_round=max_per_round,
            )

            if dataset and dataset.examples:
                new_examples = [ex.to_dict() for ex in dataset.examples]
                all_examples.extend(new_examples)
                print(f"  Generated {len(new_examples)} examples (total: {len(all_examples)})")

            # Save checkpoint after each trajectory
            save_sft_checkpoint(
                output_path=output_path,
                all_examples=all_examples,
                num_trajectories_processed=i + 1,
                total_trajectories=len(trajectory_paths),
            )
            print(f"  ✓ Checkpoint saved")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            # Still save checkpoint on error
            save_sft_checkpoint(
                output_path=output_path,
                all_examples=all_examples,
                num_trajectories_processed=i,  # Don't count failed one
                total_trajectories=len(trajectory_paths),
            )

    print(f"\n{'='*60}")
    print(f"SFT data saved to: {output_path}")
    print(f"Total examples: {len(all_examples)}")
    print(f"{'='*60}")

    return len(all_examples)


async def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse training data at scale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pilot run with scripted opponents (recommended)
  python scripts/generate_training_batch.py --num-games 10 --output-dir training_data/pilot

  # Full scale: 200 games → ~10K SFT examples
  python scripts/generate_training_batch.py --num-games 200 --output-dir training_data/batch_10k

  # Generate SFT from existing trajectories
  python scripts/generate_training_batch.py --sft-only --trajectory-dir training_data/pilot/trajectories
        """
    )
    parser.add_argument("--num-games", type=int, default=10000, help="Number of games to generate (paper: 10,000)")
    parser.add_argument("--output-dir", type=str, default="training_data/batch", help="Output directory")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model for Team A")
    parser.add_argument("--team-size", type=int, default=5, help="Team A size (agents)")
    parser.add_argument("--max-per-round", type=int, default=3, help="Max SFT examples per round (paper: 3)")
    parser.add_argument("--llm-opponents", action="store_true", help="Use LLM for Team B instead of scripted strategies")
    parser.add_argument("--sft-only", action="store_true", help="Skip game generation, only generate SFT")
    parser.add_argument("--trajectory-dir", type=str, help="Directory with existing trajectories (for --sft-only)")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT generation (only run games + export)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sft_only:
        # Just generate SFT from existing trajectories
        if not args.trajectory_dir:
            print("Error: --trajectory-dir required with --sft-only")
            return

        traj_dir = Path(args.trajectory_dir)
        trajectory_paths = list(traj_dir.glob("*.json"))

        if not trajectory_paths:
            print(f"No trajectories found in {traj_dir}")
            return

        print(f"Found {len(trajectory_paths)} trajectories")

        sft_output = output_dir / "sft_combined.json"
        await generate_sft_data(trajectory_paths, sft_output, args.max_per_round)
    else:
        # Full pipeline: generate games -> export -> SFT

        # Phase 1: Generate games
        result_paths = await generate_games(
            num_games=args.num_games,
            output_dir=output_dir,
            model=args.model,
            team_size=args.team_size,
            use_llm_opponents=args.llm_opponents,
        )

        if not result_paths:
            print("No games completed successfully")
            return

        # Phase 2: Export trajectories
        results_dir = output_dir / "results"
        trajectory_paths = export_trajectories(results_dir, output_dir)

        if not trajectory_paths:
            print("No trajectories to process")
            return

        # Phase 3: Generate SFT data
        if not args.skip_sft:
            sft_output = output_dir / "sft_combined.json"
            await generate_sft_data(trajectory_paths, sft_output, args.max_per_round)

    print("\n✓ Done!")


if __name__ == "__main__":
    asyncio.run(main())
