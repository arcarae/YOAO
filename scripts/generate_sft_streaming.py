#!/usr/bin/env python3
"""Streaming SFT data generation pipeline with parallelization.

This script processes trajectories through all phases in a streaming manner:
  1. Generate game trajectory (run simulation)
  2. Export to rbbench.v1 training format
  3. Generate SFT data using Kimi K2

Each trajectory flows through all phases before moving to the next one.
Multiple trajectories can be processed in parallel.

Usage:
    # Generate 10 trajectories with 3 parallel workers
    python scripts/generate_sft_streaming.py --num-games 10 --workers 3

    # Generate 50 trajectories with 5 parallel workers, save to custom dir
    python scripts/generate_sft_streaming.py --num-games 50 --workers 5 --output-dir training_data/streaming_batch

    # Resume from existing output (skips already completed trajectories)
    python scripts/generate_sft_streaming.py --num-games 50 --workers 5 --output-dir training_data/streaming_batch --resume

    # Use LLM opponents instead of scripted
    python scripts/generate_sft_streaming.py --num-games 10 --workers 3 --llm-opponents
"""

import argparse
import asyncio
import json
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

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
OPPONENT_STRATEGIES = {
    # Easy (cooperative) - 20%
    "always_cooperate": 0.10,
    "mostly_cooperate": 0.10,
    # Medium (mixed) - 30%
    "tit_for_tat": 0.10,
    "early_exploiter": 0.10,
    "mid_exploiter": 0.10,
    # Hard (exploitation) - 30%
    "late_betrayer": 0.10,
    "critical_exploiter": 0.10,
    "defect_critical": 0.10,
    # Very hard (sustained exploitation) - 20%
    "always_defect": 0.10,
    "early_exploiter_no_recovery": 0.10,
}

# Models
DEFAULT_MODEL = "qwen/qwen3-14b"  # For trajectory generation
THINKING_MODEL = "moonshotai/kimi-k2-thinking"  # For SFT ideal response generation


@dataclass
class TrajectoryResult:
    """Result of processing a single trajectory."""
    game_index: int
    scenario_id: str
    strategy_id: str
    success: bool
    trajectory_path: Optional[Path] = None
    training_path: Optional[Path] = None
    sft_examples: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    phases_completed: List[str] = field(default_factory=list)


class RateLimiter:
    """Simple rate limiter using token bucket algorithm.

    Prevents API overload when running many parallel workers.
    """

    def __init__(self, requests_per_second: float = 2.0):
        """Initialize rate limiter.

        Args:
            requests_per_second: Max requests per second across all workers
        """
        self.min_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until we can make another request."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self._last_request_time
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            self._last_request_time = asyncio.get_event_loop().time()


@dataclass
class StreamingProgress:
    """Progress tracker for streaming pipeline."""
    total: int
    completed: int = 0
    failed: int = 0
    in_progress: int = 0
    total_sft_examples: int = 0
    completed_ids: Set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "in_progress": self.in_progress,
            "total_sft_examples": self.total_sft_examples,
            "completed_ids": list(self.completed_ids),
        }


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
    scenarios_per_game = []
    scenario_ids = [s["id"] for s in SCENARIOS]

    for i in range(num_games):
        scenario = scenario_ids[i % len(scenario_ids)]
        strategy = sample_strategy()
        scenarios_per_game.append((scenario, strategy))

    random.shuffle(scenarios_per_game)
    return scenarios_per_game


async def run_game(
    scenario_id: str,
    strategy_id: str,
    game_index: int,
    output_dir: Path,
    model: str,
    team_size: int,
    use_llm_opponents: bool = False,
) -> Optional[Path]:
    """Run a single game and return the trajectory path.

    Phase 1: Game generation
    """
    try:
        from redblackbench.game.coordinator import GameCoordinator
        from redblackbench.game.config import GameConfig
        from redblackbench.teams.team import Team
        from redblackbench.scenarios import get_scenario
        from redblackbench.trajectory import TrajectoryCollector
        from redblackbench.agents.llm_agent import LLMAgent
        from redblackbench.cli import create_provider

        scenario = get_scenario(scenario_id)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_id}")

        config = GameConfig(
            num_rounds=10,
            multipliers={5: 3, 8: 5, 10: 10},
        )

        prompt_template = scenario.to_prompt_template()
        team_a_names = ["Dr. Sarah Chen", "Marcus Webb", "Dr. Priya Sharma", "James O'Connor", "Dr. Elena Vasquez"]
        team_b_names = ["Dr. Thomas Berg", "Linda Okonkwo", "Dr. Raj Mehta", "Susan Clarke", "Dr. Antonio Silva"]

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        # Create Team A (always LLM)
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

        # Create Team B (scripted or LLM)
        if use_llm_opponents:
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
            team_b_model = model
        else:
            from redblackbench.strategies import create_scripted_team
            team_b = create_scripted_team(
                strategy_id=strategy_id,
                team_name=f"Team B ({strategy_id})",
            )
            team_b_model = f"scripted:{strategy_id}"

        # Setup trajectory collection
        results_dir = output_dir / "results" / scenario_id
        results_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = results_dir / f"game_{game_index}_{strategy_id}.json"

        collector = TrajectoryCollector()

        coordinator = GameCoordinator(
            team_a=team_a,
            team_b=team_b,
            config=config,
            trajectory_collector=collector,
            trajectory_save_path=str(trajectory_path),
            team_a_model=model,
            team_b_model=team_b_model,
        )

        await asyncio.wait_for(coordinator.play_game(), timeout=1800)  # 30 min timeout
        return trajectory_path

    except asyncio.TimeoutError:
        raise Exception("Game timed out after 30 minutes")
    except Exception as e:
        raise


def export_trajectory(
    result_path: Path,
    output_dir: Path,
    scenario_id: str,
) -> Path:
    """Export raw trajectory to training format.

    Phase 2: Export to rbbench.v1 format
    """
    from redblackbench.training import export_trajectory_file

    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    output_path = trajectories_dir / f"{result_path.stem}_training.json"
    export_trajectory_file(
        str(result_path),
        str(output_path),
        scenario_id=scenario_id,
    )
    return output_path


async def generate_sft_for_trajectory(
    training_path: Path,
    max_per_round: int = 5,
    model: str = THINKING_MODEL,
    verbose: bool = False,
) -> List[Dict]:
    """Generate SFT examples from a training trajectory.

    Phase 3: Generate ideal responses with Kimi K2

    IMPORTANT: Creates a fresh SFTGenerator instance per trajectory to avoid
    shared client state corruption when running in parallel.
    """
    from redblackbench.training.schemas import TrainingTrajectory
    from redblackbench.training.sft_generator import SFTGenerator

    # Load trajectory
    trajectory = TrainingTrajectory.load(str(training_path))

    # Create a NEW generator instance for this trajectory
    # This ensures no shared client state between parallel workers
    generator = SFTGenerator(model=model)

    dataset = await generator.generate_from_trajectory(
        trajectory=trajectory,
        team="team_a",
        max_examples_per_round=max_per_round,
        verbose=verbose,
    )

    if dataset and dataset.examples:
        return [ex.to_dict() for ex in dataset.examples]
    return []


async def process_single_trajectory(
    game_index: int,
    scenario_id: str,
    strategy_id: str,
    output_dir: Path,
    model: str,
    team_size: int,
    max_per_round: int,
    sft_model: str,
    use_llm_opponents: bool = False,
    semaphore: Optional[asyncio.Semaphore] = None,
    rate_limiter: Optional[RateLimiter] = None,
    verbose: bool = False,
) -> TrajectoryResult:
    """Process a single trajectory through all phases.

    Phase 1: Run game
    Phase 2: Export trajectory
    Phase 3: Generate SFT data

    Each trajectory creates its own provider/generator instances to avoid
    shared state corruption between parallel workers.

    Args:
        game_index: Index of this game in the batch
        scenario_id: Which scenario to use
        strategy_id: Which opponent strategy to use
        output_dir: Where to save outputs
        model: Model for game generation
        team_size: Number of agents per team
        max_per_round: Max SFT examples per round
        sft_model: Model for SFT generation
        use_llm_opponents: Whether to use LLM for Team B
        semaphore: Semaphore for limiting concurrent workers
        rate_limiter: Rate limiter for API calls
        verbose: If True, enable detailed logging for Kimi K2 API calls
    """
    result = TrajectoryResult(
        game_index=game_index,
        scenario_id=scenario_id,
        strategy_id=strategy_id,
        success=False,
    )

    async def _process():
        # Rate limit before starting (prevents burst of parallel starts)
        if rate_limiter:
            await rate_limiter.acquire()

        # Phase 1: Run game
        print(f"  [{game_index}] Phase 1: Running game ({scenario_id} vs {strategy_id})...")
        trajectory_path = await run_game(
            scenario_id=scenario_id,
            strategy_id=strategy_id,
            game_index=game_index,
            output_dir=output_dir,
            model=model,
            team_size=team_size,
            use_llm_opponents=use_llm_opponents,
        )
        result.trajectory_path = trajectory_path
        result.phases_completed.append("game_generation")
        print(f"  [{game_index}] Phase 1 complete: {trajectory_path.name}")

        # Phase 2: Export trajectory (no API calls, no rate limiting needed)
        print(f"  [{game_index}] Phase 2: Exporting trajectory...")
        training_path = export_trajectory(
            result_path=trajectory_path,
            output_dir=output_dir,
            scenario_id=scenario_id,
        )
        result.training_path = training_path
        result.phases_completed.append("export")
        print(f"  [{game_index}] Phase 2 complete: {training_path.name}")

        # Rate limit before SFT generation (many API calls)
        if rate_limiter:
            await rate_limiter.acquire()

        # Phase 3: Generate SFT
        print(f"  [{game_index}] Phase 3: Generating SFT data with {sft_model}...")
        sft_examples = await generate_sft_for_trajectory(
            training_path=training_path,
            max_per_round=max_per_round,
            model=sft_model,
            verbose=verbose,
        )
        result.sft_examples = sft_examples
        result.phases_completed.append("sft_generation")
        print(f"  [{game_index}] Phase 3 complete: {len(sft_examples)} SFT examples")

        result.success = True

    try:
        if semaphore:
            async with semaphore:
                await _process()
        else:
            await _process()
    except Exception as e:
        result.error = str(e)
        print(f"  [{game_index}] ERROR at phase {len(result.phases_completed) + 1}: {e}")

    return result


def save_progress(
    output_dir: Path,
    progress: StreamingProgress,
    all_examples: List[Dict],
    results: List[TrajectoryResult],
):
    """Save progress checkpoint."""
    progress_path = output_dir / "streaming_progress.json"
    sft_path = output_dir / "sft_combined.json"

    # Save progress metadata
    progress_data = progress.to_dict()
    progress_data["last_updated"] = datetime.now().isoformat()
    progress_data["results"] = [
        {
            "game_index": r.game_index,
            "scenario_id": r.scenario_id,
            "strategy_id": r.strategy_id,
            "success": r.success,
            "phases_completed": r.phases_completed,
            "num_sft_examples": len(r.sft_examples),
            "error": r.error,
        }
        for r in results
    ]

    with open(progress_path, 'w') as f:
        json.dump(progress_data, f, indent=2)

    # Save SFT data
    sft_data = {
        "schema_version": "rbbench.sft.v1",
        "generator_model": THINKING_MODEL,
        "generated_at": datetime.now().isoformat(),
        "num_examples": len(all_examples),
        "num_trajectories_completed": progress.completed,
        "total_trajectories": progress.total,
        "is_complete": progress.completed >= progress.total,
        "examples": all_examples,
    }

    with open(sft_path, 'w') as f:
        json.dump(sft_data, f, indent=2)


def load_progress(output_dir: Path) -> Tuple[StreamingProgress, List[Dict], Set[int]]:
    """Load existing progress if available.

    Returns:
        Tuple of (progress, existing_examples, completed_game_indices)
    """
    progress_path = output_dir / "streaming_progress.json"
    sft_path = output_dir / "sft_combined.json"

    completed_indices = set()
    examples = []
    progress = None

    if progress_path.exists():
        try:
            with open(progress_path, 'r') as f:
                data = json.load(f)
            completed_indices = set(data.get("completed_ids", []))
            progress = StreamingProgress(
                total=data.get("total", 0),
                completed=data.get("completed", 0),
                failed=data.get("failed", 0),
                total_sft_examples=data.get("total_sft_examples", 0),
                completed_ids=completed_indices,
            )
            print(f"  Loaded progress: {progress.completed}/{progress.total} completed")
        except Exception as e:
            print(f"  Could not load progress: {e}")

    if sft_path.exists():
        try:
            with open(sft_path, 'r') as f:
                data = json.load(f)
            examples = data.get("examples", [])
            print(f"  Loaded {len(examples)} existing SFT examples")
        except Exception as e:
            print(f"  Could not load SFT data: {e}")

    return progress, examples, completed_indices


async def run_streaming_pipeline(
    num_games: int,
    output_dir: Path,
    model: str,
    team_size: int,
    max_per_round: int,
    sft_model: str,
    workers: int,
    use_llm_opponents: bool = False,
    resume: bool = False,
    requests_per_second: float = 2.0,
    verbose: bool = False,
):
    """Run the streaming pipeline with parallel processing.

    Concurrency safety:
    - Each trajectory gets its own SFTGenerator instance (no shared client state)
    - Semaphore limits concurrent workers
    - Rate limiter prevents API overload
    - All mutable state is local to each trajectory

    Args:
        num_games: Number of trajectories to generate
        output_dir: Where to save all outputs
        model: Model for game trajectory generation
        team_size: Number of agents per team
        max_per_round: Max SFT examples per round
        sft_model: Model for SFT ideal response generation
        workers: Number of parallel workers
        use_llm_opponents: Whether to use LLM for Team B
        resume: Whether to resume from existing progress
        requests_per_second: Rate limit for API calls
        verbose: If True, enable detailed logging for Kimi K2 API calls
    """
    print(f"\n{'='*60}")
    print(f"STREAMING SFT PIPELINE")
    print(f"{'='*60}")
    print(f"Games to generate: {num_games}")
    print(f"Parallel workers: {workers}")
    print(f"Rate limit: {requests_per_second} req/s")
    print(f"Game model: {model}")
    print(f"SFT model: {sft_model}")
    print(f"Opponent type: {'LLM' if use_llm_opponents else 'Scripted'}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing progress if resuming
    completed_indices: Set[int] = set()
    all_examples: List[Dict] = []

    if resume:
        print("Checking for existing progress...")
        existing_progress, all_examples, completed_indices = load_progress(output_dir)
        if completed_indices:
            print(f"Resuming: {len(completed_indices)} games already completed\n")

    # Generate game distribution
    distribution = distribute_games(num_games)

    # Filter out already completed games
    pending_games = [
        (i, scenario, strategy)
        for i, (scenario, strategy) in enumerate(distribution)
        if i not in completed_indices
    ]

    if not pending_games:
        print("All games already completed!")
        return

    print(f"Games to process: {len(pending_games)} ({len(completed_indices)} already done)\n")

    # Initialize progress tracking
    progress = StreamingProgress(
        total=num_games,
        completed=len(completed_indices),
        total_sft_examples=len(all_examples),
        completed_ids=completed_indices,
    )

    results: List[TrajectoryResult] = []

    # Create semaphore for parallel execution
    semaphore = asyncio.Semaphore(workers)

    # Create rate limiter to prevent API overload
    rate_limiter = RateLimiter(requests_per_second=requests_per_second)

    # Create tasks for all pending games
    tasks = []
    for game_index, scenario_id, strategy_id in pending_games:
        task = process_single_trajectory(
            game_index=game_index,
            scenario_id=scenario_id,
            strategy_id=strategy_id,
            output_dir=output_dir,
            model=model,
            team_size=team_size,
            max_per_round=max_per_round,
            sft_model=sft_model,
            use_llm_opponents=use_llm_opponents,
            semaphore=semaphore,
            rate_limiter=rate_limiter,
            verbose=verbose,
        )
        tasks.append(task)

    # Process with progress updates
    print(f"Starting {len(tasks)} trajectory processing tasks...\n")

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)

        if result.success:
            progress.completed += 1
            progress.completed_ids.add(result.game_index)
            progress.total_sft_examples += len(result.sft_examples)
            all_examples.extend(result.sft_examples)
            status = "SUCCESS"
        else:
            progress.failed += 1
            status = f"FAILED ({result.error})"

        print(f"\n[{progress.completed + progress.failed}/{len(pending_games)}] "
              f"Game {result.game_index}: {status}")

        # Save checkpoint after each completion
        save_progress(output_dir, progress, all_examples, results)

    # Final summary
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total games: {num_games}")
    print(f"Successful: {progress.completed}")
    print(f"Failed: {progress.failed}")
    print(f"Total SFT examples: {progress.total_sft_examples}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Streaming SFT data generation with parallelization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 trajectories with 3 parallel workers
  python scripts/generate_sft_streaming.py --num-games 10 --workers 3

  # Generate 50 trajectories with 5 workers, custom output
  python scripts/generate_sft_streaming.py --num-games 50 --workers 5 --output-dir training_data/streaming

  # Resume interrupted run
  python scripts/generate_sft_streaming.py --num-games 50 --workers 5 --output-dir training_data/streaming --resume

  # Use LLM opponents
  python scripts/generate_sft_streaming.py --num-games 10 --workers 3 --llm-opponents
        """
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=10,
        help="Number of games/trajectories to generate"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers (default: 3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_data/streaming_batch",
        help="Output directory for all data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model for game trajectory generation"
    )
    parser.add_argument(
        "--sft-model",
        type=str,
        default=THINKING_MODEL,
        help="Model for SFT ideal response generation"
    )
    parser.add_argument(
        "--team-size",
        type=int,
        default=5,
        help="Number of agents per team"
    )
    parser.add_argument(
        "--max-per-round",
        type=int,
        default=5,
        help="Max SFT examples per round"
    )
    parser.add_argument(
        "--llm-opponents",
        action="store_true",
        help="Use LLM for Team B instead of scripted strategies"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing progress (skip completed games)"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=2.0,
        help="Max API requests per second across all workers (default: 2.0)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable detailed logging for Kimi K2 API calls (timing, tokens, retries)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    await run_streaming_pipeline(
        num_games=args.num_games,
        output_dir=output_dir,
        model=args.model,
        team_size=args.team_size,
        max_per_round=args.max_per_round,
        sft_model=args.sft_model,
        workers=args.workers,
        use_llm_opponents=args.llm_opponents,
        resume=args.resume,
        requests_per_second=args.rate_limit,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    asyncio.run(main())
