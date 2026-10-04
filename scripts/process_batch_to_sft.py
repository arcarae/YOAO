#!/usr/bin/env python3
"""Process existing batch results into SFT training data.

This script takes raw game results and:
1. Exports them to rbbench.v1 trajectory format
2. Generates SFT data using Kimi K2 thinking model

Usage:
    # Process batch_10k results
    python scripts/process_batch_to_sft.py \
        --results-dir training_data/batch_10k/results \
        --output-dir training_data/batch_10k

    # Resume from checkpoint (automatic)
    python scripts/process_batch_to_sft.py \
        --results-dir training_data/batch_10k/results \
        --output-dir training_data/batch_10k

    # Skip export if trajectories already exist
    python scripts/process_batch_to_sft.py \
        --results-dir training_data/batch_10k/results \
        --output-dir training_data/batch_10k \
        --skip-export

    # Limit to games up to a certain number (e.g., game_0 to game_29)
    python scripts/process_batch_to_sft.py \
        --results-dir training_data/batch_10k/results \
        --output-dir training_data/batch_10k \
        --max-game 29

    # Process a range of games (e.g., game_30 to game_50)
    python scripts/process_batch_to_sft.py \
        --results-dir training_data/batch_10k/results \
        --output-dir training_data/batch_10k \
        --min-game 30 --max-game 50
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_game_number(filename: str) -> Optional[int]:
    """Extract game number from filename like 'game_13_mid_exploiter.json'."""
    match = re.search(r'game_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def filter_by_game_range(
    files: List[Path],
    min_game: Optional[int] = None,
    max_game: Optional[int] = None,
) -> List[Path]:
    """Filter files to only include games within the specified range.

    Args:
        files: List of file paths to filter
        min_game: Minimum game number (inclusive), None means no lower bound
        max_game: Maximum game number (inclusive), None means no upper bound

    Returns:
        Filtered list of file paths
    """
    if min_game is None and max_game is None:
        return files

    filtered = []
    for f in files:
        game_num = extract_game_number(f.name)
        if game_num is not None:
            if min_game is not None and game_num < min_game:
                continue
            if max_game is not None and game_num > max_game:
                continue
            filtered.append(f)
    return filtered


def export_results_to_trajectories(
    results_dir: Path,
    output_dir: Path,
    min_game: Optional[int] = None,
    max_game: Optional[int] = None,
) -> List[Path]:
    """Export raw game results to rbbench.v1 trajectory format.

    Args:
        results_dir: Directory containing raw game result JSONs
        output_dir: Base output directory (trajectories saved to output_dir/trajectories/)
        min_game: If set, only process games with number >= min_game
        max_game: If set, only process games with number <= max_game

    Returns:
        List of exported trajectory file paths
    """
    from redblackbench.training import export_trajectory_file

    print(f"\n{'='*60}")
    print("PHASE 1: Exporting results to trajectory format")
    if min_game is not None or max_game is not None:
        min_str = str(min_game) if min_game is not None else "0"
        max_str = str(max_game) if max_game is not None else "∞"
        print(f"Limiting to games {min_str}-{max_str}")
    print(f"{'='*60}")

    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    failed = []

    # Find all result JSON files recursively
    result_files = list(results_dir.rglob("*.json"))
    result_files = filter_by_game_range(result_files, min_game, max_game)
    print(f"Found {len(result_files)} result files (after filtering)")

    for i, result_file in enumerate(result_files):
        # Skip config files or already-exported files
        if "config" in result_file.name or "_training" in result_file.name:
            continue

        print(f"\n[{i+1}/{len(result_files)}] {result_file.relative_to(results_dir)}")

        try:
            # Extract scenario from path (e.g., results/pandemic_vaccines/game_1.json)
            scenario_id = result_file.parent.name
            output_path = trajectories_dir / f"{result_file.stem}_training.json"

            # Skip if already exported
            if output_path.exists():
                print(f"  ⏭ Already exported, skipping")
                exported.append(output_path)
                continue

            export_trajectory_file(
                str(result_file),
                str(output_path),
                scenario_id=scenario_id,
            )
            exported.append(output_path)
            print(f"  ✓ Exported: {output_path.name}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append((result_file, str(e)))

    print(f"\n{len(exported)} trajectories exported, {len(failed)} failed")

    if failed:
        print("\nFailed files:")
        for path, error in failed[:5]:
            print(f"  - {path.name}: {error}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    return exported


def save_sft_checkpoint(
    output_path: Path,
    all_examples: List[Dict],
    num_trajectories_processed: int,
    total_trajectories: int,
    generator_model: str,
):
    """Save incremental checkpoint of SFT data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined = {
        "schema_version": "rbbench.sft.v1",
        "generator_model": generator_model,
        "generated_at": datetime.now().isoformat(),
        "num_examples": len(all_examples),
        "num_trajectories_processed": num_trajectories_processed,
        "total_trajectories": total_trajectories,
        "is_complete": num_trajectories_processed >= total_trajectories,
        "examples": all_examples,
    }

    with open(output_path, 'w') as f:
        json.dump(combined, f, indent=2)


def load_sft_checkpoint(output_path: Path) -> Tuple[List[Dict], int, set]:
    """Load existing checkpoint if available.

    Returns:
        Tuple of (examples_list, num_trajectories_processed, processed_trajectory_ids)
    """
    if not output_path.exists():
        return [], 0, set()

    try:
        with open(output_path, 'r') as f:
            data = json.load(f)

        examples = data.get("examples", [])
        processed = data.get("num_trajectories_processed", 0)
        is_complete = data.get("is_complete", False)

        # Extract trajectory IDs that have been processed
        processed_ids = set(ex.get("trajectory_id", "") for ex in examples)

        if is_complete:
            print(f"  Found complete checkpoint with {len(examples)} examples")
            return examples, processed, processed_ids

        print(f"  Resuming from checkpoint: {processed} trajectories, {len(examples)} examples")
        return examples, processed, processed_ids

    except Exception as e:
        print(f"  Could not load checkpoint: {e}")
        return [], 0, set()


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, requests_per_second: float = 2.0):
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


async def process_single_trajectory_sft(
    traj_path: Path,
    traj_index: int,
    total_trajectories: int,
    max_per_round: int,
    generator_model: str,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    verbose: bool = False,
) -> Tuple[str, List[Dict], Optional[str]]:
    """Process a single trajectory to generate SFT examples.

    Args:
        traj_path: Path to trajectory file
        traj_index: Index of this trajectory
        total_trajectories: Total number of trajectories
        max_per_round: Max examples per round
        generator_model: Model for generating ideal responses
        semaphore: Semaphore for limiting concurrent workers
        rate_limiter: Rate limiter for API calls
        verbose: Enable detailed logging

    Returns:
        Tuple of (trajectory_id, examples_list, error_message or None)
    """
    from redblackbench.training.schemas import TrainingTrajectory
    from redblackbench.training.sft_generator import SFTGenerator

    traj_id = traj_path.stem.replace("_training", "")

    async with semaphore:
        await rate_limiter.acquire()

        print(f"  [{traj_index + 1}/{total_trajectories}] Processing {traj_path.name}...")

        try:
            # Load trajectory
            trajectory = TrainingTrajectory.load(str(traj_path))

            # Create a fresh generator for this trajectory (thread safety)
            generator = SFTGenerator(model=generator_model)

            dataset = await generator.generate_from_trajectory(
                trajectory=trajectory,
                team="team_a",
                max_examples_per_round=max_per_round,
                verbose=verbose,
            )

            if dataset and dataset.examples:
                examples = [ex.to_dict() for ex in dataset.examples]
                print(f"  [{traj_index + 1}/{total_trajectories}] ✓ Generated {len(examples)} examples")
                return traj_id, examples, None
            else:
                print(f"  [{traj_index + 1}/{total_trajectories}] ✓ No examples generated")
                return traj_id, [], None

        except Exception as e:
            print(f"  [{traj_index + 1}/{total_trajectories}] ✗ Error: {e}")
            return traj_id, [], str(e)


async def generate_sft_from_trajectories(
    trajectory_paths: List[Path],
    output_path: Path,
    max_per_round: int = 3,
    generator_model: str = "moonshotai/kimi-k2-thinking",
    workers: int = 3,
    requests_per_second: float = 2.0,
    verbose: bool = False,
) -> int:
    """Generate SFT data from exported trajectories with parallel processing.

    Args:
        trajectory_paths: List of trajectory JSON paths
        output_path: Where to save the combined SFT dataset
        max_per_round: Max examples to generate per round
        generator_model: Model to use for generating ideal responses
        workers: Number of parallel workers
        requests_per_second: Rate limit for API calls
        verbose: Enable detailed logging for API calls

    Returns:
        Total number of examples generated
    """
    print(f"\n{'='*60}")
    print("PHASE 2: Generating SFT data with thinking model")
    print(f"Model: {generator_model}")
    print(f"Workers: {workers}")
    print(f"Rate limit: {requests_per_second} req/s")
    print(f"{'='*60}")

    # Load checkpoint
    all_examples, start_index, processed_ids = load_sft_checkpoint(output_path)

    # Filter out already processed trajectories
    pending_trajectories = [
        (i, traj_path)
        for i, traj_path in enumerate(trajectory_paths)
        if traj_path.stem.replace("_training", "") not in processed_ids
    ]

    if not pending_trajectories:
        print(f"All {len(trajectory_paths)} trajectories already processed!")
        return len(all_examples)

    print(f"Trajectories to process: {len(pending_trajectories)} ({len(processed_ids)} already done)")

    # Create semaphore and rate limiter
    semaphore = asyncio.Semaphore(workers)
    rate_limiter = RateLimiter(requests_per_second=requests_per_second)

    # Create tasks for all pending trajectories
    tasks = [
        process_single_trajectory_sft(
            traj_path=traj_path,
            traj_index=i,
            total_trajectories=len(trajectory_paths),
            max_per_round=max_per_round,
            generator_model=generator_model,
            semaphore=semaphore,
            rate_limiter=rate_limiter,
            verbose=verbose,
        )
        for i, traj_path in pending_trajectories
    ]

    print(f"\nStarting {len(tasks)} parallel processing tasks...\n")

    # Process tasks as they complete and save checkpoints
    completed = 0
    failed = 0

    for coro in asyncio.as_completed(tasks):
        traj_id, examples, error = await coro

        if error:
            failed += 1
        else:
            completed += 1
            all_examples.extend(examples)

        # Save checkpoint after each completion
        save_sft_checkpoint(
            output_path=output_path,
            all_examples=all_examples,
            num_trajectories_processed=len(processed_ids) + completed + failed,
            total_trajectories=len(trajectory_paths),
            generator_model=generator_model,
        )

    print(f"\n{'='*60}")
    print(f"SFT generation complete!")
    print(f"Successful: {completed}, Failed: {failed}")
    print(f"Output: {output_path}")
    print(f"Total examples: {len(all_examples)}")
    print(f"{'='*60}")

    return len(all_examples)


async def main():
    parser = argparse.ArgumentParser(
        description="Process existing batch results into SFT training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing raw game result JSONs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for trajectories and SFT data"
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip export phase (use existing trajectories)"
    )
    parser.add_argument(
        "--trajectories-dir",
        type=str,
        default=None,
        help="Directory containing existing trajectories (default: output-dir/trajectories). Useful with --skip-export when trajectories are in a different location."
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export trajectories, skip SFT generation"
    )
    parser.add_argument(
        "--max-per-round",
        type=int,
        default=10,
        help="Max SFT examples per round (default: 10, set high to get all agents)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="moonshotai/kimi-k2-thinking",
        help="Model for generating ideal responses"
    )
    parser.add_argument(
        "--min-game",
        type=int,
        default=None,
        help="Only process games starting from this number (inclusive)"
    )
    parser.add_argument(
        "--max-game",
        type=int,
        default=None,
        help="Only process games up to this number (inclusive)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers (default: 3)"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=2.0,
        help="Max API requests per second (default: 2.0)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for API calls"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine trajectories directory
    if args.trajectories_dir:
        trajectories_dir = Path(args.trajectories_dir)
    else:
        trajectories_dir = output_dir / "trajectories"

    # Phase 1: Export results to trajectories
    if args.skip_export:
        print("Skipping export phase, using existing trajectories...")
        print(f"Looking in: {trajectories_dir}")
        trajectory_paths = list(trajectories_dir.glob("*.json"))
        trajectory_paths = filter_by_game_range(trajectory_paths, args.min_game, args.max_game)
        if not trajectory_paths:
            print(f"Error: No trajectories found in {trajectories_dir}")
            return
        print(f"Found {len(trajectory_paths)} existing trajectories (after filtering)")
    else:
        trajectory_paths = export_results_to_trajectories(
            results_dir, output_dir, args.min_game, args.max_game
        )

    if not trajectory_paths:
        print("No trajectories to process")
        return

    # Phase 2: Generate SFT data
    if args.export_only:
        print("\nExport complete. Skipping SFT generation (--export-only)")
        return

    sft_output = output_dir / "sft_combined.json"
    await generate_sft_from_trajectories(
        trajectory_paths=trajectory_paths,
        output_path=sft_output,
        max_per_round=args.max_per_round,
        generator_model=args.model,
        workers=args.workers,
        requests_per_second=args.rate_limit,
        verbose=args.verbose,
    )

    print("\n✓ Done!")


if __name__ == "__main__":
    asyncio.run(main())
