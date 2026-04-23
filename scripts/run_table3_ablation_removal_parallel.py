#!/usr/bin/env python3
"""
Parallel runner for Table 3 Ablation: Trained Agent Removal

Runs multiple removal experiments in parallel with controlled concurrency.

Usage:
    python run_table3_ablation_removal_parallel.py --max-parallel 10 --games-per-combo 3
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configurations
COMPOSITIONS = [
    (1, 4),  # 1 trained + 4 untrained
    (2, 3),  # 2 trained + 3 untrained
    (3, 2),  # 3 trained + 2 untrained
]

SCENARIOS = [
    "climate_cooperation",
    "agi_safety",
    "pandemic_vaccines",
    "election_crisis",
    "standards_coordination",
    "baseline",
]

TRAINED_MODEL = "gpt-4o-mini"
UNTRAINED_MODEL = "/workspace/models/Qwen3-14B"
VLLM_BASE_URL = "http://localhost:8000/v1"
REMOVAL_ROUND = 3  # Remove agents after round 3


def get_all_combinations():
    """Get all (num_trained, num_untrained, scenario) combinations."""
    return [
        (num_t, num_u, scenario)
        for num_t, num_u in COMPOSITIONS
        for scenario in SCENARIOS
    ]


async def run_single_combination(
    num_trained: int,
    num_untrained: int,
    scenario: str,
    removal_round: int,
    games_per_combo: int,
    output_dir: str,
    trained_model: str,
    untrained_model: str,
    vllm_base_url: str,
) -> Tuple[int, int, str, bool, str]:
    """
    Run evaluation for a single (num_trained, num_untrained, scenario) combination.

    Returns:
        (num_trained, num_untrained, scenario, success, output)
    """
    script_path = Path(__file__).parent / "eval_table3_ablation_removal.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--trained-count", str(num_trained),
        "--scenario", scenario,
        "--removal-round", str(removal_round),
        "--games", str(games_per_combo),
        "--output-dir", output_dir,
        "--trained-model", trained_model,
        "--untrained-model", untrained_model,
        "--vllm-base-url", vllm_base_url,
    ]

    print(f"[{num_trained}T+{num_untrained}U, {scenario}] Starting...")

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        success = process.returncode == 0
        output = stdout.decode() if stdout else stderr.decode()

        if success:
            print(f"[{num_trained}T+{num_untrained}U, {scenario}] ✓ Completed")
        else:
            print(f"[{num_trained}T+{num_untrained}U, {scenario}] ✗ Failed")
            print(f"  Error: {stderr.decode()[:200]}")

        return (num_trained, num_untrained, scenario, success, output)

    except Exception as e:
        print(f"[{num_trained}T+{num_untrained}U, {scenario}] ✗ Exception: {e}")
        return (num_trained, num_untrained, scenario, False, str(e))


async def run_parallel(
    max_parallel: int,
    games_per_combo: int,
    removal_round: int,
    output_dir: str,
    trained_model: str,
    untrained_model: str,
    vllm_base_url: str,
) -> None:
    """
    Run all combinations with limited parallelism.
    """
    combinations = get_all_combinations()
    total = len(combinations)

    print(f"\n{'='*80}")
    print(f"Table 3 Ablation: Trained Agent Removal (Parallel)")
    print(f"{'='*80}")
    print(f"Compositions: {len(COMPOSITIONS)}")
    for num_t, num_u in COMPOSITIONS:
        print(f"  - {num_t} trained + {num_u} untrained (remove after round {removal_round})")
    print(f"Total combinations: {total} ({len(COMPOSITIONS)} compositions × {len(SCENARIOS)} scenarios)")
    print(f"Games per combination: {games_per_combo}")
    print(f"Total games: {total * games_per_combo}")
    print(f"Max parallel processes: {max_parallel}")
    print(f"Output directory: {output_dir}")
    print(f"Trained model: {trained_model}")
    print(f"Untrained model: {untrained_model}")
    print(f"Removal round: {removal_round}")
    print(f"{'='*80}\n")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Track results
    start_time = datetime.now()

    # Run with semaphore to limit parallelism
    semaphore = asyncio.Semaphore(max_parallel)

    async def run_with_semaphore(num_t: int, num_u: int, scenario: str):
        async with semaphore:
            return await run_single_combination(
                num_t,
                num_u,
                scenario,
                removal_round,
                games_per_combo,
                output_dir,
                trained_model,
                untrained_model,
                vllm_base_url,
            )

    # Launch all tasks
    tasks = [
        run_with_semaphore(num_t, num_u, scenario)
        for num_t, num_u, scenario in combinations
    ]

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    duration_seconds = duration.total_seconds()

    successes = sum(1 for _, _, _, success, _ in results if success)
    failures = total - successes

    print(f"\n{'='*80}")
    print(f"Execution Complete")
    print(f"{'='*80}")
    print(f"Total time: {duration}")
    print(f"Successful: {successes}/{total}")
    print(f"Failed: {failures}/{total}")

    if failures > 0:
        print(f"\nFailed combinations:")
        for num_t, num_u, scenario, success, output in results:
            if not success:
                print(f"  - {num_t}T+{num_u}U, {scenario}")

    # Aggregate results from individual summary files
    aggregate_stats = {
        "total_games": 0,
        "successful_games": 0,
        "failed_games": 0,
        "composition_results": {},
    }

    for num_t, num_u, scenario, success, output in results:
        # Try to load individual summary
        summary_file = Path(output_dir) / "removal_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)

                aggregate_stats["total_games"] += data.get("total_games", 0)
                aggregate_stats["successful_games"] += data.get("successful_games", 0)
                aggregate_stats["failed_games"] += data.get("failed_games", 0)

                # Aggregate by composition
                for result in data.get("results", []):
                    if result.get("success"):
                        comp_key = f"{result['num_trained']}t_{result['num_untrained']}u"
                        if comp_key not in aggregate_stats["composition_results"]:
                            aggregate_stats["composition_results"][comp_key] = {
                                "num_trained": result["num_trained"],
                                "num_untrained": result["num_untrained"],
                                "games": 0,
                                "avg_coop_before": 0.0,
                                "avg_coop_after": 0.0,
                                "avg_change": 0.0,
                            }

                        comp = aggregate_stats["composition_results"][comp_key]
                        n = comp["games"]
                        comp["games"] += 1
                        # Running average
                        comp["avg_coop_before"] = (comp["avg_coop_before"] * n + result["cooperation_before_removal"]) / comp["games"]
                        comp["avg_coop_after"] = (comp["avg_coop_after"] * n + result["cooperation_after_removal"]) / comp["games"]
                        comp["avg_change"] = (comp["avg_change"] * n + result["cooperation_change"]) / comp["games"]
            except Exception as e:
                print(f"Warning: Could not parse summary file: {e}")

    print(f"\n{'='*80}")
    print(f"Aggregate Results by Composition")
    print(f"{'='*80}")
    print(f"Total games completed: {aggregate_stats['total_games']}")

    for comp_key in sorted(aggregate_stats["composition_results"].keys()):
        comp = aggregate_stats["composition_results"][comp_key]
        num_t = comp["num_trained"]
        num_u = comp["num_untrained"]

        print(f"\n{num_t}T + {num_u}U:")
        print(f"  Games: {comp['games']}")
        print(f"  Avg cooperation (WITH trained):    {comp['avg_coop_before'] * 100:.1f}%")
        print(f"  Avg cooperation (WITHOUT trained): {comp['avg_coop_after'] * 100:.1f}%")
        print(f"  Avg change:                         {comp['avg_change'] * 100:+.1f}%")

    print(f"\n{'='*80}\n")

    # Save parallel run summary
    parallel_summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration_seconds,
        "total_combinations": total,
        "games_per_combo": games_per_combo,
        "total_games": total * games_per_combo,
        "max_parallel": max_parallel,
        "successes": successes,
        "failures": failures,
        "removal_round": removal_round,
        "trained_model": trained_model,
        "untrained_model": untrained_model,
        "compositions": COMPOSITIONS,
        "aggregate_stats": aggregate_stats,
        "results": [
            {
                "num_trained": num_t,
                "num_untrained": num_u,
                "scenario": scenario,
                "success": success,
                "output_preview": output[:200] if output else "",
            }
            for num_t, num_u, scenario, success, output in results
        ],
    }

    summary_path = Path(output_dir) / "parallel_run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(parallel_summary, f, indent=2)

    print(f"Parallel run summary saved to: {summary_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel runner for Table 3 Ablation: Trained Agent Removal"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=10,
        help="Maximum parallel processes (default: 10)",
    )
    parser.add_argument(
        "--games-per-combo",
        type=int,
        default=3,
        help="Games per combination (default: 3)",
    )
    parser.add_argument(
        "--removal-round",
        type=int,
        default=REMOVAL_ROUND,
        help="Round after which to remove trained agents (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results/table3_ablation_removal",
        help="Output directory",
    )
    parser.add_argument(
        "--trained-model",
        type=str,
        default=TRAINED_MODEL,
        help="Model for trained agents",
    )
    parser.add_argument(
        "--untrained-model",
        type=str,
        default=UNTRAINED_MODEL,
        help="Model for untrained agents",
    )
    parser.add_argument(
        "--vllm-base-url",
        type=str,
        default=VLLM_BASE_URL,
        help="vLLM server base URL",
    )

    args = parser.parse_args()

    asyncio.run(run_parallel(
        max_parallel=args.max_parallel,
        games_per_combo=args.games_per_combo,
        removal_round=args.removal_round,
        output_dir=args.output_dir,
        trained_model=args.trained_model,
        untrained_model=args.untrained_model,
        vllm_base_url=args.vllm_base_url,
    ))


if __name__ == "__main__":
    main()
