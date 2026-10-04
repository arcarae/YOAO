#!/usr/bin/env python3
"""
Parallel execution script for Table 3 Ablation (Mixed Hardcoded + Untrained).

Runs all 24 combinations (4 compositions × 6 scenarios) with configurable games
per combination in parallel.

Compositions:
- 1 hardcoded + 4 untrained
- 2 hardcoded + 3 untrained
- 3 hardcoded + 2 untrained
- 4 hardcoded + 1 untrained

Usage:
    python run_table3_ablation_mixed_parallel.py --max-parallel 10 --games-per-combo 3
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import json

# Compositions to test
COMPOSITIONS = [
    (1, 4),  # 1 hardcoded + 4 untrained
    (2, 3),  # 2 hardcoded + 3 untrained
    (3, 2),  # 3 hardcoded + 2 untrained
    (4, 1),  # 4 hardcoded + 1 untrained
]

# All scenarios from Table 3
SCENARIOS = [
    "climate_cooperation",
    "agi_safety",
    "pandemic_vaccines",
    "election_crisis",
    "standards_coordination",
    "baseline",
]

from redblackbench import defaults

UNTRAINED_MODEL = defaults.BASE_MODEL
VLLM_BASE_URL = defaults.VLLM_URL


def get_all_combinations() -> List[Tuple[int, int, str]]:
    """Get all (num_hardcoded, num_untrained, scenario) combinations."""
    return [
        (num_h, num_u, scenario)
        for num_h, num_u in COMPOSITIONS
        for scenario in SCENARIOS
    ]


async def run_single_combination(
    num_hardcoded: int,
    num_untrained: int,
    scenario: str,
    games_per_combo: int,
    output_dir: str,
    untrained_model: str,
    vllm_base_url: str,
) -> Tuple[int, int, str, bool, str]:
    """
    Run evaluation for a single (num_hardcoded, num_untrained, scenario) combination.

    Returns:
        (num_hardcoded, num_untrained, scenario, success, output)
    """
    script_path = Path(__file__).parent / "eval_table3_ablation_mixed.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--hardcoded-count", str(num_hardcoded),
        "--scenario", scenario,
        "--games", str(games_per_combo),
        "--output-dir", output_dir,
        "--untrained-model", untrained_model,
        "--vllm-base-url", vllm_base_url,
    ]

    print(f"[{num_hardcoded}H+{num_untrained}U, {scenario}] Starting...")

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
            print(f"[{num_hardcoded}H+{num_untrained}U, {scenario}] ✓ Completed")
        else:
            print(f"[{num_hardcoded}H+{num_untrained}U, {scenario}] ✗ Failed")
            print(f"  Error: {stderr.decode()[:200]}")

        return (num_hardcoded, num_untrained, scenario, success, output)

    except Exception as e:
        print(f"[{num_hardcoded}H+{num_untrained}U, {scenario}] ✗ Exception: {e}")
        return (num_hardcoded, num_untrained, scenario, False, str(e))


async def run_parallel(
    max_parallel: int,
    games_per_combo: int,
    output_dir: str,
    untrained_model: str,
    vllm_base_url: str,
) -> None:
    """
    Run all combinations with limited parallelism.
    """
    combinations = get_all_combinations()
    total = len(combinations)

    print(f"\n{'='*80}")
    print(f"Table 3 Ablation: Mixed Hardcoded + Untrained (Parallel)")
    print(f"{'='*80}")
    print(f"Compositions: {len(COMPOSITIONS)}")
    for num_h, num_u in COMPOSITIONS:
        print(f"  - {num_h} hardcoded + {num_u} untrained")
    print(f"Total combinations: {total} ({len(COMPOSITIONS)} compositions × {len(SCENARIOS)} scenarios)")
    print(f"Games per combination: {games_per_combo}")
    print(f"Total games: {total * games_per_combo}")
    print(f"Max parallel processes: {max_parallel}")
    print(f"Output directory: {output_dir}")
    print(f"Untrained model: {untrained_model}")
    print(f"{'='*80}\n")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Track results
    start_time = datetime.now()

    # Run with semaphore to limit parallelism
    semaphore = asyncio.Semaphore(max_parallel)

    async def run_with_semaphore(num_h: int, num_u: int, scenario: str):
        async with semaphore:
            return await run_single_combination(
                num_h,
                num_u,
                scenario,
                games_per_combo,
                output_dir,
                untrained_model,
                vllm_base_url,
            )

    # Launch all tasks
    tasks = [
        run_with_semaphore(num_h, num_u, scenario)
        for num_h, num_u, scenario in combinations
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
        for num_h, num_u, scenario, success, output in results:
            if not success:
                print(f"  - {num_h}H+{num_u}U, {scenario}")

    # Aggregate results from individual runs
    aggregate_stats = {
        "total_games": 0,
        "successful_games": 0,
        "failed_games": 0,
        "composition_results": {},
    }

    # Load all trajectory files to compute statistics
    try:
        for num_h, num_u in COMPOSITIONS:
            comp_key = f"{num_h}h_{num_u}u"
            comp_stats = {
                "num_hardcoded": num_h,
                "num_untrained": num_u,
                "games": 0,
                "efficiencies": [],
                "cooperation_rates": [],
            }

            for scenario in SCENARIOS:
                trajectory_dir = Path(output_dir) / "trajectories" / scenario
                if trajectory_dir.exists():
                    pattern = f"mixed_{num_h}h_{num_u}u_game_*.json"
                    trajectory_files = list(trajectory_dir.glob(pattern))

                    for traj_file in trajectory_files:
                        try:
                            with open(traj_file, 'r') as f:
                                traj_data = json.load(f)

                            # Extract metrics
                            team_a_score = traj_data.get("team_a_final_score", 0)
                            team_b_score = traj_data.get("team_b_final_score", 0)
                            total_score = team_a_score + team_b_score
                            efficiency = (total_score / 150) * 100

                            # Count BLACK choices
                            black_choices = sum(
                                1 for ts in traj_data.get("timesteps", [])
                                if ts.get("type") == "ROUND_END" and
                                   ts.get("data", {}).get("team_a_choice") == "BLACK"
                            )
                            cooperation = (black_choices / 10) * 100

                            comp_stats["efficiencies"].append(efficiency)
                            comp_stats["cooperation_rates"].append(cooperation)
                            comp_stats["games"] += 1
                            aggregate_stats["successful_games"] += 1
                        except:
                            aggregate_stats["failed_games"] += 1

            if comp_stats["games"] > 0:
                comp_stats["avg_efficiency"] = sum(comp_stats["efficiencies"]) / comp_stats["games"]
                comp_stats["avg_cooperation"] = sum(comp_stats["cooperation_rates"]) / comp_stats["games"]
                del comp_stats["efficiencies"]  # Remove raw data
                del comp_stats["cooperation_rates"]

            aggregate_stats["composition_results"][comp_key] = comp_stats
            aggregate_stats["total_games"] += comp_stats["games"]

    except Exception as e:
        print(f"\nWarning: Could not aggregate all results: {e}")

    # Print composition-level results
    if aggregate_stats["composition_results"]:
        print(f"\n{'='*80}")
        print(f"Aggregate Results by Composition")
        print(f"{'='*80}")
        print(f"Total games completed: {aggregate_stats['successful_games']}")

        for comp_key, stats in sorted(aggregate_stats["composition_results"].items()):
            if stats["games"] > 0:
                print(f"\n{stats['num_hardcoded']}H + {stats['num_untrained']}U:")
                print(f"  Games: {stats['games']}")
                print(f"  Avg efficiency: {stats['avg_efficiency']:.1f}%")
                print(f"  Avg cooperation: {stats['avg_cooperation']:.1f}%")

    print(f"\n{'='*80}\n")

    # Save parallel run summary
    summary_path = Path(output_dir) / "parallel_run_summary.json"
    summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration_seconds,
        "total_combinations": total,
        "games_per_combo": games_per_combo,
        "total_games": total * games_per_combo,
        "max_parallel": max_parallel,
        "successes": successes,
        "failures": failures,
        "untrained_model": untrained_model,
        "compositions": [(h, u) for h, u in COMPOSITIONS],
        "aggregate_stats": aggregate_stats,
        "results": [
            {
                "num_hardcoded": num_h,
                "num_untrained": num_u,
                "scenario": sc,
                "success": suc,
                "output_preview": out[:200] if out else "",
            }
            for num_h, num_u, sc, suc, out in results
        ],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Parallel run summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Table 3 ablation (mixed hardcoded + untrained) in parallel"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=10,
        help="Maximum number of parallel processes (default: 10)",
    )
    parser.add_argument(
        "--games-per-combo",
        type=int,
        default=3,
        help="Number of games per combination (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results/table3_ablation_mixed",
        help="Output directory for results",
    )
    parser.add_argument(
        "--untrained-model",
        type=str,
        default=UNTRAINED_MODEL,
        help=f"Untrained model path (default: {UNTRAINED_MODEL})",
    )
    parser.add_argument(
        "--vllm-base-url",
        type=str,
        default=VLLM_BASE_URL,
        help=f"vLLM server base URL (default: {VLLM_BASE_URL})",
    )

    args = parser.parse_args()

    # Run async main
    asyncio.run(run_parallel(
        args.max_parallel,
        args.games_per_combo,
        args.output_dir,
        args.untrained_model,
        args.vllm_base_url,
    ))


if __name__ == "__main__":
    main()
