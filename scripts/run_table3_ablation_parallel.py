#!/usr/bin/env python3
"""
Parallel execution script for Table 3 Ablation (Hardcoded Agents).

Runs all 6 scenarios with configurable games per scenario in parallel.
This provides a baseline comparison to Table 3 where reasoning quality
is eliminated (all agents hardcoded to cooperate).

Usage:
    python run_table3_ablation_parallel.py --max-parallel 6 --games-per-scenario 3
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import json

# All scenarios from Table 3
SCENARIOS = [
    "climate_cooperation",
    "agi_safety",
    "pandemic_vaccines",
    "election_crisis",
    "standards_coordination",
    "baseline",
]


async def run_single_scenario(
    scenario: str,
    games_per_scenario: int,
    output_dir: str,
) -> Tuple[str, bool, str]:
    """
    Run evaluation for a single scenario with hardcoded agents.

    Returns:
        (scenario, success, output)
    """
    script_path = Path(__file__).parent / "eval_table3_ablation_hardcoded.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--scenario", scenario,
        "--games", str(games_per_scenario),
        "--output-dir", output_dir,
    ]

    print(f"[{scenario}] Starting...")

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
            print(f"[{scenario}] ✓ Completed")
        else:
            print(f"[{scenario}] ✗ Failed")
            print(f"  Error: {stderr.decode()[:200]}")

        return (scenario, success, output)

    except Exception as e:
        print(f"[{scenario}] ✗ Exception: {e}")
        return (scenario, False, str(e))


async def run_parallel(
    max_parallel: int,
    games_per_scenario: int,
    output_dir: str,
) -> None:
    """
    Run all scenarios with limited parallelism.
    """
    total = len(SCENARIOS)

    print(f"\n{'='*80}")
    print(f"Table 3 Ablation: Hardcoded Agents (Parallel)")
    print(f"{'='*80}")
    print(f"Total scenarios: {total}")
    print(f"Games per scenario: {games_per_scenario}")
    print(f"Total games: {total * games_per_scenario}")
    print(f"Max parallel processes: {max_parallel}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Track results
    start_time = datetime.now()

    # Run with semaphore to limit parallelism
    semaphore = asyncio.Semaphore(max_parallel)

    async def run_with_semaphore(scenario: str):
        async with semaphore:
            return await run_single_scenario(
                scenario,
                games_per_scenario,
                output_dir,
            )

    # Launch all tasks
    tasks = [run_with_semaphore(scenario) for scenario in SCENARIOS]

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    duration_seconds = duration.total_seconds()

    successes = sum(1 for _, success, _ in results if success)
    failures = total - successes

    print(f"\n{'='*80}")
    print(f"Execution Complete")
    print(f"{'='*80}")
    print(f"Total time: {duration}")
    print(f"Successful scenarios: {successes}/{total}")
    print(f"Failed scenarios: {failures}/{total}")

    if failures > 0:
        print(f"\nFailed scenarios:")
        for scenario, success, output in results:
            if not success:
                print(f"  - {scenario}")

    # Aggregate results from individual scenario summaries
    aggregate_stats = {
        "total_games": 0,
        "successful_games": 0,
        "failed_games": 0,
        "total_efficiency": 0,
        "total_cooperation": 0,
        "scenario_results": {},
    }

    for scenario in SCENARIOS:
        scenario_summary_path = Path(output_dir) / "trajectories" / scenario / "../ablation_summary.json"
        # Actually, summaries are at the output_dir root
        # Let's check the individual scenario results
        try:
            # Each scenario run saves trajectories in output_dir/trajectories/{scenario}/
            # But the summary is at output_dir/ablation_summary.json
            # We need to look at the trajectory files instead
            trajectory_dir = Path(output_dir) / "trajectories" / scenario
            if trajectory_dir.exists():
                trajectory_files = list(trajectory_dir.glob("hardcoded_game_*.json"))

                scenario_efficiency = []
                scenario_cooperation = []

                for traj_file in trajectory_files:
                    try:
                        with open(traj_file, 'r') as f:
                            traj_data = json.load(f)

                        # Extract metrics from trajectory
                        # The trajectory format includes game results
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

                        scenario_efficiency.append(efficiency)
                        scenario_cooperation.append(cooperation)
                        aggregate_stats["successful_games"] += 1
                    except:
                        aggregate_stats["failed_games"] += 1

                if scenario_efficiency:
                    avg_eff = sum(scenario_efficiency) / len(scenario_efficiency)
                    avg_coop = sum(scenario_cooperation) / len(scenario_cooperation)

                    aggregate_stats["scenario_results"][scenario] = {
                        "games": len(scenario_efficiency),
                        "avg_efficiency": avg_eff,
                        "avg_cooperation": avg_coop,
                    }

                    aggregate_stats["total_efficiency"] += avg_eff * len(scenario_efficiency)
                    aggregate_stats["total_cooperation"] += avg_coop * len(scenario_cooperation)
                    aggregate_stats["total_games"] += len(scenario_efficiency)
        except Exception as e:
            print(f"Warning: Could not aggregate results for {scenario}: {e}")

    # Calculate overall averages
    if aggregate_stats["total_games"] > 0:
        aggregate_stats["avg_efficiency"] = aggregate_stats["total_efficiency"] / aggregate_stats["total_games"]
        aggregate_stats["avg_cooperation"] = aggregate_stats["total_cooperation"] / aggregate_stats["total_games"]

        print(f"\n{'='*80}")
        print(f"Aggregate Results Across All Scenarios")
        print(f"{'='*80}")
        print(f"Total games completed: {aggregate_stats['successful_games']}")
        print(f"Average efficiency: {aggregate_stats['avg_efficiency']:.1f}%")
        print(f"Average cooperation rate: {aggregate_stats['avg_cooperation']:.1f}%")

        if aggregate_stats["scenario_results"]:
            print(f"\nPer-Scenario Results:")
            for scenario, stats in aggregate_stats["scenario_results"].items():
                print(f"  {scenario}:")
                print(f"    Games: {stats['games']}")
                print(f"    Efficiency: {stats['avg_efficiency']:.1f}%")
                print(f"    Cooperation: {stats['avg_cooperation']:.1f}%")

    print(f"{'='*80}\n")

    # Save parallel run summary
    summary_path = Path(output_dir) / "parallel_run_summary.json"
    summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration_seconds,
        "total_scenarios": total,
        "games_per_scenario": games_per_scenario,
        "total_games": total * games_per_scenario,
        "max_parallel": max_parallel,
        "successes": successes,
        "failures": failures,
        "aggregate_stats": aggregate_stats,
        "results": [
            {
                "scenario": sc,
                "success": suc,
                "output_preview": out[:200] if out else "",
            }
            for sc, suc, out in results
        ],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Parallel run summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Table 3 ablation (hardcoded agents) in parallel"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=6,
        help="Maximum number of parallel processes (default: 6)",
    )
    parser.add_argument(
        "--games-per-scenario",
        type=int,
        default=3,
        help="Number of games per scenario (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results/table3_ablation_hardcoded",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Run async main
    asyncio.run(run_parallel(
        args.max_parallel,
        args.games_per_scenario,
        args.output_dir,
    ))


if __name__ == "__main__":
    main()
