#!/usr/bin/env python3
"""
Parallel execution script for Table 3 meta-alignment evaluation.
Runs all 30 combinations (5 compositions × 6 scenarios) with 3 games each in parallel.
"""
import argparse
import asyncio
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import json

from redblackbench import defaults

# Table 3 configurations
TRAINED_COUNTS = [0, 1, 2, 3, 5]  # Number of trained agents
SCENARIOS = [
    # Training scenarios
    "climate_cooperation",
    "agi_safety",
    "pandemic_vaccines",
    "election_crisis",
    "standards_coordination",
    # Held-out scenarios
    "baseline",
    "trade_war",
    "gpu_contention",
]
TOTAL_AGENTS = 5

def get_all_combinations() -> List[Tuple[int, str]]:
    """Get all (trained_count, scenario) combinations for Table 3."""
    return [(t, s) for t in TRAINED_COUNTS for s in SCENARIOS]

async def run_single_combination(
    trained_count: int,
    scenario: str,
    games_per_combo: int,
    output_dir: str,
    trained_model: str,
    untrained_model: str,
    trained_goal_preset: str = "none",
    resume: bool = False,
    provider: str = "vllm",
) -> Tuple[int, str, bool, str]:
    """
    Run evaluation for a single (trained_count, scenario) combination.

    Returns:
        (trained_count, scenario, success, output)
    """
    script_path = Path(__file__).parent / "eval_meta_alignment.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--trained-count", str(trained_count),
        "--scenario", scenario,
        "--games-per-combo", str(games_per_combo),
        "--output-dir", output_dir,
        "--trained-model", trained_model,
        "--untrained-model", untrained_model,
        "--trained-goal-preset", trained_goal_preset,
        "--provider", provider,
    ]
    if resume:
        cmd.append("--resume")

    print(f"[{trained_count}T+{TOTAL_AGENTS-trained_count}U, {scenario}] Starting...")

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
            print(f"[{trained_count}T+{TOTAL_AGENTS-trained_count}U, {scenario}] ✓ Completed")
        else:
            print(f"[{trained_count}T+{TOTAL_AGENTS-trained_count}U, {scenario}] ✗ Failed")
            print(f"  Error: {stderr.decode()[:200]}")

        return (trained_count, scenario, success, output)

    except Exception as e:
        print(f"[{trained_count}T+{TOTAL_AGENTS-trained_count}U, {scenario}] ✗ Exception: {e}")
        return (trained_count, scenario, False, str(e))

async def run_parallel(
    max_parallel: int,
    games_per_combo: int,
    output_dir: str,
    trained_model: str,
    untrained_model: str,
    trained_goal_preset: str = "none",
    resume: bool = False,
    provider: str = "vllm",
) -> None:
    """
    Run all Table 3 combinations with limited parallelism.
    """
    combinations = get_all_combinations()
    total = len(combinations)

    print(f"\n{'='*80}")
    print(f"Table 3 Parallel Evaluation")
    print(f"{'='*80}")
    print(f"Total combinations: {total} ({len(TRAINED_COUNTS)} compositions × {len(SCENARIOS)} scenarios)")
    print(f"Games per combination: {games_per_combo}")
    print(f"Total games: {total * games_per_combo}")
    print(f"Max parallel processes: {max_parallel}")
    print(f"Output directory: {output_dir}")
    print(f"Trained model: {trained_model}")
    print(f"Untrained model: {untrained_model}")
    print(f"Trained goal preset: {trained_goal_preset}")
    print(f"Resume mode: {resume}")
    print(f"{'='*80}\n")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Track results and token usage
    results = []
    start_time = datetime.now()

    # Track overall token usage across all games
    total_token_usage = {
        "total_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_latency_seconds": 0.0,
    }

    # Run with semaphore to limit parallelism
    semaphore = asyncio.Semaphore(max_parallel)

    async def run_with_semaphore(trained_count: int, scenario: str):
        async with semaphore:
            return await run_single_combination(
                trained_count,
                scenario,
                games_per_combo,
                output_dir,
                trained_model,
                untrained_model,
                trained_goal_preset,
                resume,
                provider,
            )

    # Launch all tasks
    tasks = [
        run_with_semaphore(trained_count, scenario)
        for trained_count, scenario in combinations
    ]

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    duration_seconds = duration.total_seconds()

    successes = sum(1 for _, _, success, _ in results if success)
    failures = total - successes

    # Load token usage from saved progress file
    # Each eval_meta_alignment run saves results to a progress file
    progress_files = list(Path(output_dir).glob("**/eval_meta_alignment_progress.json"))
    for progress_file in progress_files:
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            for game in progress_data.get("games", []):
                usage = game.get("token_usage", {})
                total_token_usage["total_calls"] += usage.get("total_calls", 0)
                total_token_usage["total_prompt_tokens"] += usage.get("total_prompt_tokens", 0)
                total_token_usage["total_completion_tokens"] += usage.get("total_completion_tokens", 0)
                total_token_usage["total_tokens"] += usage.get("total_tokens", 0)
                total_token_usage["total_latency_seconds"] += usage.get("total_latency_seconds", 0.0)
        except:
            pass  # Silently skip if loading fails

    # Calculate throughput metrics
    tokens_per_second = total_token_usage["total_tokens"] / duration_seconds if duration_seconds > 0 else 0
    prompt_tokens_per_second = total_token_usage["total_prompt_tokens"] / duration_seconds if duration_seconds > 0 else 0
    completion_tokens_per_second = total_token_usage["total_completion_tokens"] / duration_seconds if duration_seconds > 0 else 0

    print(f"\n{'='*80}")
    print(f"Execution Complete")
    print(f"{'='*80}")
    print(f"Total time: {duration}")
    print(f"Successful: {successes}/{total}")
    print(f"Failed: {failures}/{total}")

    # Print token usage and throughput metrics
    print(f"\n{'='*80}")
    print(f"Token Usage & GPU Throughput")
    print(f"{'='*80}")
    print(f"Total API calls: {total_token_usage['total_calls']:,}")
    print(f"Total prompt tokens: {total_token_usage['total_prompt_tokens']:,}")
    print(f"Total completion tokens: {total_token_usage['total_completion_tokens']:,}")
    print(f"Total tokens: {total_token_usage['total_tokens']:,}")
    print(f"\nThroughput Metrics:")
    print(f"  Overall tokens/s: {tokens_per_second:,.1f}")
    print(f"  Prompt tokens/s: {prompt_tokens_per_second:,.1f}")
    print(f"  Completion tokens/s: {completion_tokens_per_second:,.1f}")
    print(f"  Avg latency per call: {total_token_usage['total_latency_seconds'] / total_token_usage['total_calls']:.2f}s" if total_token_usage['total_calls'] > 0 else "  Avg latency per call: N/A")

    if failures > 0:
        print(f"\nFailed combinations:")
        for trained_count, scenario, success, output in results:
            if not success:
                print(f"  - {trained_count}T+{TOTAL_AGENTS-trained_count}U, {scenario}")

    print(f"{'='*80}\n")

    # Save summary
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
        "trained_model": trained_model,
        "untrained_model": untrained_model,
        "trained_goal_preset": trained_goal_preset,
        "token_usage": total_token_usage,
        "throughput": {
            "tokens_per_second": tokens_per_second,
            "prompt_tokens_per_second": prompt_tokens_per_second,
            "completion_tokens_per_second": completion_tokens_per_second,
            "avg_latency_per_call": total_token_usage["total_latency_seconds"] / total_token_usage["total_calls"] if total_token_usage["total_calls"] > 0 else 0,
        },
        "results": [
            {
                "trained_count": tc,
                "scenario": sc,
                "success": suc,
                "output_preview": out[:200] if out else "",
            }
            for tc, sc, suc, out in results
        ],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Run Table 3 meta-alignment evaluation in parallel"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=5,
        help="Maximum number of parallel processes (default: 5)",
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
        default="eval_results/table3_parallel",
        help="Output directory for results",
    )
    parser.add_argument(
        "--trained-model",
        type=str,
        default=defaults.SFT_MODEL,
        help="Trained model ID (LoRA adapter name on the vLLM server; default $YOAO_SFT_MODEL)",
    )
    parser.add_argument(
        "--untrained-model",
        type=str,
        default=defaults.BASE_MODEL,
        help="Untrained model ID (base model name on the vLLM server; default $YOAO_BASE_MODEL)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="vllm",
        choices=["vllm", "openrouter"],
        help="Provider for both trained and untrained agents (default: vllm)",
    )
    parser.add_argument(
        "--trained-goal-preset",
        type=str,
        default="none",
        choices=["none", "altruist", "exploitative", "cooperative", "selfish", "baseline"],
        help="Goal preset for trained agents (for prompt-only alignment, default: none)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skip completed games)",
    )

    args = parser.parse_args()

    # Run async main
    asyncio.run(run_parallel(
        args.max_parallel,
        args.games_per_combo,
        args.output_dir,
        args.trained_model,
        args.untrained_model,
        args.trained_goal_preset,
        args.resume,
        args.provider,
    ))

if __name__ == "__main__":
    main()
