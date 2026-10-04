#!/usr/bin/env python3
"""Evaluation Script: Scale Effect (Table B)

Tests whether the 20% seed threshold holds at larger team sizes (N=5, N=10, N=15).
Uses vLLM server with base Qwen3-14B (untrained) and redblackbench LoRA (trained/SFT).

Setup: Mixed Team A (seed% trained + rest untrained) vs always_defect Team B
       Across team sizes N=5, N=10, N=15
       5 runs per condition for mean ± std

Key question: Does the 20% seed threshold hold at N=10, N=15?

Usage:
    # Run full evaluation (all team sizes, all seed ratios, 5 runs each)
    python scripts/eval_scale_effect.py

    # Run specific team size
    python scripts/eval_scale_effect.py --team-size 10

    # Run specific seed ratio
    python scripts/eval_scale_effect.py --seed-pct 20

    # Quick test (1 run per condition)
    python scripts/eval_scale_effect.py --runs-per-condition 1

    # Resume interrupted evaluation
    python scripts/eval_scale_effect.py --resume
"""

import argparse
import asyncio
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from redblackbench.game.coordinator import GameCoordinator
from redblackbench.game.config import GameConfig
from redblackbench.teams.team import Team
from redblackbench.scenarios import get_scenario
from redblackbench.trajectory import TrajectoryCollector
from redblackbench.agents.llm_agent import LLMAgent
from redblackbench.cli import create_provider
from redblackbench.strategies import create_scripted_team

from redblackbench import defaults

# ── vLLM Server Config (override with YOAO_VLLM_URL / YOAO_SFT_MODEL / YOAO_BASE_MODEL) ──
VLLM_BASE_URL = defaults.VLLM_URL
TRAINED_MODEL = defaults.SFT_MODEL      # cooperative LoRA adapter name on the server
UNTRAINED_MODEL = defaults.BASE_MODEL   # unmodified base model name on the server

# ── Experiment Design ────────────────────────────────────────────────
TEAM_SIZES = [5, 10, 15]
SEED_PERCENTAGES = [0, 20, 40, 60, 80, 100]  # % of team that is trained
RUNS_PER_CONDITION = 5

# Use held-out scenarios for generalization testing (matches paper Table 3 held-out col)
EVAL_SCENARIOS = [
    "baseline",
    "trade_war",
    "gpu_contention",
]

OPPONENT_STRATEGY = "always_defect"

# Agent name pool (enough for N=15)
AGENT_NAMES = [
    "Dr. Sarah Chen", "Marcus Webb", "Dr. Priya Sharma",
    "James O'Connor", "Dr. Elena Vasquez", "Dr. Amir Hassan",
    "Lisa Park", "Dr. Thomas Müller", "Fatima Al-Rashid",
    "Dr. Kenji Tanaka", "Rachel Torres", "Dr. Oluwaseun Adeyemi",
    "Viktor Petrov", "Dr. Maya Gupta", "Samuel Okafor",
]


def compute_team_composition(team_size: int, seed_pct: int) -> Tuple[int, int]:
    """Compute (num_trained, num_untrained) from team size and seed percentage."""
    num_trained = round(team_size * seed_pct / 100)
    num_untrained = team_size - num_trained
    return num_trained, num_untrained


def calculate_cooperation_rate(history: List[str]) -> float:
    """Calculate Team A's cooperation rate from move history."""
    if not history:
        return 0.0
    return sum(1 for move in history if move == "A") / len(history)


async def run_single_game(
    team_size: int,
    num_trained: int,
    num_untrained: int,
    scenario_id: str,
    run_index: int,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Run a single game with mixed Team A vs scripted Team B."""
    try:
        # Handle baseline vs real scenarios
        if scenario_id == "baseline":
            scenario = None
            prompt_template = None
        else:
            scenario = get_scenario(scenario_id)
            if not scenario:
                print(f"    Unknown scenario: {scenario_id}")
                return None
            prompt_template = scenario.to_prompt_template()

        config = GameConfig(
            num_rounds=10,
            multipliers={5: 3, 8: 5, 10: 10},
        )

        # Create mixed Team A
        agents_a = []

        # Trained agents (SFT seed)
        for i in range(num_trained):
            provider = create_provider({
                "type": "vllm",
                "model": TRAINED_MODEL,
                "base_url": VLLM_BASE_URL,
                "temperature": 0.7,
                "max_tokens": 512,
            })
            agent = LLMAgent(
                agent_id=AGENT_NAMES[i % len(AGENT_NAMES)],
                team_name="Team A",
                provider=provider,
                prompt_template=prompt_template,
                enable_thinking=False,
            )
            agents_a.append(agent)

        # Untrained agents (base model)
        for i in range(num_untrained):
            idx = num_trained + i
            provider = create_provider({
                "type": "vllm",
                "model": UNTRAINED_MODEL,
                "base_url": VLLM_BASE_URL,
                "temperature": 0.7,
                "max_tokens": 512,
            })
            agent = LLMAgent(
                agent_id=AGENT_NAMES[idx % len(AGENT_NAMES)],
                team_name="Team A",
                provider=provider,
                prompt_template=prompt_template,
                enable_thinking=False,
            )
            agents_a.append(agent)

        team_a = Team(name="Team A", agents=agents_a)

        # Scripted opponent (always defect)
        team_b = create_scripted_team(
            strategy_id=OPPONENT_STRATEGY,
            team_name=f"Team B ({OPPONENT_STRATEGY})",
        )

        # Trajectory collection
        seed_pct = round(num_trained / team_size * 100)
        results_dir = output_dir / "trajectories" / f"N{team_size}" / scenario_id
        results_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = results_dir / f"seed{seed_pct}pct_run{run_index}.json"

        collector = TrajectoryCollector()

        team_a_desc = f"mixed:{num_trained}xSFT+{num_untrained}xBase"
        coordinator = GameCoordinator(
            team_a=team_a,
            team_b=team_b,
            config=config,
            trajectory_collector=collector,
            trajectory_save_path=str(trajectory_path),
            team_a_model=team_a_desc,
            team_b_model=f"scripted:{OPPONENT_STRATEGY}",
        )

        # Run with timeout (longer for larger teams)
        timeout = 3600 + (team_size - 5) * 600  # 60min base + 10min per extra agent
        game_state = await asyncio.wait_for(coordinator.play_game(), timeout=timeout)

        # Extract results from game state
        # Cooperation = BLACK; in history, team_a_choice is Choice.BLACK or Choice.RED
        from redblackbench.game.scoring import Choice
        total_rounds = len(game_state.history)
        black_count = sum(
            1 for r in game_state.history if r.team_a_choice == Choice.BLACK
        )
        cooperation_rate = black_count / total_rounds if total_rounds > 0 else 0.0

        return {
            "team_size": team_size,
            "seed_pct": seed_pct,
            "num_trained": num_trained,
            "num_untrained": num_untrained,
            "scenario": scenario_id,
            "run_index": run_index,
            "team_a_score": game_state.team_a_total,
            "team_b_score": game_state.team_b_total,
            "cooperation_rate": cooperation_rate,
            "trajectory_path": str(trajectory_path),
            "timestamp": datetime.now().isoformat(),
        }

    except asyncio.TimeoutError:
        print(f"    TIMEOUT after {timeout}s")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_progress(output_dir: Path) -> Tuple[Dict[str, Any], set]:
    """Load existing progress from checkpoint."""
    progress_path = output_dir / "progress.json"
    completed = set()
    results = {"games": [], "config": {}}

    if progress_path.exists():
        try:
            with open(progress_path, 'r') as f:
                data = json.load(f)
            results = data
            for game in data.get("games", []):
                key = (game["team_size"], game["seed_pct"], game["scenario"], game["run_index"])
                completed.add(key)
            print(f"  Loaded progress: {len(completed)} games completed")
        except Exception as e:
            print(f"  Could not load progress: {e}")

    return results, completed


def save_progress(output_dir: Path, results: Dict[str, Any]):
    """Save progress checkpoint."""
    progress_path = output_dir / "progress.json"
    results["last_updated"] = datetime.now().isoformat()
    with open(progress_path, 'w') as f:
        json.dump(results, f, indent=2)


def generate_summary(results: Dict[str, Any]) -> str:
    """Generate the Table B markdown summary with mean ± std."""
    games = results.get("games", [])
    if not games:
        return "No results yet."

    from collections import defaultdict

    # Group: (team_size, seed_pct, scenario) -> [cooperation_rates across runs]
    by_condition = defaultdict(list)
    for game in games:
        key = (game["team_size"], game["seed_pct"], game["scenario"])
        by_condition[key].append(game["cooperation_rate"] * 100)

    # Collect all seed_pcts and team_sizes present
    all_sizes = sorted(set(g["team_size"] for g in games))
    all_pcts = sorted(set(g["seed_pct"] for g in games))
    all_scenarios = sorted(set(g["scenario"] for g in games))

    lines = []
    lines.append("# Table B — Scale Effect (Qwen3-14B, mean ± std across runs)")
    lines.append("")
    lines.append(f"vLLM server: `{VLLM_BASE_URL}`")
    lines.append(f"Trained model: `{TRAINED_MODEL}` | Untrained model: `{UNTRAINED_MODEL}`")
    lines.append(f"Opponent: `{OPPONENT_STRATEGY}` | Scenarios: {', '.join(EVAL_SCENARIOS)}")
    lines.append(f"Runs per condition: {RUNS_PER_CONDITION}")
    lines.append("")

    # Table header
    header = "| Seed % |"
    sep = "|--------|"
    for n in all_sizes:
        header += f" N={n} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    # Main table: for each (size, pct), compute per-scenario mean across runs,
    # then report mean ± std of those per-scenario means
    for pct in all_pcts:
        row = f"| {pct}% |"
        for n in all_sizes:
            scenario_means = []
            for sc in all_scenarios:
                rates = by_condition.get((n, pct, sc), [])
                if rates:
                    scenario_means.append(sum(rates) / len(rates))
            if scenario_means:
                mean = sum(scenario_means) / len(scenario_means)
                if len(scenario_means) > 1:
                    std = math.sqrt(sum((r - mean) ** 2 for r in scenario_means) / (len(scenario_means) - 1))
                    row += f" {mean:.0f}% ± {std:.0f} |"
                else:
                    row += f" {mean:.0f}% (n=1sc) |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")

    # Per-scenario breakdown
    lines.append("## Per-Scenario Breakdown")
    lines.append("")

    for scenario in EVAL_SCENARIOS:
        lines.append(f"### {scenario}")
        lines.append("")

        header = "| Seed % |"
        sep = "|--------|"
        for n in all_sizes:
            header += f" N={n} |"
            sep += "------|"
        lines.append(header)
        lines.append(sep)

        for pct in all_pcts:
            row = f"| {pct}% |"
            for n in all_sizes:
                rates = by_condition.get((n, pct, scenario), [])
                if rates:
                    mean = sum(rates) / len(rates)
                    n_runs = len(rates)
                    if n_runs > 1:
                        std = math.sqrt(sum((r - mean) ** 2 for r in rates) / (n_runs - 1))
                        row += f" {mean:.0f}% ± {std:.0f} (n={n_runs}) |"
                    else:
                        row += f" {mean:.0f}% (n=1) |"
                else:
                    row += " — |"
            lines.append(row)

        lines.append("")

    # Total games summary
    total = len(games)
    expected = len(all_sizes) * len(all_pcts) * len(EVAL_SCENARIOS) * RUNS_PER_CONDITION
    lines.append(f"---")
    lines.append(f"Completed: {total}/{expected} games")

    return "\n".join(lines)


async def main():
    global VLLM_BASE_URL, RUNS_PER_CONDITION
    parser = argparse.ArgumentParser(
        description="Evaluate scale effect on alignment propagation (Table B)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--team-size", type=int, choices=TEAM_SIZES,
                        help="Run specific team size only")
    parser.add_argument("--seed-pct", type=int, choices=SEED_PERCENTAGES,
                        help="Run specific seed percentage only")
    parser.add_argument("--scenario", type=str, choices=EVAL_SCENARIOS,
                        help="Run specific scenario only")
    parser.add_argument("--runs-per-condition", type=int, default=RUNS_PER_CONDITION,
                        help=f"Runs per condition (default: {RUNS_PER_CONDITION})")
    parser.add_argument("--output-dir", type=str, default="results/eval_scale_effect",
                        help="Output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--vllm-url", type=str, default=VLLM_BASE_URL,
                        help=f"vLLM server URL (default: {VLLM_BASE_URL})")
    args = parser.parse_args()

    VLLM_BASE_URL = args.vllm_url
    RUNS_PER_CONDITION = args.runs_per_condition

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine conditions to run
    team_sizes = [args.team_size] if args.team_size else TEAM_SIZES
    seed_pcts = [args.seed_pct] if args.seed_pct is not None else SEED_PERCENTAGES
    scenarios = [args.scenario] if args.scenario else EVAL_SCENARIOS

    # Load progress if resuming
    if args.resume:
        results, completed = load_progress(output_dir)
    else:
        results = {
            "games": [],
            "config": {
                "vllm_url": VLLM_BASE_URL,
                "trained_model": TRAINED_MODEL,
                "untrained_model": UNTRAINED_MODEL,
                "team_sizes": team_sizes,
                "seed_pcts": seed_pcts,
                "scenarios": scenarios,
                "runs_per_condition": args.runs_per_condition,
                "opponent": OPPONENT_STRATEGY,
            }
        }
        completed = set()

    # Calculate total
    total_games = len(team_sizes) * len(seed_pcts) * len(scenarios) * args.runs_per_condition
    done = len(completed)

    print(f"\n{'='*70}")
    print("TABLE B — SCALE EFFECT EVALUATION")
    print(f"{'='*70}")
    print(f"vLLM server:     {VLLM_BASE_URL}")
    print(f"Trained model:   {TRAINED_MODEL}")
    print(f"Untrained model: {UNTRAINED_MODEL}")
    print(f"Team sizes:      {team_sizes}")
    print(f"Seed %:          {seed_pcts}")
    print(f"Scenarios:        {scenarios}")
    print(f"Runs/condition:  {args.runs_per_condition}")
    print(f"Total games:     {total_games} ({done} already done)")
    print(f"Output:          {output_dir}")
    print(f"{'='*70}\n")

    current = 0
    for N in team_sizes:
        for pct in seed_pcts:
            num_trained, num_untrained = compute_team_composition(N, pct)
            for scenario in scenarios:
                for run_idx in range(args.runs_per_condition):
                    current += 1
                    key = (N, pct, scenario, run_idx)

                    if key in completed:
                        print(f"[{current}/{total_games}] SKIP N={N} seed={pct}% {scenario} run={run_idx}")
                        continue

                    label = f"N={N} | {num_trained}T+{num_untrained}U ({pct}%) | {scenario} | run {run_idx+1}/{args.runs_per_condition}"
                    print(f"[{current}/{total_games}] {label}")

                    result = await run_single_game(
                        team_size=N,
                        num_trained=num_trained,
                        num_untrained=num_untrained,
                        scenario_id=scenario,
                        run_index=run_idx,
                        output_dir=output_dir,
                    )

                    if result:
                        results["games"].append(result)
                        completed.add(key)
                        coop = result["cooperation_rate"] * 100
                        score_a = result["team_a_score"]
                        score_b = result["team_b_score"]
                        print(f"    -> Coop: {coop:.0f}% | Score: {score_a} vs {score_b}")
                    else:
                        print(f"    -> FAILED")

                    # Checkpoint after every game
                    save_progress(output_dir, results)

    # Final summary
    print(f"\n{'='*70}")
    print("RESULTS — TABLE B")
    print(f"{'='*70}\n")

    summary = generate_summary(results)
    print(summary)

    # Save summary
    summary_path = output_dir / "table_b_summary.md"
    with open(summary_path, 'w') as f:
        f.write(summary)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Full results: {output_dir / 'progress.json'}")


if __name__ == "__main__":
    asyncio.run(main())
