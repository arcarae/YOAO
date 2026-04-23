#!/usr/bin/env python3
"""Evaluation Script: Heterogeneous Model Experiment

Tests whether SFT-trained Qwen3-14B seed agents can propagate cooperative behavior
to a DIFFERENT untrained model (Llama 3.1 8B) via deliberation.

This is a stronger test of alignment propagation — cross-model transfer means
the mechanism isn't model-specific but works through natural language persuasion.

Setup: Mixed Team A (seed% Qwen3-14B-SFT via vLLM + rest Llama-3.1-8B via OpenRouter)
       vs always_defect Team B
       Across seed percentages 0-100%, 3 held-out scenarios, 3 runs each

Usage:
    # Run full evaluation
    python scripts/eval_heterogeneous.py

    # Run specific condition
    python scripts/eval_heterogeneous.py --seed-pct 20 --scenario baseline

    # Resume interrupted run
    python scripts/eval_heterogeneous.py --resume

    # Quick test (1 run)
    python scripts/eval_heterogeneous.py --runs-per-condition 1 --seed-pct 0 --scenario baseline
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

# ── Server Config ────────────────────────────────────────────────────
VLLM_BASE_URL = "http://194.68.245.87:22067/v1"
TRAINED_MODEL = "redblackbench"                    # Qwen3-14B SFT LoRA (vLLM)
UNTRAINED_MODEL = "meta-llama/llama-3.1-8b-instruct"  # Llama 3.1 8B (OpenRouter)

# ── Experiment Design ────────────────────────────────────────────────
TEAM_SIZE = 5
SEED_PERCENTAGES = [0, 20, 40, 60, 80, 100]
RUNS_PER_CONDITION = 3

EVAL_SCENARIOS = [
    "baseline",
    "trade_war",
    "gpu_contention",
]

OPPONENT_STRATEGY = "always_defect"

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


async def run_single_game(
    team_size: int,
    num_trained: int,
    num_untrained: int,
    scenario_id: str,
    run_index: int,
    output_dir: Path,
    openrouter_api_key: str,
) -> Optional[Dict[str, Any]]:
    """Run a single game with heterogeneous Team A vs scripted Team B."""
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

        # Trained agents: Qwen3-14B SFT via vLLM
        for i in range(num_trained):
            provider = create_provider({
                "type": "vllm",
                "model": TRAINED_MODEL,
                "base_url": VLLM_BASE_URL,
                "temperature": 1.0,
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

        # Untrained agents: Llama 3.1 8B via OpenRouter
        for i in range(num_untrained):
            idx = num_trained + i
            provider = create_provider({
                "type": "openrouter",
                "model": UNTRAINED_MODEL,
                "api_key": openrouter_api_key,
                "temperature": 1.0,
                "include_reasoning": False,
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

        team_a_desc = f"hetero:{num_trained}xQwen-SFT+{num_untrained}xLlama8B"
        coordinator = GameCoordinator(
            team_a=team_a,
            team_b=team_b,
            config=config,
            trajectory_collector=collector,
            trajectory_save_path=str(trajectory_path),
            team_a_model=team_a_desc,
            team_b_model=f"scripted:{OPPONENT_STRATEGY}",
        )

        # Run with timeout
        timeout = 3600
        game_state = await asyncio.wait_for(coordinator.play_game(), timeout=timeout)

        # Extract results
        from redblackbench.game.scoring import Choice
        total_rounds = len(game_state.history)
        black_count = sum(
            1 for r in game_state.history if r.team_a_choice == Choice.BLACK
        )
        cooperation_rate = black_count / total_rounds if total_rounds > 0 else 0.0

        result = {
            "team_size": team_size,
            "seed_pct": seed_pct,
            "num_trained": num_trained,
            "num_untrained": num_untrained,
            "scenario": scenario_id,
            "run_index": run_index,
            "cooperation_rate": cooperation_rate,
            "total_rounds": total_rounds,
            "team_a_score": game_state.team_a_total,
            "team_b_score": game_state.team_b_total,
            "combined_score": game_state.team_a_total + game_state.team_b_total,
            "trained_model": TRAINED_MODEL,
            "untrained_model": UNTRAINED_MODEL,
            "experiment_type": "heterogeneous",
            "timestamp": datetime.now().isoformat(),
        }

        return result

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
    """Generate summary markdown."""
    games = results.get("games", [])
    if not games:
        return "No results yet."

    from collections import defaultdict

    by_condition = defaultdict(list)
    for game in games:
        key = (game["seed_pct"], game["scenario"])
        by_condition[key].append(game["cooperation_rate"] * 100)

    all_pcts = sorted(set(g["seed_pct"] for g in games))
    all_scenarios = sorted(set(g["scenario"] for g in games))

    lines = []
    lines.append("# Heterogeneous Model Experiment")
    lines.append(f"# Seed: Qwen3-14B SFT (vLLM) | Untrained: Llama 3.1 8B (OpenRouter)")
    lines.append("")
    lines.append(f"Team size: {TEAM_SIZE} | Opponent: {OPPONENT_STRATEGY}")
    lines.append(f"Runs per condition: {RUNS_PER_CONDITION}")
    lines.append("")

    # Main table: mean ± std across scenarios
    header = "| Seed % | Mean Coop | per scenario |"
    sep = "|--------|-----------|--------------|"
    lines.append(header)
    lines.append(sep)

    for pct in all_pcts:
        scenario_means = []
        details = []
        for sc in all_scenarios:
            rates = by_condition.get((pct, sc), [])
            if rates:
                m = sum(rates) / len(rates)
                scenario_means.append(m)
                details.append(f"{sc}={m:.0f}%")
        if scenario_means:
            mean = sum(scenario_means) / len(scenario_means)
            if len(scenario_means) > 1:
                std = math.sqrt(sum((r - mean) ** 2 for r in scenario_means) / (len(scenario_means) - 1))
                lines.append(f"| {pct}% | {mean:.0f}% ± {std:.0f} | {', '.join(details)} |")
            else:
                lines.append(f"| {pct}% | {mean:.0f}% | {', '.join(details)} |")
        else:
            lines.append(f"| {pct}% | — | |")

    lines.append("")

    # Per-scenario breakdown with run-level std
    lines.append("## Per-Scenario Breakdown (mean ± std across runs)")
    lines.append("")

    for scenario in all_scenarios:
        lines.append(f"### {scenario}")
        header = "| Seed % | Coop Rate | Runs |"
        sep = "|--------|-----------|------|"
        lines.append(header)
        lines.append(sep)

        for pct in all_pcts:
            rates = by_condition.get((pct, scenario), [])
            if rates:
                mean = sum(rates) / len(rates)
                n = len(rates)
                if n > 1:
                    std = math.sqrt(sum((r - mean) ** 2 for r in rates) / (n - 1))
                    lines.append(f"| {pct}% | {mean:.0f}% ± {std:.0f} | {n} |")
                else:
                    lines.append(f"| {pct}% | {mean:.0f}% | {n} |")
            else:
                lines.append(f"| {pct}% | — | 0 |")

        lines.append("")

    total = len(games)
    expected = len(all_pcts) * len(all_scenarios) * RUNS_PER_CONDITION
    lines.append(f"---")
    lines.append(f"Completed: {total}/{expected} games")

    return "\n".join(lines)


async def main():
    global RUNS_PER_CONDITION, VLLM_BASE_URL, UNTRAINED_MODEL, TRAINED_MODEL
    parser = argparse.ArgumentParser(
        description="Evaluate cross-model alignment propagation (heterogeneous experiment)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed-pct", type=int, choices=SEED_PERCENTAGES,
                        help="Run specific seed percentage only")
    parser.add_argument("--scenario", type=str, choices=EVAL_SCENARIOS,
                        help="Run specific scenario only")
    parser.add_argument("--runs-per-condition", type=int, default=RUNS_PER_CONDITION,
                        help=f"Runs per condition (default: {RUNS_PER_CONDITION})")
    parser.add_argument("--output-dir", type=str, default="results/eval_heterogeneous",
                        help="Output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--vllm-url", type=str, default=VLLM_BASE_URL,
                        help=f"vLLM server URL (default: {VLLM_BASE_URL})")
    parser.add_argument("--openrouter-key", type=str, default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--untrained-model", type=str, default=UNTRAINED_MODEL,
                        help=f"Untrained model on OpenRouter (default: {UNTRAINED_MODEL})")
    parser.add_argument("--trained-model", type=str, default=TRAINED_MODEL,
                        help=f"Trained model name on vLLM (default: {TRAINED_MODEL})")
    args = parser.parse_args()

    VLLM_BASE_URL = args.vllm_url
    RUNS_PER_CONDITION = args.runs_per_condition
    UNTRAINED_MODEL = args.untrained_model
    TRAINED_MODEL = args.trained_model

    openrouter_api_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("ERROR: OpenRouter API key required. Set OPENROUTER_API_KEY or use --openrouter-key")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_pcts = [args.seed_pct] if args.seed_pct is not None else SEED_PERCENTAGES
    scenarios = [args.scenario] if args.scenario else EVAL_SCENARIOS

    if args.resume:
        results, completed = load_progress(output_dir)
    else:
        results = {
            "games": [],
            "config": {
                "vllm_url": VLLM_BASE_URL,
                "trained_model": TRAINED_MODEL,
                "untrained_model": UNTRAINED_MODEL,
                "team_size": TEAM_SIZE,
                "seed_pcts": seed_pcts,
                "scenarios": scenarios,
                "runs_per_condition": args.runs_per_condition,
                "opponent": OPPONENT_STRATEGY,
                "experiment_type": "heterogeneous",
            }
        }
        completed = set()

    total_games = len(seed_pcts) * len(scenarios) * args.runs_per_condition
    done = len(completed)

    print(f"\n{'='*70}")
    print("HETEROGENEOUS MODEL EXPERIMENT")
    print(f"{'='*70}")
    print(f"Seed model:      {TRAINED_MODEL} (Qwen3-14B SFT via vLLM)")
    print(f"Untrained model: {UNTRAINED_MODEL} (via OpenRouter)")
    print(f"Team size:       {TEAM_SIZE}")
    print(f"Seed %:          {seed_pcts}")
    print(f"Scenarios:       {scenarios}")
    print(f"Runs/condition:  {args.runs_per_condition}")
    print(f"Total games:     {total_games} ({done} already done)")
    print(f"Output:          {output_dir}")
    print(f"{'='*70}\n")

    current = 0
    for pct in seed_pcts:
        num_trained, num_untrained = compute_team_composition(TEAM_SIZE, pct)
        for scenario in scenarios:
            for run_idx in range(args.runs_per_condition):
                current += 1
                key = (TEAM_SIZE, pct, scenario, run_idx)

                if key in completed:
                    print(f"[{current}/{total_games}] SKIP seed={pct}% {scenario} run={run_idx}")
                    continue

                label = f"{num_trained}xQwen-SFT + {num_untrained}xLlama8B ({pct}%) | {scenario} | run {run_idx+1}/{args.runs_per_condition}"
                print(f"[{current}/{total_games}] {label}")

                result = await run_single_game(
                    team_size=TEAM_SIZE,
                    num_trained=num_trained,
                    num_untrained=num_untrained,
                    scenario_id=scenario,
                    run_index=run_idx,
                    output_dir=output_dir,
                    openrouter_api_key=openrouter_api_key,
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
    print("RESULTS — HETEROGENEOUS EXPERIMENT")
    print(f"{'='*70}\n")

    summary = generate_summary(results)
    print(summary)

    summary_path = output_dir / "heterogeneous_summary.md"
    with open(summary_path, 'w') as f:
        f.write(summary)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Full results: {output_dir / 'progress.json'}")


if __name__ == "__main__":
    asyncio.run(main())
