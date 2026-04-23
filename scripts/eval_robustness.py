#!/usr/bin/env python3
"""Evaluation Script: Robustness to Opponent Strategies (Table 2)

Tests how well each model maintains cooperation against different hardcoded opponent behaviors.

Setup: LLM Team A (5 agents) vs Scripted Team B (hardcoded strategy)

Usage:
    # Run all models against all strategies (full evaluation)
    python scripts/eval_robustness.py --output-dir results/eval_robustness

    # Run specific model against all strategies
    python scripts/eval_robustness.py --model "qwen/qwen3-14b" --output-dir results/eval_robustness

    # Run specific model against specific strategy
    python scripts/eval_robustness.py --model "qwen/qwen3-14b" --strategy always_defect

    # Run with custom scenario (default: climate_cooperation)
    python scripts/eval_robustness.py --scenario pandemic_vaccines

    # Run multiple games per combination for statistical significance
    python scripts/eval_robustness.py --games-per-combo 3

    # Resume interrupted evaluation
    python scripts/eval_robustness.py --resume --output-dir results/eval_robustness
"""

import argparse
import asyncio
import json
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
from redblackbench.scenarios import get_scenario, list_scenarios
from redblackbench.trajectory import TrajectoryCollector
from redblackbench.agents.llm_agent import LLMAgent
from redblackbench.cli import create_provider
from redblackbench.strategies import create_scripted_team, list_strategies

# Models to evaluate
DEFAULT_MODELS = [
    "openai/gpt-4o",
    "google/gemma-3-27b-it",
    "qwen/qwen3-30b-a3b-thinking-2507",
    "zhipu/glm-4.6v",
    "openai/gpt-5.2-thinking",
    "moonshotai/kimi-k2-thinking",
    "qwen/qwen3-14b",
    # Add your SFT model here:
    # "your-org/qwen3-14b-sft",
]

# Opponent strategies to test against (ordered by difficulty)
OPPONENT_STRATEGIES = [
    # Very hard (sustained exploitation)
    "always_defect",
    "early_exploiter_no_recovery",
    # Hard (exploitation)
    "defect_critical",
    "critical_exploiter",
    "late_betrayer",
    # Medium (mixed)
    "mid_exploiter",
    "early_exploiter",
    "tit_for_tat",
    # Easy (cooperative)
    "mostly_cooperate",
    "always_cooperate",
]

DEFAULT_SCENARIO = "climate_cooperation"

# Special scenario: "baseline" uses no scenario framing (just RED/BLACK game)


def calculate_cooperation_rate(team_a_score: int, team_b_score: int, num_rounds: int = 10) -> float:
    """Calculate cooperation rate from scores.

    In a 10-round game with multipliers {5: 3, 8: 5, 10: 10}:
    - Total multiplier sum = 1+1+1+1+3+1+1+5+1+10 = 25
    - Max mutual cooperation score per team = 3 * 25 = 75
    - Max sum = 150

    Cooperation rate approximation: (sum + 150) / 300
    """
    total = team_a_score + team_b_score
    # Clamp to valid range
    rate = (total + 150) / 300
    return max(0.0, min(1.0, rate))


def calculate_team_a_cooperation_rate(history: List[str]) -> float:
    """Calculate Team A's cooperation rate from move history."""
    if not history:
        return 0.0
    cooperate_count = sum(1 for move in history if move == "A")
    return cooperate_count / len(history)


async def run_single_game(
    model: str,
    strategy_id: str,
    scenario_id: str,
    game_index: int,
    output_dir: Path,
    team_size: int = 5,
    enable_thinking: bool = False,
    goal_preset: str = "none",
) -> Optional[Dict[str, Any]]:
    """Run a single game: LLM Team A vs Scripted Team B.

    Args:
        model: Model identifier (e.g., "openai/gpt-4o")
        strategy_id: Opponent strategy (e.g., "always_defect")
        scenario_id: Scenario to use (e.g., "climate_cooperation" or "baseline")
        game_index: Game number for this combination
        output_dir: Directory for output files
        team_size: Number of agents on Team A
        enable_thinking: Enable thinking/reasoning mode for supported models
        goal_preset: Goal preset for prompt-only alignment ("none", "altruist", "exploitative")

    Returns:
        Result dict with scores, cooperation rates, and metadata
    """
    try:
        # Handle baseline (no scenario framing) vs real scenarios
        if scenario_id == "baseline":
            scenario = None
            prompt_template = None  # Will use default RED/BLACK prompts
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

        # Determine provider type based on model path
        if model.startswith("/") or model.startswith("./"):
            # Local model path - use vLLM
            provider_config = {
                "type": "vllm",
                "model": model,
                "base_url": "http://localhost:8000/v1",
                "temperature": 1.0,
            }
        else:
            # Remote model - use OpenRouter
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            provider_config = {
                "type": "openrouter",
                "model": model,
                "temperature": 1.0,
                "api_key": api_key,
            }

        # Create Team A (LLM)
        team_a_names = ["Dr. Sarah Chen", "Marcus Webb", "Dr. Priya Sharma", "James O'Connor", "Dr. Elena Vasquez"]

        agents_a = []
        for i in range(team_size):
            agent_id = team_a_names[i % len(team_a_names)]
            provider = create_provider(provider_config)
            agent = LLMAgent(
                agent_id=agent_id,
                team_name="Team A",
                provider=provider,
                prompt_template=prompt_template,
                enable_thinking=enable_thinking,
                goal_preset=goal_preset,
            )
            agents_a.append(agent)
        team_a = Team(name="Team A", agents=agents_a)

        # Create Team B (Scripted)
        team_b = create_scripted_team(
            strategy_id=strategy_id,
            team_name=f"Team B ({strategy_id})",
        )

        # Setup trajectory collection
        results_dir = output_dir / "trajectories" / scenario_id
        results_dir.mkdir(parents=True, exist_ok=True)

        model_safe = model.replace("/", "_").replace(":", "_")
        trajectory_path = results_dir / f"{model_safe}_vs_{strategy_id}_game_{game_index}.json"

        collector = TrajectoryCollector()

        coordinator = GameCoordinator(
            team_a=team_a,
            team_b=team_b,
            config=config,
            trajectory_collector=collector,
        )

        # Run game without timeout (some games can take longer)
        await coordinator.play_game()

        # Save trajectory
        if collector.trajectory:
            collector.trajectory.team_a_model = model
            collector.trajectory.team_b_model = f"scripted:{strategy_id}"
            collector.save_trajectory(str(trajectory_path))

        # Extract results
        final_scores = coordinator.get_final_scores()
        team_a_history = coordinator.get_team_history("A")

        result = {
            "model": model,
            "strategy": strategy_id,
            "scenario": scenario_id,
            "game_index": game_index,
            "team_a_score": final_scores.get("Team A", 0),
            "team_b_score": final_scores.get("Team B", 0),
            "sum": final_scores.get("Team A", 0) + final_scores.get("Team B", 0),
            "team_a_cooperation_rate": calculate_team_a_cooperation_rate(team_a_history),
            "trajectory_path": str(trajectory_path),
        }

        return result

    except asyncio.TimeoutError:
        print(f"    Game timed out after 30 minutes")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def load_progress(output_dir: Path) -> Tuple[Dict[str, Any], set]:
    """Load existing progress from checkpoint.

    Returns:
        Tuple of (results dict, set of completed (model, strategy, game_index) tuples)
    """
    progress_path = output_dir / "eval_robustness_progress.json"
    completed = set()
    results = {"games": [], "summary": {}}

    if progress_path.exists():
        try:
            with open(progress_path, 'r') as f:
                data = json.load(f)
            results = data
            for game in data.get("games", []):
                key = (game["model"], game["strategy"], game["game_index"])
                completed.add(key)
            print(f"  Loaded progress: {len(completed)} games completed")
        except Exception as e:
            print(f"  Could not load progress: {e}")

    return results, completed


def save_progress(output_dir: Path, results: Dict[str, Any]):
    """Save progress checkpoint."""
    progress_path = output_dir / "eval_robustness_progress.json"
    results["last_updated"] = datetime.now().isoformat()

    with open(progress_path, 'w') as f:
        json.dump(results, f, indent=2)


def generate_summary_table(results: Dict[str, Any]) -> str:
    """Generate markdown summary table from results."""
    games = results.get("games", [])

    if not games:
        return "No results yet."

    # Group by model and strategy
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))

    for game in games:
        model = game["model"]
        strategy = game["strategy"]
        grouped[model][strategy].append(game["team_a_cooperation_rate"])

    # Build table
    models = sorted(grouped.keys())
    strategies = OPPONENT_STRATEGIES

    lines = []
    header = "| Model | " + " | ".join(strategies) + " |"
    separator = "|-------|" + "|".join(["-------"] * len(strategies)) + "|"
    lines.append(header)
    lines.append(separator)

    for model in models:
        row = f"| {model} |"
        for strategy in strategies:
            rates = grouped[model].get(strategy, [])
            if rates:
                avg_rate = sum(rates) / len(rates) * 100
                row += f" {avg_rate:.0f}% |"
            else:
                row += " - |"
        lines.append(row)

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model robustness to opponent strategies (Table 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, help="Specific model to evaluate (default: all)")
    parser.add_argument("--strategy", type=str, help="Specific strategy to test against (default: all)")
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO,
                        help="Scenario to use (use 'baseline' for abstract RED/BLACK game without scenario framing)")
    parser.add_argument("--output-dir", type=str, default="results/eval_robustness", help="Output directory")
    parser.add_argument("--games-per-combo", type=int, default=1, help="Games per model-strategy combination")
    parser.add_argument("--team-size", type=int, default=5, help="Team A size")
    parser.add_argument("--max-parallel", type=int, default=3, help="Maximum parallel games")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-strategies", action="store_true", help="List available strategies")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable thinking/reasoning mode for LLM agents (Qwen3, etc.)")
    parser.add_argument("--goal-preset", type=str, default="none",
                        choices=["none", "altruist", "exploitative", "cooperative", "selfish", "baseline"],
                        help="Goal preset for prompt-only alignment (default: none)")
    args = parser.parse_args()

    if args.list_models:
        print("Default models to evaluate:")
        for m in DEFAULT_MODELS:
            print(f"  - {m}")
        return

    if args.list_strategies:
        print("Available opponent strategies:")
        for sid, desc in list_strategies().items():
            marker = "*" if sid in OPPONENT_STRATEGIES else " "
            print(f"  {marker} {sid}: {desc}")
        print("\n* = included in default evaluation")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine models and strategies to run
    models = [args.model] if args.model else DEFAULT_MODELS
    strategies = [args.strategy] if args.strategy else OPPONENT_STRATEGIES

    # Load existing progress if resuming
    if args.resume:
        results, completed = load_progress(output_dir)
    else:
        results = {"games": [], "summary": {}}
        completed = set()

    print(f"\n{'='*60}")
    print("EVALUATION: Robustness to Opponent Strategies (Table 2)")
    print(f"{'='*60}")
    print(f"Models: {len(models)}")
    print(f"Strategies: {len(strategies)}")
    print(f"Games per combo: {args.games_per_combo}")
    print(f"Scenario: {args.scenario}")
    print(f"Goal preset: {args.goal_preset}")
    print(f"Enable thinking: {args.enable_thinking}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    total_combos = len(models) * len(strategies) * args.games_per_combo

    # Build list of tasks to run
    tasks_to_run = []
    for model in models:
        for strategy in strategies:
            for game_idx in range(args.games_per_combo):
                key = (model, strategy, game_idx)
                if key not in completed:
                    tasks_to_run.append((model, strategy, game_idx, key))

    print(f"Tasks to run: {len(tasks_to_run)}/{total_combos}")
    print(f"Max parallel: {args.max_parallel}\n")

    # Run games in parallel with semaphore to limit concurrency
    semaphore = asyncio.Semaphore(args.max_parallel)

    async def run_with_semaphore(model: str, strategy: str, game_idx: int, key: tuple, task_num: int):
        """Run a single game with semaphore limiting."""
        async with semaphore:
            print(f"[{task_num}/{len(tasks_to_run)}] {model} vs {strategy} (game {game_idx})...")

            result = await run_single_game(
                model=model,
                strategy_id=strategy,
                scenario_id=args.scenario,
                game_index=game_idx,
                output_dir=output_dir,
                team_size=args.team_size,
                enable_thinking=args.enable_thinking,
                goal_preset=args.goal_preset,
            )

            if result:
                results["games"].append(result)
                completed.add(key)
                coop_rate = result["team_a_cooperation_rate"] * 100
                print(f"    ✓ [{task_num}/{len(tasks_to_run)}] {model} vs {strategy}: Score {result['team_a_score']} vs {result['team_b_score']} | Coop: {coop_rate:.0f}%")

                # Save checkpoint after each completed game
                save_progress(output_dir, results)
            else:
                print(f"    ✗ [{task_num}/{len(tasks_to_run)}] {model} vs {strategy} FAILED")

            return result

    # Create all tasks
    tasks = [
        run_with_semaphore(model, strategy, game_idx, key, i+1)
        for i, (model, strategy, game_idx, key) in enumerate(tasks_to_run)
    ]

    # Run all games in parallel (controlled by semaphore)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    # Generate final summary
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}\n")

    summary_table = generate_summary_table(results)
    print(summary_table)

    # Save summary
    summary_path = output_dir / "summary_table.md"
    with open(summary_path, 'w') as f:
        f.write("# Table 2: Robustness to Opponent Strategies (Cooperation Rate %)\n\n")
        f.write(f"*Agent Team (Team A) vs hardcoded opponent (Team B)*\n\n")
        f.write(f"Scenario: {args.scenario}\n\n")
        f.write(summary_table)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Full results saved to: {output_dir / 'eval_robustness_progress.json'}")


if __name__ == "__main__":
    asyncio.run(main())
