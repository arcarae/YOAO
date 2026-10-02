#!/usr/bin/env python3
"""
Table 3 Ablation: Mixed Hardcoded + Untrained Agents

This script tests mixed teams with varying ratios of hardcoded "always cooperate"
agents and untrained LLM agents.

Compositions tested:
- 1 hardcoded + 4 untrained
- 2 hardcoded + 3 untrained
- 3 hardcoded + 2 untrained
- 4 hardcoded + 1 untrained

Purpose: Measure whether hardcoded cooperators influence untrained agents'
behavior, and how cooperation rates change with different mixing ratios.

Usage:
    python eval_table3_ablation_mixed.py --scenario pandemic_vaccines --games 3
    python eval_table3_ablation_mixed.py --hardcoded-count 2 --scenario baseline --games 1
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from redblackbench.game.coordinator import GameCoordinator
from redblackbench.game.config import GameConfig
from redblackbench.teams.team import Team
from redblackbench.scenarios import get_scenario
from redblackbench.trajectory import TrajectoryCollector
from redblackbench.agents.hardcoded_agent import HardcodedBlackAgent
from redblackbench.agents.llm_agent import LLMAgent
from redblackbench.agents.muted_agent import MutedAgent
from redblackbench.cli import create_provider
from redblackbench.strategies import create_scripted_team

# How the "seed" agents on Team A are built (set from --mode in main()):
#   muted     SFT seeds that vote through their own model but may only say "I vote X."
#             (the paper's mute test: argument content removed, votes kept)
#   hardcoded non-LLM agents that always vote cooperatively without any model call
SEED_MODE = "muted"


# Agent names (same as in Table 3 for consistency)
AGENT_NAMES = [
    "Dr. Sarah Chen",
    "Marcus Webb",
    "Dr. Priya Sharma",
    "James O'Connor",
    "Dr. Elena Vasquez",
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

# Mixed compositions to test
COMPOSITIONS = [
    (1, 4),  # 1 hardcoded + 4 untrained
    (2, 3),  # 2 hardcoded + 3 untrained
    (3, 2),  # 3 hardcoded + 2 untrained
    (4, 1),  # 4 hardcoded + 1 untrained
]

# Model configuration (override with YOAO_SFT_MODEL / YOAO_BASE_MODEL / YOAO_VLLM_URL)
from redblackbench import defaults

TRAINED_MODEL = defaults.SFT_MODEL      # cooperative LoRA adapter name on the vLLM server
UNTRAINED_MODEL = defaults.BASE_MODEL   # unmodified base model name on the vLLM server
VLLM_BASE_URL = defaults.VLLM_URL


def get_choice_names(scenario: Optional[str]) -> tuple[str, str]:
    """Get scenario-specific choice names.

    Returns:
        (black_choice_name, red_choice_name) tuple
    """
    if scenario is None or scenario == "baseline":
        return ("BLACK", "RED")

    scenario_obj = get_scenario(scenario)
    if scenario_obj:
        config = scenario_obj.config
        return (config.humanity_choice_name, config.tribe_choice_name)

    return ("BLACK", "RED")


async def run_single_game(
    num_hardcoded: int,
    num_untrained: int,
    scenario: str,
    game_index: int,
    output_dir: Path,
    untrained_model: str,
    vllm_base_url: str,
    enable_thinking: bool = False,
) -> dict:
    """Run a single game with mixed hardcoded + untrained agents.

    Args:
        num_hardcoded: Number of hardcoded agents on Team A
        num_untrained: Number of untrained agents on Team A
        scenario: Scenario name
        game_index: Game number for this configuration
        output_dir: Directory to save trajectories
        untrained_model: Path to untrained model
        vllm_base_url: vLLM server base URL

    Returns:
        Game result summary dict
    """
    print(f"\n{'='*80}")
    print(f"Game {game_index + 1} - {num_hardcoded}H+{num_untrained}U - Scenario: {scenario}")
    print(f"{'='*80}")

    try:
        # Setup scenario
        if scenario == "baseline":
            scenario_obj = None
            prompt_template = None
        else:
            scenario_obj = get_scenario(scenario)
            if not scenario_obj:
                print(f"  ✗ Unknown scenario: {scenario}")
                return {
                    "success": False,
                    "error": f"Unknown scenario: {scenario}",
                }
            prompt_template = scenario_obj.to_prompt_template()

        # Get scenario-specific choice names
        black_name, red_name = get_choice_names(scenario)

        # Game configuration
        config = GameConfig(
            num_rounds=10,
            multipliers={5: 3, 8: 5, 10: 10},
        )

        # Create Team A with mixed agents
        agents_a = []

        # Add hardcoded agents first
        for i in range(num_hardcoded):
            if SEED_MODE == "hardcoded":
                agent = HardcodedBlackAgent(
                    agent_id=AGENT_NAMES[i],
                    team_name="Team A",
                    choice_name_black=black_name,
                    choice_name_red=red_name,
                )
            else:
                seed_provider = create_provider({
                    "type": "vllm",
                    "model": TRAINED_MODEL,
                    "temperature": 0.7,
                    "base_url": vllm_base_url,
                })
                seed = LLMAgent(
                    agent_id=AGENT_NAMES[i],
                    team_name="Team A",
                    provider=seed_provider,
                    prompt_template=prompt_template,
                    enable_thinking=enable_thinking,
                )
                seed._is_trained = True
                seed._model_name = TRAINED_MODEL
                agent = MutedAgent(seed, choice_name_black=black_name, choice_name_red=red_name)
            agent._is_hardcoded = True
            agents_a.append(agent)

        # Add untrained agents
        for i in range(num_untrained):
            idx = num_hardcoded + i
            agent_id = AGENT_NAMES[idx]
            provider = create_provider({
                "type": "vllm",
                "model": untrained_model,
                "temperature": 0.7,
                "base_url": vllm_base_url,
            })
            agent = LLMAgent(
                agent_id=agent_id,
                team_name="Team A",
                provider=provider,
                prompt_template=prompt_template,
                enable_thinking=enable_thinking,
            )
            agent._is_trained = False
            agent._is_hardcoded = False
            agent._model_name = untrained_model
            agents_a.append(agent)

        team_a = Team(name="Team A", agents=agents_a)

        # Create Team B (always defect - same as Table 3)
        team_b = create_scripted_team(
            strategy_id="always_defect",
            team_name="Team B (always_defect)",
        )

        # Setup trajectory collection
        collector = TrajectoryCollector()

        # Create game coordinator
        coordinator = GameCoordinator(
            team_a=team_a,
            team_b=team_b,
            config=config,
            trajectory_collector=collector,
        )

        # Play game
        print(f"\nStarting game...")
        start_time = datetime.now()

        game_state = await coordinator.play_game()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Collect trajectory
        team_a_model = f"mixed:{num_hardcoded}xhardcoded+{num_untrained}x{untrained_model}"
        trajectory = collector.get_trajectory()

        # Save trajectory
        scenario_dir = output_dir / "trajectories" / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)

        trajectory_path = scenario_dir / f"mixed_{num_hardcoded}h_{num_untrained}u_game_{game_index}.json"
        if trajectory:
            # Add metadata about agent composition
            traj_dict = trajectory.get_full_trajectory()
            traj_dict["team_a_model"] = team_a_model
            traj_dict["team_b_model"] = "always_defect"
            traj_dict["num_hardcoded"] = num_hardcoded
            traj_dict["num_untrained"] = num_untrained

            with open(trajectory_path, "w") as f:
                json.dump(traj_dict, f, indent=2)

        # Calculate metrics
        team_a_score = game_state.team_a_total
        team_b_score = game_state.team_b_total
        total_score = team_a_score + team_b_score
        max_possible = 150  # 10 rounds with perfect cooperation
        efficiency = (total_score / max_possible) * 100

        # Count cooperation choices from game history
        cooperation_count = sum(
            1 for round_result in game_state.history
            if round_result.team_a_choice.value == "BLACK"
        )
        cooperation_rate = (cooperation_count / config.num_rounds) * 100

        print(f"\n{'='*80}")
        print(f"Game {game_index + 1} Complete - {num_hardcoded}H+{num_untrained}U")
        print(f"{'='*80}")
        print(f"Duration: {duration:.1f}s")
        print(f"Team A ({num_hardcoded}H+{num_untrained}U): {team_a_score}")
        print(f"Team B (Defect): {team_b_score}")
        print(f"Total: {total_score}/{max_possible} ({efficiency:.1f}% efficiency)")
        print(f"Cooperation rate: {cooperation_rate:.1f}%")
        print(f"Trajectory saved to: {trajectory_path}")

        return {
            "success": True,
            "scenario": scenario,
            "game_index": game_index,
            "num_hardcoded": num_hardcoded,
            "num_untrained": num_untrained,
            "team_a_score": team_a_score,
            "team_b_score": team_b_score,
            "total_score": total_score,
            "efficiency": efficiency,
            "cooperation_rate": cooperation_rate,
            "duration_seconds": duration,
            "trajectory_path": str(trajectory_path),
        }

    except Exception as e:
        print(f"\n✗ Game {game_index + 1} failed with error: {e}")
        import traceback
        traceback.print_exc()

        return {
            "success": False,
            "scenario": scenario,
            "game_index": game_index,
            "num_hardcoded": num_hardcoded,
            "num_untrained": num_untrained,
            "error": str(e),
        }


async def run_composition_scenario(
    num_hardcoded: int,
    num_untrained: int,
    scenario: str,
    num_games: int,
    output_dir: Path,
    untrained_model: str,
    vllm_base_url: str,
    enable_thinking: bool = False,
) -> list[dict]:
    """Run multiple games for a single composition + scenario combination.

    Args:
        num_hardcoded: Number of hardcoded agents
        num_untrained: Number of untrained agents
        scenario: Scenario name
        num_games: Number of games to run
        output_dir: Directory to save results
        untrained_model: Path to untrained model
        vllm_base_url: vLLM server base URL

    Returns:
        List of game result dicts
    """
    print(f"\n{'#'*80}")
    print(f"# Running {num_games} game(s): {num_hardcoded}H+{num_untrained}U - {scenario}")
    print(f"{'#'*80}")

    results = []
    for i in range(num_games):
        result = await run_single_game(
            num_hardcoded, num_untrained, scenario, i,
            output_dir, untrained_model, vllm_base_url, enable_thinking
        )
        results.append(result)

    return results


async def run_full_ablation(
    num_games: int,
    output_dir: Path,
    scenarios: list[str],
    compositions: list[tuple[int, int]],
    untrained_model: str,
    vllm_base_url: str,
    enable_thinking: bool = False,
) -> dict:
    """Run full ablation with all compositions and scenarios.

    Args:
        num_games: Number of games per (composition, scenario) combination
        output_dir: Directory to save results
        scenarios: List of scenarios to run
        compositions: List of (num_hardcoded, num_untrained) tuples
        untrained_model: Path to untrained model
        vllm_base_url: vLLM server base URL

    Returns:
        Summary dict with all results
    """
    total_combinations = len(compositions) * len(scenarios)

    print(f"\n{'='*80}")
    print(f"Table 3 Ablation: Mixed Hardcoded + Untrained Agents")
    print(f"{'='*80}")
    print(f"Compositions: {len(compositions)}")
    for num_h, num_u in compositions:
        print(f"  - {num_h} hardcoded + {num_u} untrained")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Games per combination: {num_games}")
    print(f"Total combinations: {total_combinations}")
    print(f"Total games: {total_combinations * num_games}")
    print(f"Output directory: {output_dir}")
    print(f"Untrained model: {untrained_model}")
    print(f"{'='*80}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    all_results = []

    # Run each combination sequentially
    for num_hardcoded, num_untrained in compositions:
        for scenario in scenarios:
            combo_results = await run_composition_scenario(
                num_hardcoded, num_untrained, scenario, num_games,
                output_dir, untrained_model, vllm_base_url, enable_thinking
            )
            all_results.extend(combo_results)

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    # Calculate summary statistics
    successful_games = [r for r in all_results if r["success"]]
    failed_games = [r for r in all_results if not r["success"]]

    summary = {
        "experiment": "table3_ablation_mixed_hardcoded_untrained",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_duration_seconds": total_duration,
        "compositions": [(h, u) for h, u in compositions],
        "scenarios": scenarios,
        "games_per_combination": num_games,
        "total_combinations": total_combinations,
        "total_games": len(all_results),
        "successful_games": len(successful_games),
        "failed_games": len(failed_games),
        "untrained_model": untrained_model,
        "results": all_results,
    }

    # Add aggregate statistics
    if successful_games:
        summary["aggregate_stats"] = {
            "avg_efficiency": sum(r["efficiency"] for r in successful_games) / len(successful_games),
            "avg_cooperation_rate": sum(r["cooperation_rate"] for r in successful_games) / len(successful_games),
            "avg_team_a_score": sum(r["team_a_score"] for r in successful_games) / len(successful_games),
            "avg_total_score": sum(r["total_score"] for r in successful_games) / len(successful_games),
        }

        # Per-composition statistics
        composition_stats = {}
        for num_h, num_u in compositions:
            comp_key = f"{num_h}h_{num_u}u"
            comp_games = [r for r in successful_games
                         if r["num_hardcoded"] == num_h and r["num_untrained"] == num_u]
            if comp_games:
                composition_stats[comp_key] = {
                    "num_hardcoded": num_h,
                    "num_untrained": num_u,
                    "games": len(comp_games),
                    "avg_efficiency": sum(r["efficiency"] for r in comp_games) / len(comp_games),
                    "avg_cooperation_rate": sum(r["cooperation_rate"] for r in comp_games) / len(comp_games),
                    "avg_team_a_score": sum(r["team_a_score"] for r in comp_games) / len(comp_games),
                }
        summary["composition_stats"] = composition_stats

        # Per-scenario statistics
        scenario_stats = {}
        for scenario in scenarios:
            scenario_games = [r for r in successful_games if r["scenario"] == scenario]
            if scenario_games:
                scenario_stats[scenario] = {
                    "games": len(scenario_games),
                    "avg_efficiency": sum(r["efficiency"] for r in scenario_games) / len(scenario_games),
                    "avg_cooperation_rate": sum(r["cooperation_rate"] for r in scenario_games) / len(scenario_games),
                    "avg_team_a_score": sum(r["team_a_score"] for r in scenario_games) / len(scenario_games),
                }
        summary["scenario_stats"] = scenario_stats

    # Save summary
    summary_path = output_dir / "ablation_mixed_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print(f"\n{'='*80}")
    print(f"Ablation Experiment Complete")
    print(f"{'='*80}")
    print(f"Total time: {total_duration:.1f}s ({total_duration/60:.1f}m)")
    print(f"Successful games: {len(successful_games)}/{len(all_results)}")
    print(f"Failed games: {len(failed_games)}")

    if successful_games:
        print(f"\nAggregate Statistics:")
        print(f"  Average efficiency: {summary['aggregate_stats']['avg_efficiency']:.1f}%")
        print(f"  Average cooperation rate: {summary['aggregate_stats']['avg_cooperation_rate']:.1f}%")
        print(f"  Average Team A score: {summary['aggregate_stats']['avg_team_a_score']:.1f}")

        if "composition_stats" in summary:
            print(f"\nPer-Composition Statistics:")
            for comp_key, stats in summary["composition_stats"].items():
                print(f"  {stats['num_hardcoded']}H+{stats['num_untrained']}U:")
                print(f"    Efficiency: {stats['avg_efficiency']:.1f}%")
                print(f"    Cooperation: {stats['avg_cooperation_rate']:.1f}%")

        if "scenario_stats" in summary:
            print(f"\nPer-Scenario Statistics:")
            for scenario, stats in summary["scenario_stats"].items():
                print(f"  {scenario}:")
                print(f"    Efficiency: {stats['avg_efficiency']:.1f}%")
                print(f"    Cooperation: {stats['avg_cooperation_rate']:.1f}%")

    print(f"\nSummary saved to: {summary_path}")
    print(f"{'='*80}\n")

    return summary


def main():
    global SEED_MODE, TRAINED_MODEL
    parser = argparse.ArgumentParser(
        description="Mute test: muted SFT seeds (or hardcoded cooperators) + unmodified agents"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=SCENARIOS + ["all"],
        default="all",
        help="Scenario to run (default: all)",
    )
    parser.add_argument(
        "--hardcoded-count",
        type=int,
        choices=[1, 2, 3, 4, None],
        default=None,
        help="Number of hardcoded agents (1-4). If not specified, runs all compositions.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=3,
        help="Number of games per (composition, scenario) combination (default: 3)",
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
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking/reasoning mode for LLM agents (Qwen3, etc.)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["muted", "hardcoded"],
        default="muted",
        help="Seed agents: 'muted' = SFT seeds that vote but only say 'I vote X.' (paper's mute test); "
             "'hardcoded' = non-LLM always-cooperate voters (default: muted)",
    )
    parser.add_argument(
        "--trained-model",
        type=str,
        default=TRAINED_MODEL,
        help=f"LoRA adapter name for muted seeds on the vLLM server (default: {TRAINED_MODEL})",
    )

    args = parser.parse_args()

    SEED_MODE = args.mode
    TRAINED_MODEL = args.trained_model

    output_dir = Path(args.output_dir)

    # Determine which scenarios to run
    if args.scenario == "all":
        scenarios = SCENARIOS
    else:
        scenarios = [args.scenario]

    # Determine which compositions to run
    if args.hardcoded_count is not None:
        num_untrained = 5 - args.hardcoded_count
        compositions = [(args.hardcoded_count, num_untrained)]
    else:
        compositions = COMPOSITIONS

    # Run evaluation
    asyncio.run(run_full_ablation(
        num_games=args.games,
        output_dir=output_dir,
        scenarios=scenarios,
        compositions=compositions,
        untrained_model=args.untrained_model,
        vllm_base_url=args.vllm_base_url,
        enable_thinking=args.enable_thinking,
    ))


if __name__ == "__main__":
    main()
