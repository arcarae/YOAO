#!/usr/bin/env python3
"""
Table 3 Ablation: Hardcoded "Always Cooperate" Agents

This script replicates the Table 3 experiment structure but replaces ALL agents
with hardcoded personas that always vote BLACK (cooperative choice).

Purpose: By comparing these results with the original Table 3, we can isolate
the effect of reasoning quality on cooperation rates. The hardcoded agents
have 100% cooperation by construction, so any difference in game outcomes
reveals the impact of reasoning and deliberation on cooperation success.

Usage:
    python eval_table3_ablation_hardcoded.py --scenario pandemic_vaccines --games 3

    Or for a single game:
    python eval_table3_ablation_hardcoded.py --scenario baseline --games 1
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
from redblackbench.strategies import create_scripted_team


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


def get_choice_names(scenario: Optional[str]) -> tuple[str, str]:
    """Get scenario-specific choice names.

    Returns:
        (black_choice_name, red_choice_name) tuple
    """
    if scenario is None or scenario == "baseline":
        # Baseline scenario uses generic names
        return ("BLACK", "RED")

    scenario_obj = get_scenario(scenario)
    if scenario_obj:
        config = scenario_obj.config
        return (config.humanity_choice_name, config.tribe_choice_name)

    # Fallback
    return ("BLACK", "RED")


async def run_single_game(
    scenario: str,
    game_index: int,
    output_dir: Path,
) -> dict:
    """Run a single game with all hardcoded agents.

    Args:
        scenario: Scenario name
        game_index: Game number for this configuration
        output_dir: Directory to save trajectories

    Returns:
        Game result summary dict
    """
    print(f"\n{'='*80}")
    print(f"Game {game_index + 1} - Scenario: {scenario}")
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

        # Create Team A with 5 hardcoded agents (all vote BLACK)
        agents_a = []
        for i in range(5):
            agent = HardcodedBlackAgent(
                agent_id=AGENT_NAMES[i],
                team_name="Team A",
                choice_name_black=black_name,
                choice_name_red=red_name,
            )
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
        trajectory = collector.end_trajectory_collection(
            team_a_model="hardcoded_black",
            team_b_model="always_defect",
        )

        # Save trajectory
        scenario_dir = output_dir / "trajectories" / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)

        trajectory_path = scenario_dir / f"hardcoded_game_{game_index}.json"
        with open(trajectory_path, "w") as f:
            json.dump(trajectory.to_dict(), f, indent=2)

        # Calculate metrics (game_state is a GameState object)
        team_a_score = game_state.team_a_total
        team_b_score = game_state.team_b_total
        total_score = team_a_score + team_b_score
        max_possible = 150  # 10 rounds with perfect cooperation
        efficiency = (total_score / max_possible) * 100

        # Count cooperation choices from trajectory
        cooperation_count = sum(
            1 for timestep in trajectory.timesteps
            if timestep.type == "ROUND_END" and timestep.data.get("team_a_choice") == "BLACK"
        )
        cooperation_rate = (cooperation_count / config.num_rounds) * 100

        print(f"\n{'='*80}")
        print(f"Game {game_index + 1} Complete")
        print(f"{'='*80}")
        print(f"Duration: {duration:.1f}s")
        print(f"Team A (Hardcoded): {team_a_score}")
        print(f"Team B (Defect): {team_b_score}")
        print(f"Total: {total_score}/{max_possible} ({efficiency:.1f}% efficiency)")
        print(f"Cooperation rate: {cooperation_rate:.1f}%")
        print(f"Trajectory saved to: {trajectory_path}")

        return {
            "success": True,
            "scenario": scenario,
            "game_index": game_index,
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
            "error": str(e),
        }


async def run_scenario_games(
    scenario: str,
    num_games: int,
    output_dir: Path,
) -> list[dict]:
    """Run multiple games for a single scenario.

    Args:
        scenario: Scenario name
        num_games: Number of games to run
        output_dir: Directory to save results

    Returns:
        List of game result dicts
    """
    print(f"\n{'#'*80}")
    print(f"# Running {num_games} game(s) for scenario: {scenario}")
    print(f"{'#'*80}")

    results = []
    for i in range(num_games):
        result = await run_single_game(scenario, i, output_dir)
        results.append(result)

    return results


async def run_all_scenarios(
    num_games: int,
    output_dir: Path,
    scenarios: list[str] = None,
) -> dict:
    """Run games for all scenarios (Table 3 ablation).

    Args:
        num_games: Number of games per scenario
        output_dir: Directory to save results
        scenarios: List of scenarios to run (default: all)

    Returns:
        Summary dict with all results
    """
    if scenarios is None:
        scenarios = SCENARIOS

    print(f"\n{'='*80}")
    print(f"Table 3 Ablation: Hardcoded Agents")
    print(f"{'='*80}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Games per scenario: {num_games}")
    print(f"Total games: {len(scenarios) * num_games}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    all_results = []

    # Run scenarios sequentially
    for scenario in scenarios:
        scenario_results = await run_scenario_games(scenario, num_games, output_dir)
        all_results.extend(scenario_results)

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    # Calculate summary statistics
    successful_games = [r for r in all_results if r["success"]]
    failed_games = [r for r in all_results if not r["success"]]

    summary = {
        "experiment": "table3_ablation_hardcoded",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_duration_seconds": total_duration,
        "scenarios": scenarios,
        "games_per_scenario": num_games,
        "total_games": len(all_results),
        "successful_games": len(successful_games),
        "failed_games": len(failed_games),
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
    summary_path = output_dir / "ablation_summary.json"
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
    parser = argparse.ArgumentParser(
        description="Table 3 Ablation: Hardcoded Always-Cooperate Agents"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=SCENARIOS + ["all"],
        default="all",
        help="Scenario to run (default: all)",
    )
    parser.add_argument(
        "--games",
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

    output_dir = Path(args.output_dir)

    # Determine which scenarios to run
    if args.scenario == "all":
        scenarios = SCENARIOS
    else:
        scenarios = [args.scenario]

    # Run evaluation
    asyncio.run(run_all_scenarios(
        num_games=args.games,
        output_dir=output_dir,
        scenarios=scenarios,
    ))


if __name__ == "__main__":
    main()
