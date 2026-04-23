#!/usr/bin/env python3
"""
Table 3 Ablation: Trained Agent Removal

This script tests whether cooperative behavior persists after trained agents
are removed mid-game.

Process:
1. Start with trained agents + untrained agents (mixed team)
2. Run for N rounds (default: 3) with trained agents present
3. Remove trained agents from the team
4. Continue game with only untrained agents
5. Measure cooperation rates before and after removal

Purpose: Determine if trained agents' influence on cooperation is sustained
after they leave, or if untrained agents revert to defection.

Compositions tested:
- 1 trained + 4 untrained (remove after round N)
- 2 trained + 3 untrained (remove after round N)
- 3 trained + 2 untrained (remove after round N)

Usage:
    python eval_table3_ablation_removal.py --scenario baseline --removal-round 3 --games 3
    python eval_table3_ablation_removal.py --trained-count 2 --scenario pandemic_vaccines
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
from redblackbench.game.scoring import Choice
from redblackbench.teams.team import Team
from redblackbench.scenarios import get_scenario
from redblackbench.trajectory import TrajectoryCollector
from redblackbench.agents.llm_agent import LLMAgent
from redblackbench.cli import create_provider
from redblackbench.strategies import create_scripted_team


# Agent names
AGENT_NAMES = [
    "Dr. Sarah Chen",
    "Marcus Webb",
    "Dr. Priya Sharma",
    "James O'Connor",
    "Dr. Elena Vasquez",
]

# Scenarios
SCENARIOS = [
    "climate_cooperation",
    "agi_safety",
    "pandemic_vaccines",
    "election_crisis",
    "standards_coordination",
    "baseline",
]

# Compositions: (num_trained, num_untrained)
COMPOSITIONS = [
    (1, 4),  # 1 trained + 4 untrained
    (2, 3),  # 2 trained + 3 untrained
    (3, 2),  # 3 trained + 2 untrained
]

# Model configuration
TRAINED_MODEL = "qwen3-14b-v2"  # LoRA fine-tuned model
UNTRAINED_MODEL = "/workspace/models/Qwen3-14B"
VLLM_BASE_URL = "http://localhost:8000/v1"


def get_choice_names(scenario: Optional[str]) -> tuple[str, str]:
    """Get scenario-specific choice names."""
    if scenario is None or scenario == "baseline":
        return ("BLACK", "RED")

    scenario_obj = get_scenario(scenario)
    if scenario_obj:
        config = scenario_obj.config
        return (config.humanity_choice_name, config.tribe_choice_name)

    return ("BLACK", "RED")


async def run_single_game(
    num_trained: int,
    num_untrained: int,
    scenario: str,
    removal_round: int,
    game_index: int,
    output_dir: Path,
    trained_model: str,
    untrained_model: str,
    vllm_base_url: str,
    enable_thinking: bool = False,
) -> dict:
    """
    Run a single game with agent removal.

    Args:
        num_trained: Number of trained agents to start with
        num_untrained: Number of untrained agents
        scenario: Scenario name
        removal_round: Round number after which to remove trained agents
        game_index: Game number for this configuration
        output_dir: Directory for output files
        trained_model: Model for trained agents
        untrained_model: Model for untrained agents
        vllm_base_url: vLLM server URL

    Returns:
        Dictionary with game results
    """
    print(f"================================================================================")
    print(f"Game {game_index + 1} - {num_trained}T+{num_untrained}U - Scenario: {scenario}")
    print(f"================================================================================")
    print(f"Removal round: {removal_round}")
    print()

    # Get scenario configuration
    black_name, red_name = get_choice_names(scenario)

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
                "scenario": scenario,
                "game_index": game_index,
                "num_trained": num_trained,
                "num_untrained": num_untrained,
            }
        prompt_template = scenario_obj.to_prompt_template()

    # Create game configuration
    config = GameConfig(
        num_rounds=10,
        multipliers={5: 3, 8: 5, 10: 10},
    )

    # Create Team A: Mixed trained + untrained agents
    agents_a = []

    # Add trained agents first (these will be removed)
    for i in range(num_trained):
        # Use LoRA fine-tuned model for trained agents
        provider = create_provider({
            "type": "openai",
            "model": trained_model,
            "temperature": 0.7,
            "base_url": vllm_base_url,
        })
        agent = LLMAgent(
            agent_id=AGENT_NAMES[i],
            team_name="Team A",
            provider=provider,
            prompt_template=prompt_template,
            enable_thinking=enable_thinking,
        )
        agent._is_trained = True
        agents_a.append(agent)

    # Add untrained agents (these will remain)
    for i in range(num_untrained):
        idx = num_trained + i
        provider = create_provider({
            "type": "openai",
            "model": untrained_model,
            "temperature": 0.7,  # Match trained agents' temperature for consistency
            "base_url": vllm_base_url,
        })
        agent = LLMAgent(
            agent_id=AGENT_NAMES[idx],
            team_name="Team A",
            provider=provider,
            prompt_template=prompt_template,
            enable_thinking=enable_thinking,
        )
        agent._is_trained = False
        agents_a.append(agent)

    team_a = Team("Team A", agents_a)

    # Team B: Always defect (baseline)
    team_b = create_scripted_team(
        strategy_id="always_defect",
        team_name="Team B (always_defect)",
    )

    # Create trajectory collector
    trajectory_collector = TrajectoryCollector()

    # Create coordinator
    coordinator = GameCoordinator(
        config=config,
        team_a=team_a,
        team_b=team_b,
        trajectory_collector=trajectory_collector,
    )

    # Run game with mid-game agent removal
    print("Starting game...\n")
    start_time = datetime.now()

    try:
        # Play rounds before removal
        for round_num in range(1, removal_round + 1):
            print(f"[Round {round_num}] Starting round with multiplier {config.get_multiplier(round_num)}x")

            # Play the round using coordinator
            round_result = await coordinator.play_round()

            print(f"[Round {round_num}] Choices: Team_A={round_result.team_a_choice.name}, Team_B={round_result.team_b_choice.name}")
            print(f"[Round {round_num}] Scores: Team_A={round_result.team_a_score:+d}, Team_B={round_result.team_b_score:+d} (multiplier {config.get_multiplier(round_num)}x)")
            print(f"[Totals] Team_A={coordinator.state.team_a_total}, Team_B={coordinator.state.team_b_total}, Combined={coordinator.state.total_score} / {coordinator.state.max_possible_score}")
            print()

        # Calculate cooperation BEFORE removal
        before_removal_coop = sum(
            1 for r in coordinator.state.history if r.team_a_choice == Choice.BLACK
        ) / len(coordinator.state.history) if coordinator.state.history else 0

        print(f"{'='*80}")
        print(f"REMOVING TRAINED AGENTS AFTER ROUND {removal_round}")
        print(f"{'='*80}")
        print(f"Cooperation rate before removal: {before_removal_coop * 100:.1f}%")
        print(f"Removing {num_trained} trained agents: {', '.join(a.agent_id for a in agents_a[:num_trained])}")
        print()

        # Remove trained agents from team
        trained_agents_removed = team_a.agents[:num_trained]
        team_a.agents = team_a.agents[num_trained:]  # Keep only untrained agents
        team_a.deliberation.agents = team_a.agents  # Update deliberation agent list

        print(f"Remaining agents: {', '.join(a.agent_id for a in team_a.agents)}")
        print(f"{'='*80}\n")

        # Continue game with only untrained agents
        for round_num in range(removal_round + 1, config.num_rounds + 1):
            print(f"[Round {round_num}] Starting round with multiplier {config.get_multiplier(round_num)}x (POST-REMOVAL)")

            # Play the round using coordinator
            round_result = await coordinator.play_round()

            print(f"[Round {round_num}] Choices: Team_A={round_result.team_a_choice.name}, Team_B={round_result.team_b_choice.name}")
            print(f"[Round {round_num}] Scores: Team_A={round_result.team_a_score:+d}, Team_B={round_result.team_b_score:+d} (multiplier {config.get_multiplier(round_num)}x)")
            print(f"[Totals] Team_A={coordinator.state.team_a_total}, Team_B={coordinator.state.team_b_total}, Combined={coordinator.state.total_score} / {coordinator.state.max_possible_score}")
            print()

        coordinator.state.is_complete = True

    except Exception as e:
        print(f"Error during game: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "scenario": scenario,
            "game_index": game_index,
            "num_trained": num_trained,
            "num_untrained": num_untrained,
        }

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Calculate final statistics
    game_state = coordinator.state

    # Split history into before and after removal
    before_removal_rounds = game_state.history[:removal_round]
    after_removal_rounds = game_state.history[removal_round:]

    # Calculate cooperation rates
    before_coop = sum(1 for r in before_removal_rounds if r.team_a_choice == Choice.BLACK) / len(before_removal_rounds) if before_removal_rounds else 0
    after_coop = sum(1 for r in after_removal_rounds if r.team_a_choice == Choice.BLACK) / len(after_removal_rounds) if after_removal_rounds else 0
    overall_coop = sum(1 for r in game_state.history if r.team_a_choice == Choice.BLACK) / len(game_state.history)

    team_a_score = game_state.team_a_total
    team_b_score = game_state.team_b_total
    total_score = game_state.total_score
    max_possible = game_state.max_possible_score
    efficiency = (total_score / max_possible * 100) if max_possible > 0 else 0

    print(f"================================================================================")
    print(f"Game {game_index + 1} Complete - {num_trained}T+{num_untrained}U")
    print(f"================================================================================")
    print(f"Duration: {duration:.1f}s")
    print(f"Team A: {team_a_score}")
    print(f"Team B (Defect): {team_b_score}")
    print(f"Total: {total_score}/{max_possible} ({efficiency:.1f}% efficiency)")
    print()
    print(f"Cooperation rate (Rounds 1-{removal_round}, WITH trained):  {before_coop * 100:.1f}%")
    print(f"Cooperation rate (Rounds {removal_round+1}-10, WITHOUT trained): {after_coop * 100:.1f}%")
    print(f"Overall cooperation rate: {overall_coop * 100:.1f}%")
    print(f"Cooperation change: {(after_coop - before_coop) * 100:+.1f}%")
    print(f"================================================================================\n")

    # Save trajectory
    trajectory = trajectory_collector.get_trajectory()
    scenario_dir = output_dir / "trajectories" / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = scenario_dir / f"removal_{num_trained}t_{num_untrained}u_game_{game_index}.json"
    if trajectory:
        traj_dict = trajectory.get_full_trajectory()
        traj_dict["removal_round"] = removal_round
        traj_dict["num_trained"] = num_trained
        traj_dict["num_untrained"] = num_untrained
        traj_dict["trained_model"] = trained_model
        traj_dict["untrained_model"] = untrained_model
        traj_dict["cooperation_before_removal"] = before_coop
        traj_dict["cooperation_after_removal"] = after_coop
        traj_dict["cooperation_change"] = after_coop - before_coop

        with open(trajectory_path, "w") as f:
            json.dump(traj_dict, f, indent=2)

    return {
        "success": True,
        "scenario": scenario,
        "game_index": game_index,
        "num_trained": num_trained,
        "num_untrained": num_untrained,
        "removal_round": removal_round,
        "team_a_score": team_a_score,
        "team_b_score": team_b_score,
        "total_score": total_score,
        "efficiency": efficiency,
        "cooperation_before_removal": before_coop,
        "cooperation_after_removal": after_coop,
        "cooperation_change": after_coop - before_coop,
        "overall_cooperation": overall_coop,
        "duration": duration,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Table 3 Ablation: Trained Agent Removal Experiment"
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS + ["all"],
        default="all",
        help="Scenario to run (default: all)",
    )
    parser.add_argument(
        "--trained-count",
        type=int,
        choices=[1, 2, 3, None],
        default=None,
        help="Number of trained agents (default: test all compositions)",
    )
    parser.add_argument(
        "--removal-round",
        type=int,
        default=3,
        help="Round after which to remove trained agents (default: 3)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=3,
        help="Number of games per combination (default: 3)",
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
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking/reasoning mode for LLM agents (Qwen3, etc.)",
    )

    args = parser.parse_args()

    # Determine compositions to test
    if args.trained_count is not None:
        num_untrained = 5 - args.trained_count
        compositions = [(args.trained_count, num_untrained)]
    else:
        compositions = COMPOSITIONS

    # Determine scenarios to test
    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Table 3 Ablation: Trained Agent Removal")
    print(f"{'='*80}")
    print(f"Compositions: {len(compositions)}")
    for num_t, num_u in compositions:
        print(f"  - {num_t} trained + {num_u} untrained (remove trained after round {args.removal_round})")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Games per combination: {args.games}")
    print(f"Total combinations: {len(compositions) * len(scenarios)}")
    print(f"Total games: {len(compositions) * len(scenarios) * args.games}")
    print(f"Output directory: {output_dir}")
    print(f"Trained model: {args.trained_model}")
    print(f"Untrained model: {args.untrained_model}")
    print(f"{'='*80}\n")

    # Run experiments
    all_results = []
    start_time = datetime.now()

    for num_trained, num_untrained in compositions:
        for scenario in scenarios:
            print(f"\n{'#'*80}")
            print(f"# Running {args.games} game(s): {num_trained}T+{num_untrained}U - {scenario}")
            print(f"{'#'*80}\n")

            for game_idx in range(args.games):
                result = await run_single_game(
                    num_trained=num_trained,
                    num_untrained=num_untrained,
                    scenario=scenario,
                    removal_round=args.removal_round,
                    game_index=game_idx,
                    output_dir=output_dir,
                    trained_model=args.trained_model,
                    untrained_model=args.untrained_model,
                    vllm_base_url=args.vllm_base_url,
                    enable_thinking=args.enable_thinking,
                )
                all_results.append(result)

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    # Aggregate statistics
    successful = [r for r in all_results if r["success"]]
    failed = [r for r in all_results if not r["success"]]

    print(f"\n{'='*80}")
    print(f"Ablation Experiment Complete")
    print(f"{'='*80}")
    print(f"Total time: {total_duration:.1f}s ({total_duration / 60:.1f}m)")
    print(f"Successful games: {len(successful)}/{len(all_results)}")
    print(f"Failed games: {len(failed)}")

    if successful:
        # Aggregate by composition
        from collections import defaultdict
        comp_stats = defaultdict(lambda: {
            "games": 0,
            "before_coop": [],
            "after_coop": [],
            "change": [],
        })

        for r in successful:
            key = f"{r['num_trained']}T+{r['num_untrained']}U"
            comp_stats[key]["games"] += 1
            comp_stats[key]["before_coop"].append(r["cooperation_before_removal"])
            comp_stats[key]["after_coop"].append(r["cooperation_after_removal"])
            comp_stats[key]["change"].append(r["cooperation_change"])

        print(f"\nAggregate Statistics:")
        for comp in sorted(comp_stats.keys()):
            stats = comp_stats[comp]
            avg_before = sum(stats["before_coop"]) / len(stats["before_coop"]) * 100
            avg_after = sum(stats["after_coop"]) / len(stats["after_coop"]) * 100
            avg_change = sum(stats["change"]) / len(stats["change"]) * 100

            print(f"\n{comp}:")
            print(f"  Games: {stats['games']}")
            print(f"  Avg cooperation (WITH trained):    {avg_before:.1f}%")
            print(f"  Avg cooperation (WITHOUT trained): {avg_after:.1f}%")
            print(f"  Avg change:                         {avg_change:+.1f}%")

    # Save summary
    summary = {
        "experiment": "table3_ablation_removal",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_duration_seconds": total_duration,
        "removal_round": args.removal_round,
        "compositions": compositions,
        "scenarios": scenarios,
        "games_per_combination": args.games,
        "total_games": len(all_results),
        "successful_games": len(successful),
        "failed_games": len(failed),
        "trained_model": args.trained_model,
        "untrained_model": args.untrained_model,
        "results": all_results,
    }

    summary_path = output_dir / "removal_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
