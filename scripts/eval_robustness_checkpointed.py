#!/usr/bin/env python3
"""Evaluation Script with Mid-Game Checkpointing: Robustness to Opponent Strategies (Table 2)

Enhanced version that saves progress after each round and can resume interrupted games.
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
]

# Opponent strategies to test against
OPPONENT_STRATEGIES = [
    "always_defect",
    "early_exploiter_no_recovery",
    "defect_critical",
    "critical_exploiter",
    "late_betrayer",
    "mid_exploiter",
    "early_exploiter",
    "tit_for_tat",
    "mostly_cooperate",
    "always_cooperate",
]

DEFAULT_SCENARIO = "climate_cooperation"


def calculate_team_a_cooperation_rate(history: List[str]) -> float:
    """Calculate Team A's cooperation rate from move history."""
    if not history:
        return 0.0
    cooperate_count = sum(1 for move in history if move == "A")
    return cooperate_count / len(history)


def check_existing_trajectory(trajectory_path: Path) -> Optional[Dict[str, Any]]:
    """Check if a trajectory file exists and extract its completion status.

    Returns:
        Dict with 'completed_rounds' and 'is_complete' if file exists, None otherwise
    """
    if not trajectory_path.exists():
        return None

    try:
        with open(trajectory_path, 'r') as f:
            data = json.load(f)

        # Check timesteps to count completed rounds
        timesteps = data.get('timesteps', [])
        # Count unique rounds from ROUND_END timesteps
        completed_rounds = len([t for t in timesteps if t.get('timestep_type') == 'round_end'])
        is_complete = completed_rounds >= 10 or data.get('final_outcome') is not None

        return {
            'completed_rounds': completed_rounds,
            'is_complete': is_complete,
            'trajectory_path': trajectory_path,
        }
    except Exception as e:
        print(f"    Warning: Could not read trajectory {trajectory_path}: {e}")
        return None


def restore_agent_state_from_trajectory(
    agent: LLMAgent,
    agent_id: str,
    trajectory_data: Dict[str, Any]
) -> None:
    """Restore an agent's conversation history from trajectory data.

    Args:
        agent: The LLMAgent instance to restore
        agent_id: The agent's ID
        trajectory_data: The loaded trajectory JSON data
    """
    from redblackbench.trajectory import GameTrajectory

    # Find the latest timestep with team snapshots
    timesteps = trajectory_data.get('timesteps', [])
    for timestep in reversed(timesteps):
        team_a = timestep.get('team_a_snapshot')
        if team_a:
            # Find this agent in the team snapshot
            for agent_snap in team_a.get('agents', []):
                if agent_snap['agent_id'] == agent_id:
                    # Restore conversation history with thinking tags stripped
                    import re
                    history = agent_snap.get('conversation_history', [])
                    for exchange in history:
                        content = exchange['content']

                        # Strip <think> tags to prevent context overflow
                        if exchange['role'] == 'assistant':
                            # Remove <think>...</think> blocks
                            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                            # Handle unclosed thinking tags
                            content = re.sub(r'<think>.*$', '', content, flags=re.DOTALL)
                            content = content.strip()

                        agent.conversation_history.append({
                            'role': exchange['role'],
                            'content': content,
                        })
                    return


async def run_single_game(
    model: str,
    strategy_id: str,
    scenario_id: str,
    game_index: int,
    output_dir: Path,
    team_size: int = 5,
    checkpoint_interval: int = 2,
) -> Optional[Dict[str, Any]]:
    """Run a single game with round-by-round checkpointing.

    Args:
        checkpoint_interval: Save trajectory every N rounds (default: 2)

    Returns:
        Result dict with scores, cooperation rates, and metadata
    """
    try:
        # Setup paths
        results_dir = output_dir / "trajectories" / scenario_id
        results_dir.mkdir(parents=True, exist_ok=True)

        model_safe = model.replace("/", "_").replace(":", "_")
        trajectory_path = results_dir / f"{model_safe}_vs_{strategy_id}_game_{game_index}.json"

        # Check if game already completed
        existing = check_existing_trajectory(trajectory_path)
        if existing and existing['is_complete']:
            print(f"    ✓ Game already complete (10 rounds), loading results...")
            with open(trajectory_path, 'r') as f:
                data = json.load(f)

            # Extract results from round outcomes in timesteps
            timesteps = data.get('timesteps', [])
            round_outcomes = [ts['outcome'] for ts in timesteps if ts.get('outcome') and ts['outcome'].get('outcome_type') == 'round']

            team_a_cumulative = sum(o.get('team_a_score', 0) for o in round_outcomes)
            team_b_cumulative = sum(o.get('team_b_score', 0) for o in round_outcomes)

            # Calculate cooperation rate from choices
            team_a_choices = [o.get('team_a_choice') for o in round_outcomes]
            coop_count = sum(1 for c in team_a_choices if c == 'BLACK')
            coop_rate = coop_count / len(team_a_choices) if team_a_choices else 0

            return {
                "model": model,
                "strategy": strategy_id,
                "scenario": scenario_id,
                "game_index": game_index,
                "team_a_score": team_a_cumulative,
                "team_b_score": team_b_cumulative,
                "sum": team_a_cumulative + team_b_cumulative,
                "team_a_cooperation_rate": coop_rate,
                "trajectory_path": str(trajectory_path),
                "resumed": True,
            }

        # Check for partial progress and prepare to resume
        resume_from_round = None
        trajectory_data = None
        if existing and existing['completed_rounds'] > 0:
            resume_from_round = existing['completed_rounds'] + 1
            print(f"    → Resuming from round {resume_from_round}/10 (completed {existing['completed_rounds']} rounds)...")
            with open(trajectory_path, 'r') as f:
                trajectory_data = json.load(f)

        # Handle baseline vs scenarios
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

        # Determine provider type
        if model.startswith("/") or model.startswith("./"):
            provider_config = {
                "type": "vllm",
                "model": model,
                "base_url": "http://localhost:8000/v1",
                "temperature": 1.0,
            }
        else:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            provider_config = {
                "type": "openrouter",
                "model": model,
                "temperature": 1.0,
                "api_key": api_key,
            }

        # Create teams
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
            )
            agents_a.append(agent)
        team_a = Team(name="Team A", agents=agents_a)

        team_b = create_scripted_team(
            strategy_id=strategy_id,
            team_name=f"Team B ({strategy_id})",
        )

        collector = TrajectoryCollector()

        coordinator = GameCoordinator(
            team_a=team_a,
            team_b=team_b,
            config=config,
            trajectory_collector=collector,
        )

        # Restore state if resuming
        if resume_from_round and trajectory_data:
            print(f"    🔄 Restoring agent states and game progress...")
            # Restore agent conversation histories
            for agent in agents_a:
                restore_agent_state_from_trajectory(agent, agent.agent_id, trajectory_data)

            # Restore game state scores and history
            from redblackbench.game.scoring import Choice, RoundResult

            # Extract cumulative scores and history from timesteps
            team_a_cumulative = 0
            team_b_cumulative = 0

            for ts in trajectory_data.get('timesteps', []):
                if ts.get('outcome') and ts.get('timestep_type') == 'round_end':
                    outcome = ts['outcome']
                    # Extract this round's choices and scores
                    team_a_choice = Choice.BLACK if outcome.get('team_a_choice') == 'BLACK' else Choice.RED
                    team_b_choice = Choice.BLACK if outcome.get('team_b_choice') == 'BLACK' else Choice.RED
                    team_a_score = outcome.get('team_a_score', 0)
                    team_b_score = outcome.get('team_b_score', 0)
                    multiplier = outcome.get('multiplier', 1)

                    # Calculate this round's individual scores (not cumulative)
                    round_a_score = team_a_score - team_a_cumulative
                    round_b_score = team_b_score - team_b_cumulative
                    team_a_cumulative = team_a_score
                    team_b_cumulative = team_b_score

                    # Add to coordinator history
                    round_result = RoundResult(
                        round_num=outcome.get('round_num', len(coordinator.state.history) + 1),
                        team_a_choice=team_a_choice,
                        team_b_choice=team_b_choice,
                        team_a_score=round_a_score,
                        team_b_score=round_b_score,
                        multiplier=multiplier
                    )
                    coordinator.state.history.append(round_result)

            # Set cumulative scores
            coordinator.state.team_a_score = team_a_cumulative
            coordinator.state.team_b_score = team_b_cumulative

            # Set the coordinator to start from the resume round
            coordinator.state.current_round = resume_from_round - 1  # Will increment to resume_from_round
            print(f"    ✓ Restored state: Round {resume_from_round}, Scores A:{team_a_cumulative} B:{team_b_cumulative}")

        # Run game with checkpointing
        # Start trajectory collection
        if collector:
            if resume_from_round:
                # Load existing trajectory and continue
                from redblackbench.trajectory import GameTrajectory
                collector.trajectory = GameTrajectory.load(str(trajectory_path))
            else:
                # Start new trajectory
                collector.start_game(config, team_a, team_b)

        # Play rounds manually with checkpoint saving
        start_round = resume_from_round or 1
        while not coordinator.state.is_complete:
            await coordinator.play_round()
            current_round = coordinator.state.current_round

            # Save checkpoint every N rounds
            if current_round % checkpoint_interval == 0 and collector.trajectory:
                collector.trajectory.team_a_model = model
                collector.trajectory.team_b_model = f"scripted:{strategy_id}"
                collector.get_trajectory().save(str(trajectory_path))
                print(f"    💾 Checkpoint saved at round {current_round}/10")

        # Final save
        if collector.trajectory:
            collector.trajectory.team_a_model = model
            collector.trajectory.team_b_model = f"scripted:{strategy_id}"
            collector.get_trajectory().save(str(trajectory_path))

        # Extract results
        final_scores = {
            "Team A": coordinator.state.team_a_total,
            "Team B": coordinator.state.team_b_total,
        }
        # Get team history from state
        team_a_history = ["A" if r.team_a_choice.value == "BLACK" else "B" for r in coordinator.state.history]

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
        print(f"    Game timed out")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_progress(output_dir: Path) -> Tuple[Dict[str, Any], set]:
    """Load existing progress from checkpoint."""
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

    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))

    for game in games:
        model = game["model"]
        strategy = game["strategy"]
        grouped[model][strategy].append(game["team_a_cooperation_rate"])

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
        description="Evaluate model robustness (Table 2) with checkpointing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, help="Specific model to evaluate")
    parser.add_argument("--strategy", type=str, help="Specific strategy to test against")
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    parser.add_argument("--output-dir", type=str, default="results/eval_robustness", help="Output directory")
    parser.add_argument("--games-per-combo", type=int, default=1)
    parser.add_argument("--team-size", type=int, default=5)
    parser.add_argument("--max-parallel", type=int, default=3, help="Maximum parallel games")
    parser.add_argument("--checkpoint-interval", type=int, default=2, help="Save checkpoint every N rounds")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else DEFAULT_MODELS
    strategies = [args.strategy] if args.strategy else OPPONENT_STRATEGIES

    if args.resume:
        results, completed = load_progress(output_dir)
    else:
        results = {"games": [], "summary": {}}
        completed = set()

    print(f"\n{'='*60}")
    print("EVALUATION: Robustness (Table 2) with Checkpointing")
    print(f"{'='*60}")
    print(f"Models: {len(models)}")
    print(f"Strategies: {len(strategies)}")
    print(f"Games per combo: {args.games_per_combo}")
    print(f"Scenario: {args.scenario}")
    print(f"Max parallel: {args.max_parallel}")
    print(f"Checkpoint interval: every {args.checkpoint_interval} rounds")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    total_combos = len(models) * len(strategies) * args.games_per_combo

    # Build task list
    tasks_to_run = []
    for model in models:
        for strategy in strategies:
            for game_idx in range(args.games_per_combo):
                key = (model, strategy, game_idx)
                if key not in completed:
                    tasks_to_run.append((model, strategy, game_idx, key))

    print(f"Tasks to run: {len(tasks_to_run)}/{total_combos}\n")

    # Run with semaphore
    semaphore = asyncio.Semaphore(args.max_parallel)

    async def run_with_semaphore(model: str, strategy: str, game_idx: int, key: tuple, task_num: int):
        async with semaphore:
            print(f"[{task_num}/{len(tasks_to_run)}] {model} vs {strategy} (game {game_idx})...")

            result = await run_single_game(
                model=model,
                strategy_id=strategy,
                scenario_id=args.scenario,
                game_index=game_idx,
                output_dir=output_dir,
                team_size=args.team_size,
                checkpoint_interval=args.checkpoint_interval,
            )

            if result:
                results["games"].append(result)
                completed.add(key)
                coop_rate = result["team_a_cooperation_rate"] * 100
                resumed_marker = " (loaded from checkpoint)" if result.get("resumed") else ""
                print(f"    ✓ [{task_num}/{len(tasks_to_run)}] {model} vs {strategy}: Score {result['team_a_score']} vs {result['team_b_score']} | Coop: {coop_rate:.0f}%{resumed_marker}")

                save_progress(output_dir, results)
            else:
                print(f"    ✗ [{task_num}/{len(tasks_to_run)}] {model} vs {strategy} FAILED")

            return result

    tasks = [
        run_with_semaphore(model, strategy, game_idx, key, i+1)
        for i, (model, strategy, game_idx, key) in enumerate(tasks_to_run)
    ]

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    # Generate summary
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}\n")

    summary_table = generate_summary_table(results)
    print(summary_table)

    summary_path = output_dir / "summary_table.md"
    with open(summary_path, 'w') as f:
        f.write("# Table 2: Robustness to Opponent Strategies (Cooperation Rate %)\\n\n")
        f.write(f"*Agent Team (Team A) vs hardcoded opponent (Team B)*\n\n")
        f.write(f"Scenario: {args.scenario}\n\n")
        f.write(summary_table)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Full results saved to: {output_dir / 'eval_robustness_progress.json'}")


if __name__ == "__main__":
    asyncio.run(main())
