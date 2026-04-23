"""Command-line interface for RedBlackBench."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional
import yaml

from redblackbench.game.config import GameConfig
from redblackbench.game.coordinator import GameCoordinator
from redblackbench.agents.llm_agent import LLMAgent
from redblackbench.teams.team import Team
from redblackbench.logging.game_logger import GameLogger
from redblackbench.logging.metrics import MetricsCollector
from redblackbench.trajectory.collector import TrajectoryCollector
from redblackbench.scenarios import get_scenario, SCENARIOS


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_provider(provider_config: dict):
    """Create an LLM provider from configuration.
    
    Args:
        provider_config: Provider configuration dictionary
        
    Returns:
        Configured LLM provider
    """
    provider_type = provider_config.get("type", "openai")
    model = provider_config.get("model")
    temperature = provider_config.get("temperature", 0.7)
    api_key = provider_config.get("api_key")
    base_url = provider_config.get("base_url")
    # Enable reasoning capture for OpenRouter by default
    include_reasoning = provider_config.get("include_reasoning", True)

    if provider_type == "openai":
        from redblackbench.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(
            model=model or "gpt-4",
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )
    elif provider_type == "anthropic":
        from redblackbench.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            model=model or "claude-3-opus-20240229",
            temperature=temperature,
            api_key=api_key,
        )
    elif provider_type == "openrouter":
        from redblackbench.providers.openrouter_provider import OpenRouterProvider
        return OpenRouterProvider(
            model=model,
            temperature=temperature,
            api_key=api_key,
            include_reasoning=include_reasoning,
        )
    elif provider_type == "vllm":
        from redblackbench.providers.vllm_provider import VLLMProvider
        base_url = provider_config.get("base_url", "http://localhost:8000/v1")
        return VLLMProvider(
            model=model or "/workspace/models/Qwen3-14B",
            base_url=base_url,
            temperature=temperature,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


def create_team(
    team_config: dict,
    team_name: str,
    default_provider_config: dict,
    scenario=None,
    discussion_rounds: int = 1,
) -> tuple[Team, str]:
    """Create a team from configuration.

    Args:
        team_config: Team configuration dictionary
        team_name: Name for the team
        default_provider_config: Default provider config if not specified
        scenario: Optional scenario to use for prompts
        discussion_rounds: Number of discussion rounds before final vote

    Returns:
        Tuple of (Configured Team, model name string)
    """
    team_size = team_config.get("size", 5)
    provider_config = team_config.get("provider", default_provider_config)

    # Extract model name for tracking
    model_name = provider_config.get("model", "unknown")

    # Get prompt template from scenario if provided
    prompt_template = scenario.to_prompt_template() if scenario else None

    # Realistic advisor names for each team
    team_a_names = ["Dr. Sarah Chen", "Marcus Webb", "Dr. Priya Sharma", "James O'Connor", "Dr. Elena Vasquez",
                   "Michael Torres", "Dr. Aisha Patel", "Robert Kim", "Dr. Catherine Moore", "David Nguyen"]
    team_b_names = ["Dr. Thomas Berg", "Linda Okonkwo", "Dr. Raj Mehta", "Susan Clarke", "Dr. Antonio Silva",
                   "Jennifer Park", "Dr. William Foster", "Maria Santos", "Dr. Daniel Lee", "Rachel Adams"]

    names = team_a_names if "North" in team_name else team_b_names

    agents = []
    for i in range(team_size):
        agent_id = names[i % len(names)]
        provider = create_provider(provider_config)
        agent = LLMAgent(
            agent_id=agent_id,
            team_name=team_name,
            provider=provider,
            prompt_template=prompt_template,
        )
        agents.append(agent)

    return Team(name=team_name, agents=agents, discussion_rounds=discussion_rounds), model_name


async def run_experiment(config: dict, save_trajectory: bool = True, resume_from: Optional[str] = None) -> None:
    """Run an experiment based on configuration.

    Args:
        config: Experiment configuration dictionary
        save_trajectory: Whether to save full trajectory data
        resume_from: Optional path to trajectory file to resume from
    """
    # Extract configurations
    game_config_dict = config.get("game", {})
    team_a_config = config.get("team_a", {})
    team_b_config = config.get("team_b", {})
    default_provider = config.get("default_provider", {"type": "openai", "model": "gpt-4"})
    output_dir = config.get("output_dir", "results")
    experiment_name = config.get("experiment_name", "experiment")
    num_games = config.get("num_games", 1)

    # Load scenario if specified
    scenario_config = config.get("scenario", {})
    scenario_id = scenario_config.get("id") if scenario_config else None
    scenario = None
    if scenario_id:
        scenario = get_scenario(scenario_id)
        if scenario:
            print(f"Using scenario: {scenario.config.name} ({scenario_id})")
        else:
            print(f"Warning: Scenario '{scenario_id}' not found, using default prompts")
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trajectories_path = output_path / "trajectories"
    if save_trajectory:
        trajectories_path.mkdir(parents=True, exist_ok=True)
    
    # Create game configuration
    discussion_rounds = game_config_dict.get("discussion_rounds", 1)
    game_config = GameConfig(
        num_rounds=game_config_dict.get("num_rounds", 10),
        team_size=game_config_dict.get("team_size", 5),
        multipliers=game_config_dict.get("multipliers", {5: 3, 8: 5, 10: 10}),
        discussion_rounds=discussion_rounds,
    )
    if discussion_rounds > 1:
        print(f"Discussion rounds: {discussion_rounds} (agents will discuss before voting)")
    
    # Create logger
    logger = GameLogger(output_dir=output_dir, experiment_name=experiment_name)

    # Run games
    start_game = 1
    for game_num in range(start_game, num_games + 1):
        print(f"\n{'='*60}")
        print(f"Starting Game {game_num} of {num_games}")
        print(f"{'='*60}")

        # Create fresh teams for each game (using neutral nation names for realism)
        team_a, team_a_model = create_team(team_a_config, "Northland", default_provider, scenario=scenario, discussion_rounds=discussion_rounds)
        team_b, team_b_model = create_team(team_b_config, "Southland", default_provider, scenario=scenario, discussion_rounds=discussion_rounds)
        print(f"Team A model: {team_a_model}")
        print(f"Team B model: {team_b_model}")

        # Create a sanitized model identifier for filenames
        def sanitize_model_name(model: str) -> str:
            """Convert model name to safe filename component."""
            # Replace slashes and other unsafe chars, keep it short
            return model.replace("/", "_").replace(":", "_").replace(" ", "_")[:50]

        # Create trajectory collector if enabled
        trajectory_collector = None
        model_suffix = f"_{sanitize_model_name(team_a_model)}" if team_a_model != team_b_model else f"_{sanitize_model_name(team_a_model)}"
        if save_trajectory:
            trajectory_id = f"{experiment_name}{model_suffix}_game_{game_num}"
            trajectory_collector = TrajectoryCollector(trajectory_id=trajectory_id)

        # Set up trajectory save path for incremental saving - include model to prevent cross-model contamination
        trajectory_file = trajectories_path / f"{experiment_name}{model_suffix}_game_{game_num}.json" if save_trajectory else None

        # Create coordinator
        coordinator = GameCoordinator(
            team_a=team_a,
            team_b=team_b,
            config=game_config,
            logger=logger,
            trajectory_collector=trajectory_collector,
            trajectory_save_path=str(trajectory_file) if trajectory_file else None,
            team_a_model=team_a_model,
            team_b_model=team_b_model,
        )

        # Set up checkpoint path for crash recovery
        checkpoint_path = str(output_path / "checkpoints" / f"{experiment_name}_game_{game_num}_checkpoint.json")

        # Check if we should resume from existing trajectory
        resume_path = None
        if resume_from and game_num == 1:
            # Only resume for game 1, use provided path
            resume_path = resume_from
            print(f"Resuming from: {resume_path}")
        elif save_trajectory and trajectory_file and trajectory_file.exists():
            # Auto-resume if trajectory file exists
            resume_path = str(trajectory_file)
            print(f"Found existing trajectory, resuming from: {resume_path}")

        # Play the game
        try:
            final_state = await coordinator.play_game(resume_from_path=resume_path)
            
            # Print results
            summary = coordinator.get_summary()
            print(f"\nGame {game_num} Complete!")
            print(f"Team A ({summary['team_a']['name']}): {summary['team_a']['score']} points")
            print(f"Team B ({summary['team_b']['name']}): {summary['team_b']['score']} points")
            print(f"Combined Total: {summary['total_score']} / {summary['max_possible_score']}")
            print(f"Efficiency: {summary['efficiency']:.1%}")
            print(f"Cooperation Rate: {summary['cooperation_rate']:.1%}")

            # Save trajectory if enabled
            if save_trajectory and coordinator.get_trajectory():
                trajectory = coordinator.get_trajectory()
                trajectory_file = trajectories_path / f"{trajectory.trajectory_id}.json"
                trajectory.save(str(trajectory_file))
                print(f"Trajectory saved to: {trajectory_file}")

                # Print trajectory summary
                traj_summary = trajectory.get_summary()
                print(f"  Timesteps: {traj_summary.get('total_timesteps', 0)}")
                print(f"  Total Actions: {traj_summary.get('total_actions', 0)}")
                print(f"  Dialogue Exchanges: {traj_summary.get('total_dialogue_exchanges', 0)}")

            # Clean up checkpoint on successful completion
            checkpoint_file = Path(checkpoint_path)
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                trajectory_checkpoint = Path(checkpoint_path.replace(".json", "_trajectory.json"))
                if trajectory_checkpoint.exists():
                    trajectory_checkpoint.unlink()

        except Exception as e:
            print(f"Error during game {game_num}: {e}")
            print(f"\n[TIP] Partial results saved. Check: {checkpoint_path}")
            raise
    
    print(f"\n{'='*60}")
    print(f"Experiment Complete! Results saved to: {output_dir}")
    if save_trajectory:
        print(f"Trajectories saved to: {trajectories_path}")
    print(f"{'='*60}")


async def analyze_results(results_dir: str) -> None:
    """Analyze results from a directory of game logs.
    
    Args:
        results_dir: Path to directory containing game logs
    """
    collector = MetricsCollector()
    collector.load_from_directory(results_dir)
    
    print(collector.generate_summary())


async def analyze_trajectory(trajectory_path: str) -> None:
    """Analyze a single trajectory file.
    
    Args:
        trajectory_path: Path to trajectory JSON file
    """
    from redblackbench.trajectory import GameTrajectory
    
    trajectory = GameTrajectory.load(trajectory_path)
    
    print(f"\n{'='*60}")
    print(f"Trajectory Analysis: {trajectory.trajectory_id}")
    print(f"{'='*60}")
    
    print(f"\nGame Configuration:")
    for key, value in trajectory.game_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nTeams: {trajectory.team_a_name} vs {trajectory.team_b_name}")
    print(f"Start Time: {trajectory.start_time}")
    print(f"End Time: {trajectory.end_time}")
    
    # Summary stats
    summary = trajectory.get_summary()
    print(f"\nSummary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if "rate" in key or "efficiency" in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Action sequence
    actions = trajectory.get_action_sequence()
    print(f"\nAction Sequence ({len(actions)} total actions):")
    team_choices = [a for a in actions if a.action_type == "team_choice"]
    for action in team_choices:
        print(f"  Round {action.round_num}: {action.actor} chose {action.choice}")
    
    # Outcomes
    outcomes = trajectory.get_outcomes()
    print(f"\nRound Outcomes:")
    for outcome in outcomes:
        if outcome.outcome_type == "round":
            coop = "✓" if outcome.both_cooperated else ("✗" if outcome.both_defected else "~")
            print(f"  Round {outcome.round_num} ({outcome.multiplier}x): "
                  f"A={outcome.team_a_choice} ({outcome.team_a_score:+d}), "
                  f"B={outcome.team_b_choice} ({outcome.team_b_score:+d}) [{coop}]")
    
    if trajectory.final_outcome:
        fo = trajectory.final_outcome
        print(f"\nFinal Outcome:")
        print(f"  Team A Score: {fo.team_a_score}")
        print(f"  Team B Score: {fo.team_b_score}")
        print(f"  Total Score: {fo.total_score} / {fo.max_possible_score}")
        print(f"  Efficiency: {fo.efficiency:.1%}")
    print(f"  Cooperation Rate: {fo.cooperation_rate:.1%}")


def export_training_data(
    input_path: str,
    output_path: str,
    scenario_id: Optional[str] = None,
    logging_mode: str = "lite",
    auto_label: bool = False,
) -> None:
    """Export raw trajectories to rbbench.v1 training format.

    Args:
        input_path: Path to trajectory file or directory
        output_path: Path for output training data
        scenario_id: Scenario identifier
        logging_mode: 'lite' or 'full'
        auto_label: Whether to auto-generate training labels
    """
    from redblackbench.trajectory import GameTrajectory
    from redblackbench.training import TrainingDataExporter, label_trajectory

    input_p = Path(input_path)
    output_p = Path(output_path)

    exporter = TrainingDataExporter(logging_mode=logging_mode)

    if input_p.is_file():
        # Single file export
        print(f"Exporting {input_path} -> {output_path}")
        trajectory = GameTrajectory.load(str(input_p))
        training_traj = exporter.export(trajectory, scenario_id=scenario_id)

        if auto_label:
            training_traj = label_trajectory(training_traj)
            print("  Labels generated")

        output_p.parent.mkdir(parents=True, exist_ok=True)
        training_traj.save(str(output_p))
        print(f"  Saved: {output_p}")

    elif input_p.is_dir():
        # Batch export
        output_p.mkdir(parents=True, exist_ok=True)
        files = list(input_p.glob("*.json"))
        print(f"Exporting {len(files)} trajectories from {input_path}")

        for f in files:
            try:
                trajectory = GameTrajectory.load(str(f))
                training_traj = exporter.export(trajectory, scenario_id=scenario_id)

                if auto_label:
                    training_traj = label_trajectory(training_traj)

                out_file = output_p / f"training_{f.name}"
                training_traj.save(str(out_file))
                print(f"  Exported: {f.name}")
            except Exception as e:
                print(f"  Error exporting {f.name}: {e}")

        print(f"\nExported to: {output_path}")
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)


def generate_comparisons(
    input_dir: str,
    output_path: str,
    metrics: list,
    include_rounds: bool = False,
) -> None:
    """Generate comparisons.jsonl for preference learning.

    Args:
        input_dir: Directory containing training trajectory files
        output_path: Path for comparisons.jsonl output
        metrics: List of metrics to compare by
        include_rounds: Include round-level comparisons
    """
    from redblackbench.training import generate_comparisons_from_trajectories

    print(f"Generating comparisons from {input_dir}")
    print(f"  Metrics: {metrics}")
    print(f"  Include round comparisons: {include_rounds}")

    counts = generate_comparisons_from_trajectories(
        trajectory_dir=input_dir,
        output_path=output_path,
        metrics=metrics,
        include_round_comparisons=include_rounds,
    )

    print(f"\nComparisons generated:")
    for key, count in counts.items():
        print(f"  {key}: {count}")
    print(f"\nSaved to: {output_path}")


def generate_prompts_registry(output_path: str) -> None:
    """Generate prompts.json registry from all scenarios.

    Args:
        output_path: Path to save prompts.json
    """
    from redblackbench.training.prompts import create_registry_from_scenarios

    print(f"Generating prompts registry from all scenarios...")
    registry = create_registry_from_scenarios(output_path)
    print(f"  Registered {len(registry)} prompts")
    print(f"  Saved to: {output_path}")


def show_training_stats(input_dir: str) -> None:
    """Show statistics about training data.

    Args:
        input_dir: Directory containing training trajectory files
    """
    from redblackbench.training import TrainingTrajectory

    input_p = Path(input_dir)
    files = list(input_p.glob("*.json"))

    if not files:
        print(f"No training files found in {input_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Training Data Statistics: {input_dir}")
    print(f"{'='*60}\n")

    total_rounds = 0
    total_messages = 0
    total_trajectories = len(files)
    cooperation_rates = []
    efficiency_scores = []
    scenarios = set()

    for f in files:
        try:
            traj = TrainingTrajectory.load(str(f))
            total_rounds += len(traj.rounds)

            for r in traj.rounds:
                total_messages += len(r.team_a_deliberation)
                if r.team_b_deliberation:
                    total_messages += len(r.team_b_deliberation)

            if traj.final:
                coop_rate = traj.final.cooperation_rates.get("team_a", 0)
                cooperation_rates.append(coop_rate)

                max_score = traj.final.scores.get("max_possible", 150)
                total_score = traj.final.scores.get("sum", 0)
                if max_score > 0:
                    efficiency_scores.append(total_score / max_score)

            scenarios.add(traj.task.scenario_id)

        except Exception as e:
            print(f"  Warning: Could not load {f.name}: {e}")

    print(f"Trajectories: {total_trajectories}")
    print(f"Total Rounds: {total_rounds}")
    print(f"Total Messages: {total_messages}")
    print(f"Scenarios: {', '.join(scenarios) if scenarios else 'N/A'}")

    if cooperation_rates:
        avg_coop = sum(cooperation_rates) / len(cooperation_rates)
        print(f"\nTeam A Cooperation Rate:")
        print(f"  Average: {avg_coop:.1%}")
        print(f"  Min: {min(cooperation_rates):.1%}")
        print(f"  Max: {max(cooperation_rates):.1%}")

    if efficiency_scores:
        avg_eff = sum(efficiency_scores) / len(efficiency_scores)
        print(f"\nEfficiency (score/max):")
        print(f"  Average: {avg_eff:.1%}")
        print(f"  Min: {min(efficiency_scores):.1%}")
        print(f"  Max: {max(efficiency_scores):.1%}")

    # Check for labels
    labeled_count = 0
    for f in files:
        try:
            traj = TrainingTrajectory.load(str(f))
            if traj.labels and traj.labels.trajectory_quality:
                labeled_count += 1
        except:
            pass

    print(f"\nLabeled trajectories: {labeled_count}/{total_trajectories}")


async def generate_sft_training_data(
    input_path: str,
    output_path: str,
    model: str = "moonshotai/kimi-k2-thinking",
    api_key: Optional[str] = None,
    team: str = "team_a",
    max_per_round: int = 3,
) -> None:
    """Generate SFT training data using a thinking model.

    Args:
        input_path: Path to training trajectory JSON file
        output_path: Output path for SFT data (.json or .jsonl)
        model: Model for generating ideal responses
        api_key: OpenRouter API key
        team: Team to generate examples for
        max_per_round: Max examples per round
    """
    from redblackbench.training.sft_generator import generate_sft_data

    print(f"Generating SFT data from {input_path}")
    print(f"  Model: {model}")
    print(f"  Team: {team}")
    print(f"  Max per round: {max_per_round}")

    dataset = await generate_sft_data(
        trajectory_path=input_path,
        output_path=output_path,
        model=model,
        api_key=api_key,
        team=team,
        max_per_round=max_per_round,
    )

    print(f"\nGenerated {len(dataset.examples)} SFT examples")
    print(f"Saved to: {output_path}")


async def provider_check(provider: str, model: str, api_key: Optional[str], max_tokens: int = 64) -> None:
    """Check provider connectivity and attempt minimal completion."""
    if provider == "openrouter":
        from redblackbench.providers.openrouter_provider import OpenRouterProvider
        # Enable reasoning for check to verify it works
        prov = OpenRouterProvider(model=model, api_key=api_key, temperature=0.0, max_tokens=max_tokens, include_reasoning=True)
        # List a few models
        try:
            models = await prov._client.models.list()
            print(f"Models available (first 5): {[m.id for m in models.data[:5]]}")
        except Exception as e:
            print(f"Model listing failed: {e}")
        # Retrieve target model
        try:
            m = await prov._client.models.retrieve(model)
            print(f"Model retrieve OK: {m.id}")
        except Exception as e:
            print(f"Model retrieve failed for '{model}': {e}")
        try:
            resp = await prov.generate(system_prompt="ping", messages=[{"role":"user","content":"ping"}])
            print(f"Chat completion succeeded!")
            # Check for hidden thinking delimiters
            if "__THINKING_START__" in resp:
                print("✓ Reasoning/Thinking tokens captured successfully (hidden from final output)")
                print(f"Raw output preview: {resp[:100]}...")
            else:
                print("⚠ No reasoning tokens found in response (Model might not support it or didn't think)")
                print(f"Response preview: {resp[:80]}...")
        except Exception as e:
            print(f"Chat completion failed: {e}")
            print("If error code is 402, you need OpenRouter credits: https://openrouter.ai/settings/credits")
    else:
        print(f"Unsupported provider: {provider}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="RedBlackBench: Multi-Agent Game Theory Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment from config file
  redblackbench run --config experiments/configs/example.yaml
  
  # Run with trajectory collection disabled
  redblackbench run --config experiments/configs/example.yaml --no-trajectory
  
  # Analyze results
  redblackbench analyze --results-dir results/
  
  # Analyze a specific trajectory
  redblackbench trajectory --file results/trajectories/game_1.json
  
  # Quick test with default settings
  redblackbench run --quick-test
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file"
    )
    run_parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test game with default settings"
    )
    run_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for results"
    )
    run_parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable trajectory collection (saves memory/disk)"
    )
    run_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to trajectory file to resume from"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze experiment results")
    analyze_parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="results",
        help="Directory containing game log files"
    )
    
    # Trajectory command
    traj_parser = subparsers.add_parser("trajectory", help="Analyze a trajectory file")
    traj_parser.add_argument(
        "--file", "-f",
        type=str,
        required=True,
        help="Path to trajectory JSON file"
    )
    # Provider check command
    pc_parser = subparsers.add_parser("provider-check", help="Check provider connectivity")
    pc_parser.add_argument("--provider", required=True, type=str, help="Provider type (e.g., openrouter)")
    pc_parser.add_argument("--model", required=True, type=str, help="Model ID to test")
    pc_parser.add_argument("--api-key", required=False, type=str, help="API key override")
    pc_parser.add_argument("--max-tokens", required=False, type=int, default=64, help="Max output tokens for the check (default: 64)")

    # Scenarios command - list available scenarios
    scenarios_parser = subparsers.add_parser("scenarios", help="List available real-world scenarios")
    scenarios_parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed information about each scenario"
    )

    # Training data commands
    training_parser = subparsers.add_parser("training", help="Training data generation commands")
    training_subparsers = training_parser.add_subparsers(dest="training_command", help="Training subcommands")

    # training export - convert raw trajectories to training format
    export_parser = training_subparsers.add_parser("export", help="Export trajectories to training format")
    export_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input trajectory file or directory"
    )
    export_parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file or directory for training data"
    )
    export_parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario ID for the trajectories"
    )
    export_parser.add_argument(
        "--mode",
        type=str,
        choices=["lite", "full"],
        default="lite",
        help="Logging mode: 'lite' (Team A full, Team B summary) or 'full' (both teams full)"
    )
    export_parser.add_argument(
        "--label",
        action="store_true",
        help="Auto-generate training labels (rewards, adherence scores)"
    )

    # training compare - generate comparisons for preference learning
    compare_parser = training_subparsers.add_parser("compare", help="Generate comparisons for preference learning")
    compare_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Directory containing training trajectory files"
    )
    compare_parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for comparisons.jsonl"
    )
    compare_parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["collective_welfare", "cooperation_rate", "principle_adherence"],
        help="Metrics to compare by"
    )
    compare_parser.add_argument(
        "--include-rounds",
        action="store_true",
        help="Include round-level comparisons"
    )

    # training prompts - generate prompts.json registry
    prompts_parser = training_subparsers.add_parser("prompts", help="Generate prompts.json registry")
    prompts_parser.add_argument(
        "--output", "-o",
        type=str,
        default="prompts.json",
        help="Output path for prompts.json"
    )

    # training stats - show statistics about training data
    stats_parser = training_subparsers.add_parser("stats", help="Show training data statistics")
    stats_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Directory containing training trajectory files"
    )

    # training sft - generate SFT (input, output) pairs using thinking model
    sft_parser = training_subparsers.add_parser("sft", help="Generate SFT data using a thinking model")
    sft_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input training trajectory JSON file"
    )
    sft_parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for SFT data (.json or .jsonl)"
    )
    sft_parser.add_argument(
        "--model", "-m",
        type=str,
        default="moonshotai/kimi-k2-thinking",
        help="Model to generate ideal responses (default: moonshotai/kimi-k2-thinking)"
    )
    sft_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )
    sft_parser.add_argument(
        "--team",
        type=str,
        default="team_a",
        choices=["team_a", "team_b"],
        help="Which team to generate examples for (default: team_a)"
    )
    sft_parser.add_argument(
        "--max-per-round",
        type=int,
        default=3,
        help="Max examples to generate per round (default: 3)"
    )

    args = parser.parse_args()
    
    if args.command == "run":
        if args.config:
            config = load_config(args.config)
        elif args.quick_test:
            # Default config for quick testing
            config = {
                "experiment_name": "quick_test",
                "output_dir": args.output_dir,
                "num_games": 1,
                "default_provider": {
                    "type": "openai",
                    "model": "gpt-4",
                    "temperature": 0.7,
                },
                "game": {
                    "num_rounds": 10,
                    "team_size": 5,
                },
            }
        else:
            print("Error: Either --config or --quick-test is required")
            sys.exit(1)
        
        save_trajectory = not getattr(args, 'no_trajectory', False)
        resume_from = getattr(args, 'resume', None)
        asyncio.run(run_experiment(config, save_trajectory=save_trajectory, resume_from=resume_from))
        
    elif args.command == "analyze":
        asyncio.run(analyze_results(args.results_dir))
    
    elif args.command == "trajectory":
        asyncio.run(analyze_trajectory(args.file))
    elif args.command == "provider-check":
        asyncio.run(provider_check(args.provider, args.model, args.api_key, args.max_tokens))

    elif args.command == "scenarios":
        print("\n" + "="*60)
        print("Available Real-World Scenarios")
        print("="*60 + "\n")

        for scenario_id, scenario in SCENARIOS.items():
            print(f"  {scenario_id}")
            print(f"    Name: {scenario.config.name}")
            print(f"    Domain: {scenario.config.domain}")

            if getattr(args, 'details', False):
                print(f"    Roles: {scenario.config.team_a_role} vs {scenario.config.team_b_role}")
                print(f"    Choices: {scenario.config.humanity_choice_name} (cooperate) vs {scenario.config.tribe_choice_name} (defect)")
                print(f"    Principle: {scenario.config.constitution_line[:80]}...")
            print()

        print("Usage: redblackbench run --config experiments/configs/scenario_<name>.yaml")
        print("       redblackbench scenarios --details  # for more info")

    elif args.command == "training":
        if args.training_command == "export":
            export_training_data(
                input_path=args.input,
                output_path=args.output,
                scenario_id=args.scenario,
                logging_mode=args.mode,
                auto_label=args.label,
            )
        elif args.training_command == "compare":
            generate_comparisons(
                input_dir=args.input,
                output_path=args.output,
                metrics=args.metrics,
                include_rounds=getattr(args, 'include_rounds', False),
            )
        elif args.training_command == "prompts":
            generate_prompts_registry(output_path=args.output)
        elif args.training_command == "stats":
            show_training_stats(input_dir=args.input)
        elif args.training_command == "sft":
            asyncio.run(generate_sft_training_data(
                input_path=args.input,
                output_path=args.output,
                model=args.model,
                api_key=args.api_key,
                team=args.team,
                max_per_round=args.max_per_round,
            ))
        else:
            training_parser.print_help()
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
