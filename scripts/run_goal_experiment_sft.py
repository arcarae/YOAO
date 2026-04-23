"""Run Sugarscape experiments with SFT v2 fine-tuned model.

This script is identical to run_goal_experiment.py but uses the
redblackbench-qwen3-14b-sft-v2 LoRA adapter as the backbone model.

The vLLM server must be started with LoRA support first — see
VLLM_INTEGRATION.md for the server setup command.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sugarscape.simulation import SugarSimulation
from sugarscape.config import SugarscapeConfig


# SFT v2 LoRA adapter name (must match --lora-modules name in vLLM server)
SFT_MODEL_NAME = "redblackbench-qwen3-14b-sft-v2"


def run_goal_experiment(
    goal_preset: str,
    ticks: int = 300,
    seed: int = 42,
    population: int = 100,
    enable_trade: bool = True,
    checkpoint_interval: int = 0,
    provider_type: str = "vllm",
    model: str = None,
    vllm_url: str = "http://localhost:8000/v1"
):
    """Run a single experiment with a specific goal preset using SFT v2 model.

    Default settings optimized for emergent phenomena:
    - 100 agents (critical mass for complex interactions)
    - 300 ticks (enough time for patterns to emerge)
    - Trade enabled (economic emergence)

    Args:
        goal_preset: One of "none", "survival", "wealth", "egalitarian", "utilitarian"
        ticks: Number of simulation ticks to run
        seed: Random seed for reproducibility
        population: Number of agents
        enable_trade: Whether to enable trade system
        checkpoint_interval: Save checkpoint every N ticks (0 = no checkpoints)
        provider_type: LLM provider type (default: "vllm" for SFT model)
        model: Model name/path (default: SFT v2 LoRA adapter)
        vllm_url: vLLM server URL (only used if provider_type="vllm")
    """

    print(f"\n{'='*60}")
    print(f"Running Goal Experiment (SFT v2): {goal_preset.upper()}")
    print(f"{'='*60}")

    # Use SFT v2 model by default
    if model is None:
        if provider_type == "vllm":
            model = SFT_MODEL_NAME
        else:
            model = "openai/gpt-4o"

    # Configure simulation - optimized for emergent phenomena
    config = SugarscapeConfig(
        initial_population=population,
        max_ticks=ticks,
        width=50,  # Keep 50x50 for classic Sugarscape comparison
        height=50,
        seed=seed,
        enable_llm_agents=True,
        llm_agent_ratio=1.0,  # All agents are LLM
        llm_provider_type=provider_type,
        llm_provider_model=model,
        llm_vllm_base_url=vllm_url,
        llm_goal_preset=goal_preset,
        enable_spice=True,
        enable_trade=enable_trade,  # Enable trade for economic emergence
        trade_mode="dialogue",  # Use dialogue trade for richer interactions
        trade_dialogue_rounds=2,  # Reduced from 4 for faster experiments
        trade_allow_fraud=True,  # Allow deception for trust dynamics

        # Moderate resource scarcity for interesting dynamics
        sugar_growback_rate=1,
        max_sugar_capacity=4,
        spice_growback_rate=1,
        max_spice_capacity=4,

        # Debug logging enabled by default for analysis
        enable_debug_logging=True,
        debug_log_decisions=True,
        debug_log_llm=True,
        debug_log_trades=True,
        debug_log_deaths=True,
        debug_log_efficiency=True,
    )

    print(f"Goal: {goal_preset}")
    print(f"Provider: {provider_type} ({model})")
    print(f"Model Type: SFT v2 Fine-tuned")
    print(f"Goal Prompt: {config.llm_goal_prompt[:100]}...")
    print(f"Seed: {seed}, Ticks: {ticks}, Population: {population}")
    print(f"Trade: {'enabled' if enable_trade else 'disabled'}, Spice: enabled")
    if checkpoint_interval > 0:
        print(f"Checkpoints: every {checkpoint_interval} ticks")

    # Use sft_ prefix in experiment name to distinguish from base model runs
    sim = SugarSimulation(config=config, experiment_name=f"sft_goal_{goal_preset}")

    print("\nInitial Stats:")
    initial_stats = sim.get_stats()
    print(f"  Population: {initial_stats['population']}")
    print(f"  Mean Wealth: {initial_stats['mean_wealth']:.2f}")
    print(f"  Survival Rate: {initial_stats['survival_rate']:.2f}")

    print("\nRunning simulation...")

    # Use checkpoint-enabled run if interval specified
    if checkpoint_interval > 0:
        sim.run_with_checkpoints(steps=ticks, checkpoint_interval=checkpoint_interval)
    else:
        sim.run(steps=ticks)

    print("\nFinal Stats:")
    final_stats = sim.get_stats()
    print(f"  Population: {final_stats['population']}")
    print(f"  Mean Wealth: {final_stats['mean_wealth']:.2f}")
    print(f"  Survival Rate: {final_stats['survival_rate']:.2f}")
    print(f"  Utilitarian Welfare: {final_stats['utilitarian_welfare']:.2f}")
    print(f"  Nash Welfare: {final_stats['nash_welfare']:.2f}")
    print(f"  Rawlsian Welfare: {final_stats['rawlsian_welfare']:.2f}")
    print(f"  Welfare Gini: {final_stats['welfare_gini']:.3f}")
    print(f"  Mean Lifespan Utilization: {final_stats['mean_lifespan_utilization']:.3f}")

    return sim.logger.run_dir, final_stats


def resume_experiment(checkpoint_path: str, additional_ticks: int = None, checkpoint_interval: int = 50):
    """Resume an experiment from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file (.pkl)
        additional_ticks: Number of additional ticks to run (default: run to original max_ticks)
        checkpoint_interval: Save checkpoint every N ticks
    """
    print(f"\n{'='*60}")
    print(f"Resuming SFT Experiment from Checkpoint")
    print(f"{'='*60}")

    sim = SugarSimulation.load_checkpoint(checkpoint_path)

    print(f"\nResumed at tick {sim.tick}")
    print(f"Original max_ticks: {sim.config.max_ticks}")

    stats = sim.get_stats()
    print(f"\nCurrent Stats:")
    print(f"  Population: {stats['population']}")
    print(f"  Mean Wealth: {stats['mean_wealth']:.2f}")
    print(f"  Survival Rate: {stats['survival_rate']:.2f}")

    # Determine how many ticks to run
    if additional_ticks:
        steps = additional_ticks
    else:
        steps = sim.config.max_ticks - sim.tick

    if steps <= 0:
        print("Simulation already complete (at or beyond max_ticks)")
        return sim.logger.run_dir, stats

    print(f"\nRunning {steps} more ticks...")
    sim.run_with_checkpoints(steps=steps, checkpoint_interval=checkpoint_interval)

    print("\nFinal Stats:")
    final_stats = sim.get_stats()
    print(f"  Population: {final_stats['population']}")
    print(f"  Mean Wealth: {final_stats['mean_wealth']:.2f}")
    print(f"  Survival Rate: {final_stats['survival_rate']:.2f}")
    print(f"  Utilitarian Welfare: {final_stats['utilitarian_welfare']:.2f}")
    print(f"  Nash Welfare: {final_stats['nash_welfare']:.2f}")
    print(f"  Rawlsian Welfare: {final_stats['rawlsian_welfare']:.2f}")
    print(f"  Welfare Gini: {final_stats['welfare_gini']:.3f}")

    return sim.logger.run_dir, final_stats


def compare_goals(goals_to_test, ticks=300, seed=42, population=100, enable_trade=True,
                  checkpoint_interval=0, provider_type="vllm", model=None, vllm_url="http://localhost:8000/v1"):
    """Run experiments with multiple goal presets and compare results."""

    results = {}

    print(f"\n{'='*80}")
    print("GOAL COMPARISON EXPERIMENT (SFT v2) - Emergent Phenomena Mode")
    print(f"{'='*80}")
    print(f"Testing {len(goals_to_test)} goal presets with SFT v2 model")
    print(f"Config: {population} agents, {ticks} ticks, trade={'enabled' if enable_trade else 'disabled'}, seed={seed}")
    print(f"Provider: {provider_type}")
    print(f"Model: {model or SFT_MODEL_NAME}")

    for goal in goals_to_test:
        try:
            run_dir, final_stats = run_goal_experiment(
                goal, ticks, seed, population, enable_trade, checkpoint_interval,
                provider_type, model, vllm_url
            )
            results[goal] = {
                'run_dir': run_dir,
                'final_stats': final_stats
            }
        except Exception as e:
            print(f"Error running {goal}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY (SFT v2)")
    print(f"{'='*80}")

    print(f"{'Goal':<12}{'Gini':<8}{'Utilitarian':<12}{'Nash':<12}{'Rawlsian':<12}")
    print("-" * 80)

    for goal, data in results.items():
        stats = data['final_stats']
        print(f"{goal:<12}{stats['welfare_gini']:<8.2f}{stats['utilitarian_welfare']:<12.2f}"
              f"{stats['nash_welfare']:<12.2f}{stats['rawlsian_welfare']:<12.2f}")

    print("-" * 80)

    # Generate comparison plots if we have multiple results
    if len(results) >= 2:
        print("\nGenerating comparison plots...")

        # Create comparison directory
        try:
            from sugarscape.welfare_plots import WelfarePlotter
        except ImportError:
            print("Warning: Matplotlib not available. Skipping plot generation.")
            return results

        comparison_dir = Path("results/sugarscape/sft_goal_comparison")
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # Get CSV paths for the first two goals (or more if desired)
        csv_paths = []
        goal_names = []
        for goal, data in list(results.items())[:4]:  # Compare up to 4 goals
            csv_path = Path(data['run_dir']) / "metrics.csv"
            if csv_path.exists():
                csv_paths.append(str(csv_path))
                goal_names.append(goal)

        if len(csv_paths) >= 2:
            WelfarePlotter.generate_comparison_plots(
                llm_csv_path=csv_paths[0],
                baseline_csv_path=csv_paths[1],
                output_dir=str(comparison_dir)
            )

            print(f"Comparison plots saved to: {comparison_dir}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Sugarscape Goal Experiments with SFT v2 Fine-tuned Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script uses the SFT v2 fine-tuned LoRA adapter by default.
Make sure the vLLM server is running with LoRA support:
  ./scripts/setup_and_run.sh start

Examples:
  # Quick smoke test with SFT model (10 ticks)
  python run_goal_experiment_sft.py --single wealth --population 50 --ticks 10 --checkpoint-interval 5

  # Baseline experiment with SFT v2
  python run_goal_experiment_sft.py --single wealth --population 50 --ticks 100

  # Full emergent phenomena
  python run_goal_experiment_sft.py --single wealth --population 100 --ticks 300

  # Resume from checkpoint
  python run_goal_experiment_sft.py --resume /path/to/checkpoint_tick_50.pkl

  # Compare multiple goals
  python run_goal_experiment_sft.py --goals wealth egalitarian --population 100 --ticks 300
        """)

    parser.add_argument("--goals", nargs="+",
                        choices=["none", "survival", "wealth", "egalitarian", "utilitarian"],
                        default=["none", "wealth", "utilitarian"],
                        help="Goal presets to test (none = no explicit goal baseline)")

    parser.add_argument("--ticks", type=int, default=300,
                        help="Number of simulation ticks (default: 300 for emergent phenomena)")

    parser.add_argument("--population", type=int, default=100,
                        help="Number of agents (default: 100 for emergent phenomena)")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    parser.add_argument("--single", type=str,
                        choices=["none", "survival", "wealth", "egalitarian", "utilitarian"],
                        help="Run single goal experiment (overrides --goals)")

    parser.add_argument("--no-trade", action="store_true",
                        help="Disable trade (faster, but less emergent phenomena)")

    parser.add_argument("--checkpoint-interval", type=int, default=0,
                        help="Save checkpoint every N ticks (0 = no checkpoints, default: 0)")

    parser.add_argument("--resume", type=str, metavar="CHECKPOINT_PATH",
                        help="Resume from a checkpoint file (.pkl)")

    # Provider options - default to vllm for SFT model
    parser.add_argument("--provider", type=str, default="vllm",
                        choices=["openrouter", "vllm"],
                        help="LLM provider type (default: vllm for SFT model)")

    parser.add_argument("--model", type=str, default=None,
                        help=f"Model name/path (default: {SFT_MODEL_NAME})")

    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server URL (default: http://localhost:8000/v1)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Check for API key only if using OpenRouter
    if args.provider == "openrouter" and not args.resume:
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("ERROR: OPENROUTER_API_KEY environment variable not set!")
            print("Please export it: export OPENROUTER_API_KEY='sk-...'")
            print("Or use --provider vllm to use local vLLM server (default for SFT)")
            sys.exit(1)

    # Handle resume mode
    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
            sys.exit(1)

        # If --ticks specified with --resume, treat as additional ticks
        additional_ticks = args.ticks if args.ticks != 300 else None
        checkpoint_interval = args.checkpoint_interval if args.checkpoint_interval > 0 else 50

        resume_experiment(
            str(checkpoint_path),
            additional_ticks=additional_ticks,
            checkpoint_interval=checkpoint_interval
        )
        return

    enable_trade = not args.no_trade

    if args.single:
        # Run single experiment
        run_goal_experiment(
            args.single,
            args.ticks,
            args.seed,
            args.population,
            enable_trade,
            args.checkpoint_interval,
            args.provider,
            args.model,
            args.vllm_url
        )
    else:
        # Run comparison
        compare_goals(
            args.goals,
            args.ticks,
            args.seed,
            args.population,
            enable_trade,
            args.checkpoint_interval,
            args.provider,
            args.model,
            args.vllm_url
        )


if __name__ == "__main__":
    main()
