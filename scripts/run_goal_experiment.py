"""Run Sugarscape experiments with different LLM agent goals to study their impact on welfare metrics."""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sugarscape.simulation import SugarSimulation
from sugarscape.config import SugarscapeConfig
from sugarscape import presets

# Default model names on the local vLLM server (override with YOAO_BASE_MODEL / YOAO_SFT_MODEL /
# YOAO_VLLM_URL). With --provider openrouter, pass an OpenRouter model id via --model instead.
QWEN3_14B_BASE = os.environ.get("YOAO_BASE_MODEL", "Qwen/Qwen3-14B")
QWEN3_14B_SFT = os.environ.get("YOAO_SFT_MODEL", "redblackbench-qwen3-14b-sft-v2")
VLLM_URL = os.environ.get("YOAO_VLLM_URL", "http://localhost:8000/v1")


def run_goal_experiment(goal_preset: str, ticks: int = 100, seed: int = 42,
                       model: str = QWEN3_14B_BASE,
                       population: int = 100, width: int = 30, height: int = 30,
                       difficulty: str = "standard", trade_rounds: int = 4,
                       provider: str = "vllm", use_lora: bool = False,
                       use_mixed_identity: bool = False,
                       identity_distribution: dict = None,
                       enable_survival_pressure: bool = True,
                       social_memory_visible: bool = True,
                       experiment_name_suffix: str = "",
                       # LLM Evaluation options
                       enable_llm_evaluation: bool = True,
                       llm_evaluator_model: str = "openai/gpt-4o-mini",
                       llm_evaluator_provider: str = "openrouter",
                       # Resource abundance options
                       initial_wealth_min: int = None,
                       initial_wealth_max: int = None,
                       growback_rate: int = None,
                       # Paper environment and extra evaluator wiring
                       paper_env: bool = False,
                       resource_specialization: bool = None,
                       moral_evaluator_model: str = None,
                       moral_evaluator_provider: str = None,
                       vllm_url: str = VLLM_URL):
    """Run a single experiment with a specific goal preset.

    Ablation flags:
        enable_survival_pressure: If False, agents don't die from starvation (only old age)
        social_memory_visible: If False, agents can't see trade history or partner reputation
    paper_env: start from ``sugarscape.presets.paper_config`` (20x20, endowments 45-85,
        metabolism specialisation, 100 ticks, vLLM evaluators) and apply the other arguments on top.
    """

    print(f"\n{'='*60}")
    print(f"Running Goal Experiment: {goal_preset.upper()}")
    print(f"{'='*60}")

    # Use the cooperative LoRA adapter if specified
    if use_lora:
        model = QWEN3_14B_SFT
        print(f"Using cooperative SFT adapter: {model}")

    if paper_env:
        config = presets.paper_config(
            provider=provider,
            model=model,
            vllm_url=vllm_url,
            goal_preset=goal_preset,
            identity_distribution=identity_distribution,
            ticks=ticks,
            population=population,
            seed=seed,
            enable_llm_evaluation=enable_llm_evaluation,
            evaluator_provider=llm_evaluator_provider,
            evaluator_model=llm_evaluator_model,
            moral_evaluator_provider=moral_evaluator_provider or "vllm",
            moral_evaluator_model=moral_evaluator_model,
            resource_specialization=True if resource_specialization is None else resource_specialization,
        )
        config.enable_survival_pressure = enable_survival_pressure
        config.social_memory_visible = social_memory_visible
        config.trade_dialogue_rounds = trade_rounds
        width, height = config.width, config.height
    else:
        # Configure simulation
        config = SugarscapeConfig(
            initial_population=population,
            max_ticks=ticks,
            width=width,
            height=height,
            seed=seed,
            enable_llm_agents=True,
            llm_agent_ratio=1.0,  # All agents are LLM
            llm_provider_type=provider,
            llm_provider_model=model,
            llm_vllm_base_url=vllm_url,
            llm_goal_preset=goal_preset,
            enable_spice=True,
            enable_trade=True,  # Enable trade for goal experiments
            trade_dialogue_rounds=trade_rounds,
            # Trade robustness features to reduce timeouts
            trade_dialogue_repair_json=True,
            trade_dialogue_repair_attempts=2,
            trade_dialogue_coerce_protocol=True,  # Force valid actions to prevent timeouts
            trade_dialogue_two_stage=True,
            # Explicitly enable small talk and new encounter protocol
            enable_new_encounter_protocol=True,
            small_talk_rounds=2,
            # Ablation flags
            enable_survival_pressure=enable_survival_pressure,
            social_memory_visible=social_memory_visible,
            # LLM Evaluation settings
            enable_llm_evaluation=enable_llm_evaluation,
            llm_evaluator_model=llm_evaluator_model,
            llm_evaluator_provider=llm_evaluator_provider,
        )
        if resource_specialization is not None:
            config.enable_resource_specialization = resource_specialization
        if moral_evaluator_provider:
            config.external_moral_evaluator_provider = moral_evaluator_provider
        if moral_evaluator_model:
            config.external_moral_evaluator_model = moral_evaluator_model

    # Print ablation status
    if not enable_survival_pressure:
        print("⚠️ ABLATION: Survival pressure DISABLED (agents won't die from starvation)")
    if not social_memory_visible:
        print("⚠️ ABLATION: Social memory DISABLED (no trade history, reputation hidden)")
    
    # Print evaluation status
    if enable_llm_evaluation:
        print(f"✓ LLM Evaluation ENABLED (model: {llm_evaluator_model})")
    else:
        print("⚠️ LLM Evaluation DISABLED (behavioral metrics only)")

    if use_mixed_identity or identity_distribution:
        config.enable_origin_identity = True
        if identity_distribution:
            config.origin_identity_distribution = identity_distribution
            dist_str = ", ".join(f"{k}: {v*100:.0f}%" for k, v in identity_distribution.items())
            print(f"Enabled Origin Identity System ({dist_str})")
        else:
            # Default distribution
            print("Enabled Origin Identity System (80% Exploiter / 20% Altruist)")

    # Apply difficulty preset
    if difficulty == "easy":
        config.sugar_growback_rate = 2
        config.max_sugar_capacity = 6
        config.spice_growback_rate = 2
        config.max_spice_capacity = 6
    elif difficulty == "harsh":
        config.sugar_growback_rate = 1
        config.max_sugar_capacity = 2
        config.spice_growback_rate = 1
        config.max_spice_capacity = 2
    elif difficulty == "desert":
        config.sugar_growback_rate = 0
        config.max_sugar_capacity = 4
        config.spice_growback_rate = 0
        config.max_spice_capacity = 4

    # Override resource settings if specified
    if initial_wealth_min is not None and initial_wealth_max is not None:
        config.initial_wealth_range = (initial_wealth_min, initial_wealth_max)
        config.initial_spice_range = (initial_wealth_min, initial_wealth_max)
        print(f"✓ Initial resources: {initial_wealth_min}-{initial_wealth_max} (sugar & spice)")
    
    if growback_rate is not None:
        config.sugar_growback_rate = growback_rate
        config.spice_growback_rate = growback_rate
        config.max_sugar_capacity = max(6, growback_rate * 2)  # Scale capacity with rate
        config.max_spice_capacity = max(6, growback_rate * 2)
        print(f"✓ Growback rate: {growback_rate} (capacity: {config.max_sugar_capacity})")

    print(f"Goal: {goal_preset}")
    print(f"Goal Prompt: {config.llm_goal_prompt[:100]}...")
    print(f"Grid: {width}x{height} ({width*height} cells)")
    print(f"Population: {population} agents ({population/(width*height)*100:.1f}% density)")
    print(f"Seed: {seed}, Ticks: {ticks}")

    exp_name = f"goal_{goal_preset}"
    if experiment_name_suffix:
        exp_name += f"_{experiment_name_suffix}"
    sim = SugarSimulation(config=config, experiment_name=exp_name)

    print("\nInitial Stats:")
    initial_stats = sim.get_stats()
    print(f"  Population: {initial_stats['population']}")
    print(f"  Mean Wealth: {initial_stats['mean_wealth']:.2f}")
    print(f"  Survival Rate: {initial_stats['survival_rate']:.2f}")

    print("\nRunning simulation...")
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

    # Print reputation stats
    if hasattr(sim.env, 'agent_reputation') and sim.env.agent_reputation:
        reps = list(sim.env.agent_reputation.values())
        print(f"\nReputation Stats:")
        print(f"  Mean Reputation: {sum(reps)/len(reps):.3f}")
        print(f"  Min Reputation: {min(reps):.3f}")
        print(f"  Max Reputation: {max(reps):.3f}")

    return sim.logger.run_dir, final_stats


def compare_goals(goals_to_test, ticks=100, seed=42, model=QWEN3_14B_BASE,
                  population=100, width=30, height=30, difficulty="standard",
                  trade_rounds=4, provider="vllm", use_lora=False):
    """Run experiments with multiple goal presets and compare results."""

    results = {}

    print(f"\n{'='*80}")
    print("GOAL COMPARISON EXPERIMENT")
    print(f"{'='*80}")
    print(f"Testing {len(goals_to_test)} goal presets with {ticks} ticks, seed={seed}")

    for goal in goals_to_test:
        try:
            run_dir, final_stats = run_goal_experiment(
                goal, ticks, seed, model, population, width, height,
                difficulty, trade_rounds, provider, use_lora
            )
            results[goal] = {
                'run_dir': run_dir,
                'final_stats': final_stats
            }
        except Exception as e:
            print(f"Error running {goal}: {e}")
            continue

    # Print comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    print(f"{'Goal':<12}{'Pop':>8}{'Welfare':>12}{'Nash':>12}{'Gini':>12}")
    print("-" * 80)

    for goal, data in results.items():
        stats = data['final_stats']
        print(f"{goal:<12}{stats['population']:>8}{stats['utilitarian_welfare']:>12.2f}"
              f"{stats['nash_welfare']:>12.2f}{stats['welfare_gini']:>12.3f}")

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

        comparison_dir = Path("results/sugarscape/goal_comparison")
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
    parser = argparse.ArgumentParser(description="Run Sugarscape Goal Experiments")

    parser.add_argument("--goals", nargs="+",
                        choices=["none", "survival", "wealth", "altruist"],
                        default=["none", "survival", "wealth", "altruist"],
                        help="Goal presets to test (altruist = merged egalitarian/utilitarian/samaritan)")

    parser.add_argument("--ticks", type=int, default=100,
                        help="Number of simulation ticks (default: 100)")

    parser.add_argument("--population", type=int, default=100,
                        help="Initial population size (default: 100)")

    parser.add_argument("--width", type=int, default=30,
                        help="Grid width (default: 30)")
    parser.add_argument("--height", type=int, default=30,
                        help="Grid height (default: 30)")

    parser.add_argument("--difficulty", type=str, choices=["standard", "easy", "harsh", "desert"],
                        default="standard", help="Difficulty preset")

    parser.add_argument("--trade-rounds", type=int, default=4,
                        help="Maximum dialogue rounds during trade negotiations (default: 4)")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    parser.add_argument("--single", type=str,
                        choices=["none", "survival", "wealth", "altruist"],
                        help="Run single goal experiment (overrides --goals)")

    parser.add_argument("--provider", type=str, choices=["openrouter", "vllm"],
                        default="vllm", help="LLM provider to use (default: vllm)")

    parser.add_argument("--model", type=str, default=QWEN3_14B_BASE,
                        help=f"Model path for vLLM (default: {QWEN3_14B_BASE})")

    parser.add_argument("--lora", action="store_true",
                        help="Use LoRA fine-tuned model instead of base model")

    parser.add_argument("--use-mixed-identity", action="store_true",
                        help="Enable mixed origin identity (80%% Exploiter, 20%% Altruist)")

    parser.add_argument("--identity-distribution", type=str, default=None,
                        help="Custom identity distribution as 'type1:pct,type2:pct,...' "
                             "e.g., 'altruist:20,survivor:80' or 'altruist:20,exploiter:60,survivor:20'. "
                             "Types: altruist, exploiter, survivor")

    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick smoke test: 5 ticks, 10 agents on 10x10 grid")

    # Ablation study flags
    parser.add_argument("--no-survival-pressure", action="store_true",
                        help="ABLATION: Disable survival pressure (agents don't die from starvation)")
    parser.add_argument("--no-social-memory", action="store_true",
                        help="ABLATION: Disable social memory (no trade history, reputation hidden)")
    parser.add_argument("--experiment-suffix", type=str, default="",
                        help="Suffix to add to experiment name (for ablation labeling)")
    
    # LLM Evaluation options
    parser.add_argument("--no-llm-evaluation", action="store_true",
                        help="Disable independent LLM evaluation (saves API costs)")
    parser.add_argument("--evaluator-model", type=str, default="openai/gpt-4o-mini",
                        help="Model for independent evaluation (default: openai/gpt-4o-mini)")
    parser.add_argument("--evaluator-provider", type=str, default="openrouter",
                        choices=["openrouter", "vllm"],
                        help="Provider for evaluator model (default: openrouter)")
    
    # Resource abundance options
    parser.add_argument("--initial-wealth-min", type=int, default=None,
                        help="Minimum initial sugar/spice per agent (default: 5)")
    parser.add_argument("--initial-wealth-max", type=int, default=None,
                        help="Maximum initial sugar/spice per agent (default: 25)")
    parser.add_argument("--growback-rate", type=int, default=None,
                        help="Resource growback rate per tick (higher = more abundant)")

    # Paper environment and evaluator wiring
    parser.add_argument("--paper-env", action="store_true",
                        help="Use the paper's Sugarscape environment (20x20 torus, 100 agents, 100 ticks, "
                             "endowments 45-85, metabolism specialisation, 4-turn trades, event-triggered "
                             "identity review); other flags are applied on top")
    parser.add_argument("--sft", action="store_true",
                        help=f"Use the cooperative SFT adapter ({QWEN3_14B_SFT}) instead of the base model "
                             "(same as --lora)")
    parser.add_argument("--resource-specialization", dest="resource_specialization", action="store_true",
                        default=None, help="Half the agents need more sugar, half more spice")
    parser.add_argument("--no-resource-specialization", dest="resource_specialization",
                        action="store_false", help="Disable metabolism specialisation")
    parser.add_argument("--moral-evaluator-provider", type=str, default=None,
                        choices=["openrouter", "vllm"],
                        help="Provider for the per-reflection moral evaluator (default: vllm with "
                             "--paper-env, otherwise the config default openrouter)")
    parser.add_argument("--moral-evaluator-model", type=str, default=None,
                        help="Model for the per-reflection moral evaluator (default: the agent model)")
    parser.add_argument("--vllm-url", type=str, default=VLLM_URL,
                        help=f"vLLM server base URL (default: {VLLM_URL})")

    return parser.parse_args()


IDENTITY_TYPES = ("altruist", "exploiter", "survivor")
IDENTITY_ALIASES = {"normie": "survivor", "neutral": "survivor"}


def parse_identity_distribution(dist_str: str) -> dict:
    """Parse an identity distribution such as ``'altruist:20,survivor:80'`` into fractions.

    Values may be percentages (``altruist:20``) or fractions (``altruist:0.2``); the set must
    sum to 100% / 1.0. ``normie`` is accepted as an alias of ``survivor`` (the paper's name).
    """
    if not dist_str:
        return None

    result = {}
    for item in dist_str.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid identity distribution format: {item}. Expected 'type:share'")
        identity_type = IDENTITY_ALIASES.get(parts[0].strip().lower(), parts[0].strip().lower())
        if identity_type not in IDENTITY_TYPES:
            raise ValueError(f"Unknown identity type: {identity_type}. Valid: {', '.join(IDENTITY_TYPES)}")
        result[identity_type] = float(parts[1].strip())

    total = sum(result.values())
    if abs(total - 100.0) <= 0.5:
        result = {k: v / 100.0 for k, v in result.items()}
    elif abs(total - 1.0) > 0.01:
        raise ValueError(f"Identity distribution must sum to 100% or 1.0, got {total:g}")

    for t in IDENTITY_TYPES:
        result.setdefault(t, 0.0)
    return result


def main():
    args = parse_args()

    # Handle smoke test - override settings for quick test
    if args.smoke_test:
        print("\n" + "="*60)
        print("SMOKE TEST MODE")
        print("="*60)
        args.ticks = 5
        args.population = 10
        args.width = 10
        args.height = 10
        args.single = "survival"
        print(f"Running quick test: {args.ticks} ticks, {args.population} agents on {args.width}x{args.height} grid")

    # Parse identity distribution if provided
    identity_distribution = None
    if args.identity_distribution:
        try:
            identity_distribution = parse_identity_distribution(args.identity_distribution)
            print(f"Custom identity distribution: {identity_distribution}")
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    use_sft = args.lora or args.sft

    # Determine model based on provider
    if args.provider == "vllm":
        model = QWEN3_14B_SFT if use_sft else args.model
        print(f"Using vLLM provider ({args.vllm_url}) with model: {model}")
    else:
        # Check for API key
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("ERROR: OPENROUTER_API_KEY environment variable not set!")
            print("Please export it: export OPENROUTER_API_KEY='sk-...'")
            sys.exit(1)
        model = args.model
        if model == QWEN3_14B_BASE:
            model = "qwen/qwen3-14b"  # the served vLLM name is not an OpenRouter id
        print(f"Using OpenRouter provider with model: {model}")

    # With --paper-env the evaluators default to the local vLLM server too, unless overridden.
    evaluator_provider = args.evaluator_provider
    evaluator_model = args.evaluator_model
    if args.paper_env and args.provider == "vllm" and evaluator_provider == "openrouter" \
            and evaluator_model == "openai/gpt-4o-mini":
        evaluator_provider, evaluator_model = "vllm", model

    if args.single:
        # Run single experiment
        run_goal_experiment(args.single, args.ticks, args.seed, model,
                          args.population, args.width, args.height, args.difficulty,
                          args.trade_rounds, args.provider, False,
                          args.use_mixed_identity, identity_distribution,
                          enable_survival_pressure=not args.no_survival_pressure,
                          social_memory_visible=not args.no_social_memory,
                          experiment_name_suffix=args.experiment_suffix,
                          enable_llm_evaluation=not args.no_llm_evaluation,
                          llm_evaluator_model=evaluator_model,
                          llm_evaluator_provider=evaluator_provider,
                          initial_wealth_min=args.initial_wealth_min,
                          initial_wealth_max=args.initial_wealth_max,
                          growback_rate=args.growback_rate,
                          paper_env=args.paper_env,
                          resource_specialization=args.resource_specialization,
                          moral_evaluator_model=args.moral_evaluator_model,
                          moral_evaluator_provider=args.moral_evaluator_provider,
                          vllm_url=args.vllm_url)
    else:
        # Run comparison
        compare_goals(args.goals, args.ticks, args.seed, model,
                     args.population, args.width, args.height, args.difficulty,
                     args.trade_rounds, args.provider, False)


if __name__ == "__main__":
    main()
