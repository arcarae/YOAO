import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sugarscape.simulation import SugarSimulation
from sugarscape.config import SugarscapeConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Run Sugarscape Simulation")
    
    parser.add_argument("--mode", type=str, choices=["basic", "llm"], default="basic",
                        help="Simulation mode: 'basic' (standard agents) or 'llm' (LLM-driven agents)")
    
    parser.add_argument("--ticks", type=int, default=100,
                        help="Number of simulation ticks to run")
    
    parser.add_argument("--population", type=int, default=50,
                        help="Initial population size")

    # Environment Arguments
    parser.add_argument("--variant", type=str, choices=["sugar", "spice"], default="spice",
                        help="Environment variant: 'sugar' (Classic V1) or 'spice' (Sugar+Spice V2)")

    parser.add_argument("--width", type=int, default=50, help="Grid width")
    parser.add_argument("--height", type=int, default=50, help="Grid height")

    parser.add_argument("--difficulty", type=str, choices=["standard", "easy", "harsh", "desert"], default="standard",
                        help="Difficulty preset affecting resource density and growback")

    # Agent Attributes
    parser.add_argument("--wealth-min", type=int, default=5, help="Minimum initial wealth")
    parser.add_argument("--wealth-max", type=int, default=25, help="Maximum initial wealth")
    parser.add_argument("--metabolism-min", type=int, default=1, help="Minimum metabolism rate")
    parser.add_argument("--metabolism-max", type=int, default=4, help="Maximum metabolism rate")
    parser.add_argument("--vision-min", type=int, default=1, help="Minimum vision range")
    parser.add_argument("--vision-max", type=int, default=6, help="Maximum vision range")

    # LLM Arguments
    parser.add_argument("--provider", type=str, choices=["vllm", "openrouter"], default="vllm",
                        help="Where --model is served: local vLLM (default) or OpenRouter")
    parser.add_argument("--model", type=str,
                        default=os.environ.get("YOAO_BASE_MODEL", "Qwen/Qwen3-14B"),
                        help="Served model name (vLLM) or OpenRouter model id")
    parser.add_argument("--vllm-url", type=str,
                        default=os.environ.get("YOAO_VLLM_URL", "http://localhost:8000/v1"),
                        help="vLLM server base URL")

    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of LLM agents (0.0 - 1.0) when in LLM mode")

    parser.add_argument("--goal-preset", type=str,
                        choices=["none", "survival", "wealth", "altruist"],
                        default="survival",
                        help="Goal preset for LLM agents")

    parser.add_argument("--custom-goal", type=str,
                        default="",
                        help="Custom goal prompt for LLM agents (overrides --goal-preset)")

    # Trade Configuration
    parser.add_argument("--trade-rounds", type=int, default=4,
                        help="Maximum dialogue rounds during trade negotiations")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Initializing Sugarscape in '{args.mode.upper()}' mode...")
    
    # Configure Simulation
    config = SugarscapeConfig(
        initial_population=args.population,
        max_ticks=args.ticks,
        width=args.width,
        height=args.height,
        seed=args.seed,
        initial_wealth_range=(args.wealth_min, args.wealth_max),
        metabolism_range=(args.metabolism_min, args.metabolism_max),
        vision_range=(args.vision_min, args.vision_max),
        trade_dialogue_rounds=args.trade_rounds
    )
    
    # Apply Difficulty Presets
    if args.difficulty == "easy":
        config.sugar_growback_rate = 2
        config.max_sugar_capacity = 6
        config.spice_growback_rate = 2
        config.max_spice_capacity = 6
    elif args.difficulty == "harsh":
        config.sugar_growback_rate = 1
        config.max_sugar_capacity = 2
        config.spice_growback_rate = 1
        config.max_spice_capacity = 2
    elif args.difficulty == "desert":
        config.sugar_growback_rate = 0
        config.max_sugar_capacity = 4
        config.spice_growback_rate = 0
        config.max_spice_capacity = 4
    # standard is default (growback=1, capacity=4)
    
    # Apply Variant (spice enables the two-resource economy and, in LLM mode, trading)
    config.enable_spice = (args.variant == "spice")

    if args.mode == "llm":
        print(f"Configuring LLM Agents with model: {args.model} via {args.provider}")
        config.enable_llm_agents = True
        config.llm_agent_ratio = args.ratio
        config.llm_provider_type = args.provider
        config.llm_provider_model = args.model
        config.llm_vllm_base_url = args.vllm_url
        config.enable_trade = config.enable_spice
        if args.provider == "vllm":
            # keep the run fully local: evaluators on the same server as the agents
            config.llm_evaluator_provider = "vllm"
            config.llm_evaluator_model = args.model
            config.external_moral_evaluator_provider = "vllm"
            config.external_moral_evaluator_model = args.model

        # Set goal based on preset or custom goal
        if args.custom_goal:
            config.llm_goal_preset = "custom"
            config.llm_goal_prompt = args.custom_goal
        else:
            config.llm_goal_preset = args.goal_preset
            # The __post_init__ will set the goal prompt automatically

        print(f"Agent Goal: {config.llm_goal_preset.upper()}")
        print(f"Goal Prompt: {config.llm_goal_prompt[:100]}...")
        
        # Check API Key
        if args.provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
            sys.exit("ERROR: OPENROUTER_API_KEY is not set; export it or use --provider vllm")
            
    sim = SugarSimulation(config=config)
    
    print("Initial Stats:")
    print(sim.get_stats())
    
    print(f"\nRunning for {args.ticks} ticks...")
    for i in range(args.ticks):
        sim.step()
        if (i+1) % 10 == 0:
            stats = sim.get_stats()
            print(f"Tick {i+1}: Pop={stats['population']}, Mean W={stats['mean_wealth']:.2f}, Gini={stats['gini']:.2f}")
            
    print("\nFinal Stats:")
    print(sim.get_stats())
    
    # Check if agents are actually moving/harvesting
    # We can check total wealth in the system
    total_wealth = sum(a.wealth for a in sim.agents)
    print(f"Total Wealth: {total_wealth}")

if __name__ == "__main__":
    main()
