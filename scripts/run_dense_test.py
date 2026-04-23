"""Quick dense environment test for trade features."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sugarscape.simulation import SugarSimulation
from sugarscape.config import SugarscapeConfig

def run_dense_test():
    """Run a dense environment test to verify trade features."""
    print("\n" + "="*60)
    print("DENSE ENVIRONMENT TEST")
    print("="*60)
    
    config = SugarscapeConfig(
        # Dense environment: 10 agents on 10x10 grid = 10% density
        initial_population=10,
        width=10,
        height=10,
        max_ticks=10,
        seed=42,
        
        # All LLM agents
        enable_llm_agents=True,
        llm_agent_ratio=1.0,
        llm_provider_type="vllm",
        llm_provider_model="/workspace/models/Qwen3-14B",
        llm_goal_preset="survival",
        
        # Enable trade
        enable_spice=True,
        enable_trade=True,
        trade_mode="dialogue",
        trade_dialogue_rounds=4,
        trade_dialogue_two_stage=True,
        trade_dialogue_repair_json=True,
        trade_dialogue_coerce_protocol=True,
    )
    
    print(f"Grid: {config.width}x{config.height} ({config.width * config.height} cells)")
    print(f"Population: {config.initial_population}")
    print(f"Density: {config.initial_population / (config.width * config.height) * 100:.1f}%")
    print(f"Ticks: {config.max_ticks}")
    
    sim = SugarSimulation(config=config, experiment_name="dense_trade_test")
    
    print(f"\nExperiment dir: {sim.logger.run_dir}")
    print("\nRunning simulation...")
    sim.run(steps=config.max_ticks)
    
    print("\nFinal Stats:")
    stats = sim.get_stats()
    print(f"  Population: {stats['population']}")
    print(f"  Mean Wealth: {stats['mean_wealth']:.2f}")
    print(f"  Survival Rate: {stats['survival_rate']:.2f}")
    
    # Reputation stats
    if sim.env.agent_reputation:
        reps = list(sim.env.agent_reputation.values())
        print(f"\nReputation Stats:")
        print(f"  Mean: {sum(reps)/len(reps):.3f}")
        print(f"  Min: {min(reps):.3f}")
        print(f"  Max: {max(reps):.3f}")
    
    # Check trade history
    debug_dir = sim.logger.run_dir / "debug"
    trade_csv = debug_dir / "trade_history.csv"
    if trade_csv.exists():
        import csv
        with open(trade_csv) as f:
            rows = list(csv.reader(f))
        print(f"\nTrade History: {len(rows)-1} trades logged")
        if len(rows) > 1:
            print("  Sample row columns:", rows[0][-10:])  # Last 10 columns (new ones)
    
    return sim.logger.run_dir

if __name__ == "__main__":
    run_dense_test()
