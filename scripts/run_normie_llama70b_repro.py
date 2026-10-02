#!/usr/bin/env python3
"""Reviewer rebuttal: the 100% Normie Sugarscape baseline with Llama-3.3-70B via OpenRouter.

The paper's Normie baseline uses Qwen3-14B served locally. This control swaps in a much larger
model (meta-llama/llama-3.3-70b-instruct) for the agents and both evaluators, keeping every
other setting of the paper environment (see ``sugarscape.presets.paper_config``).

Requires OPENROUTER_API_KEY.

Usage:
    python scripts/run_normie_llama70b_repro.py [--ticks 100] [--seed 42]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sugarscape.simulation import SugarSimulation
from sugarscape import presets

MODEL = "meta-llama/llama-3.3-70b-instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticks", type=int, default=presets.PAPER_TICKS, help="Simulation ticks (default: 100)")
    parser.add_argument("--population", type=int, default=presets.PAPER_POPULATION, help="Agents (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default=MODEL, help=f"OpenRouter model id (default: {MODEL})")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("ERROR: OPENROUTER_API_KEY is not set. Export it before running.")

    config = presets.paper_config(
        provider="openrouter",
        model=args.model,
        goal_preset="survival",
        identity_distribution=presets.IDENTITY_ALL_NORMIE,
        ticks=args.ticks,
        population=args.population,
        seed=args.seed,
        evaluator_provider="openrouter",
        evaluator_model=args.model,
        moral_evaluator_provider="openrouter",
        moral_evaluator_model=args.model,
    )

    sim = SugarSimulation(config=config, experiment_name="normie_100pct_llama70b")
    print(f"Experiment dir: {sim.logger.run_dir}")
    print(f"Agents + evaluators: {args.model} via OpenRouter; {args.population} agents, {args.ticks} ticks")
    sim.run(steps=args.ticks)

    stats = sim.get_stats()
    print("\nFinal stats:")
    for key in ("population", "mean_wealth", "survival_rate", "utilitarian_welfare", "welfare_gini"):
        print(f"  {key}: {stats[key]}")
    print(f"\nResults: {sim.logger.run_dir}")


if __name__ == "__main__":
    main()
