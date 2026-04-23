# YOAO: You Only Align Once

**Alignment Propagation in Multi-Agent LLM Systems**
ICLR 2026 LLA Workshop

Can a small number of aligned (fine-tuned) LLM agents pull a larger population of unaligned agents toward cooperation?

This repo contains the code for both experimental suites in the paper, as sibling top-level packages:

| Suite | Package | Used for |
|---|---|---|
| **Red-Black Game** | [`redblackbench/`](redblackbench/) | Tables 2, 3, and B (robustness, meta-alignment, scale effect, heterogeneous models, ablations) |
| **Sugarscape** | [`sugarscape/`](sugarscape/) | §4.2 (zero-shot transfer from SFT exploiters) and §4.3 (normie baseline + altruist seeding tipping point at 50%) |

Across both environments we find a consistent tipping point: once ≥50% of agents carry the aligned policy, the rest of the population follows, even though unaligned agents receive no fine-tuning and no explicit coordination signal.

## The two suites

### Red-Black Game

Two teams of 5 play 10 rounds. Each round every team simultaneously chooses BLACK (cooperate) or RED (defect).

| Team A | Team B |  A  |  B  | Total |
|--------|--------|-----|-----|-------|
| BLACK  | BLACK  | +3  | +3  | **+6** |
| RED    | RED    | −3  | −3  | **−6** |
| RED    | BLACK  | +6  | −6  | 0 |
| BLACK  | RED    | −6  | +6  | 0 |

Rounds 5 / 8 / 10 have multipliers 3× / 5× / 10×. Full mutual cooperation yields a maximum total of **150**. Each team decides via a two-phase deliberation (opinions → final votes → majority).

Alignment propagation is measured by seeding teams with a mix of aligned (SFT) and unaligned agents and tracking the team's cooperation rate as the seed proportion varies.

### Sugarscape

100 LLM agents live on a 20×20 torus, harvest sugar and spice, and trade via multi-round dialogue. Per tick:

1. Growback — resources regenerate on the grid.
2. Rule-based agents move (vision-limited greedy harvesting).
3. LLM agents move (async batched inference).
4. Trade phase — agents in the same cell enter a dialogue protocol (small talk → intent → negotiation → execution), optionally preceded by a third-party *broker*.
5. Reflection — each agent updates dual-track beliefs and an editable policy list.
6. Metabolism / aging / death.

Alignment propagation is measured by varying the fraction of *altruist*-identity agents in a majority-survivor population and tracking welfare, inequality (Gini), and trade fairness.

## Installation

Python ≥ 3.10 is required.

```bash
git clone https://github.com/arcarae/YOAO && cd YOAO

# Full install with both suites + all LLM providers
pip install -e ".[all]"

# Or install only the providers for one suite
pip install -e ".[redblackbench]"
pip install -e ".[sugarscape]"

# Dev tools
pip install -e ".[dev]"
```

### API keys

```bash
export OPENAI_API_KEY=...         # for gpt-4o evaluators
export OPENROUTER_API_KEY=...     # for qwen/llama/etc. via OpenRouter
export ANTHROPIC_API_KEY=...      # optional, for Claude agents
```

For local Qwen3-14B / LoRA inference via vLLM, see `VLLM_INTEGRATION.md`.

## Running experiments

### Red-Black Game

```bash
# CLI (installed as console script)
redblackbench run --config experiments/configs/example.yaml
redblackbench analyze --results-dir results/
redblackbench scenarios --details

# Table 2 — robustness vs scripted opponents
python3.10 scripts/eval_robustness.py \
    --model "qwen/qwen3-14b" --games-per-combo 3 \
    --output-dir results/eval_robustness

# Table 3 — meta-alignment with mixed teams (parallel orchestrator)
python3.10 scripts/run_table3_parallel.py --games-per-combo 3

# Table B — scale effect at N=5 / 10 / 15 with seed percentages
python3.10 scripts/eval_scale_effect.py

# Table 3 ablations
python3.10 scripts/run_table3_ablation_parallel.py
python3.10 scripts/run_table3_ablation_removal_parallel.py
python3.10 scripts/run_table3_ablation_mixed_parallel.py

# Cross-model heterogeneous transfer
python3.10 scripts/eval_heterogeneous.py --phase 1
```

All evaluators accept `--resume` for checkpointed re-runs and write progress to `<output-dir>/progress.json`.

### Sugarscape

```bash
# Minimal smoke test: rule-based agents, no LLM
python3.10 scripts/run_sugarscape.py --mode basic --ticks 50 --population 30

# LLM agents with a goal preset (survival / wealth / altruist / none)
python3.10 scripts/run_sugarscape.py --mode llm --goal-preset survival --ticks 100

# Goal comparison sweep (paper §4.3)
python3.10 scripts/run_goal_experiment.py \
    --goals survival wealth altruist \
    --population 100 --ticks 200 \
    --provider openrouter --model "qwen/qwen3-14b"

# Altruist seeding at 20% / 40% / 50% (paper §4.3 tipping-point table)
python3.10 scripts/run_goal_experiment.py \
    --single survival \
    --identity-distribution "altruist:0.2,survivor:0.8" \
    --experiment-suffix "seed_20pct"

# SFT-v2 LoRA variant (requires local vLLM)
python3.10 scripts/run_goal_experiment_sft.py --ticks 200

# Reviewer rebuttal: 100% normie with Llama-3.3-70B
python3.10 scripts/run_normie_gpt4o_repro.py
```

Full CLI flag reference: `COMMAND_LINE_GUIDE.md` or `python3.10 scripts/<script>.py --help`.

## Reproducing the paper

### Red-Black Game

| Paper table | Script | Notes |
|---|---|---|
| Table 2 (robustness vs strategies) | `eval_robustness.py` | 3+ games per (model × strategy × scenario) combination |
| Table 3 (meta-alignment with mixed teams) | `run_table3_parallel.py` | 30 combinations × 3 games, parallel across scenarios |
| Table 3 ablations | `run_table3_ablation_{parallel,removal_parallel,mixed_parallel}.py` | hardcoded / removal / mixed controls |
| Table B (scale effect) | `eval_scale_effect.py` | N=5, 10, 15; seed % 0 → 100 in 20% steps; 3–5 runs/condition |
| Heterogeneous transfer | `eval_heterogeneous.py` | `--phase {1,2}` for the two-stage protocol |

Ablation design: `ABLATION_EXPERIMENT.md`.

### Sugarscape

| Paper section | Experiment | Runner | Key flags |
|---|---|---|---|
| §4.2 | Zero-shot transfer (trained vs. untrained exploiter) | `run_goal_experiment_sft.py` | `--ticks 100 --population 100` |
| §4.3, 0% seed | 100% normie baseline | `run_goal_experiment.py` | `--single survival --identity-distribution "survivor:1.0"` |
| §4.3, 20/40/50% seed | Altruist seeding | `run_goal_experiment.py` | `--identity-distribution "altruist:0.X,survivor:(1-0.X)"` |
| Rebuttal | 100% normie, Llama-3.3-70B | `run_normie_gpt4o_repro.py` | (hardcoded config inside script) |

Sugarscape LLM-experiment runs produce a timestamped directory under `results/sugarscape/<experiment_name>/experiment_<ts>/` with at minimum:

- `config.json` — full `SugarscapeConfig` snapshot
- `metrics.csv` — per-tick population / welfare / Gini time series
- `initial_state.json`, `final_state.json` — grid + agent snapshots
- `checkpoints/checkpoint_tick_N.pkl` — full pickled state at each `checkpoint_interval`
- `debug/*.csv` and `*.jsonl` — per-agent decisions, trade dialogues, reflections, moral evaluations, death records
- `plots/` — welfare / Gini / survival visualizations
- `trajectory_<name>.json` — full LLM prompt/response record (for RL post-training)

## Repository structure

```
YOAO/
├── redblackbench/                 # Red-Black Game package
│   ├── cli.py                     #   `redblackbench` CLI
│   ├── game/                      #   GameConfig, GameCoordinator, scoring
│   ├── agents/                    #   LLMAgent + prompts
│   ├── teams/                     #   Team deliberation
│   ├── scenarios/                 #   Scenario reskins (climate, pandemic, agi_safety, ...)
│   ├── strategies/                #   Scripted opponents
│   ├── trajectory/                #   Trajectory recording
│   ├── logging/                   #   Game + metrics logging
│   ├── training/                  #   SFT / RL training data pipeline
│   └── providers/                 #   LLM provider adapters
├── sugarscape/                    # Sugarscape package
│   ├── simulation.py              #   Tick loop, checkpoints
│   ├── config.py                  #   SugarscapeConfig — ~90 typed fields
│   ├── environment.py             #   Torus grid + resource growback
│   ├── agent.py                   #   SugarAgent
│   ├── llm_agent.py               #   LLMSugarAgent (async batched inference)
│   ├── trade.py                   #   Dialogue + MRS trade system; broker phase
│   ├── prompts.py                 #   Encounter / reflection / broker prompts
│   ├── moral_evaluator.py         #   Structured moral evaluator
│   ├── evaluator.py               #   Real-time trade fairness scorer
│   ├── welfare.py                 #   Cobb-Douglas + social welfare metrics
│   ├── welfare_plots.py           #   Welfare / Gini / survival visualization
│   ├── trajectory.py              #   RL post-training trajectory capture
│   └── providers/                 #   LLM provider adapters (sugarscape-local copy)
├── scripts/                       # All paper runners (both suites)
├── experiments/configs/           # YAML configs for Red-Black CLI runner
├── tests/                         # 50 tests (39 Red-Black + 11 Sugarscape)
├── findings/                      # Analysis markdown
├── ABLATION_EXPERIMENT.md         # Red-Black Table 3 ablation design
├── VLLM_INTEGRATION.md            # Local vLLM (Qwen3-14B + LoRA) setup
└── COMMAND_LINE_GUIDE.md          # Full CLI reference for sugarscape runners
```

## Programmatic usage

```python
# Red-Black game
import asyncio
from redblackbench import GameConfig, GameCoordinator
from redblackbench.agents.llm_agent import LLMAgent
from redblackbench.teams.team import Team
from redblackbench.providers.openai_provider import OpenAIProvider

async def rb_game():
    provider = OpenAIProvider(model="gpt-4o", temperature=0.7)
    team_a = Team("A", [LLMAgent(f"a{i}", "A", provider) for i in range(5)])
    team_b = Team("B", [LLMAgent(f"b{i}", "B", provider) for i in range(5)])
    config = GameConfig(num_rounds=10, team_size=5,
                        multipliers={5: 3, 8: 5, 10: 10})
    state = await GameCoordinator(team_a, team_b, config).play_game()
    print(f"Cooperation rate: {state.cooperation_rate:.0%}")

asyncio.run(rb_game())
```

```python
# Sugarscape
from sugarscape import SugarSimulation, SugarscapeConfig

config = SugarscapeConfig(
    width=20, height=20, initial_population=100, max_ticks=100,
    enable_spice=True, enable_trade=True, trade_mode="dialogue",
    enable_llm_agents=True, llm_provider_type="openrouter",
    llm_provider_model="qwen/qwen3-14b",
    llm_goal_preset="survival",
    enable_origin_identity=True,
    origin_identity_distribution={"altruist": 0.2, "survivor": 0.8},
)
sim = SugarSimulation(config=config, experiment_name="seed_20pct_demo")
sim.run()
```

## Tests

```bash
pytest tests/
```

50 tests total — 39 Red-Black (`test_game`, `test_scoring`, `test_trajectory`) + 11 Sugarscape (`test_e2e_mock_llm`, `test_trade_dialogue`, `test_sugarscape_smoke`).

## Citation

```bibtex
@inproceedings{yoao2026,
  title     = {YOAO: You Only Align Once -- Alignment Propagation in Multi-Agent Systems},
  booktitle = {ICLR 2026 LLA Workshop},
  year      = {2026}
}
```

## License

MIT.
