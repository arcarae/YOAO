# Red-Black Game ablations (paper RQ3)

All three probes use the paper's game (N=5, 10 rounds, multipliers 3x/5x/10x at rounds
5/8/10) against a scripted `always_defect` Team B, on the local vLLM server
([`docs/vllm.md`](vllm.md)). Cooperation rate is the fraction of rounds in which Team A votes
BLACK (cooperate). Votes that fail after retries are recorded as abstentions and excluded;
a round in which every vote fails aborts the game instead of being scored.

## Mute test: seeds vote but cannot argue

`scripts/eval_table3_ablation_mixed.py` builds Team A from `k` seed agents plus `5-k`
unmodified agents. In the default `--mode muted`, each seed is the SFT adapter wrapped in
`redblackbench.agents.muted_agent.MutedAgent`: it still reasons and votes through its own
model, but every message it broadcasts is replaced by `"I vote <choice>."`. If cooperation
collapses toward the unmodified baseline, the argument content, not the vote itself, is what
propagates.

```bash
python scripts/eval_table3_ablation_mixed.py --scenario all --games 3            # muted seeds
python scripts/eval_table3_ablation_mixed.py --scenario all --games 3 --mode hardcoded
```

`--mode hardcoded` swaps the seeds for non-LLM agents that always vote cooperatively and say
only `"I vote <cooperative choice>."` (no model call), the "presence without reasoning" control.
`scripts/run_table3_ablation_mixed_parallel.py` runs all compositions and scenarios in parallel.

## Norm persistence after seed removal

`scripts/eval_table3_ablation_removal.py` starts Team A with `k` seeds plus `5-k` unmodified
agents, plays until the removal point, then **replaces** every seed with a fresh unmodified
agent (the team stays at five), and plays the remaining rounds. It reports cooperation before
and after the swap.

```bash
# remove after the first round in which Team A has cooperated 2 rounds in a row (paper protocol)
python scripts/eval_table3_ablation_removal.py --scenario all --games 3 --remove-after-stable 2
# or at a fixed round
python scripts/eval_table3_ablation_removal.py --scenario all --games 3 --removal-round 3
```

The replacement agents have no memory of earlier deliberations; what survives the swap is
only what the remaining unmodified agents internalised. `run_table3_ablation_removal_parallel.py`
is the parallel driver.

## Influence shift

Vote changes are read off the trajectories rather than computed by a dedicated script: each
`initial_opinions` timestep records every agent's stated recommendation and each
`final_votes` timestep its final vote, so an unmodified agent whose opinion was RED and whose
vote is BLACK after hearing a seed counts as a RED-to-BLACK shift (and vice versa).

```python
import json
t = json.load(open("results/.../seed20pct_run0.json"))
for step in t["timesteps"]:
    if step["timestep_type"] in ("initial_opinions", "final_votes"):
        print(step["round_num"], step["timestep_type"], [a["choice"] for a in step["actions"]])
```

## Output layout

```
eval_results/<experiment>/
  ablation_summary.json        per-game results and aggregates
  trajectories/<scenario>/     one JSON per game (round-by-round choices, every message,
                               abstentions, scores)
```

## Control adapter, uncooperative seed, LLaMA seed

Three more adapters back the paper's "Non-Cooperative SFT Training" section, the "Propagation is
bidirectional" paragraph and the LLaMA-3.1-8B cross-architecture result. All share the seed's LoRA
shape (r=128, alpha=256, all attention and MLP projections); each model card states its data and
the recorded training settings.

| adapter | base | data | paper result |
|---|---|---|---|
| [`Arcarae/redblackbench-qwen3-14b-control`](https://huggingface.co/Arcarae/redblackbench-qwen3-14b-control) | Qwen3-14B | 10,607 generic-reasoning samples from OpenThoughts-114k ([dataset](https://huggingface.co/datasets/Arcarae/redblackbench-control-openthoughts-10k)) | 0-5 control agents per team: 25, 41, 35, 30, 29, 31 % cooperation; no scaling |
| [`Arcarae/redblackbench-qwen3-14b-uncoop`](https://huggingface.co/Arcarae/redblackbench-qwen3-14b-uncoop) | Qwen3-14B | 7,686 defection-persuasion deliberations ([dataset](https://huggingface.co/datasets/Arcarae/redblackbench-sft-uncoop-v1)) | one seed among unmodified LLaMA-3.1-8B: 62 % -> 13 % |
| [`Arcarae/redblackbench-llama-3.1-8b-coop-seed`](https://huggingface.co/Arcarae/redblackbench-llama-3.1-8b-coop-seed) | LLaMA-3.1-8B-Instruct | 10,607 cooperative deliberations, 270 games ([dataset](https://huggingface.co/datasets/Arcarae/redblackbench-sft-batches-2-3-10k)) | one seed among unmodified LLaMA-3.1-8B: 62.7 % -> 97.0 % |

Serve them beside the seed (vLLM loads several adapters on one base; the LLaMA seed needs a
second server on its own base):

```bash
hf download Arcarae/redblackbench-qwen3-14b-control --local-dir models/Qwen3-14B-control
hf download Arcarae/redblackbench-qwen3-14b-uncoop   --local-dir models/Qwen3-14B-uncoop
vllm serve models/Qwen3-14B --served-model-name Qwen/Qwen3-14B --enable-lora --max-lora-rank 128 \
  --lora-modules redblackbench-qwen3-14b-sft-v2=models/Qwen3-14B-LoRA \
                 redblackbench-qwen3-14b-control=models/Qwen3-14B-control \
                 redblackbench-qwen3-14b-uncoop=models/Qwen3-14B-uncoop \
  --max-model-len 32768 --port 8000

# control adapter, 0-5 copies per five-agent team, three held-out scenarios
python scripts/eval_meta_alignment.py --trained-model redblackbench-qwen3-14b-control \
    --runs-per-condition 10 --output-dir results/control

# uncooperative seed among unmodified LLaMA-3.1-8B teammates (teammates via OpenRouter)
python scripts/eval_heterogeneous.py --phase 1 --trained-model redblackbench-qwen3-14b-uncoop \
    --seed-pct 20 --output-dir results/uncoop

# LLaMA seed among unmodified LLaMA teammates: serve the adapter on meta-llama/Llama-3.1-8B-Instruct
hf download Arcarae/redblackbench-llama-3.1-8b-coop-seed --local-dir models/Llama-3.1-8B-seed
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-lora --max-lora-rank 128 \
  --lora-modules llama-3.1-8b-coop-seed=models/Llama-3.1-8B-seed --port 8001
python scripts/eval_heterogeneous.py --phase 1 --trained-model llama-3.1-8b-coop-seed \
    --vllm-url http://localhost:8001/v1 --seed-pct 20 --output-dir results/llama_seed
```
