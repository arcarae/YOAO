# Sugarscape Experiment Result Examination Guide

Quick guide for examining experiment results and spotting potential issues.

## 1. Locate Results

```bash
# Current completed experiment
cd /workspace/RedBlackBench/results/sugarscape/goal_none/experiment_20260109_083440
```

## 2. Key Files

| File | Purpose |
|------|---------|
| `config.json` | Experiment settings (verify correct setup) |
| `metrics.csv` | Time-series data every 10 ticks |
| `initial_state.json` | Starting agent positions/wealth |
| `final_state.json` | Ending agent positions/wealth |
| `trajectory_*.json` | Full LLM prompts/responses for RL |
| **debug/** | |
| `debug/debug_summary.json` | Totals: deaths, trades, decisions |
| `debug/death_records.csv` | All deaths with cause and tick |
| `debug/trade_history.csv` | All trades with amounts |
| `debug/llm_interactions.jsonl` | Every LLM prompt/response |
| `debug/agent_decisions.csv` | Parsed moves per agent per tick |
| `debug/resource_efficiency.csv` | Resource gathering stats |

## 3. Quick Health Checks

### Check config was correct
```bash
cat config.json | python -m json.tool | head -30
```

Verify:
- `num_llm_agents`: 100
- `max_ticks`: 100
- `enable_trade`: true
- `goal_prompt`: should mention "Observe your situation"

### Check metrics summary
```bash
# View metrics header and last few rows
head -1 metrics.csv && tail -5 metrics.csv
```

### Plot key metrics (if matplotlib available)
```bash
python -c "
import pandas as pd
df = pd.read_csv('metrics.csv')
print('=== METRICS SUMMARY ===')
print(f'Ticks recorded: {len(df)}')
print(f'Final population: {df.population.iloc[-1]}')
print(f'Wealth: {df.mean_wealth.iloc[0]:.1f} -> {df.mean_wealth.iloc[-1]:.1f}')
print(f'Gini: {df.gini.iloc[0]:.3f} -> {df.gini.iloc[-1]:.3f}')
print(f'Utilitarian welfare: {df.utilitarian_welfare.iloc[-1]:.1f}')
print()
print('=== TRENDS ===')
print(f'Wealth change: {((df.mean_wealth.iloc[-1]/df.mean_wealth.iloc[0])-1)*100:+.1f}%')
print(f'Gini change: {((df.gini.iloc[-1]/df.gini.iloc[0])-1)*100:+.1f}%')
"
```

## 4. Red Flags to Look For

### A. Population Issues
```bash
python -c "
import pandas as pd
df = pd.read_csv('metrics.csv')
if df.population.min() < 50:
    print('WARNING: Population dropped below 50!')
if df.population.std() > 20:
    print('WARNING: High population variance')
print(f'Population range: {df.population.min()} - {df.population.max()}')
"
```

### B. Wealth Distribution Problems
```bash
python -c "
import pandas as pd
df = pd.read_csv('metrics.csv')
# Check for extreme inequality
if df.gini.iloc[-1] > 0.6:
    print('WARNING: Very high inequality (Gini > 0.6)')
# Check for wealth collapse
if df.mean_wealth.iloc[-1] < df.mean_wealth.iloc[0] * 0.5:
    print('WARNING: Wealth collapsed by >50%')
# Check Rawlsian (worst-off agent)
if df.rawlsian_welfare.iloc[-1] < 1:
    print('WARNING: Poorest agent near death')
print(f'Final Gini: {df.gini.iloc[-1]:.3f}')
print(f'Final Rawlsian: {df.rawlsian_welfare.iloc[-1]:.2f}')
"
```

### C. Death Analysis
```bash
python -c "
import csv
from pathlib import Path

death_file = Path('debug/death_records.csv')
if not death_file.exists():
    print('No death records file')
    exit()

with open(death_file) as f:
    deaths = list(csv.DictReader(f))

# Count by cause
causes = {}
for d in deaths:
    c = d.get('cause', 'unknown')
    causes[c] = causes.get(c, 0) + 1

print(f'Total deaths: {len(deaths)}')
print('\nBy cause:')
for cause, count in sorted(causes.items(), key=lambda x: -x[1]):
    print(f'  {cause}: {count} ({count/len(deaths)*100:.0f}%)')

# Check for spikes (many deaths in one tick)
tick_deaths = {}
for d in deaths:
    t = d.get('tick', '?')
    tick_deaths[t] = tick_deaths.get(t, 0) + 1

max_tick = max(tick_deaths.items(), key=lambda x: x[1])
print(f'\nWorst tick: {max_tick[0]} ({max_tick[1]} deaths)')
"
```

### D. LLM Response Quality
```bash
# Sample some LLM responses to check for issues
python -c "
import json
from pathlib import Path

debug_dir = Path('debug')
llm_file = debug_dir / 'llm_interactions.jsonl'

with open(llm_file) as f:
    lines = f.readlines()

print(f'Total LLM interactions: {len(lines)}')

# Sample from middle of experiment
mid = len(lines) // 2
for i in range(mid, min(mid+3, len(lines))):
    data = json.loads(lines[i])
    print(f'\n--- Agent {data.get(\"agent_id\", \"?\")} (tick {data.get(\"tick\", \"?\")}) ---')
    response = data.get('response', '')[:200]
    print(f'Response: {response}...')
"
```

### E. Trade Activity
```bash
python -c "
import csv
from pathlib import Path

trade_file = Path('debug/trade_history.csv')
if not trade_file.exists():
    print('No trade history file')
    exit()

with open(trade_file) as f:
    trades = list(csv.DictReader(f))

print(f'Total trades: {len(trades)}')

if trades:
    # Show column names to understand structure
    print(f'Columns: {list(trades[0].keys())}')

    # Sample trade
    sample = trades[len(trades)//2]
    print(f'\nSample trade:')
    for k, v in sample.items():
        print(f'  {k}: {v}')
"
```

## 5. Common Issues & What They Mean

| Symptom | Possible Cause |
|---------|---------------|
| Many `parsed_move: null` | LLM response format issues |
| High starvation deaths | Agents not finding resources |
| Gini > 0.5 | Wealth concentrating (may be intended) |
| Low trade count | Agents not engaging economically |
| Wealth declining | Resource scarcity or bad decisions |
| Population crashes | Starvation epidemic |

## 6. Compare With Expected Baseline

For NONE goal (no explicit objective), expect:
- Moderate wealth growth (10-30%)
- Gini around 0.25-0.40
- Mixed death causes (not all one type)
- Some trading activity (500-2000 trades)
- Survival rate ~1.0 (constant population replacement)

## 7. Full Trajectory Analysis (Advanced)

```bash
# Check trajectory file size (should be large with LLM data)
ls -lh trajectory_*.json

# Sample a trajectory entry
python -c "
import json
traj = json.load(open('trajectory_experiment_20260109_083440.json'))
print(f'Total entries: {len(traj)}')
if traj:
    sample = traj[len(traj)//2]  # Middle entry
    print(f'Sample tick: {sample.get(\"tick\")}')
    print(f'Keys: {list(sample.keys())}')
"
```

## 8. Quick Sanity Check Script

Run this for a fast overview (no pandas required):

```bash
cd /path/to/experiment_folder

python -c "
import csv
import json
from pathlib import Path

print('='*50)
print('EXPERIMENT SANITY CHECK')
print('='*50)

# Config
config = json.load(open('config.json'))
print(f'\nConfig: max_ticks={config.get(\"max_ticks\")}, trade={config.get(\"enable_trade\")}')

# Metrics
with open('metrics.csv') as f:
    rows = list(csv.DictReader(f))
first, last = rows[0], rows[-1]

print(f'\n--- Key Metrics ---')
w0, w1 = float(first['mean_wealth']), float(last['mean_wealth'])
print(f'Wealth: {w0:.1f} -> {w1:.1f} ({((w1/w0)-1)*100:+.1f}%)')
print(f'Gini: {float(first[\"gini\"]):.3f} -> {float(last[\"gini\"]):.3f}')
print(f'Utilitarian: {float(last[\"utilitarian_welfare\"]):.1f}')
print(f'Nash: {float(last[\"nash_welfare\"]):.2f}')
print(f'Rawlsian: {float(last[\"rawlsian_welfare\"]):.2f}')

# Debug summary
debug_dir = Path('debug')
if (debug_dir / 'debug_summary.json').exists():
    summary = json.load(open(debug_dir / 'debug_summary.json'))
    print(f'\n--- Activity Summary ---')
    print(f'LLM interactions: {summary.get(\"total_llm_interactions\", \"?\")}')
    print(f'Trades: {summary.get(\"total_trades\", \"?\")}')
    print(f'Deaths: {summary.get(\"total_deaths\", \"?\")}')

# Death breakdown
if (debug_dir / 'death_records.csv').exists():
    with open(debug_dir / 'death_records.csv') as f:
        deaths = list(csv.DictReader(f))
    causes = {}
    for d in deaths:
        c = d.get('cause', 'unknown')
        causes[c] = causes.get(c, 0) + 1
    print(f'\n--- Death Causes ---')
    for c, n in sorted(causes.items(), key=lambda x: -x[1]):
        print(f'  {c}: {n}')

# Warnings
print(f'\n--- Warnings ---')
warnings = []
if float(last['gini']) > 0.5:
    warnings.append('High inequality (Gini > 0.5)')
if float(last['rawlsian_welfare']) < 2:
    warnings.append('Very poor worst-off agent (Rawlsian < 2)')
if w1 < w0:
    warnings.append('Wealth declined overall')

if warnings:
    for w in warnings:
        print(f'  ! {w}')
else:
    print('  None - looks healthy')
"
```

## Questions to Answer

After examination, document:

1. Did the experiment run to completion (100 ticks)?
2. Are metrics reasonable (no NaN, no extreme values)?
3. Did LLM responses parse correctly (check debug logs)?
4. What were the main death causes?
5. Did agents trade? Was it beneficial?
6. Any anomalies in wealth/welfare trends?
