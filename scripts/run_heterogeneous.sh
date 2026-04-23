#!/bin/bash
# Launch heterogeneous experiment: 3 scenarios in parallel
# Qwen3-14B SFT (seed) + Llama 3.1 8B (untrained) via OpenRouter
#
# Usage: OPENROUTER_API_KEY=sk-xxx bash scripts/run_heterogeneous.sh

cd /workspace/RedBlackBench

# ── Config ───────────────────────────────────────────────────────────
VLLM_URL="http://194.68.245.87:22067/v1"
SCENARIOS="baseline trade_war gpu_contention"
RUNS=3
OUTBASE="results/eval_heterogeneous"

# ── Validate ─────────────────────────────────────────────────────────
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set"
    exit 1
fi

# Check vLLM
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "${VLLM_URL}/models" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: vLLM not responding (HTTP $HTTP_CODE)"
    exit 1
fi
echo "$(date) | vLLM OK"

# Quick OpenRouter test
TEST=$(python3.10 -c "
from openai import OpenAI
c = OpenAI(api_key='${OPENROUTER_API_KEY}', base_url='https://openrouter.ai/api/v1')
r = c.chat.completions.create(model='meta-llama/llama-3.1-8b-instruct', messages=[{'role':'user','content':'Say OK'}], max_tokens=8)
print(r.choices[0].message.content[:10])
" 2>&1)
if echo "$TEST" | grep -qi "ok"; then
    echo "$(date) | OpenRouter OK (Llama 3.1 8B responded)"
else
    echo "ERROR: OpenRouter test failed: $TEST"
    exit 1
fi

# ── Launch parallel scenarios ────────────────────────────────────────
echo "$(date) | Launching heterogeneous experiment (3 parallel scenarios, $RUNS runs each)"

PIDS=""
for scenario in $SCENARIOS; do
    outdir="${OUTBASE}_${scenario}"
    mkdir -p "$outdir"
    nohup python3.10 scripts/eval_heterogeneous.py \
        --scenario "$scenario" \
        --runs-per-condition "$RUNS" \
        --output-dir "$outdir" \
        --openrouter-key "$OPENROUTER_API_KEY" \
        > "${outdir}/run.log" 2>&1 &
    PID=$!
    PIDS="$PIDS $PID"
    echo "$(date) | Launched ${scenario}: PID $PID -> ${outdir}/run.log"
done

echo ""
echo "$(date) | All scenarios launched! PIDs:$PIDS"
echo ""
echo "Monitor with:"
echo "  tail -f ${OUTBASE}_*/run.log"
echo ""
echo "Check progress:"
echo "  for d in ${OUTBASE}_*/progress.json; do echo \$d; python3.10 -c \"import json; d=json.load(open('\$d')); print(f'  {len(d.get(\\\"games\\\",[]))}/18 games')\"; done"
echo ""
echo "Estimated: 18 games/scenario × ~25min = ~7.5 hrs per scenario process"

# ── Monitor and report ───────────────────────────────────────────────
echo ""
echo "$(date) | Monitoring progress (checking every 5 min)..."

while true; do
    all_done=true
    for pid in $PIDS; do
        if kill -0 $pid 2>/dev/null; then
            all_done=false
            break
        fi
    done

    if $all_done; then
        echo ""
        echo "$(date) | All scenarios finished!"
        break
    fi

    # Progress report
    for scenario in $SCENARIOS; do
        outdir="${OUTBASE}_${scenario}"
        done_count=$(python3.10 -c "
import json
try:
    with open('${outdir}/progress.json') as f:
        data = json.load(f)
    print(len(data.get('games', [])))
except:
    print('?')
" 2>/dev/null)
        printf "  %s: %s/18 games  " "$scenario" "$done_count"
    done
    echo ""
    sleep 300
done

# ── Merge results ────────────────────────────────────────────────────
echo "$(date) | Merging results..."
python3.10 -c "
import json
from collections import defaultdict

all_games = []
for sc in ['baseline', 'trade_war', 'gpu_contention']:
    path = '${OUTBASE}_' + sc + '/progress.json'
    try:
        with open(path) as f:
            data = json.load(f)
        all_games.extend(data.get('games', []))
        print(f'  {sc}: {len(data.get(\"games\", []))} games')
    except Exception as e:
        print(f'  {sc}: ERROR {e}')

# Deduplicate
seen = set()
unique = []
for g in all_games:
    key = (g['team_size'], g['seed_pct'], g['scenario'], g['run_index'])
    if key not in seen:
        seen.add(key)
        unique.append(g)

merged = {
    'games': unique,
    'config': {
        'trained_model': 'redblackbench',
        'untrained_model': 'meta-llama/llama-3.1-8b-instruct',
        'team_size': 5,
        'seed_pcts': [0, 20, 40, 60, 80, 100],
        'scenarios': ['baseline', 'trade_war', 'gpu_contention'],
        'runs_per_condition': $RUNS,
        'opponent': 'always_defect',
        'experiment_type': 'heterogeneous',
    }
}

import os
os.makedirs('${OUTBASE}', exist_ok=True)
with open('${OUTBASE}/progress_merged.json', 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Merged: {len(unique)} unique games -> ${OUTBASE}/progress_merged.json')
"

echo "$(date) | Done! Results in ${OUTBASE}/"
