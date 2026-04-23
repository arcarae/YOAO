#!/bin/bash
# Heterogeneous experiment #2: Qwen3-14B SFT (seed) + Mistral Small 3.1 24B (untrained)
# Full 5-run auto-chain: runs all 5 at once
#
# Usage: OPENROUTER_API_KEY=sk-xxx bash scripts/run_heterogeneous_mistral.sh

cd /workspace/RedBlackBench

# ── Config ───────────────────────────────────────────────────────────
VLLM_URL="http://194.68.245.87:22067/v1"
OPENROUTER_KEY="${OPENROUTER_API_KEY}"
SCENARIOS="baseline trade_war gpu_contention"
OUTBASE="results/eval_heterogeneous_mistral"
RUNS=5
UNTRAINED_MODEL="mistralai/mistral-small-3.1-24b-instruct"

if [ -z "$OPENROUTER_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set"
    exit 1
fi

# ── Health checks ────────────────────────────────────────────────────
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "${VLLM_URL}/models" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    echo "$(date) | ERROR: vLLM not responding (HTTP $HTTP_CODE)"
    exit 1
fi
echo "$(date) | vLLM OK"

TEST=$(python3.10 -c "
from openai import OpenAI
c = OpenAI(api_key='${OPENROUTER_KEY}', base_url='https://openrouter.ai/api/v1')
r = c.chat.completions.create(model='${UNTRAINED_MODEL}', messages=[{'role':'user','content':'Say OK'}], max_tokens=8)
print(r.choices[0].message.content[:10])
" 2>&1)
if echo "$TEST" | grep -qi "ok"; then
    echo "$(date) | OpenRouter OK (${UNTRAINED_MODEL} responded)"
else
    echo "$(date) | ERROR: OpenRouter test failed: $TEST"
    exit 1
fi

# ── Launch 3 parallel scenarios, 5 runs each ────────────────────────
echo "$(date) | Launching Mistral heterogeneous experiment (3 scenarios × ${RUNS} runs × 6 seed%)"
echo "$(date) | Untrained model: ${UNTRAINED_MODEL}"

PIDS=""
for scenario in $SCENARIOS; do
    outdir="${OUTBASE}_${scenario}"
    mkdir -p "$outdir"
    nohup python3.10 scripts/eval_heterogeneous.py \
        --scenario "$scenario" \
        --runs-per-condition "$RUNS" \
        --output-dir "$outdir" \
        --openrouter-key "$OPENROUTER_KEY" \
        --untrained-model "$UNTRAINED_MODEL" \
        > "${outdir}/run.log" 2>&1 &
    PID=$!
    PIDS="$PIDS $PID"
    echo "$(date) | Launched ${scenario}: PID $PID -> ${outdir}/run.log"
done

echo ""
echo "$(date) | All scenarios launched! PIDs:$PIDS"
echo "$(date) | Total games: 3 scenarios × 6 seed% × ${RUNS} runs = $((3 * 6 * RUNS))"
echo ""

# ── Monitor ──────────────────────────────────────────────────────────
TOTAL=$((6 * RUNS))  # per scenario
echo "$(date) | Monitoring progress (every 5 min)..."

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
        printf "  %s: %s/%s  " "$scenario" "$done_count" "$TOTAL"
    done
    echo ""
    sleep 300
done

# ── Merge results ────────────────────────────────────────────────────
echo "$(date) | Merging results..."
python3.10 -c "
import json, os

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
        'untrained_model': '${UNTRAINED_MODEL}',
        'team_size': 5,
        'seed_pcts': [0, 20, 40, 60, 80, 100],
        'scenarios': ['baseline', 'trade_war', 'gpu_contention'],
        'runs_per_condition': $RUNS,
        'opponent': 'always_defect',
        'experiment_type': 'heterogeneous_mistral',
    }
}

os.makedirs('${OUTBASE}', exist_ok=True)
with open('${OUTBASE}/progress_merged.json', 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Merged: {len(unique)} unique games -> ${OUTBASE}/progress_merged.json')
"

echo "$(date) | DONE! Mistral heterogeneous: ${RUNS} runs × 3 scenarios × 6 seed% = $((3 * 6 * RUNS)) games"
echo "$(date) | Results in ${OUTBASE}*/"
