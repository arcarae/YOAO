#!/bin/bash
# Phase 2: Wait for phase 1 (3 runs) to finish, then add runs 4-5
# Auto-chain: monitors PIDs from phase 1, then launches --resume with 5 runs
#
# Usage: OPENROUTER_API_KEY=sk-xxx bash scripts/run_heterogeneous_phase2.sh

cd /workspace/RedBlackBench

VLLM_URL="http://194.68.245.87:22067/v1"
OPENROUTER_KEY="${OPENROUTER_API_KEY}"
SCENARIOS="baseline trade_war gpu_contention"
OUTBASE="results/eval_heterogeneous"
TOTAL_RUNS=5

if [ -z "$OPENROUTER_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set"
    exit 1
fi

# ── Wait for phase 1 processes to finish ─────────────────────────────
echo "$(date) | Phase 2: Waiting for phase 1 (3-run) processes to finish..."

# Find phase 1 PIDs (eval_heterogeneous.py with --runs-per-condition 3)
while true; do
    RUNNING=$(ps aux | grep "eval_heterogeneous.py" | grep "runs-per-condition 3" | grep -v grep | wc -l)
    if [ "$RUNNING" -eq 0 ]; then
        echo "$(date) | Phase 1 complete! All 3-run processes finished."
        break
    fi
    echo "$(date) | Still waiting... $RUNNING phase-1 processes running"

    # Progress check
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
        printf "  %s: %s/18  " "$scenario" "$done_count"
    done
    echo ""
    sleep 120
done

# ── Health checks before phase 2 ────────────────────────────────────
echo ""
echo "$(date) | Running health checks..."

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "${VLLM_URL}/models" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    echo "$(date) | WARNING: vLLM not responding (HTTP $HTTP_CODE). Waiting 60s and retrying..."
    sleep 60
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "${VLLM_URL}/models" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" != "200" ]; then
        echo "$(date) | ERROR: vLLM still not responding. Aborting phase 2."
        exit 1
    fi
fi
echo "$(date) | vLLM OK"

TEST=$(python3.10 -c "
from openai import OpenAI
c = OpenAI(api_key='${OPENROUTER_KEY}', base_url='https://openrouter.ai/api/v1')
r = c.chat.completions.create(model='meta-llama/llama-3.1-8b-instruct', messages=[{'role':'user','content':'Say OK'}], max_tokens=8)
print(r.choices[0].message.content[:10])
" 2>&1)
if echo "$TEST" | grep -qi "ok"; then
    echo "$(date) | OpenRouter OK"
else
    echo "$(date) | WARNING: OpenRouter test failed: $TEST"
    echo "$(date) | Proceeding anyway (may recover)..."
fi

# ── Launch phase 2: resume with 5 total runs ────────────────────────
echo ""
echo "$(date) | Launching phase 2: extending to ${TOTAL_RUNS} runs per condition (adding runs 4-5)"

PIDS=""
for scenario in $SCENARIOS; do
    outdir="${OUTBASE}_${scenario}"

    # Check current progress
    current=$(python3.10 -c "
import json
with open('${outdir}/progress.json') as f:
    data = json.load(f)
print(len(data.get('games', [])))
" 2>/dev/null)
    echo "$(date) | ${scenario}: ${current}/18 games from phase 1, extending to ${TOTAL_RUNS} runs (30 total games)"

    nohup python3.10 scripts/eval_heterogeneous.py \
        --scenario "$scenario" \
        --runs-per-condition "$TOTAL_RUNS" \
        --output-dir "$outdir" \
        --openrouter-key "$OPENROUTER_KEY" \
        --resume \
        > "${outdir}/run_phase2.log" 2>&1 &
    PID=$!
    PIDS="$PIDS $PID"
    echo "$(date) | Launched ${scenario} phase 2: PID $PID -> ${outdir}/run_phase2.log"
done

echo ""
echo "$(date) | Phase 2 launched! PIDs:$PIDS"
echo ""

# ── Monitor phase 2 ─────────────────────────────────────────────────
echo "$(date) | Monitoring phase 2 progress..."

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
        echo "$(date) | Phase 2 complete! All scenarios finished 5 runs."
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
        printf "  %s: %s/30 games  " "$scenario" "$done_count"
    done
    echo ""
    sleep 300
done

# ── Final merge ──────────────────────────────────────────────────────
echo "$(date) | Merging all 5-run results..."
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
        'runs_per_condition': $TOTAL_RUNS,
        'opponent': 'always_defect',
        'experiment_type': 'heterogeneous',
    }
}

os.makedirs('${OUTBASE}', exist_ok=True)
with open('${OUTBASE}/progress_merged_5runs.json', 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Merged: {len(unique)} unique games -> ${OUTBASE}/progress_merged_5runs.json')
"

echo "$(date) | ALL DONE! 5 runs × 3 scenarios × 6 seed pcts = 90 total games"
echo "$(date) | Results in ${OUTBASE}/"
