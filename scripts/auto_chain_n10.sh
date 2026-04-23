#!/bin/bash
# Auto-chain: wait for N=5 runs to finish, then launch N=10
# Checks vLLM health before proceeding

cd /workspace/RedBlackBench

VLLM_URL="http://194.68.245.87:22067/v1"
SCENARIOS="baseline trade_war gpu_contention"
N5_PIDS="9764 9765 9766"

echo "$(date) | Monitoring N=5 processes: $N5_PIDS"

# ── Phase 1: Wait for N=5 to finish ──────────────────────────────────
while true; do
    all_done=true
    for pid in $N5_PIDS; do
        if kill -0 $pid 2>/dev/null; then
            all_done=false
            break
        fi
    done
    if $all_done; then
        echo "$(date) | All N=5 processes finished!"
        break
    fi
    # Progress check
    for sc in $SCENARIOS; do
        done_count=$(python3.10 -c "
import json
with open('results/eval_scale_effect_${sc}/progress.json') as f:
    data = json.load(f)
print(len(data['games']))
" 2>/dev/null || echo "?")
        printf "  %s: %s/18 games  " "$sc" "$done_count"
    done
    echo ""
    sleep 300  # check every 5 min
done

# ── Phase 2: Check vLLM health ───────────────────────────────────────
echo "$(date) | Checking vLLM server..."
MAX_RETRIES=10
RETRY=0
while true; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "${VLLM_URL}/models" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "$(date) | vLLM server OK (HTTP $HTTP_CODE)"
        # Quick inference test
        TEST_RESULT=$(python3.10 -c "
import urllib.request, json
req = urllib.request.Request('${VLLM_URL}/chat/completions',
    data=json.dumps({
        'model': 'Qwen/Qwen3-14B',
        'messages': [{'role':'user','content':'Say OK'}],
        'max_tokens': 8
    }).encode(),
    headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req, timeout=30)
data = json.loads(resp.read())
print('INFERENCE_OK')
" 2>&1)
        if echo "$TEST_RESULT" | grep -q "INFERENCE_OK"; then
            echo "$(date) | Inference test passed"
            break
        else
            echo "$(date) | Inference test FAILED: $TEST_RESULT"
        fi
    else
        echo "$(date) | vLLM unreachable (HTTP $HTTP_CODE)"
    fi
    RETRY=$((RETRY + 1))
    if [ $RETRY -ge $MAX_RETRIES ]; then
        echo "$(date) | FATAL: vLLM not responding after $MAX_RETRIES retries. Aborting."
        exit 1
    fi
    echo "$(date) | Retrying in 60s... ($RETRY/$MAX_RETRIES)"
    sleep 60
done

# ── Phase 3: Merge N=5 results ────────────────────────────────────────
echo "$(date) | Merging N=5 results..."
python3.10 -c "
import json
from collections import defaultdict

all_games = []
for sc in ['baseline', 'trade_war', 'gpu_contention']:
    path = f'results/eval_scale_effect_{sc}/progress.json'
    try:
        with open(path) as f:
            data = json.load(f)
        all_games.extend(data['games'])
        print(f'  {sc}: {len(data[\"games\"])} games')
    except Exception as e:
        print(f'  {sc}: ERROR {e}')

# Deduplicate by (team_size, seed_pct, scenario, run_index)
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
        'vllm_url': '${VLLM_URL}',
        'trained_model': 'redblackbench',
        'untrained_model': 'Qwen/Qwen3-14B',
        'team_sizes': [5],
        'seed_pcts': [0, 20, 40, 60, 80, 100],
        'scenarios': ['baseline', 'trade_war', 'gpu_contention'],
        'runs_per_condition': 3,
        'opponent': 'always_defect',
    }
}
with open('results/eval_scale_effect/progress_n5_3runs.json', 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Merged: {len(unique)} unique games saved to progress_n5_3runs.json')
"

# ── Phase 4: Launch N=10 x 5 runs (3 parallel scenario processes) ────
echo "$(date) | Launching N=10 evaluation (5 runs, 3 parallel scenarios)..."

for scenario in $SCENARIOS; do
    outdir="results/eval_scale_effect_n10_${scenario}"
    mkdir -p "$outdir"
    nohup python3.10 scripts/eval_scale_effect.py \
        --team-size 10 \
        --scenario "$scenario" \
        --runs-per-condition 5 \
        --output-dir "$outdir" \
        > "${outdir}/run.log" 2>&1 &
    echo "$(date) | Launched N=10 ${scenario}: PID $!"
done

echo "$(date) | N=10 launched! Monitor with:"
echo "  tail -f results/eval_scale_effect_n10_*/run.log"
echo ""
echo "Estimated: 30 games/scenario × ~65min = ~32.5 hrs per process"
