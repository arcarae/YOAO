#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Scale Effect N=15: 15-agent teams, 6 seed%, 3 scenarios, 5 runs
# Total: 90 games | Auto-resume on failure | vLLM health checks
#
# Usage:
#   bash scripts/run_scale_n15.sh                    # default vLLM URL
#   VLLM_URL=http://x.x.x.x:8000/v1 bash scripts/run_scale_n15.sh
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Config ────────────────────────────────────────────────────────────
VLLM_URL="${VLLM_URL:-${YOAO_VLLM_URL:-http://localhost:8000/v1}}"
SCENARIOS="baseline trade_war gpu_contention"
RUNS=5
TEAM_SIZE=15
OUTBASE="results/eval_scale_effect_n15"
TOTAL_PER_SCENARIO=$((6 * RUNS))  # 6 seed% × 5 runs = 30
MAX_RETRIES=10
RETRY_WAIT=60
MONITOR_INTERVAL=300  # 5 min

# ── Logging ───────────────────────────────────────────────────────────
MASTER_LOG="${OUTBASE}_master.log"
mkdir -p "$(dirname "$MASTER_LOG")"

log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') | $*" | tee -a "$MASTER_LOG"; }

log "═══════════════════════════════════════════════════════════"
log "SCALE EFFECT N=${TEAM_SIZE} — 3 scenarios × 6 seed% × ${RUNS} runs = $((TOTAL_PER_SCENARIO * 3)) games"
log "vLLM: ${VLLM_URL}"
log "Output: ${OUTBASE}_<scenario>/"
log "═══════════════════════════════════════════════════════════"

# ── vLLM Health Check with Retry ──────────────────────────────────────
check_vllm() {
    local retry=0
    while true; do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "${VLLM_URL}/models" 2>/dev/null || echo "000")
        if [ "$HTTP_CODE" = "200" ]; then
            # Quick inference test
            TEST_RESULT=$(python3 -c "
from openai import OpenAI
c = OpenAI(api_key='dummy', base_url='${VLLM_URL}')
r = c.chat.completions.create(
    model='Qwen/Qwen3-14B',
    messages=[{'role':'user','content':'Say OK'}],
    max_tokens=8,
    extra_body={'chat_template_kwargs': {'enable_thinking': False}}
)
print('INFERENCE_OK', r.choices[0].message.content[:20])
" 2>&1) || true
            if echo "$TEST_RESULT" | grep -q "INFERENCE_OK"; then
                log "vLLM OK — inference test passed"
                return 0
            else
                log "vLLM HTTP OK but inference FAILED: $TEST_RESULT"
            fi
        else
            log "vLLM unreachable (HTTP $HTTP_CODE)"
        fi

        retry=$((retry + 1))
        if [ $retry -ge $MAX_RETRIES ]; then
            log "FATAL: vLLM not responding after $MAX_RETRIES retries. Aborting."
            return 1
        fi
        log "Retrying in ${RETRY_WAIT}s... ($retry/$MAX_RETRIES)"
        sleep $RETRY_WAIT
    done
}

# ── Progress Counter ──────────────────────────────────────────────────
count_progress() {
    local outdir="$1"
    python3 -c "
import json
try:
    with open('${outdir}/progress.json') as f:
        data = json.load(f)
    print(len(data.get('games', [])))
except:
    print('0')
" 2>/dev/null
}

# ── Health Check ──────────────────────────────────────────────────────
check_vllm || exit 1

# ── Launch 3 Scenarios in Parallel ────────────────────────────────────
log "Launching ${TEAM_SIZE}-agent evaluation..."

PIDS=""
for scenario in $SCENARIOS; do
    outdir="${OUTBASE}_${scenario}"
    mkdir -p "$outdir"

    nohup python3 scripts/eval_scale_effect.py \
        --team-size $TEAM_SIZE \
        --scenario "$scenario" \
        --runs-per-condition $RUNS \
        --output-dir "$outdir" \
        --vllm-url "$VLLM_URL" \
        --resume \
        > "${outdir}/run.log" 2>&1 &
    PID=$!
    PIDS="$PIDS $PID"
    log "  ${scenario}: PID $PID → ${outdir}/run.log"
done

log "All scenarios launched! PIDs:$PIDS"

# ── Monitor Loop ──────────────────────────────────────────────────────
log "Monitoring every $((MONITOR_INTERVAL/60)) min..."

while true; do
    all_done=true
    for pid in $PIDS; do
        if kill -0 $pid 2>/dev/null; then
            all_done=false
            break
        fi
    done

    if $all_done; then
        log "All scenario processes finished!"
        break
    fi

    # Progress report
    status=""
    total_done=0
    for scenario in $SCENARIOS; do
        outdir="${OUTBASE}_${scenario}"
        done=$(count_progress "$outdir")
        total_done=$((total_done + done))
        status="${status}  ${scenario}: ${done}/${TOTAL_PER_SCENARIO}"
    done
    log "Progress [${total_done}/$((TOTAL_PER_SCENARIO * 3))] |${status}"

    # vLLM liveness check — warn but don't kill
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "${VLLM_URL}/models" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" != "200" ]; then
        log "WARNING: vLLM health check failed (HTTP $HTTP_CODE) — processes may stall"
    fi

    sleep $MONITOR_INTERVAL
done

# ── Check for Failures & Auto-Retry ──────────────────────────────────
log "Checking for incomplete scenarios..."
NEED_RETRY=false

for scenario in $SCENARIOS; do
    outdir="${OUTBASE}_${scenario}"
    done=$(count_progress "$outdir")
    if [ "$done" -lt "$TOTAL_PER_SCENARIO" ]; then
        log "  ${scenario}: ${done}/${TOTAL_PER_SCENARIO} — INCOMPLETE, will retry"
        NEED_RETRY=true
    else
        log "  ${scenario}: ${done}/${TOTAL_PER_SCENARIO} — COMPLETE ✓"
    fi
done

if $NEED_RETRY; then
    log "── Auto-Retry Phase ──────────────────────────────────────"
    check_vllm || { log "vLLM down, cannot retry. Manual intervention needed."; exit 1; }

    PIDS=""
    for scenario in $SCENARIOS; do
        outdir="${OUTBASE}_${scenario}"
        done=$(count_progress "$outdir")
        if [ "$done" -lt "$TOTAL_PER_SCENARIO" ]; then
            nohup python3 scripts/eval_scale_effect.py \
                --team-size $TEAM_SIZE \
                --scenario "$scenario" \
                --runs-per-condition $RUNS \
                --output-dir "$outdir" \
                --vllm-url "$VLLM_URL" \
                --resume \
                > "${outdir}/run_retry.log" 2>&1 &
            PID=$!
            PIDS="$PIDS $PID"
            log "  Retry ${scenario}: PID $PID"
        fi
    done

    # Wait for retries
    for pid in $PIDS; do
        wait $pid 2>/dev/null || true
    done
    log "Retry phase complete."
fi

# ── Final Summary ─────────────────────────────────────────────────────
log "═══════════════════════════════════════════════════════════"
log "FINAL STATUS — N=${TEAM_SIZE}"
log "═══════════════════════════════════════════════════════════"

GRAND_TOTAL=0
for scenario in $SCENARIOS; do
    outdir="${OUTBASE}_${scenario}"
    done=$(count_progress "$outdir")
    GRAND_TOTAL=$((GRAND_TOTAL + done))
    log "  ${scenario}: ${done}/${TOTAL_PER_SCENARIO}"
done
log "  TOTAL: ${GRAND_TOTAL}/$((TOTAL_PER_SCENARIO * 3))"

# Merge all results into one summary
python3 -c "
import json, math
from collections import defaultdict

all_games = []
for sc in ['baseline', 'trade_war', 'gpu_contention']:
    path = 'results/eval_scale_effect_n15_' + sc + '/progress.json'
    try:
        with open(path) as f:
            data = json.load(f)
        all_games.extend(data['games'])
    except:
        pass

# Deduplicate
seen = set()
unique = []
for g in all_games:
    key = (g['team_size'], g['seed_pct'], g['scenario'], g['run_index'])
    if key not in seen:
        seen.add(key)
        unique.append(g)

# Generate summary table
by_cond = defaultdict(list)
for g in unique:
    by_cond[(g['seed_pct'], g['scenario'])].append(g['cooperation_rate'] * 100)

pcts = sorted(set(g['seed_pct'] for g in unique))
scenarios = ['baseline', 'trade_war', 'gpu_contention']

lines = ['# Scale Effect N=15 — Cooperation Rate (mean +/- std, 5 runs)', '']
lines.append('| Seed% | ' + ' | '.join(scenarios) + ' | Overall |')
lines.append('|-------|' + '|'.join(['-------'] * (len(scenarios)+1)) + '|')

for pct in pcts:
    row = f'| {pct}% |'
    sc_means = []
    for sc in scenarios:
        rates = by_cond.get((pct, sc), [])
        if rates:
            m = sum(rates)/len(rates)
            sc_means.append(m)
            if len(rates) > 1:
                s = math.sqrt(sum((r-m)**2 for r in rates)/(len(rates)-1))
                row += f' {m:.0f}+/-{s:.0f} (n={len(rates)}) |'
            else:
                row += f' {m:.0f} (n=1) |'
        else:
            row += ' -- |'
    if sc_means:
        om = sum(sc_means)/len(sc_means)
        row += f' {om:.0f} |'
    else:
        row += ' -- |'
    lines.append(row)

lines.append(f'')
lines.append(f'Total: {len(unique)} games')

summary = '\n'.join(lines)
print(summary)

with open('results/eval_scale_effect_n15_summary.md', 'w') as f:
    f.write(summary)
print(f'\nSaved to results/eval_scale_effect_n15_summary.md')
"

log "Done! Summary: results/eval_scale_effect_n15_summary.md"
