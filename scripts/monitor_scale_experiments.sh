#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Monitor & Auto-Repair for N=10 and N=15 Scale Effect Experiments
#
# Features:
#   1. Checks all 6 processes (3 scenarios × 2 team sizes) every 5 min
#   2. vLLM health checks on both endpoints
#   3. Auto-restart dead processes with --resume
#   4. Progress reporting
#   5. Merge results when all done
#   6. Call Claude Code to diagnose & fix when auto-repair fails
#
# Usage:
#   nohup bash scripts/monitor_scale_experiments.sh \
#       > results/monitor_scale.log 2>&1 &
# ═══════════════════════════════════════════════════════════════════════

cd /workspace/RedBlackBench

# ── Config ────────────────────────────────────────────────────────────
VLLM_N10="https://mum6vd8kfdw86u-8888.proxy.runpod.net/v1"
VLLM_N15="https://ollgpp3ltvuum0-8888.proxy.runpod.net/v1"

SCENARIOS="baseline trade_war gpu_contention"
RUNS=5
GAMES_PER_SCENARIO=30  # 6 seed% × 5 runs

MONITOR_INTERVAL=300   # 5 min
MAX_AUTO_RETRIES=3     # per scenario, before calling Claude
VLLM_RETRY_WAIT=120    # 2 min wait if vLLM is down

LOG="results/monitor_scale.log"
mkdir -p results

# Track retries per job
declare -A RETRY_COUNT
declare -A PIDS

# ── Logging ───────────────────────────────────────────────────────────
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') | $*"; }

# ── vLLM Health Check ─────────────────────────────────────────────────
check_vllm() {
    local url="$1"
    local label="$2"

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 15 "${url}/models" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        # Quick inference test
        TEST=$(python3.10 -c "
from openai import OpenAI
c = OpenAI(api_key='dummy', base_url='${url}')
r = c.chat.completions.create(
    model='/root/models/Qwen3-14B',
    messages=[{'role':'user','content':'Say OK'}],
    max_tokens=8,
    extra_body={'chat_template_kwargs': {'enable_thinking': False}}
)
print('OK')
" 2>&1) || true
        if echo "$TEST" | grep -q "OK"; then
            return 0
        else
            log "WARNING: ${label} HTTP OK but inference FAILED: $TEST"
            return 1
        fi
    else
        log "WARNING: ${label} unreachable (HTTP $HTTP_CODE)"
        return 1
    fi
}

# ── Progress Counter ──────────────────────────────────────────────────
count_games() {
    local outdir="$1"
    python3.10 -c "
import json
try:
    with open('${outdir}/progress.json') as f:
        data = json.load(f)
    print(len(data.get('games', [])))
except:
    print('0')
" 2>/dev/null
}

# ── Get last error from run.log ───────────────────────────────────────
get_last_error() {
    local logfile="$1"
    # Grab last 50 lines, look for ERROR/Traceback/Exception
    tail -50 "$logfile" 2>/dev/null | grep -iE "error|traceback|exception|failed|timeout" | tail -10
}

# ── Call Claude Code for diagnosis ────────────────────────────────────
call_claude_for_help() {
    local team_size="$1"
    local scenario="$2"
    local outdir="$3"
    local vllm_url="$4"
    local error_context="$5"

    log "═══ CALLING CLAUDE FOR HELP ═══"
    log "  Job: N=${team_size} ${scenario}"
    log "  Dir: ${outdir}"
    log "  Error: ${error_context}"

    # Write a diagnosis request file
    local diag_file="${outdir}/claude_diagnosis_request.md"
    cat > "$diag_file" << DIAGEOF
# Auto-Repair Failed — Claude Diagnosis Needed

## Job Info
- Team size: N=${team_size}
- Scenario: ${scenario}
- Output dir: ${outdir}
- vLLM URL: ${vllm_url}
- Auto-retries exhausted: ${MAX_AUTO_RETRIES}

## Last Error
\`\`\`
${error_context}
\`\`\`

## What to check
1. Is the vLLM endpoint responding? \`curl ${vllm_url}/models\`
2. Check run.log for the root cause: \`tail -100 ${outdir}/run.log\`
3. Check if there's a Python error in eval_scale_effect.py
4. Check disk space: \`df -h /workspace\`
5. Check if progress.json is corrupted

## Expected fix
Diagnose the issue, fix it, and restart with:
\`\`\`bash
nohup python3.10 scripts/eval_scale_effect.py \\
    --team-size ${team_size} --scenario ${scenario} \\
    --runs-per-condition ${RUNS} --output-dir ${outdir} \\
    --vllm-url "${vllm_url}" --resume \\
    > ${outdir}/run.log 2>&1 &
\`\`\`
DIAGEOF

    # Call Claude Code non-interactively
    if command -v claude &>/dev/null; then
        log "Invoking Claude Code..."
        local claude_prompt="An experiment process died and auto-retry failed ${MAX_AUTO_RETRIES} times. \
Job: N=${team_size} ${scenario}, dir: ${outdir}, vLLM: ${vllm_url}. \
Last error: ${error_context}. \
Please: 1) Read ${outdir}/run.log (last 200 lines) to diagnose. \
2) Check vLLM health: curl ${vllm_url}/models. \
3) Fix the issue if possible. \
4) Restart the job with --resume. \
5) Write the new PID to ${outdir}/new_pid.txt so the monitor can track it."

        timeout 300 claude --print -p "$claude_prompt" \
            > "${outdir}/claude_repair.log" 2>&1 || true

        # Check if Claude wrote a new PID
        if [ -f "${outdir}/new_pid.txt" ]; then
            local new_pid=$(cat "${outdir}/new_pid.txt" | tr -d '[:space:]')
            if kill -0 "$new_pid" 2>/dev/null; then
                log "Claude restarted job with PID $new_pid"
                PIDS["${team_size}_${scenario}"]=$new_pid
                RETRY_COUNT["${team_size}_${scenario}"]=0
                return 0
            fi
        fi
        log "Claude repair attempt completed — check ${outdir}/claude_repair.log"
    else
        log "Claude Code CLI not found — manual intervention needed"
        log "See diagnosis request: ${diag_file}"
    fi
    return 1
}

# ── Restart a failed job ──────────────────────────────────────────────
restart_job() {
    local team_size="$1"
    local scenario="$2"
    local outdir="$3"
    local vllm_url="$4"
    local job_key="${team_size}_${scenario}"

    local retries=${RETRY_COUNT[$job_key]:-0}

    if [ $retries -ge $MAX_AUTO_RETRIES ]; then
        local error_ctx=$(get_last_error "${outdir}/run.log")
        call_claude_for_help "$team_size" "$scenario" "$outdir" "$vllm_url" "$error_ctx"
        return
    fi

    # Check vLLM first
    local vllm_label="vLLM-N${team_size}"
    if ! check_vllm "$vllm_url" "$vllm_label"; then
        log "  vLLM down for N=${team_size}, waiting ${VLLM_RETRY_WAIT}s before restart..."
        sleep $VLLM_RETRY_WAIT
        if ! check_vllm "$vllm_url" "$vllm_label"; then
            log "  vLLM still down — skipping restart, will retry next cycle"
            return
        fi
    fi

    retries=$((retries + 1))
    RETRY_COUNT[$job_key]=$retries
    log "  Auto-restart N=${team_size} ${scenario} (attempt ${retries}/${MAX_AUTO_RETRIES})..."

    # Rotate log
    if [ -f "${outdir}/run.log" ]; then
        cp "${outdir}/run.log" "${outdir}/run_crashed_$(date -u +%H%M%S).log"
    fi

    nohup python3.10 scripts/eval_scale_effect.py \
        --team-size "$team_size" \
        --scenario "$scenario" \
        --runs-per-condition $RUNS \
        --output-dir "$outdir" \
        --vllm-url "$vllm_url" \
        --resume \
        > "${outdir}/run.log" 2>&1 &
    local new_pid=$!
    PIDS[$job_key]=$new_pid
    log "  Restarted: PID $new_pid"
}

# ── Merge results for a team size ─────────────────────────────────────
merge_results() {
    local team_size="$1"
    log "Merging N=${team_size} results..."

    python3.10 -c "
import json, math
from collections import defaultdict

all_games = []
for sc in ['baseline', 'trade_war', 'gpu_contention']:
    path = f'results/eval_scale_effect_n${team_size}_{sc}/progress.json'
    try:
        with open(path) as f:
            data = json.load(f)
        all_games.extend(data['games'])
        print(f'  {sc}: {len(data[\"games\"])} games')
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

# Save merged
merged = {
    'games': unique,
    'config': {
        'team_size': ${team_size},
        'seed_pcts': [0, 20, 40, 60, 80, 100],
        'scenarios': ['baseline', 'trade_war', 'gpu_contention'],
        'runs_per_condition': ${RUNS},
    }
}
outpath = f'results/eval_scale_effect_n${team_size}_merged.json'
with open(outpath, 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Merged: {len(unique)} unique games -> {outpath}')

# Summary table
by_cond = defaultdict(list)
for g in unique:
    by_cond[(g['seed_pct'], g['scenario'])].append(g['cooperation_rate'] * 100)

pcts = sorted(set(g['seed_pct'] for g in unique))
scenarios = ['baseline', 'trade_war', 'gpu_contention']

lines = [f'# Scale Effect N=${team_size} — Cooperation Rate (mean ± std, ${RUNS} runs)', '']
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
                row += f' {m:.0f}±{s:.0f} (n={len(rates)}) |'
            else:
                row += f' {m:.0f} (n=1) |'
        else:
            row += ' — |'
    if sc_means:
        om = sum(sc_means)/len(sc_means)
        row += f' {om:.0f} |'
    else:
        row += ' — |'
    lines.append(row)

lines.append(f'')
lines.append(f'Total: {len(unique)} games')
summary = '\n'.join(lines)
print()
print(summary)

sumpath = f'results/eval_scale_effect_n${team_size}_summary.md'
with open(sumpath, 'w') as f:
    f.write(summary)
print(f'\nSaved: {sumpath}')
"
}

# ═══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════

log "═══════════════════════════════════════════════════════════"
log "SCALE EXPERIMENT MONITOR — N=10 + N=15"
log "═══════════════════════════════════════════════════════════"
log "N=10 vLLM: $VLLM_N10"
log "N=15 vLLM: $VLLM_N15"
log "Check interval: $((MONITOR_INTERVAL/60)) min"
log "Max auto-retries: $MAX_AUTO_RETRIES (then calls Claude)"
log "═══════════════════════════════════════════════════════════"

# ── Detect existing PIDs ──────────────────────────────────────────────
log "Detecting running experiment processes..."
for team_size in 10 15; do
    for scenario in $SCENARIOS; do
        job_key="${team_size}_${scenario}"
        RETRY_COUNT[$job_key]=0

        # Find PID by matching command line
        pid=$(pgrep -f "eval_scale_effect.py.*--team-size ${team_size}.*--scenario ${scenario}" 2>/dev/null | head -1)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            PIDS[$job_key]=$pid
            log "  Found N=${team_size} ${scenario}: PID $pid"
        else
            PIDS[$job_key]=""
            log "  N=${team_size} ${scenario}: not running"
        fi
    done
done

# ── Track completion ──────────────────────────────────────────────────
N10_DONE=false
N15_DONE=false

while true; do
    log "── Status Check ────────────────────────────────────────"

    n10_total=0
    n15_total=0

    for team_size in 10 15; do
        if [ "$team_size" = "10" ]; then
            vllm_url="$VLLM_N10"
        else
            vllm_url="$VLLM_N15"
        fi

        for scenario in $SCENARIOS; do
            job_key="${team_size}_${scenario}"
            outdir="results/eval_scale_effect_n${team_size}_${scenario}"
            pid="${PIDS[$job_key]}"
            done_count=$(count_games "$outdir")

            if [ "$team_size" = "10" ]; then
                n10_total=$((n10_total + done_count))
            else
                n15_total=$((n15_total + done_count))
            fi

            # Check if complete
            if [ "$done_count" -ge "$GAMES_PER_SCENARIO" ]; then
                log "  N=${team_size} ${scenario}: ${done_count}/${GAMES_PER_SCENARIO} COMPLETE"
                continue
            fi

            # Check if process is alive
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                log "  N=${team_size} ${scenario}: ${done_count}/${GAMES_PER_SCENARIO} running (PID $pid)"
            else
                if [ "$done_count" -lt "$GAMES_PER_SCENARIO" ]; then
                    log "  N=${team_size} ${scenario}: ${done_count}/${GAMES_PER_SCENARIO} DEAD — restarting..."
                    restart_job "$team_size" "$scenario" "$outdir" "$vllm_url"
                fi
            fi
        done
    done

    log "  TOTALS: N=10: ${n10_total}/90 | N=15: ${n15_total}/90"

    # Check if N=10 just finished
    if [ "$n10_total" -ge 90 ] && [ "$N10_DONE" = "false" ]; then
        N10_DONE=true
        log "═══ N=10 ALL COMPLETE! Merging results... ═══"
        merge_results 10
    fi

    # Check if N=15 just finished
    if [ "$n15_total" -ge 90 ] && [ "$N15_DONE" = "false" ]; then
        N15_DONE=true
        log "═══ N=15 ALL COMPLETE! Merging results... ═══"
        merge_results 15
    fi

    # All done?
    if [ "$N10_DONE" = "true" ] && [ "$N15_DONE" = "true" ]; then
        log "═══════════════════════════════════════════════════════════"
        log "ALL EXPERIMENTS COMPLETE! 180 games total."
        log "  N=10: results/eval_scale_effect_n10_summary.md"
        log "  N=15: results/eval_scale_effect_n15_summary.md"
        log "═══════════════════════════════════════════════════════════"
        break
    fi

    sleep $MONITOR_INTERVAL
done
