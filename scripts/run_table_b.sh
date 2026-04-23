#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# Claude-Orchestrated Runner for Table B — Scale Effect
#
# Claude Code IS the brain. It:
#   1. Validates setup, fixes issues, runs sanity checks
#   2. Runs each batch of games by team size
#   3. Monitors results, detects anomalies
#   4. Diagnoses and fixes crashes
#   5. Writes interim and final analysis
#
# vLLM server is health-checked by Claude every 5 min on the
# other pod. If it goes down, we poll every 30s here. Worst case
# it's back in ~10 min.
#
# Usage:
#   nohup bash scripts/run_table_b.sh > /dev/null 2>&1 &
#   tail -f results/eval_scale_effect/orchestrator.log
# ══════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/results/eval_scale_effect"
ORCH_LOG="$OUTPUT_DIR/orchestrator.log"
VLLM_URL="http://194.68.245.87:22067/v1"

mkdir -p "$OUTPUT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$ORCH_LOG"
}

wait_for_vllm() {
    local max_wait=900  # 15 min max (server restarts within ~10 min)
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s --connect-timeout 5 "$VLLM_URL/models" 2>&1 | grep -q "redblackbench"; then
            return 0
        fi
        sleep 30
        waited=$((waited + 30))
        log "  vLLM not reachable (${waited}s/${max_wait}s)..."
    done
    return 1
}

log "══════════════════════════════════════════"
log "CLAUDE-ORCHESTRATED TABLE B EXPERIMENT"
log "══════════════════════════════════════════"

cd "$PROJECT_DIR"

# ── Phase 1: Claude validates setup & runs sanity check ──────────
log ""
log "PHASE 1: Setup validation & sanity check"

# Wait for vLLM first
if ! curl -s --connect-timeout 5 "$VLLM_URL/models" 2>&1 | grep -q "redblackbench"; then
    log "vLLM not up yet, waiting..."
    if ! wait_for_vllm; then
        log "FATAL: vLLM not reachable. Exiting."
        exit 1
    fi
fi

claude -p "$(cat <<'PROMPT'
You are orchestrating the Table B Scale Effect experiment for the YOAO/Alignment Propagation ICLR 2026 LLA Workshop paper.

Working directory: /workspace/RedBlackBench
Script: scripts/eval_scale_effect.py
Output: results/eval_scale_effect/
Python: python3.10
vLLM: http://194.68.245.87:22067/v1 (models: Qwen/Qwen3-14B base, redblackbench SFT LoRA)
The vLLM server has a health-check from the other pod every 5 minutes. If it goes down it auto-restarts — worst case ~10 min downtime.

PHASE 1 — VALIDATION & SANITY CHECK

Do these steps in order. If any step fails, fix the issue and retry before moving on.

1. **Verify vLLM**: curl http://194.68.245.87:22067/v1/models — confirm both models listed.

2. **Syntax check**: Run `python3.10 -c "import ast; ast.parse(open('scripts/eval_scale_effect.py').read()); print('OK')"`. If it fails, read the file, fix the syntax error, retry.

3. **Import check**: Run `cd /workspace/RedBlackBench && python3.10 -c "import sys; sys.path.insert(0,'.'); from scripts.eval_scale_effect import *; print('Imports OK')"`. Fix any import errors.

4. **Single game test**: Run ONE game to validate the full pipeline:
   ```
   python3.10 scripts/eval_scale_effect.py --team-size 5 --seed-pct 0 --scenario baseline --runs-per-condition 1 --output-dir results/eval_scale_effect
   ```
   - If it crashes, read the traceback carefully
   - Read the relevant source files (eval_scale_effect.py, coordinator.py, team.py, etc.)
   - Fix the bug
   - Retry. Keep fixing until it works. You have up to 5 attempts.

5. **Validate result**: Read results/eval_scale_effect/progress.json:
   - Did a game complete with cooperation_rate between 0.0 and 1.0?
   - Is team_a_score a reasonable integer?

6. **N=5 quick sweep**: Run all seed percentages at N=5 with 1 run each to compare against paper:
   ```
   python3.10 scripts/eval_scale_effect.py --team-size 5 --runs-per-condition 1 --resume --output-dir results/eval_scale_effect
   ```
   This should produce ~18 games (6 seed_pcts × 3 scenarios). Fix crashes and retry if needed.

7. **Sanity check**: Read progress.json. Paper reports held-out cooperation:
   - 0% seed → ~26%
   - 20% seed → ~62%
   - 40% seed → ~83%
   Pattern should be monotonically increasing. If wildly off, investigate model assignment logic.

8. **Write status**: Write results/eval_scale_effect/phase1_status.txt with findings and any issues fixed.
PROMPT
)" --allowedTools "Bash(*),Read,Edit,Write,Glob,Grep" 2>&1 | tee -a "$ORCH_LOG"

log "Phase 1 exit: $?"

# ── Phase 2: Full experiment, batched by team size ───────────────
for TEAM_SIZE in 5 10 15; do
    log ""
    log "══ PHASE 2: N=$TEAM_SIZE (5 runs per condition) ══"

    # Pre-check vLLM before each batch
    if ! curl -s --connect-timeout 5 "$VLLM_URL/models" 2>&1 | grep -q "redblackbench"; then
        log "vLLM down before N=$TEAM_SIZE batch, waiting..."
        if ! wait_for_vllm; then
            log "FATAL: vLLM gone. Exiting."
            exit 1
        fi
    fi

    claude -p "$(cat <<PROMPT
You are orchestrating the Table B Scale Effect experiment.

Working directory: /workspace/RedBlackBench
Output: results/eval_scale_effect/
Python: python3.10
vLLM: http://194.68.245.87:22067/v1

The vLLM server is health-checked every 5 min on the other pod and auto-restarts. If you get a connection error, wait 30 seconds and retry. It should be back within ~10 min worst case.

PHASE 2 — RUN N=${TEAM_SIZE} BATCH

You need to get ALL games for N=${TEAM_SIZE} completed. That's 6 seed_pcts × 3 scenarios × 5 runs = 90 games.

RETRY STRATEGY — this is critical:
- Always use --resume so completed games are skipped
- On connection/timeout errors: wait 60 seconds, check vLLM with curl, retry
- On code bugs: read the error, read the source, fix the bug, retry
- Track how many games complete after each attempt
- Keep retrying until all 90 games are done or you've tried 15 times with no progress

STEPS:

1. **Check existing progress**: Read results/eval_scale_effect/progress.json, count games where team_size==${TEAM_SIZE}.

2. **Run the batch**:
   \`\`\`
   python3.10 scripts/eval_scale_effect.py --team-size ${TEAM_SIZE} --runs-per-condition 5 --resume --output-dir results/eval_scale_effect 2>&1
   \`\`\`

3. **If it crashes**:
   a. Check the error. Is it a connection/timeout issue or a code bug?
   b. For connection issues:
      - Run: curl -s http://194.68.245.87:22067/v1/models
      - If down, wait 60s and check again (up to 10 min)
      - Once back, retry the command
   c. For code bugs:
      - Read the traceback
      - Read the relevant source files
      - Fix the bug (minimal change, don't refactor)
      - Retry

4. **After completion or max retries**: Read progress.json, count N=${TEAM_SIZE} games.

5. **Interim analysis**: Write results/eval_scale_effect/analysis_N${TEAM_SIZE}.md with:
   - Mean ± std cooperation rate per seed %
   - Does 20% seed threshold hold at N=${TEAM_SIZE}?
   - Quick comparison: at what seed % does cooperation cross 50%?
   - How many games completed vs expected (90)?
   - Any failed/missing conditions?

6. **Print current table** so it shows in the log:
   \`\`\`
   python3.10 -c "
import json, math
from collections import defaultdict
d = json.load(open('results/eval_scale_effect/progress.json'))
grouped = defaultdict(list)
for g in d['games']:
    grouped[(g['team_size'], g['seed_pct'])].append(g['cooperation_rate']*100)
sizes = sorted(set(g['team_size'] for g in d['games']))
pcts = sorted(set(g['seed_pct'] for g in d['games']))
print()
print('| Seed % |' + ''.join(f' N={n} |' for n in sizes))
print('|--------|' + '------|' * len(sizes))
for p in pcts:
    row = f'| {p}% |'
    for n in sizes:
        rates = grouped.get((n,p), [])
        if rates:
            m = sum(rates)/len(rates)
            s = math.sqrt(sum((r-m)**2 for r in rates)/(len(rates)-1)) if len(rates)>1 else 0
            row += f' {m:.1f} ± {s:.1f} |'
        else:
            row += ' — |'
    print(row)
print()
"
   \`\`\`
PROMPT
)" --allowedTools "Bash(*),Read,Edit,Write,Glob,Grep" 2>&1 | tee -a "$ORCH_LOG"

    log "N=$TEAM_SIZE batch exit: $?"
done

# ── Phase 3: Final analysis ──────────────────────────────────────
log ""
log "PHASE 3: Final analysis"

claude -p "$(cat <<'PROMPT'
You are doing final analysis for the Table B Scale Effect experiment.

Working directory: /workspace/RedBlackBench
Output: results/eval_scale_effect/

PHASE 3 — FINAL ANALYSIS & REPORT

1. **Load all results** from results/eval_scale_effect/progress.json

2. **Completeness report**: How many of 270 expected games completed? List any missing conditions.

3. **Generate Table B** with mean ± std:

| Seed % | N=5 | N=10 | N=15 |
|--------|-----|------|------|
| 0%     | ?   | ?    | ?    |
| 20%    | ?   | ?    | ?    |
| ...    |     |      |      |

4. **Key research question**: Does the 20% seed threshold hold at N=10, N=15?
   - Compare cooperation at 20% seed across team sizes
   - At what seed % does each team size first cross 50% cooperation?
   - Does the monotonic increase hold at all team sizes?

5. **Per-scenario breakdown**: Table for each scenario (baseline, trade_war, gpu_contention)

6. **Comparison to paper**: N=5 results vs published Table 3 held-out column:
   - 0+5U: 26%, 1+4U: 62%, 2+3U: 83%, 3+2U: 79%, 4+1U: 97%, 5+0U: 96%

7. **Write final report**: Write results/eval_scale_effect/TABLE_B_FINAL.md with everything above, formatted for the paper appendix.

8. **Print the final table** to stdout.
PROMPT
)" --allowedTools "Bash(*),Read,Edit,Write,Glob,Grep" 2>&1 | tee -a "$ORCH_LOG"

log ""
log "══════════════════════════════════════════"
log "EXPERIMENT COMPLETE"
log "══════════════════════════════════════════"
log "Results:    $OUTPUT_DIR/progress.json"
log "Table B:    $OUTPUT_DIR/TABLE_B_FINAL.md"
log "Phase logs: $OUTPUT_DIR/analysis_N*.md"
log "Full log:   $ORCH_LOG"

# Count final results
if [ -f "$OUTPUT_DIR/progress.json" ]; then
    TOTAL=$(python3.10 -c "import json; d=json.load(open('$OUTPUT_DIR/progress.json')); print(len(d.get('games',[])))" 2>/dev/null)
    log "Total games completed: $TOTAL/270"
fi
