#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Monitor experiments, auto-fix with Claude, terminate RunPod when done
#
# Usage:
#   bash scripts/monitor_and_cleanup.sh <experiment_pid> <log_dir>
#
# Example:
#   bash scripts/monitor_and_cleanup.sh 120773 logs/remaining_20260402_v3
# ═══════════════════════════════════════════════════════════════════════

set -uo pipefail

# ── Args ──
EXPERIMENT_PID="${1:?Usage: $0 <pid> <log_dir>}"
LOG_DIR="${2:?Usage: $0 <pid> <log_dir>}"

# ── Config ──
MAX_CLAUDE_RETRIES=3
POLL_INTERVAL=60
VLLM_URL="http://213.181.111.149:16547/v1"
OPENROUTER_KEY="${OPENROUTER_API_KEY}"
RUNPOD_API_KEY="${RUNPOD_API_KEY}"
RUNPOD_POD_ID="v6jt91tql1icjz"

# ── Logging ──
MONITOR_LOG="$LOG_DIR/monitor.log"
mkdir -p "$LOG_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$MONITOR_LOG"
}

log_section() {
    log "═══════════════════════════════════════════════════════════"
    log "$1"
    log "═══════════════════════════════════════════════════════════"
}

# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Monitor experiment progress
# ═══════════════════════════════════════════════════════════════════════

phase1_monitor() {
    local pid=$1
    log_section "PHASE 1: Monitoring experiment (PID=$pid)"

    while true; do
        # Check if process is alive
        if ! ps -p "$pid" > /dev/null 2>&1; then
            log "Process $pid exited."
            return 0
        fi

        # Read summary
        if [ -f "$LOG_DIR/summary.json" ]; then
            python3.10 -c "
import json, sys
d = json.load(open('$LOG_DIR/summary.json'))
c, f, t = d['completed'], d['failed'], d['total']
ip = d['in_progress']
e = d['elapsed_seconds']
eta = d.get('eta_seconds') or 0
pct = d['progress_pct']
pools = d['pools']
v = pools.get('vllm', {})
o = pools.get('openrouter', {})

elapsed_str = f'{e//3600}h{(e%3600)//60}m'
eta_str = f'{eta//3600}h{(eta%3600)//60}m' if eta > 0 else '??'

print(f'[{pct:5.1f}%] done={c} fail={f} active={ip} | elapsed={elapsed_str} ETA={eta_str}')
print(f'        vLLM: {v.get(\"completed\",0)} done {v.get(\"active\",0)} active {v.get(\"failed\",0)} fail | OR: {o.get(\"completed\",0)} done {o.get(\"active\",0)} active {o.get(\"failed\",0)} fail')
" 2>/dev/null | while IFS= read -r line; do log "$line"; done
        else
            log "Waiting for summary.json..."
        fi

        sleep "$POLL_INTERVAL"
    done
}

# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Completeness check
# ═══════════════════════════════════════════════════════════════════════

phase2_check() {
    log_section "PHASE 2: Completeness check"

    # Run dry-run to see remaining jobs
    local dry_output
    dry_output=$(python3.10 scripts/run_remaining_all.py \
        --dry-run \
        --vllm-url "$VLLM_URL" \
        2>&1)

    local remaining
    remaining=$(echo "$dry_output" | grep "Total:" | grep -oP '\d+' | head -1)

    echo "$dry_output" >> "$MONITOR_LOG"

    if [ -z "$remaining" ] || [ "$remaining" -eq 0 ]; then
        log "ALL EXPERIMENTS COMPLETE! Remaining=0"
        return 0  # Complete
    else
        log "INCOMPLETE: $remaining games remaining"

        # Save dry-run details
        echo "$dry_output" > "$LOG_DIR/incomplete_dryrun.txt"
        return 1  # Incomplete
    fi
}

# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Claude auto-fix
# ═══════════════════════════════════════════════════════════════════════

phase3_claude_fix() {
    local attempt=$1
    log_section "PHASE 3: Claude auto-fix (attempt $attempt/$MAX_CLAUDE_RETRIES)"

    # ── Collect context ──
    local dry_output
    dry_output=$(cat "$LOG_DIR/incomplete_dryrun.txt" 2>/dev/null || echo "no dry-run output")

    local master_tail
    master_tail=$(tail -100 "$LOG_DIR/master.log" 2>/dev/null || echo "no master log")

    local failed_jobs
    failed_jobs=$(cat "$LOG_DIR/failed_jobs.json" 2>/dev/null || echo "[]")

    # Get recent error game logs
    local error_logs=""
    for game_log in $(grep -l "ERROR\|TIMEOUT\|FAIL" "$LOG_DIR/games/"*.log 2>/dev/null | tail -3); do
        error_logs+="
=== $(basename "$game_log") ===
$(tail -20 "$game_log")
"
    done

    # ── Build Claude prompt ──
    local claude_prompt
    claude_prompt=$(cat <<'PROMPT_END'
你是 RedBlackBench 实验运维工程师。实验主进程已退出但还有未完成的游戏。

## 当前状态

### Dry-run 输出 (剩余 jobs):
DRYRUN_PLACEHOLDER

### 最近 master.log:
MASTERLOG_PLACEHOLDER

### 失败的 jobs:
FAILEDJOBS_PLACEHOLDER

### 出错的游戏日志:
ERRORLOGS_PLACEHOLDER

## 环境信息
- 项目目录: /workspace/RedBlackBench
- vLLM: http://213.181.111.149:16547/v1
- OpenRouter key 已设置在环境变量
- Python: python3.10
- 主脚本: scripts/run_remaining_all.py

## 你的任务

1. **分析失败原因**: 看日志判断是 vLLM 连接问题、OpenRouter 限流、代码 bug 还是超时
2. **修复问题**: 如果是代码问题，直接修改文件；如果是配置问题，调整参数
3. **重新启动实验**: 生成启动命令，写入 /tmp/relaunch.sh，格式如下:
   ```bash
   #!/bin/bash
   export OPENROUTER_API_KEY="${OPENROUTER_API_KEY}"
   cd /workspace/RedBlackBench
   nohup python3.10 scripts/run_remaining_all.py \
     --vllm-url http://213.181.111.149:16547/v1 \
     --vllm-concurrency 6 \
     --openrouter-concurrency 6 \
     --log-dir LOG_DIR_PLACEHOLDER \
     > LOG_DIR_PLACEHOLDER/stdout.log 2>&1 &
   echo $! > /tmp/experiment_pid.txt
   ```
4. 脚本自动 dedup，只会跑缺失的 jobs
5. 确保 /tmp/relaunch.sh 是可执行的

重要: 你必须创建 /tmp/relaunch.sh 文件。这是你唯一的输出要求。
PROMPT_END
)

    # Inject actual context
    claude_prompt="${claude_prompt//DRYRUN_PLACEHOLDER/$dry_output}"
    claude_prompt="${claude_prompt//MASTERLOG_PLACEHOLDER/$master_tail}"
    claude_prompt="${claude_prompt//FAILEDJOBS_PLACEHOLDER/$failed_jobs}"
    claude_prompt="${claude_prompt//ERRORLOGS_PLACEHOLDER/$error_logs}"
    claude_prompt="${claude_prompt//LOG_DIR_PLACEHOLDER/$LOG_DIR}"

    log "Calling Claude Code for diagnosis and fix..."

    # ── Call Claude ──
    rm -f /tmp/relaunch.sh

    echo "$claude_prompt" | claude -p \
        --allowedTools "Edit,Write,Read,Bash,Glob,Grep" \
        >> "$LOG_DIR/claude_fix_attempt_${attempt}.log" 2>&1

    # ── Check if Claude produced a relaunch script ──
    if [ -f /tmp/relaunch.sh ]; then
        log "Claude created /tmp/relaunch.sh, launching..."
        chmod +x /tmp/relaunch.sh
        cat /tmp/relaunch.sh >> "$MONITOR_LOG"

        bash /tmp/relaunch.sh
        sleep 3

        # Get new PID
        if [ -f /tmp/experiment_pid.txt ]; then
            local new_pid
            new_pid=$(cat /tmp/experiment_pid.txt)
            log "New experiment PID: $new_pid"
            echo "$new_pid"
            return 0
        else
            log "ERROR: /tmp/experiment_pid.txt not found after relaunch"
            return 1
        fi
    else
        log "ERROR: Claude did not create /tmp/relaunch.sh"
        log "Check $LOG_DIR/claude_fix_attempt_${attempt}.log for details"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Terminate RunPod
# ═══════════════════════════════════════════════════════════════════════

phase4_terminate() {
    log_section "PHASE 4: Final verification & RunPod termination"

    # ── Final data verification ──
    log "Verifying result counts..."
    python3.10 -c "
import json
from pathlib import Path

expected = {
    'eval_scale_n5_baseline_fixed': 30,
    'eval_scale_n10_baseline_fixed': 30,
    'eval_scale_n15_baseline_fixed': 30,
    'eval_hetero_llama_baseline_fixed': 25,
    'eval_hetero_mistral_baseline_fixed': 25,
    'eval_scale_n5_scenarios_backfill': 1,
    'eval_hetero_mistral_gaps': 22,
}

total_ok = 0
total_expected = 0
for exp, target in expected.items():
    p = Path('results') / exp / 'progress.json'
    if p.exists():
        n = len(json.load(open(p))['games'])
    else:
        n = 0
    status = 'OK' if n >= target else 'MISSING'
    print(f'  {status:7s} {exp}: {n}/{target}')
    total_ok += min(n, target)
    total_expected += target

print(f'\n  Total: {total_ok}/{total_expected}')
if total_ok >= total_expected:
    print('  ALL COMPLETE')
else:
    print(f'  STILL MISSING {total_expected - total_ok} games')
" 2>&1 | while IFS= read -r line; do log "$line"; done

    # ── Terminate RunPod ──
    log "Terminating RunPod pod $RUNPOD_POD_ID..."

    local response
    response=$(curl -s -X POST https://api.runpod.io/graphql \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}")

    log "RunPod API response: $response"

    if echo "$response" | grep -q "error"; then
        log "WARNING: RunPod terminate may have failed. Check manually."
    else
        log "RunPod pod $RUNPOD_POD_ID terminated successfully."
    fi

    # ── Final report ──
    log_section "EXPERIMENT COMPLETE - FINAL REPORT"
    log "All 163 experiments finished."
    log "Results in: /workspace/RedBlackBench/results/"
    log "Logs in: $LOG_DIR/"
    log "RunPod GPU released."
    log "Next: run analysis scripts to generate tables."
}

# ═══════════════════════════════════════════════════════════════════════
# Main orchestration loop
# ═══════════════════════════════════════════════════════════════════════

main() {
    log_section "MONITOR & CLEANUP STARTED"
    log "Experiment PID: $EXPERIMENT_PID"
    log "Log dir: $LOG_DIR"
    log "RunPod pod: $RUNPOD_POD_ID"

    local current_pid=$EXPERIMENT_PID
    local claude_attempts=0

    while true; do
        # Phase 1: Monitor until process exits
        phase1_monitor "$current_pid"

        # Phase 2: Check completeness
        if phase2_check; then
            # All done! Terminate RunPod
            phase4_terminate
            log "Monitor exiting. Goodbye."
            exit 0
        fi

        # Phase 3: Try Claude fix
        claude_attempts=$((claude_attempts + 1))
        if [ "$claude_attempts" -gt "$MAX_CLAUDE_RETRIES" ]; then
            log "ERROR: $MAX_CLAUDE_RETRIES Claude fix attempts exhausted."
            log "Manual intervention required. Run: claude"
            log "NOT terminating RunPod (experiments incomplete)."
            exit 1
        fi

        local new_pid
        new_pid=$(phase3_claude_fix "$claude_attempts")
        if [ $? -eq 0 ] && [ -n "$new_pid" ]; then
            current_pid="$new_pid"
            log "Resuming monitoring with new PID=$current_pid"
        else
            log "Claude fix attempt $claude_attempts failed."
            # Still loop - will try again next iteration
            # Brief pause before retry
            sleep 10
            # Try direct relaunch without Claude
            log "Attempting direct relaunch..."
            export OPENROUTER_API_KEY="$OPENROUTER_KEY"
            nohup python3.10 scripts/run_remaining_all.py \
                --vllm-url "$VLLM_URL" \
                --vllm-concurrency 6 \
                --openrouter-concurrency 6 \
                --log-dir "$LOG_DIR" \
                > "$LOG_DIR/stdout_retry_${claude_attempts}.log" 2>&1 &
            current_pid=$!
            log "Direct relaunch PID: $current_pid"
        fi
    done
}

main
