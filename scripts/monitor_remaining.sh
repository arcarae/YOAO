#!/bin/bash
# Monitor RedBlackBench remaining experiments in real-time
#
# Usage:
#   bash scripts/monitor_remaining.sh [log_dir]
#
# If log_dir is not specified, uses the most recent logs/remaining_* directory.

set -euo pipefail

# Find log directory
if [ $# -ge 1 ]; then
    LOG_DIR="$1"
else
    LOG_DIR=$(ls -td logs/remaining_* 2>/dev/null | head -1)
    if [ -z "$LOG_DIR" ]; then
        echo "No log directory found. Start experiments first:"
        echo "  python3.10 scripts/run_remaining_all.py --vllm-url ..."
        exit 1
    fi
fi

SUMMARY="$LOG_DIR/summary.json"
HEALTH_LOG="$LOG_DIR/vllm_health.log"
MASTER_LOG="$LOG_DIR/master.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No color

format_time() {
    local seconds=$1
    if [ "$seconds" -lt 60 ]; then
        echo "${seconds}s"
    elif [ "$seconds" -lt 3600 ]; then
        echo "$((seconds / 60))m $((seconds % 60))s"
    else
        echo "$((seconds / 3600))h $((seconds % 3600 / 60))m"
    fi
}

# Main loop
REFRESH=10
echo -e "${BOLD}Monitoring: $LOG_DIR${NC}"
echo -e "Refresh every ${REFRESH}s. Press Ctrl+C to stop.\n"

while true; do
    clear

    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  RedBlackBench — Experiment Monitor${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "  Log dir: ${CYAN}$LOG_DIR${NC}"
    echo ""

    if [ ! -f "$SUMMARY" ]; then
        echo -e "  ${YELLOW}Waiting for experiments to start...${NC}"
        echo -e "  (summary.json not found yet)"
        sleep $REFRESH
        continue
    fi

    # Parse summary.json
    TOTAL=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('total',0))" 2>/dev/null || echo "?")
    COMPLETED=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('completed',0))" 2>/dev/null || echo "?")
    FAILED=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('failed',0))" 2>/dev/null || echo "?")
    SKIPPED=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('skipped',0))" 2>/dev/null || echo "?")
    REMAINING=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('remaining',0))" 2>/dev/null || echo "?")
    IN_PROGRESS=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('in_progress',0))" 2>/dev/null || echo "?")
    PROGRESS=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('progress_pct',0))" 2>/dev/null || echo "?")
    ELAPSED=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('elapsed_seconds',0))" 2>/dev/null || echo "0")
    ETA=$(python3 -c "import json; d=json.load(open('$SUMMARY')); e=d.get('eta_seconds'); print(int(e) if e else -1)" 2>/dev/null || echo "-1")
    CONSEC_FAIL=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('consecutive_failures',0))" 2>/dev/null || echo "0")
    TIMESTAMP=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('timestamp','?'))" 2>/dev/null || echo "?")

    # Progress bar
    if [ "$TOTAL" != "?" ] && [ "$TOTAL" -gt 0 ]; then
        DONE_COUNT=$((COMPLETED + SKIPPED))
        BAR_WIDTH=40
        FILLED=$((DONE_COUNT * BAR_WIDTH / TOTAL))
        EMPTY=$((BAR_WIDTH - FILLED))
        BAR=$(printf '█%.0s' $(seq 1 $FILLED 2>/dev/null) 2>/dev/null || true)
        SPACE=$(printf '░%.0s' $(seq 1 $EMPTY 2>/dev/null) 2>/dev/null || true)
        echo -e "  ${BOLD}Progress:${NC} [${GREEN}${BAR}${NC}${SPACE}] ${PROGRESS}%"
    fi
    echo ""

    # Stats
    echo -e "  ${BOLD}总进度:${NC}     ${GREEN}$COMPLETED${NC} completed  ${RED}$FAILED${NC} failed  ${YELLOW}$SKIPPED${NC} skipped  ${BLUE}$IN_PROGRESS${NC} active  ${NC}$REMAINING remaining"
    echo -e "  ${BOLD}总数:${NC}       $TOTAL games"

    if [ "$ELAPSED" != "0" ]; then
        echo -e "  ${BOLD}已用时:${NC}     $(format_time $ELAPSED)"
    fi
    if [ "$ETA" != "-1" ] && [ "$ETA" -gt 0 ]; then
        echo -e "  ${BOLD}预计剩余:${NC}   $(format_time $ETA)"
    fi
    echo ""

    # Pool stats
    echo -e "  ${BOLD}── Pools ──${NC}"
    VLLM_ACTIVE=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('pools',{}).get('vllm',{}).get('active',0))" 2>/dev/null || echo "?")
    VLLM_DONE=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('pools',{}).get('vllm',{}).get('completed',0))" 2>/dev/null || echo "?")
    VLLM_FAIL=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('pools',{}).get('vllm',{}).get('failed',0))" 2>/dev/null || echo "?")
    VLLM_PAUSED=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('pools',{}).get('vllm',{}).get('paused',False))" 2>/dev/null || echo "False")

    OR_ACTIVE=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('pools',{}).get('openrouter',{}).get('active',0))" 2>/dev/null || echo "?")
    OR_DONE=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('pools',{}).get('openrouter',{}).get('completed',0))" 2>/dev/null || echo "?")
    OR_FAIL=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('pools',{}).get('openrouter',{}).get('failed',0))" 2>/dev/null || echo "?")
    OR_PAUSED=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('pools',{}).get('openrouter',{}).get('paused',False))" 2>/dev/null || echo "False")

    VLLM_STATUS="${GREEN}running${NC}"
    if [ "$VLLM_PAUSED" = "True" ]; then VLLM_STATUS="${RED}PAUSED${NC}"; fi
    OR_STATUS="${GREEN}running${NC}"
    if [ "$OR_PAUSED" = "True" ]; then OR_STATUS="${RED}PAUSED${NC}"; fi

    echo -e "  vLLM:       ${VLLM_ACTIVE} active  ${VLLM_DONE} done  ${VLLM_FAIL} fail  [$VLLM_STATUS]"
    echo -e "  OpenRouter: ${OR_ACTIVE} active  ${OR_DONE} done  ${OR_FAIL} fail  [$OR_STATUS]"
    echo ""

    # vLLM health
    echo -e "  ${BOLD}── vLLM Health ──${NC}"
    if [ -f "$HEALTH_LOG" ]; then
        LAST_HEALTH=$(tail -1 "$HEALTH_LOG" 2>/dev/null || echo "unknown")
        if echo "$LAST_HEALTH" | grep -q "OK"; then
            echo -e "  Status: ${GREEN}OK${NC}  ($(echo $LAST_HEALTH | cut -d' ' -f1))"
        else
            echo -e "  Status: ${RED}FAIL${NC}  ($LAST_HEALTH)"
        fi
    else
        echo -e "  Status: ${YELLOW}no data${NC}"
    fi
    echo ""

    # Consecutive failures warning
    if [ "$CONSEC_FAIL" -ge 2 ]; then
        echo -e "  ${RED}${BOLD}⚠️  连续失败 $CONSEC_FAIL 次！运行: claude 来诊断问题${NC}"
        echo ""
    fi

    # Recent failures
    RECENT_FAILS=$(python3 -c "
import json
d = json.load(open('$SUMMARY'))
fails = d.get('recent_failures', [])
for f in fails[-5:]:
    print(f'  {f.get(\"timestamp\",\"?\")[:19]}  {f.get(\"job_id\",\"?\")}')
" 2>/dev/null || true)

    if [ -n "$RECENT_FAILS" ]; then
        echo -e "  ${BOLD}── Recent Failures ──${NC}"
        echo "$RECENT_FAILS"
        echo ""
    fi

    # Recent log lines
    echo -e "  ${BOLD}── Recent Activity ──${NC}"
    if [ -f "$MASTER_LOG" ]; then
        tail -8 "$MASTER_LOG" 2>/dev/null | while IFS= read -r line; do
            if echo "$line" | grep -q "DONE"; then
                echo -e "  ${GREEN}$line${NC}"
            elif echo "$line" | grep -q "FAIL\|ERROR"; then
                echo -e "  ${RED}$line${NC}"
            elif echo "$line" | grep -q "START"; then
                echo -e "  ${BLUE}$line${NC}"
            else
                echo -e "  $line"
            fi
        done
    fi
    echo ""

    echo -e "  ${CYAN}Last updated: $TIMESTAMP${NC}"
    echo -e "  ${CYAN}Refreshing every ${REFRESH}s... (Ctrl+C to stop)${NC}"

    sleep $REFRESH
done
