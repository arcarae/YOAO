#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# MASTER SCRIPT: Run N=10 then N=15 scale effect experiments
# Total: 180 games (90 per team size)
# Designed for H200 141GB GPU with vLLM serving Qwen3-14B + LoRA
#
# Usage:
#   # On the H200 machine, after starting vLLM:
#   VLLM_URL=http://localhost:8000/v1 nohup bash scripts/run_scale_n10_n15_full.sh \
#       > results/scale_n10_n15_master.log 2>&1 &
#
# Architecture:
#   Phase 1: N=10 (3 scenarios × 6 seed% × 5 runs = 90 games, ~5-6 hr)
#   Phase 2: N=15 (3 scenarios × 6 seed% × 5 runs = 90 games, ~8-10 hr)
#   Auto-retry on failure, resume-safe
#
# Expected output:
#   results/eval_scale_effect_n10_baseline/     (30 games)
#   results/eval_scale_effect_n10_trade_war/    (30 games)
#   results/eval_scale_effect_n10_gpu_contention/ (30 games)
#   results/eval_scale_effect_n10_summary.md
#   results/eval_scale_effect_n15_baseline/     (30 games)
#   results/eval_scale_effect_n15_trade_war/    (30 games)
#   results/eval_scale_effect_n15_gpu_contention/ (30 games)
#   results/eval_scale_effect_n15_summary.md
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail
cd /workspace/RedBlackBench

export VLLM_URL="${VLLM_URL:-http://194.68.245.87:22067/v1}"

echo "$(date -u) | ══════════════════════════════════════════════════"
echo "$(date -u) | SCALE EFFECT FULL RUN: N=10 + N=15"
echo "$(date -u) | vLLM: ${VLLM_URL}"
echo "$(date -u) | Total: 180 games (~14-16 hours on H200)"
echo "$(date -u) | ══════════════════════════════════════════════════"

echo ""
echo "$(date -u) | ── PHASE 1: N=10 (90 games) ──────────────────────"
bash scripts/run_scale_n10.sh
echo "$(date -u) | ── PHASE 1 COMPLETE ───────────────────────────────"

echo ""
echo "$(date -u) | ── PHASE 2: N=15 (90 games) ──────────────────────"
bash scripts/run_scale_n15.sh
echo "$(date -u) | ── PHASE 2 COMPLETE ───────────────────────────────"

echo ""
echo "$(date -u) | ══════════════════════════════════════════════════"
echo "$(date -u) | ALL DONE! 180 games complete."
echo "$(date -u) | Results:"
echo "$(date -u) |   N=10: results/eval_scale_effect_n10_summary.md"
echo "$(date -u) |   N=15: results/eval_scale_effect_n15_summary.md"
echo "$(date -u) | ══════════════════════════════════════════════════"
