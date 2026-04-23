# Table 2 Re-Run Setup Status

**Date:** 2026-01-14
**Status:** ✅ Parser fixed, ⚠️ Additional setup required

---

## ✅ Completed

### 1. Comprehensive Parser Fix Implemented

Location: `/workspace/RedBlackBench/redblackbench/agents/llm_agent.py`

**Patterns now supported:**
- ✅ `VOTE: RED` / `VOTE: BLACK`
- ✅ `VOTE: A` / `VOTE: B` (CRITICAL FIX)
- ✅ `VOTE: OPTION A` / `VOTE: OPTION B`
- ✅ `RECOMMENDATION: RED/BLACK/A/B`
- ✅ Numbered options: `VOTE: 1` / `VOTE: 2`
- ✅ Markdown formatting: `VOTE: **A**`
- ✅ Strong patterns: "I recommend A", "Voting B", etc.
- ✅ Warning detection: "If we choose B, we lose..." → wants A
- ✅ Enhanced error logging

**Test results:** 27/27 tests passed ✓

### 2. Test Suite Created

Location: `/workspace/test_comprehensive_parser.py`

Validates all parser patterns work correctly.

---

## ⚠️ Missing Components (Need to Copy from nicole/dev branch)

### 1. Evaluation Script

**Needed:** `scripts/eval_robustness.py`

**Found on:** nicole/dev branch at `scripts/eval_robustness.py` (404 lines)

**What it does:**
- Runs LLM Team A (5 agents) vs Scripted Team B (hardcoded strategies)
- Tests against 10 opponent strategies
- Supports multiple scenarios and games per combo
- Generates trajectory files and cooperation rate tables

**Command to copy:**
```bash
cd /workspace/RedBlackBench
git show nicole/dev:scripts/eval_robustness.py > scripts/eval_robustness.py
chmod +x scripts/eval_robustness.py
```

### 2. Strategies Module

**Needed:** `redblackbench/strategies/` directory with scripted opponent implementations

**What it contains:**
- `__init__.py` - Exports `create_scripted_team()` and `list_strategies()`
- Strategy implementations for:
  - always_defect
  - always_cooperate
  - tit_for_tat
  - early_exploiter
  - early_exploiter_no_recovery
  - mid_exploiter
  - late_betrayer
  - critical_exploiter
  - defect_critical
  - mostly_cooperate

**Commands to copy:**
```bash
cd /workspace/RedBlackBench
git show nicole/dev:redblackbench/strategies/ | grep "\.py$" | while read file; do
    git show "nicole/dev:redblackbench/strategies/$file" > "redblackbench/strategies/$file"
done
```

### 3. Scenarios Module (if using scenario-based evaluation)

**Status:** Already exists but may be empty

**Needed files from nicole/dev:**
- `redblackbench/scenarios/__init__.py`
- `redblackbench/scenarios/base.py`
- `redblackbench/scenarios/climate.py`
- `redblackbench/scenarios/pandemic.py`
- etc.

---

## 🚫 Blockers

### 1. vLLM Server Not Running

**Issue:** `curl http://localhost:8000/health` fails

**Models needed:**
- `/workspace/models/Qwen3-14B` (base model)
- `qwen3-14b-v2` (trained model)

**How to start:**
```bash
# Check if models exist
ls -la /workspace/models/

# Start vLLM with both models
# (exact command depends on your vLLM setup)
```

---

## 📋 Setup Checklist

- [x] 1. Implement comprehensive parser fix
- [x] 2. Test parser with examples
- [ ] 3. Copy `eval_robustness.py` from nicole/dev
- [ ] 4. Copy `strategies/` module from nicole/dev
- [ ] 5. (Optional) Copy/verify `scenarios/` module
- [ ] 6. Start vLLM server with required models
- [ ] 7. Test with 1-2 games
- [ ] 8. Run full Table 2 (20 games: 2 models × 10 strategies)

---

## 🎯 Table 2 Specification

**Evaluation:** Robustness to Opponent Strategies

**Setup:**
- Team A: 5 LLM agents (all same model)
- Team B: Scripted opponent (hardcoded strategy)
- 10 rounds per game
- Multipliers: Round 5 (3x), Round 8 (5x), Round 10 (10x)

**Models to test:**
1. Qwen3-14B (base, untrained)
2. qwen3-14b-v2 (LoRA trained)

**Opponent strategies (10):**
1. always_defect
2. early_exploiter_no_recovery
3. defect_critical
4. critical_exploiter
5. late_betrayer
6. mid_exploiter
7. early_exploiter
8. tit_for_tat
9. mostly_cooperate
10. always_cooperate

**Total games:** 2 models × 10 strategies = 20 games

**Expected runtime:** ~2-3 hours (depending on model inference speed)

**Output:**
- Trajectory files with full game history
- Cooperation rate table (model × strategy)
- Location: `/workspace/eval_results/table2_rerun_fixed/`

---

## 🚀 Quick Start (Once Setup Complete)

```bash
cd /workspace/RedBlackBench

# Run full Table 2 evaluation
python3 scripts/eval_robustness.py \
    --model "/workspace/models/Qwen3-14B" \
    --output-dir "/workspace/eval_results/table2_rerun_fixed" \
    --scenario baseline \
    --games-per-combo 1

python3 scripts/eval_robustness.py \
    --model "qwen3-14b-v2" \
    --output-dir "/workspace/eval_results/table2_rerun_fixed" \
    --scenario baseline \
    --games-per-combo 1

# Or use the shell script
bash /workspace/run_table2_fixed.sh
```

---

## ✅ Why This Will Work Now

**Problems with old Table 2:**
1. ❌ Vote data missing in 16/20 files (resume bug)
2. ❌ Parser couldn't handle A/B patterns → inflated cooperation rates

**Fixed in new run:**
1. ✅ Fresh games (no resume) → no data loss
2. ✅ Comprehensive parser handles all patterns
3. ✅ Enhanced error logging → we'll see any parsing failures

**Result:** Clean, reliable Table 2 data

---

**Next Steps:** Copy missing modules from nicole/dev and start vLLM server.
