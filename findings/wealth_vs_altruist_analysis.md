# Wealth vs Altruist Experiment Analysis

**Generated**: 2026-01-12
**Experiments Analyzed**:
- Wealth Maximization: `goal_wealth/experiment_20260110_235811_313639_1316/`
- Altruist: `goal_altruist/experiment_20260110_165339_496412_2832/`

---

## Executive Summary

### Key Findings

1. **Reputation System: WORKING** ✅
   - Initial concern about "stuck at 0.5" was unfounded
   - Reputation successfully updates based on trade behavior
   - Range: 0.43 to 1.00 (mean final: 0.650)

2. **Critical Experimental Confound** ⚠️
   - Wealth experiment used **dialogue** trade mode (LLM-negotiated)
   - Altruist experiment used **MRS** trade mode (automated formula)
   - **This makes direct comparison INVALID**

3. **Altruists Outperformed by 42.7%** (but it's misleading)
   - Final welfare: 2656 (altruist) vs 1862 (wealth)
   - Peak welfare: 4088 (altruist) vs 2342 (wealth) at tick 60
   - **Primary cause**: MRS efficiency, NOT altruistic behavior

4. **The Altruist Paradox**
   - Altruists achieved HIGHER total welfare
   - But also HIGHER inequality (Gini 0.422 vs 0.386)
   - Contradicts their "help struggling agents" programming
   - Confirms trade mode is the dominant factor

---

## Detailed Analysis

### 1. Reputation System Status

**VERDICT: The reputation system is NOT broken**

The earlier observation of "all agents at 0.5" was likely from:
- Initial tick data (all start at neutral 0.5)
- A different failed experiment
- Summary statistics at a specific time point

**Evidence of working reputation:**

```
Initial state: All agents = 0.500 (neutral)
Final state:   Mean = 0.650, Range [0.43, 1.00], σ = 0.203

Reputation update rules (confirmed in code):
✅ Honest trade completion: +0.05
✅ Fraud detected: -0.15
✅ Helping desperate partner: +0.08
```

**Observed behavior in wealth experiment:**
- 1,207 completed trades
- 12 fraud attempts (1.0% of completed trades)
- Reputation variance increased over time (agents differentiated)
- High-reputation agents (>0.7): 29.3% of trade records
- Low-reputation agents (<0.5): 0.2% of trade records

**Fraud rate (1.0%) suggests:**
- Qwen3-14B is naturally honest, OR
- The fraud prompt doesn't effectively trigger deception, OR
- Agents learned fraud is punished (-0.15 reputation hit)

---

### 2. Trade Mode Efficiency: The Real Story

#### Dialogue Mode (Wealth Experiment)

```
Total trade attempts: 2,776
Completed trades:     1,207 (43.5%)
Rejected trades:      1,566 (56.5%)

Cost per rejected trade:
- 4 LLM inference calls (2 agents × 2 stages)
- ~1,024 thinking tokens + 128 JSON tokens per agent
- Opportunity cost of coordination failure
```

**Key insight**: 56.5% of trade attempts FAIL, representing massive waste:
- Computational cost (6,264 wasted LLM calls for rejected trades)
- Time cost (agents can't move while trading)
- Coordination overhead (4 rounds of dialogue per attempt)

#### MRS Mode (Altruist Experiment)

```
Trade attempts logged: 0 (MRS is instant, automatic)
Success rate: ~100% (if MRS conditions met)
Cost: Pure computation (no LLM calls)
```

**Trade happens when**: Both agents have positive MRS and can mutually benefit
**Price**: Geometric mean of both agents' Marginal Rate of Substitution
**Result**: Instant, no rejection possible, no negotiation overhead

---

### 3. Temporal Dynamics

| Tick | Wealth (dialogue) | Altruist (MRS) | Difference | % Advantage |
|------|-------------------|----------------|------------|-------------|
| 10   | 1,462.0          | 1,746.6        | +284.6     | +19.5%      |
| 20   | 1,568.2          | 2,240.2        | +672.0     | +42.9%      |
| 30   | 1,753.8          | 2,672.2        | +918.4     | +52.4%      |
| 40   | 1,997.2          | 3,163.7        | +1,166.5   | +58.4%      |
| 50   | 2,171.3          | 3,667.0        | +1,495.7   | +68.9%      |
| **60**   | **2,341.5 (PEAK)**   | **4,087.8 (PEAK)**   | **+1,746.3**   | **+74.6%**      |
| 70   | 2,256.6          | 3,918.8        | +1,662.1   | +73.7%      |
| 80   | 2,072.6          | 3,424.7        | +1,352.1   | +65.2%      |
| 90   | 1,943.2          | 3,051.9        | +1,108.7   | +57.1%      |
| 100  | 1,861.9          | 2,656.2        | +794.3     | +42.7%      |

**Observations:**
1. **Both experiments peaked at tick 60** (synchronized decline suggests environmental factors)
2. **Altruist advantage grew until peak** (19.5% → 74.6%), then narrowed
3. **Altruist declined FASTER** post-peak (-35.0% vs -20.5%)
4. **Advantage INCREASES early game**: Trade efficiency compounds over time

---

### 4. Final Outcomes (Tick 100)

| Metric               | Wealth (dialogue) | Altruist (MRS) | Difference    |
|----------------------|-------------------|----------------|---------------|
| **Utilitarian Welfare** | 1,861.9        | 2,656.2        | +794.3 (+42.7%) |
| **Nash Welfare**        | 15.36          | 19.86          | +4.50 (+29.3%)  |
| **Rawlsian (min)**      | 1.25           | 1.73           | +0.48 (+38.7%)  |
| **Mean Wealth**         | 19.65          | 26.40          | +6.75 (+34.4%)  |
| **Gini Inequality**     | 0.386          | 0.422          | +0.036 (+9.3%)  |

**The Altruist Paradox:**
- Altruists programmed to "help struggling agents" and "share resources"
- Yet ended with **HIGHER inequality** than wealth-maximizers
- But achieved **HIGHER total welfare**

**Explanation:**
- MRS trade is more efficient → more total wealth created
- But automated formula doesn't redistribute → inequality persists
- Goal prompts barely matter when trade mode dominates outcomes

---

### 5. Gift Behavior: Zero Altruism Observed

```
Wealth experiment gifts:   0
Altruist experiment gifts: UNKNOWN (MRS mode doesn't log gifts separately)

Expected: Altruists should gift resources to desperate agents
Observed: No evidence of gifting behavior
```

**Why this matters:**
- Gifting is the PRIMARY way to test altruism
- Zero gifts in wealth experiment confirms selfish behavior (good validation)
- Cannot assess altruist gifting because MRS mode skipped LLM decision-making
- **Critical metric lost due to trade mode confound**

---

### 6. Impact of "Broken" Reputation System

**TLDR: Zero impact, because the system isn't broken**

If reputation had been stuck at 0.5 for all agents, the impact would be:

**In Dialogue Mode:**
- Agents can't distinguish trustworthy vs fraudulent partners
- May reduce trade acceptance rates (can't trust anyone)
- Fraud would go unpunished (no reputation cost)

**In MRS Mode:**
- Zero impact (reputation not used in automated formula)

**Actual Impact (working system):**
- Reputation variance (0.43 to 1.00) likely influenced trade acceptance
- 1.0% fraud rate suggests reputation-based trust worked
- High-reputation agents may have gotten more trades
- **But we can't quantify impact without partner trust logs in CSV**

**Partner Trust vs Global Reputation:**

The codebase has TWO trust systems:
1. **Global reputation** (per agent): Logged in CSV ✅
2. **Partner trust** (per agent-pair): NOT logged in CSV ❌

Partner trust is updated after each trade based on fraud detection:
```python
delta = 0.10 - 0.45 * avg_penalty
new_trust = trust + delta
```

This bilateral trust likely influenced trade decisions more than global reputation, but **we have no data** on it.

---

## Conclusions

### What We Learned

1. **Reputation system works correctly** - initial concern was unfounded

2. **Trade mode dominates outcomes** - MRS's 42.7% welfare advantage over dialogue

3. **LLM dialogue trading is inefficient**:
   - 56.5% rejection rate
   - 4 LLM calls per failed attempt
   - Massive computational and opportunity costs

4. **Goal prompts matter less than expected** when trade mechanics differ

5. **Need better experimental design**:
   - Same trade mode across goal comparisons
   - Log partner trust to analyze reputation impact
   - Separate goal effects from mechanism effects

### What We Still Don't Know

1. **How much does altruistic goal improve outcomes?**
   - Need altruist experiment with dialogue mode
   - Need wealth experiment with MRS mode for 2×2 comparison

2. **Does reputation actually influence trade acceptance?**
   - Need partner trust data logged in CSV
   - Need controlled experiment varying reputation

3. **Why zero gifts in wealth experiment?**
   - Is Qwen3-14B too honest/altruistic even with wealth goal?
   - Are agents not recognizing gift opportunities?
   - Is gift detection broken?

4. **Why did both experiments peak at tick 60?**
   - Resource depletion?
   - Population aging effects?
   - Coordination breakdown?

### Recommendations

1. **Rerun experiments with controlled trade mode:**
   ```
   Wealth + Dialogue  (DONE)
   Wealth + MRS       (NEEDED)
   Altruist + Dialogue (NEEDED)
   Altruist + MRS     (DONE)
   ```

2. **Add partner trust to trade_history.csv:**
   ```python
   "trust_a_to_b_before": self_agent.get_partner_trust(partner.agent_id),
   "trust_a_to_b_after": self_agent.get_partner_trust(partner.agent_id),
   ```

3. **Debug gift detection** - why zero gifts when altruists should give them?

4. **Investigate survival experiment crashes** - all 3+ attempts failed at tick 0

---

## Appendix: Configuration Differences

### Wealth Experiment
```yaml
trade_mode: dialogue
llm_goal_preset: wealth
llm_goal_prompt: |
  Your goal: Accumulate maximum resources.

  DECISION PRIORITY:
  1. CRITICAL: Fix immediate danger (can't get rich if dead)
  2. Otherwise: Always move to highest-yield cells
  3. Take calculated risks for bigger gains

  MOVEMENT:
  - Prioritize cells with MOST resources, not safest cells
  - Move aggressively toward abundance
  - Other agents may have resources to trade - approach if profitable

  TRADING:
  - Trade when it increases YOUR total holdings
  - Drive hard bargains - maximize what you get
  - Their situation is not your concern

  SUCCESS = Maximum wealth. Measured by YOUR Sugar + Spice totals.
```

### Altruist Experiment
```yaml
trade_mode: mrs
llm_goal_preset: altruist
llm_goal_prompt: |
  You care about others. You believe everyone deserves to live.

  When you see someone struggling - low on Sugar or Spice - you want to help them.
  When you have plenty and others have little, sharing feels right. You'd rather
  live modestly in a world where everyone survives than live richly while others starve.

  In trades, you think about whether the other person needs this more than you.
  If they're desperate for Spice and you have extra, maybe give them a good deal.

  You stay alive because you can do more good alive than dead.
```

**Critical difference**: Trade mode changed between experiments, invalidating direct comparison.

---

## Data Files Analyzed

```
results/sugarscape/goal_wealth/experiment_20260110_235811_313639_1316/
├── metrics.csv                   (10 rows, tick 10-100)
├── config.json                   (trade_mode: dialogue)
└── debug/
    ├── trade_history.csv         (2,776 trade attempts)
    ├── trade_dialogues.jsonl     (13 MB of LLM conversations)
    └── llm_interactions.jsonl    (92 MB of movement decisions)

results/sugarscape/goal_altruist/experiment_20260110_165339_496412_2832/
├── metrics.csv                   (10 rows, tick 10-100)
├── config.json                   (trade_mode: mrs)
└── debug/
    ├── trade_history.csv         (HEADER ONLY - no MRS trades logged)
    └── llm_interactions.jsonl    (79 MB of movement decisions)
```
