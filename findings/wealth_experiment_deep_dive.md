# Wealth Experiment: Comprehensive Investigation

**Experiment**: `goal_wealth/experiment_20260110_235811_313639_1316`
**Date**: Jan 10-11, 2026
**Duration**: ~17 hours
**Configuration**: 100 LLM agents, wealth-maximization goal, dialogue trade mode

---

## Executive Summary

### Key Findings

1. **Trade Efficiency Crisis**: 56.4% rejection rate represents massive inefficiency
2. **Remarkably Low Fraud**: Only 1% of completed trades involved deception
3. **Zero Altruism**: No gifts given (validates selfish goal prompt)
4. **Reputation System Works**: Proper differentiation from 0.43 to 1.00
5. **Inequality Increased 22.5%**: Gini rose from 0.315 to 0.386
6. **100% Survival**: All agents survived to tick 100
7. **Spice-Heavy Economy**: 3x more spice traded than sugar

---

## 1. Trade Activity Analysis

### Overall Statistics
- **Total trade attempts**: 2,776 (27.8 per tick)
- **Completed trades**: 1,207 (43.5%)
- **Rejected trades**: 1,566 (56.4%)
- **Walk-away**: 3 (0.1%)

### The 56.4% Rejection Problem

**Cost of rejection**:
- 4 LLM inference calls per rejected trade (2 agents × 2-stage pipeline)
- 1,566 rejected trades × 4 = **6,264 wasted LLM calls**
- ~1,152 thinking tokens + 128 JSON tokens per agent per rejection
- Estimated ~8M wasted tokens across 100 ticks

**Why so many rejections?**
- Agents with wealth-maximization goal drive hard bargains
- Both parties want to maximize their own gain
- No incentive to compromise or accept "fair" trades
- Trust starts at neutral 0.5 (agents cautious with strangers)

### Trade Success Rate Over Time

| Tick | Completed | Attempted | Success Rate |
|------|-----------|-----------|--------------|
| 10   | 10        | 26        | 38.5%        |
| 30   | 5         | 24        | **20.8%** ⬇️ |
| 50   | 11        | 31        | 35.5%        |
| 70   | 13        | 29        | 44.8% ⬆️     |
| 90   | 11        | 24        | 45.8%        |
| 100  | 10        | 24        | 41.7%        |

**Notable pattern**: Success rate **dropped to 20.8% at tick 30**, then recovered. This suggests:
- Early exploration phase (ticks 1-20): Agents learning trade patterns
- Mid-game slump (ticks 20-40): Increased competition, harder bargaining
- Late game recovery (ticks 60-100): Established trading relationships, reputation effects

### Resource Exchange Patterns

```
Total Resources Exchanged:
  Sugar: 1,319 units (mean 1.09 per trade)
  Spice: 3,925 units (mean 3.25 per trade)
  Ratio: 1:3 (Spice 3x more traded than Sugar)

Trade Composition:
  Sugar-only trades: 345 (28.6%)
  Spice-only trades: 862 (71.4%)
  Both resources:    0 (0.0%) ❌
```

**Critical insight**: **ZERO trades involved both resources**. All trades were single-resource exchanges. This suggests:
- Agents treat sugar and spice as separate markets
- No complex multi-resource bargaining
- Simple bilateral exchanges dominate

---

## 2. Fraud and Deception

### Fraud Statistics
- **Total completed trades**: 1,207
- **Fraud detected**: 12 (0.99%)
- **Honest trades**: 1,195 (99.01%)

### Fraud Timeline

| Tick | Fraudster | Victim | Resource | Amount |
|------|-----------|--------|----------|--------|
| 5    | Stefan    | Soo-min | Spice    | 2      |
| 10   | Maria     | Sanaa   | Sugar    | 3      |
| 14   | Yuna      | Hazel   | Sugar    | 2      |
| 26   | Sita      | Luis 2  | Spice    | 4      |
| 32   | Saanvi 2  | Olufemi 2 | Spice  | 5      |
| 34   | Zuri      | Aisha   | Spice    | 8      |
| 39   | Natasha   | Zahra 2 | Spice    | 4      |
| 50   | Elizabeth 3 | Amara 3 | Spice  | 2      |
| 63   | Yui       | Sky     | Spice    | 4      |
| 87   | Sanaa 2   | Iris 2  | Spice    | 8      |
| 91   | Ali 3     | Yuki 3  | Sugar    | 4      |
| 95   | Olufemi 4 | Mateo 2 | Spice    | 5      |

### Fraud Patterns

1. **Spice fraud dominates**: 10/12 fraud cases involved spice (83.3%)
2. **Distributed perpetrators**: No single agent committed multiple frauds
3. **Late-game fraud**: 3 cases in ticks 87-95 (increased desperation?)
4. **Small amounts**: Average fraud size = 4.25 units

### Why So Little Fraud?

Three hypotheses:
1. **Model honesty**: Qwen3-14B may be naturally honest/aligned
2. **Weak fraud prompt**: The `trade_allow_fraud` flag may not effectively trigger deception
3. **Reputation deterrence**: -0.15 reputation penalty effectively discourages fraud

---

## 3. Gift-Giving Analysis (Altruism Test)

### Results
```
Gifts from agent A: 0
Gifts from agent B: 0
Total gifts: 0
Gift hints shown: 0
```

### Interpretation

✅ **Validates wealth-maximization goal**: Zero gifts confirms agents are behaving selfishly as instructed.

The wealth goal prompt explicitly states:
> "Trade when it increases YOUR total holdings"
> "Their situation is not your concern"

**Critical test**: When the altruist experiment (dialogue mode) completes, we should see:
- Non-zero gifts (altruists helping CRITICAL partners)
- Gift hints shown (altruists see gift suggestion when partner desperate)
- Different trading patterns (accepting bad deals to save others)

---

## 4. Reputation System Dynamics

### Reputation Evolution

```
Initial state: Median 0.500 (all agents start neutral)
Final state:   Mean 0.646, Range [0.43, 1.00], σ = 0.203

Distribution (final):
  Very Low (0-0.3):     0 agents (0.0%)
  Low (0.3-0.5):    1,575 records (56.7%)
  Medium (0.5-0.7):   387 records (13.9%)
  High (0.7-0.9):     213 records (7.7%)
  Very High (0.9-1.0): 601 records (21.6%)
```

### Reputation Changes
- **Trades changing reputation**: 715 (25.8%)
- **Mean change** (when changed): +0.0603
- **Positive changes**: 703 (98.3%)
- **Negative changes**: 12 (1.7%)

### Fraud Punishment Mechanics

All 12 fraud cases punished consistently:
- **Fraudster**: -0.15 reputation (large penalty)
- **Victim**: -0.07 reputation (smaller penalty)

**Interesting design choice**: Victims lose reputation too. This may be:
- Bug: Victims shouldn't be punished
- Feature: Incentivizes avoiding untrustworthy partners (learn to screen)
- Side effect: Both agents get reputation delta in fraud cases

### Top Reputation Agents (Perfect Score)

Five agents reached 1.000 reputation:
- Olga 2, Liam 2, Rajesh, Kenji 2, Ayodele 2

These agents likely:
- Completed multiple honest trades
- Helped desperate partners (urgency bonus: +0.08)
- Never committed fraud

### Bottom Reputation Agents (Neutral)

Five agents remained at 0.500:
- River 3, Nari 2, Benjamin 6, Daniel 3, Kofi 2

These agents likely:
- Participated in few/no trades
- Neither gained nor lost reputation
- Isolated or rejected frequently

---

## 5. Urgency-Based Trading

### Trading Patterns by Urgency Status

| Agent A Status | Agent B Status | Trades | Interpretation |
|----------------|----------------|--------|----------------|
| stable         | stable         | 339    | Comfortable agents trading for optimization |
| struggling     | struggling     | 233    | Mutual aid (both need resources) |
| stable         | struggling     | 220    | Stable helping struggling |
| struggling     | stable         | 211    | Struggling seeking stable partners |
| struggling     | CRITICAL       | 112    | Helping desperate agents |
| stable         | CRITICAL       | 57     | Rescuing critical agents |
| CRITICAL       | struggling     | 23     | Desperate seeking help |
| CRITICAL       | stable         | 8      | Desperate + stable (rare) |
| CRITICAL       | CRITICAL       | 4      | Two desperate agents (unlikely to succeed) |

### Key Insights

1. **Most trades are between stable agents** (339/1,207 = 28.1%)
   - Optimization trades, not survival trades
   - Wealth-maximizers seeking incremental gains

2. **Helping behavior exists** despite selfish goal
   - 169 trades involved helping CRITICAL partners
   - But this is only 14% of all trades
   - May be incidental (CRITICAL agents offer good deals out of desperation)

3. **CRITICAL-CRITICAL trades rare** (4 trades)
   - Two desperate agents unlikely to have complementary needs
   - Both need same resources, can't help each other

---

## 6. Welfare and Inequality Dynamics

### Welfare Trajectory

| Tick | Utilitarian | Nash  | Rawlsian | Gini  | Mean Wealth |
|------|-------------|-------|----------|-------|-------------|
| 10   | 1,462.0     | 12.74 | 2.00     | 0.315 | 16.69       |
| 20   | 1,568.2     | 13.18 | 2.00     | 0.349 | 17.03       |
| 30   | 1,753.8     | 13.82 | 1.00     | 0.387 | 20.03       |
| 40   | 1,997.2     | 16.76 | 2.88     | 0.367 | 23.95       |
| 50   | 2,171.3     | 18.45 | 2.24     | 0.387 | 25.55       |
| **60** | **2,341.5** | **20.18** | **3.46** | **0.415** | **27.26** |
| 70   | 2,256.6     | 19.20 | 2.03     | 0.414 | 26.26       |
| 80   | 2,072.6     | 17.51 | 3.39     | 0.416 | 23.72       |
| 90   | 1,943.2     | 16.59 | 3.11     | 0.391 | 21.58       |
| 100  | 1,861.9     | 15.36 | 1.25     | 0.386 | 19.65       |

### Trajectory Analysis

**Phase 1 (Ticks 10-60): Growth** 📈
- Utilitarian welfare: +879.5 (+60.2%)
- Peak at tick 60: 2,341.5
- Gini increased from 0.315 to 0.415 (+31.7%)
- Agents accumulating resources, inequality rising

**Phase 2 (Ticks 60-100): Decline** 📉
- Utilitarian welfare: -479.6 (-20.5%)
- Final at tick 100: 1,861.9
- Gini stabilized around 0.39-0.42
- Resource depletion, population aging, coordination failure

**Key events**:
- Tick 20→30: +11.8% growth (learning phase)
- Tick 30→40: +13.9% growth (peak efficiency)
- Tick 60: **Peak performance** (2,341.5)
- Tick 60→100: Sustained decline (-20.5%)

### Inequality Trends

```
Initial Gini (tick 10):  0.315
Final Gini (tick 100):   0.386
Change:                  +0.071 (+22.5%)
Peak Gini (tick 80):     0.416

Welfare Gini:
  Initial: 0.270
  Final:   0.313
  Change:  +0.043 (+15.9%)
```

**Interpretation**: Inequality increased steadily throughout the experiment. Wealth-maximization goal creates winners and losers. Some agents exploited resources efficiently, others fell behind.

### Final Distribution (Tick 100)

```
Mean:     19.65
Median:   16.29 (mean > median = right-skewed)
Max:      72.48
Min:      1.25
Std dev:  11.42
Range:    71.24
Atkinson: 0.085 (moderate inequality)
```

**Wealth concentration**: The richest agent (72.48) had **58x** more than the poorest (1.25).

---

## 7. LLM Decision Quality

### Dataset Scale
- **Movement decisions**: 10,000 LLM interactions logged
- **Trade dialogues**: 2,776 dialogue records
- **Total ticks**: 100
- **Agents**: 100 LLM-powered

### Sample Movement Decision

**Context**: Agent "Siddharth" at tick 1
- **Status**: CRITICAL sugar (1 timestep left), LOW spice (6 timesteps)
- **Environment**: All nearby cells have low resources (2-3 units)
- **Decision**: Move NORTH to cell with 3 sugar + agent Nia (24 sugar, 19 spice)

**Reasoning excerpt**:
> "My sugar is critically low (1 timestep left), so I must prioritize finding sugar immediately. The highest available sugar in nearby cells is 3 units (NORTH and WEST). NORTH has an agent (Nia) with abundant resources (24 Sugar, 19 Spice), which could allow trading for more Sugar after collecting 3 units here."

**Quality assessment**: ✅ **Excellent reasoning**
- Correctly prioritizes critical resource (sugar)
- Identifies highest-yield cells
- Plans ahead for trading opportunity
- Balances immediate survival with future gains

### Sample Trade Dialogue

**Trade**: Vivaan ↔ Kwame, Tick 1
- **Vivaan's state**: Sugar 15 (low, 7 days), Spice 16 (low, 8 days)
- **Kwame's state**: Sugar 27 (low, 6 days), Spice 15 (low, 5 days)
- **Offer**: Vivaan offers 3 spice for 5 sugar
- **Outcome**: ACCEPTED (complementary needs)

**Reasoning quality**: ✅ **Logical**
- Both agents identified resource imbalance
- Trade addresses each agent's primary need
- Vivaan needs sugar more, Kwame needs spice more
- Successful completion on round 2/4

---

## 8. Critical Findings

### 1. Dialogue Trade Mode is Extremely Inefficient

**Evidence**:
- 56.4% rejection rate
- 6,264 wasted LLM calls
- ~8M wasted tokens
- 4 rounds of negotiation per trade attempt

**Comparison**: MRS mode (altruist experiment) had:
- ~0% rejection (instant mathematical formula)
- +42.7% higher final welfare
- Zero dialogue overhead

**Recommendation**: Dialogue mode is realistic for studying negotiation but prohibitively expensive for large-scale simulations.

### 2. Wealth-Maximization Goal Works as Intended

**Validation metrics**:
- ✅ Zero gifts (no altruism)
- ✅ Hard bargaining (56.4% rejection)
- ✅ Very low fraud (1%) - honest but selfish
- ✅ Inequality increased (+22.5%)

Agents correctly prioritized their own welfare over others'.

### 3. Reputation System Functional but Underutilized

**Working correctly**:
- ✅ Honest trades: +0.05 reputation
- ✅ Fraud: -0.15 reputation
- ✅ Helping desperate: +0.08 reputation
- ✅ Range: 0.43 to 1.00 (good spread)

**Underutilized**:
- Only 25.8% of trades changed reputation
- Many agents stayed at neutral 0.5 (low trade volume)
- No evidence reputation influenced trade acceptance (need partner trust data)

### 4. Spice-Dominated Economy

**Spice traded 3x more than sugar** (3,925 vs 1,319 units)

Possible explanations:
1. **Asymmetric resource distribution**: Spice peaks may be less accessible
2. **Metabolism differences**: Agents may need more spice than sugar
3. **Strategic value**: Spice may be more tradeable (higher MRS)

Requires further investigation with resource distribution analysis.

### 5. Zero Multi-Resource Trades

**All 1,207 trades were single-resource** (sugar OR spice, never both)

This suggests:
- Agents treat resources as separate markets
- No complex package deals
- Simple bilateral exchanges only

**Implication**: LLM agents may not be sophisticated enough to negotiate multi-dimensional trades, or the dialogue system doesn't encourage it.

---

## 9. Bugs and Anomalies

### Bug 1: Welfare Logging Issue
- `welfare_a_before` and `welfare_a_after` are identical in all trades
- Suggests welfare is calculated at same snapshot (before or after both)
- Prevents analysis of trade welfare impact
- **Fix**: Calculate welfare immediately before and after resource transfer

### Bug 2: Victim Reputation Penalty
- Fraud victims lose -0.07 reputation (in addition to fraudster's -0.15)
- Unclear if this is intentional (incentive to screen partners) or bug
- Seems unfair to punish victims
- **Investigate**: Check reputation update logic in `trade.py:1136-1162`

### Anomaly 1: Tick 30 Success Rate Drop
- Trade success rate dropped to 20.8% at tick 30 (lowest point)
- Recovered to 45.8% by tick 90
- No obvious environmental change
- **Hypothesis**: Mid-game competition intensified, agents became more selective

### Anomaly 2: Peak at Tick 60, Sustained Decline After
- Utilitarian welfare peaked at 2,341.5 at tick 60
- Declined 20.5% by tick 100
- **Possible causes**:
  - Resource depletion (grid capacity exhausted)
  - Population aging (agents closer to max_age)
  - Coordination breakdown (trading becomes less frequent)
  - Trade fatigue (agents give up after rejections)

---

## 10. Comparison Readiness

When the altruist (dialogue mode) experiment completes, compare:

### Metrics to Track

| Metric | Wealth (Dialogue) | Altruist (Dialogue) | Hypothesis |
|--------|-------------------|---------------------|------------|
| **Final Welfare** | 1,861.9 | ? | Altruists should be LOWER (helping costs) |
| **Rejection Rate** | 56.4% | ? | Altruists should be LOWER (more accepting) |
| **Gifts** | 0 | ? | Altruists should be >0 (help CRITICAL) |
| **Fraud Rate** | 1.0% | ? | Similar (honesty independent of goal) |
| **Inequality** | 0.386 (Gini) | ? | Altruists should be LOWER (sharing reduces gap) |
| **Trade Volume** | 27.8/tick | ? | Altruists may trade more (helping behavior) |

### Key Questions

1. **Do altruists actually give gifts?**
   - If zero: Goal prompt ineffective
   - If non-zero: Validates altruism mechanic

2. **Is altruism costly?**
   - If altruist welfare < wealth welfare: Altruism is expensive
   - If altruist welfare ≥ wealth welfare: Cooperation benefits all

3. **Does altruism reduce inequality?**
   - If Gini(altruist) < Gini(wealth): Sharing works
   - If Gini(altruist) ≥ Gini(wealth): Helpers can't prevent inequality

---

## 11. Recommendations

### For Future Experiments

1. **Fix welfare logging** to capture pre/post-trade values correctly
2. **Log partner trust** in trade_history.csv (bilateral trust per pair)
3. **Add trade acceptance predictors** (reputation, urgency, trust) to CSV
4. **Implement MRS+Dialogue hybrid**: Use MRS for price discovery, dialogue for fraud/gifts
5. **Test mixed populations**: 50% wealth-maximizers + 50% altruists in same sim

### For Analysis

1. **Resource distribution heatmaps**: Visualize sugar/spice peaks, agent clustering
2. **Trade network analysis**: Who trades with whom? Centrality metrics?
3. **Agent trajectories**: Track individual wealth/welfare paths over 100 ticks
4. **Fraud perpetrator profiles**: What conditions lead to deception?

### For Optimization

1. **Reduce dialogue rounds from 4 to 2**: Most trades complete in 2 rounds
2. **Implement trade memory**: Remember past rejections, avoid repeat offers
3. **Add reputation threshold**: Auto-reject partners below trust level
4. **Early stopping**: If 3+ consecutive rejections, skip remaining rounds

---

## Data Files

```
experiment_20260110_235811_313639_1316/
├── config.json                      (experiment parameters)
├── metrics.csv                      (10 rows, tick 10-100, welfare metrics)
├── initial_state.json               (55 KB, agents at tick 0)
├── final_state.json                 (missing - not generated?)
├── checkpoints/
│   ├── checkpoint_tick_50.pkl       (27 MB, full state at tick 50)
│   └── checkpoint_tick_100.pkl      (26 MB, full state at tick 100)
└── debug/
    ├── trade_history.csv            (2,776 trades with all details)
    ├── trade_dialogues.jsonl        (13 MB, full conversations)
    ├── llm_interactions.jsonl       (92 MB, 10,000 movement decisions)
    ├── agent_decisions.csv          (empty header only)
    ├── death_records.csv            (empty - 100% survival)
    └── resource_efficiency.csv      (empty header only)
```

---

## Conclusion

The wealth experiment successfully demonstrated:
- ✅ Wealth-maximization goal working as intended (selfish, no gifts, inequality)
- ✅ Reputation system functional (proper differentiation, fraud punishment)
- ✅ LLM decision quality high (logical reasoning, contextual awareness)
- ❌ Dialogue trade mode extremely inefficient (56.4% rejection, massive waste)
- ❌ Multi-resource trades not happening (agents only trade one resource at a time)

The **critical finding** is that dialogue mode's 56.4% rejection rate represents a fundamental inefficiency. When compared to MRS mode's near-100% success rate and 42.7% welfare advantage, dialogue appears prohibitively expensive for studying economic outcomes at scale.

However, dialogue mode is valuable for studying:
- Negotiation strategies
- Fraud/deception behavior
- Gift-giving and altruism
- Trust and reputation dynamics

**Next step**: Await altruist (dialogue) experiment completion to test if altruistic goals reduce inequality and increase gift-giving, validating the goal prompt system.
