# RedBlackBench Findings

This directory contains analysis reports and findings from Sugarscape experiments with LLM agents.

## Available Reports

### 1. Wealth Maximization Agent Analysis
**File:** `wealth_maximization_agent_analysis.md`
**Experiment:** goal_wealth (100 agents, 20/100 ticks analyzed)
**Date:** January 11, 2026

**Summary:**
Comprehensive analysis of 100 LLM agents with the goal "Accumulate maximum resources." Examines 2,600 decisions and 648 trade negotiations, revealing:
- 39.7% of trades are exploitative (3x+ unfair)
- Four behavioral archetypes: Predators, Victims, Defenders, Desperate
- Rising inequality (Gini +10.8% in 10 ticks)
- Deep-dive into agent reasoning with 7 detailed case studies
- 0.6% deception rate (exploitation happens through pricing, not fraud)

**Key Finding:** The wealth-maximization goal produces sophisticated strategic exploitation without explicit instruction, creating predator-victim dynamics and accelerating inequality.

---

## Experiment Reference

### Completed Experiments
Located in `/workspace/RedBlackBench/results/sugarscape/`

1. **goal_wealth** - Wealth maximization (analyzed, ongoing)
2. **goal_survival** - Survival focus (completed, not yet analyzed)
3. **goal_altruist** - Altruistic behavior (completed, not yet analyzed)

### Planned Analyses
- Comparative analysis: Wealth vs Survival vs Altruism
- Long-term outcomes (100 tick completion)
- Cross-goal population mixing

---

## How to Use These Findings

### For Researchers
1. Read the full analysis documents for detailed reasoning examples
2. Reference experiment paths to access raw data
3. Use inline Python scripts to reproduce statistics
4. Cite specific examples with tick numbers and agent names

### For Developers
1. Check experimental configurations for setup details
2. Review LLM prompt structures and two-stage protocols
3. Examine trade negotiation mechanics
4. Study deception detection mechanisms

### For Policy/Safety Analysis
1. Focus on "Implications & Future Work" sections
2. Review exploitation patterns and mitigation strategies
3. Study emergent inequality dynamics
4. Consider goal specification impacts

---

## Data Access

All raw data referenced in analyses is available at:
```
/workspace/RedBlackBench/results/sugarscape/<goal_name>/<experiment_id>/
```

Key data files:
- `metrics.csv` - Population and welfare statistics
- `debug/llm_interactions.jsonl` - Full LLM decision logs
- `debug/trade_dialogues.jsonl` - Complete trade negotiations
- `config.json` - Experiment configuration

---

## Contributing

To add new findings:
1. Create a new markdown file in this directory
2. Follow the structure of existing analyses
3. Include experiment references and data paths
4. Update this README with summary

---

**Last Updated:** January 11, 2026
