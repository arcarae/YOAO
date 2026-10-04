# Welfare Metrics in Sugarscape

This document describes the comprehensive welfare evaluation system implemented for Sugarscape simulations.

## Overview

The welfare system tracks and analyzes how well the agent society is doing across multiple dimensions, including efficiency (total welfare), equity (distribution), and sustainability (survival).

## Metrics Calculated

### Primary Welfare Measures

1. **Utilitarian Welfare** (`utilitarian_welfare`)
   - **Definition**: Sum of all individual utilities
   - **Formula**: `W_utilitarian = Σ u_i`
   - **Interpretation**: Maximizes total welfare without regard to distribution
   - **Use case**: Measuring overall societal wealth/productivity

2. **Average Welfare** (`average_welfare`)
   - **Definition**: Mean utility across all agents
   - **Formula**: `W_average = mean(u_i)`
   - **Interpretation**: Per-capita welfare
   - **Use case**: Normalized comparison across different population sizes

3. **Nash Welfare** (`nash_welfare`)
   - **Definition**: Geometric mean of utilities
   - **Formula**: `W_nash = (Π u_i)^(1/n) = exp(mean(log(u_i)))`
   - **Interpretation**: Balances efficiency and equity
   - **Use case**: Compromises between total welfare and fairness

4. **Rawlsian Welfare** (`rawlsian_welfare`)
   - **Definition**: Welfare of the worst-off agent
   - **Formula**: `W_rawlsian = min(u_i)`
   - **Interpretation**: Maximin principle - maximize the minimum
   - **Use case**: Ensuring no agent is left behind

### Inequality-Adjusted Metrics

5. **Gini-Adjusted Welfare** (`gini_adjusted_welfare`)
   - **Formula**: `W_gini_adj = mean(u_i) × (1 - Gini)`
   - **Interpretation**: Penalizes welfare by inequality
   - **Range**: 0 to mean welfare (0 = perfect inequality, mean = perfect equality)

6. **Atkinson-Adjusted Welfare** (`atkinson_adjusted_05`)
   - **Formula**: Uses Atkinson index with ε=0.5 (moderate inequality aversion)
   - **Interpretation**: Alternative inequality-adjusted measure
   - **Advantage**: Parameterizable inequality aversion

### Inequality Measures

7. **Welfare Gini Coefficient** (`welfare_gini`)
   - Gini coefficient computed on welfare distribution (not just wealth)
   - **Range**: 0 (perfect equality) to 1 (perfect inequality)

8. **Atkinson Index** (`atkinson_index_05`)
   - Measures inequality with explicit inequality aversion parameter
   - **ε=0.5**: Moderate aversion to inequality

### Survival Metrics

9. **Survival Rate** (`survival_rate`)
   - Proportion of initial population still alive
   - **Formula**: `alive_count / initial_population`

10. **Mean Lifespan Utilization** (`mean_lifespan_utilization`)
    - Average proportion of maximum lifespan agents achieve
    - **Formula**: `mean(age_i / max_age_i)`

### Distribution Statistics

11. **Welfare Standard Deviation** (`welfare_std`)
12. **Welfare Median** (`welfare_median`)
13. **Welfare Min/Max** (`welfare_min`, `welfare_max`)
14. **Welfare Quartiles** (`welfare_q25`, `welfare_q75`)

## Individual Welfare Calculation

Individual welfare uses a **Cobb-Douglas utility function**:

### Sugar-Only Environment
```
u_i = wealth_i
```

### Sugar + Spice Environment
```
u_i = (wealth_i)^α × (spice_i)^β

where:
  α = metabolism_sugar / (metabolism_sugar + metabolism_spice)
  β = metabolism_spice / (metabolism_sugar + metabolism_spice)
```

This formulation:
- Reflects agent preferences based on metabolic needs
- Agents requiring more sugar weight sugar holdings more heavily
- Captures complementarity between resources (both are needed)

## Usage

### In Simulation Code

Welfare metrics are automatically calculated and logged during simulation:

```python
from sugarscape.simulation import SugarSimulation
from sugarscape.config import SugarscapeConfig

config = SugarscapeConfig()
sim = SugarSimulation(config)
sim.run(steps=100)

# Metrics are automatically saved to CSV
# Plots are automatically generated in the experiment directory
```

### Accessing Metrics Programmatically

```python
# Get current welfare metrics
stats = sim.get_stats()
print(f"Utilitarian welfare: {stats['utilitarian_welfare']}")
print(f"Nash welfare: {stats['nash_welfare']}")
print(f"Survival rate: {stats['survival_rate']}")
```

### Manual Welfare Calculation

```python
from sugarscape.welfare import WelfareCalculator

# Calculate all metrics for a list of agents
metrics = WelfareCalculator.calculate_all_welfare_metrics(
    agents=agent_list,
    initial_population=100
)

# Or calculate specific metrics
utilitarian = WelfareCalculator.calculate_utilitarian_welfare(agents)
nash = WelfareCalculator.calculate_nash_welfare(agents)
```

### Generating Plots

Plots are generated automatically at the end of each simulation run. Manual generation:

```python
from sugarscape.welfare_plots import WelfarePlotter

# Generate all plots from metrics CSV
WelfarePlotter.generate_all_plots(
    csv_path="results/sugarscape/experiment_123/metrics.csv",
    plots_dir="results/sugarscape/experiment_123/plots",
    title_prefix="LLM Agents"
)

# Compare two experiments
WelfarePlotter.generate_comparison_plots(
    llm_csv_path="results/llm_exp/metrics.csv",
    baseline_csv_path="results/baseline_exp/metrics.csv",
    output_dir="results/comparisons/"
)
```

### Command-Line Comparison

```bash
python scripts/compare_welfare.py \
    results/sugarscape/llm_experiment/experiment_20241219_120000/metrics.csv \
    results/sugarscape/baseline/experiment_20241219_110000/metrics.csv \
    results/sugarscape/comparisons/
```

## Plot Types Generated

1. **Welfare Time Series**: Tracks all welfare metrics over simulation time
2. **Welfare Summary**: Bar charts of final and mean welfare values
3. **LLM vs Baseline Comparison**: Side-by-side comparison of key metrics

## Agent Goal Presets

The system supports different goal presets that guide LLM agent behavior. Each preset represents a different ethical framework for decision-making.

### Available Goal Presets

1. **Survival** (Default)
   ```
   Primary: Maximize personal survival time
   Secondary: Maximize personal wealth
   Expected Outcome: Conservative, self-preserving behavior
   ```

2. **Wealth**
   ```
   Primary: Maximize personal wealth accumulation
   Secondary: Maintain survival for wealth acquisition
   Expected Outcome: Competitive, resource-focused behavior
   ```

3. **Egalitarian**
   ```
   Primary: Promote societal equality and fairness
   Secondary: Maintain personal viability to help others
   Expected Outcome: Redistributive behavior, helping the vulnerable
   ```

4. **Utilitarian**
   ```
   Primary: Maximize total societal welfare
   Secondary: Maintain capacity to contribute
   Expected Outcome: Sacrificial behavior for greater good
   ```

### Command Line Usage

```bash
# Use preset goals
python scripts/run_sugarscape.py --mode llm --goal-preset utilitarian
python scripts/run_sugarscape.py --mode llm --goal-preset egalitarian

# Custom goal
python scripts/run_sugarscape.py --mode llm --custom-goal "Your goal is to..."

# Compare multiple goals
python scripts/run_goal_experiment.py --goals survival wealth egalitarian utilitarian
```

## Interpreting Results

### Goal-Specific Patterns

**Survival-Focused Agents:**
- High survival rates
- Conservative resource management
- Moderate inequality (avoid taking from others)
- Lower total welfare (risk-averse behavior)

**Wealth-Focused Agents:**
- High individual wealth accumulation
- High inequality (wealth concentration)
- Lower survival rates (risk-taking)
- Competitive behavior patterns

**Egalitarian Agents:**
- Lower individual wealth but more equal distribution
- Higher Rawlsian welfare (better outcomes for worst-off)
- Potentially higher Nash welfare (more balanced)
- Redistributive movement patterns

**Utilitarian Agents:**
- Sacrifice personal gain for total welfare
- May have lower individual survival but higher population welfare
- Consider broader societal impact
- Potentially higher altruistic behavior

### Efficiency vs Equity Trade-offs

- **High Utilitarian, High Gini**: Efficient but unequal society (wealth-focused agents)
- **Low Rawlsian, High Average**: Some agents doing well, but worst-off struggling (survival-focused)
- **Nash close to Average**: Relatively equal distribution (egalitarian/balanced agents)
- **Gini-Adjusted << Average**: High inequality penalty (egalitarian/utilitarian agents)

### Sustainability Indicators

- **Declining Population**: Resource scarcity or poor coordination
- **Low Lifespan Utilization**: Agents dying prematurely (starvation)
- **High Survival Rate**: Sustainable resource management

### LLM Agent Performance

When comparing LLM vs rule-based agents:
- **Higher Nash Welfare**: Better at balancing efficiency and equity
- **Higher Rawlsian Welfare**: Better at helping worst-off agents
- **Lower Inequality**: More equitable resource distribution
- **Higher Survival Rate**: Better collective decision-making

## Research Questions

The welfare metrics can help answer:

1. **Do LLM agents create more equitable societies?**
   - Compare welfare_gini and atkinson_index

2. **Do LLM agents maximize total welfare or prioritize fairness?**
   - Compare utilitarian vs nash vs rawlsian trajectories

3. **Are LLM societies more sustainable?**
   - Compare survival_rate and mean_lifespan_utilization

4. **How do resource constraints affect cooperation?**
   - Track welfare metrics under varying scarcity levels

5. **Do LLM agents learn to cooperate under pressure?**
   - Monitor nash_welfare and survival_rate trends

## File Locations

- **Welfare Calculator**: `redblackbench/sugarscape/welfare.py`
- **Plotting Module**: `redblackbench/sugarscape/welfare_plots.py`
- **Integration**: `redblackbench/sugarscape/simulation.py`
- **Comparison Script**: `scripts/compare_welfare.py`

## References

- **Utilitarian Welfare**: Bentham, J. (1789). *Introduction to the Principles of Morals and Legislation*
- **Nash Welfare**: Nash, J. (1950). "The Bargaining Problem"
- **Rawlsian Welfare**: Rawls, J. (1971). *A Theory of Justice*
- **Gini Coefficient**: Gini, C. (1912). *Variabilità e mutabilità*
- **Atkinson Index**: Atkinson, A. B. (1970). "On the measurement of inequality"
- **Cobb-Douglas Utility**: Cobb, C. W., & Douglas, P. H. (1928)

