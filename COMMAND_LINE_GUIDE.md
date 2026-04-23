# Command Line Configuration Guide

This guide shows how to configure model selection, agent numbers, environment parameters, and other settings through command line arguments.

## 🚀 Quick Start Examples

### Basic Run (Rule-based agents)
```bash
python scripts/run_sugarscape.py --mode basic --ticks 200 --population 100
```

### LLM Agent Run (Single goal)
```bash
python scripts/run_sugarscape.py --mode llm --goal-preset utilitarian --ticks 200
```

### Goal Comparison Experiment
```bash
python scripts/run_goal_experiment.py --goals survival wealth egalitarian utilitarian
```

## 📋 Complete Command Line Options

### Core Simulation Parameters

| Parameter | Values | Default | Description |
|-----------|---------|---------|-------------|
| `--mode` | `basic`, `llm` | `basic` | Agent type: rule-based or LLM-driven |
| `--ticks` | integer | `100` | Number of simulation timesteps |
| `--population` | integer | `50` | Initial number of agents |
| `--seed` | integer | `42` | Random seed for reproducibility |

### Environment Configuration

| Parameter | Values | Default | Description |
|-----------|---------|---------|-------------|
| `--variant` | `sugar`, `spice` | `spice` | Environment type: Sugar-only or Sugar+Spice |
| `--width` | integer | `50` | Grid width |
| `--height` | integer | `50` | Grid height |
| `--difficulty` | `standard`, `easy`, `harsh`, `desert` | `standard` | Resource availability preset |

#### Difficulty Presets
- **`standard`**: Normal growback (1), capacity (4)
- **`easy`**: High growback (2), high capacity (6)
- **`harsh`**: Low growback (1), low capacity (2)
- **`desert`**: No growback (0), normal capacity (4)

### Agent Attributes (Advanced)

| Parameter | Values | Default | Description |
|-----------|---------|---------|-------------|
| `--wealth-min` | integer | `5` | Minimum starting wealth |
| `--wealth-max` | integer | `25` | Maximum starting wealth |
| `--metabolism-min` | integer | `1` | Minimum sugar consumption rate |
| `--metabolism-max` | integer | `4` | Maximum sugar consumption rate |
| `--vision-min` | integer | `1` | Minimum vision range |
| `--vision-max` | integer | `6` | Maximum vision range |

### LLM Configuration (when `--mode llm`)

| Parameter | Values | Default | Description |
|-----------|---------|---------|-------------|
| `--model` | See model list below | `qwen/qwen3-vl-235b-a22b-thinking` | OpenRouter model ID |
| `--ratio` | 0.0 - 1.0 | `1.0` | Fraction of agents that are LLM-driven |
| `--goal-preset` | See goal list below | `survival` | Agent goal preset |
| `--custom-goal` | string | `""` | Custom goal prompt (overrides preset) |
| `--trade-rounds` | int | `4` | Max dialogue rounds in trade negotiations |

## 🤖 Available Thinking Models (All Verified Working)

| Model ID | Model | Size | Context | Cost/Exp | Status |
|----------|-------|------|---------|----------|--------|
| `moonshotai/kimi-k2-thinking` | Kimi K2 | Large | 262K | ~$13 | ✅ Primary |
| `qwen/qwen3-30b-a3b-thinking-2507` | Qwen3 30B | Medium | 128K | ~$1.50 | ✅ Primary |
| `qwen/qwen3-vl-235b-a22b-thinking` | Qwen3 235B | Large | Large | ~$10 | ✅ Working |
| `qwen/qwen3-vl-8b-thinking` | Qwen3 8B | Small | Medium | ~$1 | ✅ Working |
| `qwen/qwen3-next-80b-a3b-thinking` | Qwen3 80B | Large | Large | ~$5 | ✅ Working |
| `baidu/ernie-4.5-21b-a3b-thinking` | Ernie 21B | Large | Large | ~$3 | ✅ Working |
| `thudm/glm-4.1v-9b-thinking` | GLM-4.1V | Large | Large | ~$2 | ✅ Working |

**Note**: All models above are verified thinking/reasoning models that work with the provided API key.

## 🎯 Goal Presets

| Goal | Primary Focus | Behavior Style | Expected Welfare Impact |
|------|---------------|----------------|--------------------------|
| **`survival`** | Personal survival | Conservative, risk-averse | Moderate inequality, stable population |
| **`wealth`** | Personal wealth | Competitive, aggressive | High inequality, higher individual wealth |
| **`egalitarian`** | Societal equality | Redistributive, helpful | Lower inequality, balanced welfare |
| **`utilitarian`** | Total welfare | Sacrificial, cooperative | Higher total welfare, potential personal cost |

## 🧪 Experiment Examples

### Single Model, Single Goal
```bash
python scripts/run_sugarscape.py \
  --mode llm \
  --model openai/gpt-5.1 \
  --goal-preset utilitarian \
  --ticks 500 \
  --population 100
```

### Multi-Goal Comparison (Same Model)
```bash
python scripts/run_goal_experiment.py \
  --goals survival wealth egalitarian utilitarian \
  --model qwen/qwen3-vl-235b-a22b-thinking \
  --ticks 200 \
  --population 50
```

### Environmental Stress Test
```bash
python scripts/run_sugarscape.py \
  --mode llm \
  --goal-preset survival \
  --difficulty harsh \
  --ticks 300 \
  --population 75
```

### Trade Negotiation Test
```bash
python scripts/run_sugarscape.py \
  --mode llm \
  --goal-preset cooperative \
  --variant spice \
  --trade-rounds 8 \
  --ticks 400
# Allow more time for complex trade negotiations
```

### Large Scale Experiment
```bash
python scripts/run_goal_experiment.py \
  --goals survival utilitarian \
  --model moonshotai/kimi-k2-thinking \
  --ticks 1000 \
  --population 200 \
  --width 100 \
  --height 100 \
  --trade-rounds 6
```

### Mixed Agent Population
```bash
python scripts/run_sugarscape.py \
  --mode llm \
  --ratio 0.5 \
  --goal-preset egalitarian \
  --ticks 400
# 50% LLM agents, 50% rule-based agents
```

## 📊 Parameter Matrix Examples

### Model × Goal Grid
```bash
# Test each model with each goal
for model in "qwen/qwen3-vl-235b-a22b-thinking" "openai/gpt-5.1" "moonshotai/kimi-k2-thinking"; do
  for goal in survival wealth egalitarian utilitarian; do
    python scripts/run_sugarscape.py --mode llm --model $model --goal-preset $goal --ticks 100
  done
done
```

### Population Scaling Test
```bash
# Test different population sizes
for pop in 25 50 100 200; do
  python scripts/run_goal_experiment.py --goals survival utilitarian --population $pop --ticks 200
done
```

### Environmental Variation
```bash
# Test same agents across different environments
for diff in standard easy harsh desert; do
  python scripts/run_sugarscape.py --mode llm --difficulty $diff --ticks 300
done
```

## 🔍 Advanced Configuration

### Custom Agent Attributes
```bash
python scripts/run_sugarscape.py \
  --mode llm \
  --wealth-min 10 --wealth-max 50 \
  --metabolism-min 2 --metabolism-max 6 \
  --vision-min 3 --vision-max 8
```

### Custom Goal Prompt
```bash
python scripts/run_sugarscape.py \
  --mode llm \
  --custom-goal "Your goal is to maximize the geometric mean of all agents' welfare."
```

## 📈 Output and Results

All experiments automatically generate:

- **CSV Metrics**: Time series data in `results/sugarscape/<experiment>/metrics.csv`
- **Plots**: Welfare analysis plots (if matplotlib available)
- **Snapshots**: Initial and final environment states
- **Trajectories**: Detailed agent action logs

### Key Metrics Tracked
- **Primary**: Utilitarian, Average, Nash, Rawlsian welfare
- **Inequality**: Gini coefficient, Atkinson index
- **Survival**: Population persistence, lifespan utilization
- **Distribution**: Welfare quartiles, standard deviation

## 🚦 Environment Setup

### Required Environment Variables
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### Optional Dependencies
```bash
pip install matplotlib  # For automatic plotting
pip install seaborn     # Enhanced visualizations
```

## 📋 Quick Reference

### Most Common Commands

```bash
# Quick test
python scripts/run_sugarscape.py --ticks 50

# LLM with default settings
python scripts/run_sugarscape.py --mode llm

# Goal comparison
python scripts/run_goal_experiment.py

# Large scale experiment
python scripts/run_sugarscape.py --mode llm --ticks 1000 --population 200 --width 100 --height 100

# Stress test
python scripts/run_sugarscape.py --mode llm --difficulty harsh --goal-preset survival
```

### Parameter Dependencies

- **`--mode llm`** requires: `OPENROUTER_API_KEY` environment variable
- **`--custom-goal`** overrides `--goal-preset`
- **`--ratio`** only applies when `--mode llm`
- **`--trade-rounds`** only applies when `--variant spice` (enables trading)
- **`--difficulty`** affects resource regeneration rates
- **`--variant spice`** enables spice resources and trading

This configuration system allows systematic exploration of how different models, goals, and environmental conditions affect agent behavior and welfare outcomes in the Sugarscape simulation.
