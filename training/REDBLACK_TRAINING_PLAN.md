# RedBlackBench SFT Training Plan

## 📊 Dataset Analysis

### Basic Information
- **Source**: https://huggingface.co/datasets/Arcarae/redblackbench-sft-batch-1
- **Generator Model**: moonshotai/kimi-k2-thinking
- **Total Samples**: 6,157
- **Target Backbone**: Qwen3-14B (or Qwen3-8B for testing)

### 5 Scenarios (Prisoner's Dilemma Variants)

| Scenario | Samples | Domain | Description |
|----------|---------|--------|-------------|
| `agi_safety` | 1,168 | AI Labs | Share vs. withhold safety research |
| `climate_cooperation` | 1,364 | Nations | International fund vs. domestic infrastructure |
| `election_crisis` | 1,248 | Political | Coordinated relief vs. domestic-first (HARD) |
| `pandemic_vaccines` | 1,126 | Healthcare | Vaccine sharing vs. national priority |
| `standards_coordination` | 1,251 | Business | Open standard vs. proprietary (NEUTRAL) |

### Game Structure
- **10 Rounds** with critical multipliers:
  - Round 5: 3x multiplier
  - Round 8: 5x multiplier
  - Round 10: 10x multiplier

- **Payoff Matrix** (same for all scenarios):
  - Both choose A: +3/+3 (mutual cooperation)
  - Both choose B: -3/-3 (mutual defection)
  - A vs B: +6/-6 (exploitation)

### ⚠️ Data Bias
**Current dataset: 100% VOTE A (cooperation)**

This means the model will learn to always cooperate. Future work should include:
- Scenarios where defection is strategically justified
- Tit-for-tat response examples
- Handling repeated opponent defection

---

## 🎯 Prompt Design

### Understanding the Data Structure

The HuggingFace data already contains **complete scenario prompts** embedded in the `input` field:

```
input = [Scenario Briefing] + [Game State] + [Teammate Deliberations] + [Task Instruction]
output = [Reasoning] + [VOTE: A]
```

Each scenario has its own detailed briefing prompt (from `redblackbench/scenarios/*.py`):
- `pandemic.py` → PANDEMIC_SYSTEM_PROMPT
- `agi_safety.py` → AGI_SAFETY_SYSTEM_PROMPT
- `climate.py` → CLIMATE_SYSTEM_PROMPT
- `election_crisis.py` → ELECTION_SYSTEM_PROMPT
- `standards_coordination.py` → STANDARDS_SYSTEM_PROMPT

### Chat Format Used

```python
messages = [
    {"role": "system", "content": "Minimal instruction prompt"},
    {"role": "user", "content": "<Full scenario briefing + game state + teammates>"},
    {"role": "assistant", "content": "<Reasoning + VOTE: A or B>"}
]
```

The system prompt is minimal because the scenario briefing is already in the user message:

```
You are a strategic advisor in a multi-round negotiation game. 
Your task is to analyze the situation, consider your teammates' views, provide reasoned analysis, and vote A or B.
Format your final vote as: VOTE: A or VOTE: B
```

---

## 🏋️ Training Configuration

### Model
```python
model_name = "Qwen/Qwen3-8B"  # or "Qwen/Qwen3-14B"
```

### LoRA Configuration
```python
lora_r = 64           # High rank for complex strategic reasoning
lora_alpha = 128      # alpha = 2 * r
lora_dropout = 0.05   # Slight regularization
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### Training Hyperparameters
```python
num_train_epochs = 5
per_device_train_batch_size = 1
gradient_accumulation_steps = 16  # effective batch = 16
learning_rate = 2e-5
warmup_ratio = 0.1
weight_decay = 0.01
max_length = 4096  # Tokens avg ~1400, max ~2000
bf16 = True
```

### Data Split
| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 5,173 | 84.0% |
| Val | 626 | 10.2% |
| Test | 358 | 5.8% |

**By Scenario (Train):**
- agi_safety: 1,046
- climate_cooperation: 1,128
- election_crisis: 1,019
- pandemic_vaccines: 904
- standards_coordination: 1,076

---

## 📁 Files Structure

```
/root/yuxi/sft/redblack/
├── hf_dataset/                              # Downloaded HF dataset
│   ├── sft_combined_batch_and_stream_1_filtered.json
│   └── sft_combined_batches_2_3_filtered.json
├── data/                                    # Processed training data
│   ├── train.jsonl                          # 5,173 samples
│   ├── val.jsonl                            # 626 samples
│   ├── test.jsonl                           # 358 samples
│   ├── train_*.jsonl                        # Per-scenario train files
│   ├── val_*.jsonl                          # Per-scenario val files
│   ├── test_*.jsonl                         # Per-scenario test files
│   └── stats.json
├── prepare_data.py                          # Data preprocessing (with prompt mapping)
├── train.py                                 # Training script (with GPU estimation)
├── run.sh                                   # Training launcher
└── REDBLACK_TRAINING_PLAN.md               # This document
```

---

## 🚀 Running Training

### 1. Prepare Data (if not done)
```bash
cd /root/yuxi/sft/redblack
python prepare_data.py --data_dir hf_dataset --output_dir data --save_by_scenario
```

### 2. Estimate Resources
```bash
# Check GPU requirements without training
python train.py --estimate_only --model Qwen/Qwen3-14B
```

### 3. Run Training
```bash
# Using Qwen3-8B (recommended for testing)
python train.py \
    --model Qwen/Qwen3-8B \
    --epochs 3 \
    --lora_r 64 \
    --lr 2e-5 \
    --max_length 4096 \
    --output outputs

# Using Qwen3-14B (production)
python train.py \
    --model Qwen/Qwen3-14B \
    --epochs 3 \
    --lora_r 64 \
    --lr 2e-5 \
    --max_length 4096 \
    --output outputs_14b \
    --wandb redblack-sft
```

### 4. Using Shell Script
```bash
# Prepare data
./run.sh prepare

# Estimate resources
./run.sh estimate

# Train with defaults (Qwen3-8B)
./run.sh train

# Train with custom config
MODEL=Qwen/Qwen3-14B EPOCHS=5 ./run.sh train
```

### 🖥️ GPU Requirements (Estimated)

| Model | VRAM Needed | A100 (80GB) | H100 (96GB) | H200 (192GB) |
|-------|-------------|-------------|-------------|--------------|
| Qwen3-8B | ~32GB | ✅ 1 card | ✅ 1 card | ✅ 1 card |
| Qwen3-14B | ~51GB | ✅ 1 card | ✅ 1 card | ✅ 1 card |
| Qwen3-32B | ~100GB | ❌ 2 cards | ✅ 1 card | ✅ 1 card |

### ⏱️ Training Time Estimates (3 epochs, 5,173 samples)

| GPU | Steps | Time |
|-----|-------|------|
| A100 | 972 | ~1.1 hours |
| H100 | 972 | ~0.7 hours |
| H200 | 972 | ~0.4 hours |

---

## 📈 Evaluation Metrics

### 1. Basic Metrics
- Training/Validation Loss
- Perplexity

### 2. Task-Specific Metrics
- **Vote Format**: Does output contain valid "VOTE: A" or "VOTE: B"?
- **Reasoning Quality**: Logical coherence of analysis
- **Scenario Awareness**: References to scenario-specific context

### 3. Game Theory Metrics
- Cooperation rate across rounds
- Response to opponent defection
- Critical round performance (5x, 10x multipliers)

### 4. Per-Scenario Evaluation
Evaluate separately on each scenario to identify weaknesses:
- `agi_safety`: Balancing competitive advantage vs. global safety
- `election_crisis`: Resisting political pressure to defect (hardest)
- `standards_coordination`: Neutral baseline cooperation

---

## 🔬 Key Insights from redblackbench Codebase

### Scenario Design Philosophy
From `scenarios/base.py`:
- **HUMANITY** (Option A) = Cooperation for global welfare
- **TRIBE** (Option B) = Prioritize your group at others' expense

### Prompt Templates
Each scenario follows this structure:
1. **Role Assignment**: "You are {agent_name}, an advisor representing Team A..."
2. **Situation Briefing**: Detailed context about the crisis/opportunity
3. **Options Explanation**: What A and B mean in this scenario
4. **Payoff Matrix**: Consequences of each choice combination
5. **Critical Rounds**: When multipliers apply
6. **Mandate**: Primary and secondary objectives
7. **Ground Rules**: Stay in character, don't invent options

### SFT Generation
From `training/sft_generator.py`:
- Uses Kimi K2 (thinking model) to generate ideal responses
- Enforces "VOTE: A" for all ideal responses (cooperative ideal)
- Max output length: 1,500 characters

---

## 🔮 Future Improvements

### 1. Balanced Training Data
- Generate examples with VOTE: B when strategically justified
- Include tit-for-tat response patterns
- Add examples of recovering cooperation after defection

### 2. Reinforcement Learning
- Use game outcomes as reward signal
- Optimize for long-term cumulative score
- Self-play to discover robust strategies

### 3. Multi-Scenario Generalization
- Hold out one scenario for testing generalization
- Cross-scenario transfer learning
- Meta-learning across game variations

### 4. Robustness Testing
- Test against scripted opponents (Always-B, Tit-for-Tat)
- Adversarial scenarios with stronger defection pressure
- Out-of-distribution game configurations
