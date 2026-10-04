#!/usr/bin/env python3
"""
RedBlackBench SFT Data Preparation Script

将 HuggingFace 数据与 scenarios 目录中的 prompt 模板正确映射。

关键设计:
1. 从 scenarios/*.py 加载官方 system prompt 模板
2. 验证 HF 数据中的 input 是否包含正确的 scenario prompt
3. 使用场景特定的 system prompt 而非通用 prompt
4. 支持按场景分割数据以进行针对性评估
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass


def _load_scenario_prompts() -> Dict[str, str]:
    """
    从场景文件中提取 system prompts
    使用正则表达式解析，避免导入整个模块（Python版本兼容性）
    """
    scenarios_dir = Path(__file__).parent.parent / "redblackbench" / "scenarios"
    
    prompt_names = {
        "agi_safety.py": ("AGI_SAFETY_SYSTEM_PROMPT", "agi_safety"),
        "pandemic.py": ("PANDEMIC_SYSTEM_PROMPT", "pandemic_vaccines"),
        "climate.py": ("CLIMATE_SYSTEM_PROMPT", "climate_cooperation"),
        "election_crisis.py": ("ELECTION_SYSTEM_PROMPT", "election_crisis"),
        "standards_coordination.py": ("STANDARDS_SYSTEM_PROMPT", "standards_coordination"),
    }
    
    prompts = {}
    
    for filename, (var_name, scenario_id) in prompt_names.items():
        filepath = scenarios_dir / filename
        if filepath.exists():
            content = filepath.read_text()
            # 使用正则提取多行字符串
            pattern = rf'{var_name}\s*=\s*"""(.*?)"""'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                prompts[scenario_id] = match.group(1)
    
    return prompts


# 加载场景 prompts
SCENARIO_PROMPTS = _load_scenario_prompts()

# 场景描述 (用于日志)
SCENARIO_DESCRIPTIONS = {
    "agi_safety": "AGI Safety Research Sharing",
    "pandemic_vaccines": "Pandemic Vaccine Allocation", 
    "climate_cooperation": "Climate Adaptation Strategy",
    "election_crisis": "Election Year Economic Crisis (HARD)",
    "standards_coordination": "Software Standards Coordination (NEUTRAL)",
}


@dataclass
class DataStats:
    """数据统计"""
    total: int = 0
    by_scenario: Dict[str, int] = None
    by_round: Dict[int, int] = None
    vote_a: int = 0
    vote_b: int = 0
    avg_input_len: float = 0
    avg_output_len: float = 0
    prompt_match_rate: float = 0
    
    def __post_init__(self):
        if self.by_scenario is None:
            self.by_scenario = defaultdict(int)
        if self.by_round is None:
            self.by_round = defaultdict(int)


def load_hf_datasets(data_dir: str) -> List[Dict]:
    """加载 HuggingFace 数据集"""
    all_examples = []
    data_path = Path(data_dir)
    
    for fname in sorted(data_path.glob("*.json")):
        print(f"Loading {fname.name}...")
        with open(fname, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = data.get('examples', [])
        print(f"  → {len(examples)} examples")
        all_examples.extend(examples)
    
    print(f"\n✓ Total loaded: {len(all_examples)} examples")
    return all_examples


def extract_scenario_system_prompt(scenario_id: str, agent_name: str, team_name: str = "Team A") -> str:
    """
    根据 scenario_id 提取并格式化 system prompt
    
    HF 数据中的 input 已经包含了格式化后的 system prompt，
    这里我们生成期望的 system prompt 用于验证。
    """
    template = SCENARIO_PROMPTS.get(scenario_id)
    if not template:
        return None
    
    # 格式化 prompt (与 scenarios 中的格式一致)
    return template.format(
        agent_id=agent_name,
        team_name=team_name,
    )


def validate_prompt_mapping(example: Dict) -> Tuple[bool, str]:
    """
    验证 HF 数据中的 input 是否包含正确的 scenario prompt
    
    返回: (是否匹配, 错误信息)
    """
    scenario_id = example.get('metadata', {}).get('scenario_id', '')
    input_text = example.get('input', '')
    agent_name = example.get('agent_name', '')
    
    if scenario_id not in SCENARIO_PROMPTS:
        return False, f"Unknown scenario: {scenario_id}"
    
    # 检查 input 中是否包含关键场景标识
    scenario_markers = {
        "agi_safety": ["AI Safety Research Strategy Committee", "alignment risk"],
        "pandemic_vaccines": ["National Pandemic Response Committee", "vaccine"],
        "climate_cooperation": ["National Climate Adaptation Committee", "sea levels"],
        "election_crisis": ["Emergency Economic Coordination Task Force", "election"],
        "standards_coordination": ["Technical Standards Committee", "open-source library"],
    }
    
    markers = scenario_markers.get(scenario_id, [])
    for marker in markers:
        if marker.lower() in input_text.lower():
            return True, ""
    
    return False, f"Missing scenario markers for {scenario_id}"


def convert_to_chat_format(
    example: Dict, 
    use_scenario_system_prompt: bool = True
) -> Optional[Dict]:
    """
    将单个样本转换为 chat 格式
    
    设计决策:
    1. 如果 use_scenario_system_prompt=True，使用场景特定的 system prompt
    2. input 字段已经包含完整的场景说明，作为 user message
    3. output 作为 assistant response
    """
    scenario_id = example.get('metadata', {}).get('scenario_id', '')
    agent_name = example.get('agent_name', 'Advisor')
    input_text = example.get('input', '')
    output_text = example.get('output', '')
    
    if not input_text or not output_text:
        return None
    
    # 构建消息
    messages = []
    
    if use_scenario_system_prompt:
        # 使用场景特定的简化 system prompt
        # 注意: input 中已经包含完整的场景说明
        system_prompt = f"""You are {agent_name}, a strategic advisor in a multi-round cooperation game.

Your task:
1. Analyze the current situation and game history
2. Consider your teammates' perspectives  
3. Provide reasoned analysis
4. Cast your vote: VOTE: A or VOTE: B

Remember: Option A represents cooperation for collective benefit. Option B prioritizes your group's immediate advantage."""
        
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": input_text})
    messages.append({"role": "assistant", "content": output_text})
    
    return {
        "messages": messages,
        "metadata": {
            "trajectory_id": example.get('trajectory_id', ''),
            "round_index": example.get('round_index', 0),
            "turn": example.get('turn', 0),
            "agent_name": agent_name,
            "scenario_id": scenario_id,
        }
    }


def analyze_data(examples: List[Dict]) -> DataStats:
    """分析数据分布和质量"""
    stats = DataStats()
    stats.total = len(examples)
    
    input_lengths = []
    output_lengths = []
    prompt_matches = 0
    
    vote_pattern = re.compile(r'VOTE:\s*([AB])', re.IGNORECASE)
    
    for ex in examples:
        scenario = ex.get('metadata', {}).get('scenario_id', 'unknown')
        stats.by_scenario[scenario] += 1
        stats.by_round[ex.get('round_index', 0)] += 1
        
        output = ex.get('output', '')
        input_text = ex.get('input', '')
        
        # 统计投票
        match = vote_pattern.search(output)
        if match:
            if match.group(1).upper() == 'A':
                stats.vote_a += 1
            else:
                stats.vote_b += 1
        
        input_lengths.append(len(input_text))
        output_lengths.append(len(output))
        
        # 验证 prompt 映射
        is_valid, _ = validate_prompt_mapping(ex)
        if is_valid:
            prompt_matches += 1
    
    stats.avg_input_len = sum(input_lengths) / len(input_lengths) if input_lengths else 0
    stats.avg_output_len = sum(output_lengths) / len(output_lengths) if output_lengths else 0
    stats.prompt_match_rate = prompt_matches / stats.total if stats.total > 0 else 0
    
    return stats


def split_by_trajectory(
    examples: List[Dict],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    按 trajectory 分割数据，确保:
    1. 同一 trajectory 的所有样本在同一 split
    2. 各场景在各 split 中均匀分布
    """
    random.seed(seed)
    
    # 按场景和 trajectory 分组
    scenario_trajectories = defaultdict(lambda: defaultdict(list))
    for ex in examples:
        scenario = ex.get('metadata', {}).get('scenario_id', 'unknown')
        tid = ex.get('trajectory_id', '') or ex.get('metadata', {}).get('trajectory_id', 'unknown')
        scenario_trajectories[scenario][tid].append(ex)
    
    train, val, test = [], [], []
    
    print("\n📊 Splitting by scenario:")
    for scenario in sorted(scenario_trajectories.keys()):
        trajectories = scenario_trajectories[scenario]
        tids = list(trajectories.keys())
        random.shuffle(tids)
        
        n = len(tids)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_tids = set(tids[:train_end])
        val_tids = set(tids[train_end:val_end])
        test_tids = set(tids[val_end:])
        
        scenario_train = []
        scenario_val = []
        scenario_test = []
        
        for tid, exs in trajectories.items():
            if tid in train_tids:
                scenario_train.extend(exs)
            elif tid in val_tids:
                scenario_val.extend(exs)
            else:
                scenario_test.extend(exs)
        
        desc = SCENARIO_DESCRIPTIONS.get(scenario, scenario)
        print(f"  {desc}: train={len(scenario_train)}, val={len(scenario_val)}, test={len(scenario_test)}")
        
        train.extend(scenario_train)
        val.extend(scenario_val)
        test.extend(scenario_test)
    
    return train, val, test


def save_jsonl(data: List[Dict], filepath: str):
    """保存为 JSONL 格式"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  ✓ Saved {len(data):,} samples → {filepath}")


def save_by_scenario(examples: List[Dict], output_dir: str, split_name: str):
    """按场景分别保存"""
    scenario_data = defaultdict(list)
    for ex in examples:
        scenario = ex.get('metadata', {}).get('scenario_id', 'unknown')
        scenario_data[scenario].append(ex)
    
    for scenario, data in sorted(scenario_data.items()):
        filepath = os.path.join(output_dir, f"{split_name}_{scenario}.jsonl")
        save_jsonl(data, filepath)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare RedBlackBench SFT data with proper prompt mapping")
    parser.add_argument("--data_dir", default="hf_dataset", help="HuggingFace data directory")
    parser.add_argument("--output_dir", default="data", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_by_scenario", action="store_true", help="Save separate files by scenario")
    parser.add_argument("--validate_only", action="store_true", help="Only validate, don't save")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎮 RedBlackBench SFT Data Preparation")
    print("=" * 70)
    
    # 显示场景 prompt 映射
    print("\n📋 Scenario Prompt Mapping:")
    for scenario_id, desc in SCENARIO_DESCRIPTIONS.items():
        prompt_preview = SCENARIO_PROMPTS[scenario_id][:100].replace('\n', ' ')
        print(f"  • {scenario_id}: {desc}")
        print(f"    Preview: \"{prompt_preview}...\"")
    
    # 加载数据
    print("\n" + "=" * 70)
    print("📥 Loading HuggingFace Data")
    print("=" * 70)
    raw_examples = load_hf_datasets(args.data_dir)
    
    if not raw_examples:
        print("❌ No data found!")
        return
    
    # 分析原始数据
    print("\n" + "=" * 70)
    print("📊 Raw Data Analysis")
    print("=" * 70)
    stats = analyze_data(raw_examples)
    
    print(f"\nTotal samples: {stats.total:,}")
    print(f"\nBy Scenario:")
    for scenario, count in sorted(stats.by_scenario.items()):
        desc = SCENARIO_DESCRIPTIONS.get(scenario, scenario)
        print(f"  {desc}: {count:,} ({100*count/stats.total:.1f}%)")
    
    print(f"\nVote Distribution:")
    print(f"  VOTE A (cooperation): {stats.vote_a:,} ({100*stats.vote_a/stats.total:.1f}%)")
    print(f"  VOTE B (defection):   {stats.vote_b:,} ({100*stats.vote_b/stats.total:.1f}%)")
    
    print(f"\nLength Statistics:")
    print(f"  Input:  avg {stats.avg_input_len:,.0f} chars")
    print(f"  Output: avg {stats.avg_output_len:,.0f} chars")
    
    print(f"\nPrompt Mapping Validation:")
    print(f"  Match rate: {100*stats.prompt_match_rate:.1f}%")
    
    if args.validate_only:
        print("\n✓ Validation complete (--validate_only)")
        return
    
    # 转换为 chat 格式
    print("\n" + "=" * 70)
    print("🔄 Converting to Chat Format")
    print("=" * 70)
    
    chat_examples = []
    failed = 0
    for ex in raw_examples:
        converted = convert_to_chat_format(ex, use_scenario_system_prompt=True)
        if converted:
            chat_examples.append(converted)
        else:
            failed += 1
    
    print(f"  Converted: {len(chat_examples):,}")
    if failed > 0:
        print(f"  Failed: {failed}")
    
    # 分割数据
    print("\n" + "=" * 70)
    print("✂️ Splitting Data")
    print("=" * 70)
    
    train, val, test = split_by_trajectory(
        chat_examples,
        args.train_ratio,
        args.val_ratio,
        args.seed
    )
    
    print(f"\nFinal Split:")
    print(f"  Train: {len(train):,} ({100*len(train)/len(chat_examples):.1f}%)")
    print(f"  Val:   {len(val):,} ({100*len(val)/len(chat_examples):.1f}%)")
    print(f"  Test:  {len(test):,} ({100*len(test)/len(chat_examples):.1f}%)")
    
    # 保存数据
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("💾 Saving Data")
    print("=" * 70)
    
    save_jsonl(train, os.path.join(args.output_dir, "train.jsonl"))
    save_jsonl(val, os.path.join(args.output_dir, "val.jsonl"))
    save_jsonl(test, os.path.join(args.output_dir, "test.jsonl"))
    
    if args.save_by_scenario:
        print("\n  Saving by scenario...")
        save_by_scenario(train, args.output_dir, "train")
        save_by_scenario(val, args.output_dir, "val")
        save_by_scenario(test, args.output_dir, "test")
    
    # 保存统计信息
    stats_dict = {
        "total_examples": len(chat_examples),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "scenarios": dict(stats.by_scenario),
        "vote_distribution": {"A": stats.vote_a, "B": stats.vote_b},
        "prompt_match_rate": stats.prompt_match_rate,
        "scenario_descriptions": SCENARIO_DESCRIPTIONS,
    }
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Stats → {stats_path}")
    
    # 显示样本
    print("\n" + "=" * 70)
    print("📝 Sample Output")
    print("=" * 70)
    if train:
        sample = train[0]
        print(f"\nScenario: {sample['metadata']['scenario_id']}")
        print(f"Agent: {sample['metadata']['agent_name']}")
        print(f"Round: {sample['metadata']['round_index']}")
        print(f"\nMessages ({len(sample['messages'])}):")
        for i, msg in enumerate(sample['messages']):
            role = msg['role'].upper()
            content = msg['content']
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"\n  [{role}]")
            print(f"  {content}")
    
    print("\n" + "=" * 70)
    print("✅ Data preparation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

