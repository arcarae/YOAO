#!/usr/bin/env python3
"""
RedBlackBench SFT Training Script

训练 Qwen3 模型进行囚徒困境场景的战略推理。

GPU 需求估算 (Qwen3-14B, LoRA r=64):
- 模型权重 (BF16): ~28GB
- 梯度: ~28GB  
- 优化器状态: ~0.4GB (LoRA only)
- 激活值 (with checkpointing): ~10GB
- 总计: ~67GB

推荐配置:
- A100 (80GB): 1张足够，可用 batch_size=2
- H100 (96GB): 1张足够，可用 batch_size=4
- H200 (192GB): 1张，可用更大 batch_size

对于 Qwen3-8B (~16GB 权重):
- A100 (80GB): 1张，batch_size=4
- H100 (96GB): 1张，batch_size=8
"""

import os
import sys
import json
import argparse
import math
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset

# 添加 redblackbench 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TrainingConfig:
    """训练配置"""
    # Model
    model_name: str = "Qwen/Qwen3-14B"
    
    # LoRA
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Data
    train_path: str = "data/train.jsonl"
    val_path: str = "data/val.jsonl"
    max_length: int = 4096
    
    # Training (paper: lr 1e-5 cosine, effective batch 16 = 2 x 8, 3 epochs, 200 warmup steps,
    # weight decay 0.01, FP16, max length 4096)
    output_dir: str = "outputs"
    epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 8
    lr: float = 1e-5
    warmup_steps: int = 200
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Optimization (paper: FP16)
    bf16: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    # Early stopping
    patience: int = 5
    threshold: float = 0.001


class RedBlackDataset(Dataset):
    """RedBlackBench 数据集"""
    
    def __init__(self, path: str, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"  Loaded {len(self.data):,} samples from {path}")
        self._show_length_stats()
    
    def _show_length_stats(self):
        """显示 token 长度统计"""
        sample_size = min(100, len(self.data))
        lengths = []
        
        for item in self.data[:sample_size]:
            text = self.tokenizer.apply_chat_template(
                item['messages'], tokenize=False, add_generation_prompt=False
            )
            lengths.append(len(self.tokenizer.encode(text)))
        
        if lengths:
            over_limit = sum(1 for l in lengths if l > self.max_length)
            print(f"    Token lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")
            if over_limit > 0:
                print(f"    ⚠️  {over_limit}/{sample_size} samples exceed max_length={self.max_length}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = self.tokenizer.apply_chat_template(
            item['messages'], tokenize=False, add_generation_prompt=False
        )
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': encoded['input_ids'].copy()
        }


def collate_fn(batch, pad_token_id: int):
    """动态填充"""
    max_len = max(len(x['input_ids']) for x in batch)
    
    input_ids = []
    attention_mask = []
    labels = []
    
    for x in batch:
        pad_len = max_len - len(x['input_ids'])
        input_ids.append(x['input_ids'] + [pad_token_id] * pad_len)
        attention_mask.append(x['attention_mask'] + [0] * pad_len)
        labels.append(x['labels'] + [-100] * pad_len)
    
    return {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(attention_mask),
        'labels': torch.tensor(labels)
    }


def estimate_gpu_memory(model_name: str, lora_r: int, max_length: int) -> Dict:
    """
    估算 GPU 显存需求
    
    基于经验值:
    - Qwen3-8B: ~16GB 模型权重 (BF16), 训练时峰值 ~40GB
    - Qwen3-14B: ~28GB 模型权重 (BF16), 训练时峰值 ~60GB
    - Qwen3-32B: ~64GB 模型权重 (BF16), 训练时峰值 ~100GB
    """
    # 模型配置 (参数量, hidden_dim, num_layers)
    model_configs = {
        "8B": (8e9, 4096, 32),
        "14B": (14e9, 5120, 40),
        "32B": (32e9, 6144, 64),
        "72B": (72e9, 8192, 80),
    }
    
    # 从模型名提取大小
    params, hidden_dim, num_layers = 8e9, 4096, 32
    for size_str, config in model_configs.items():
        if size_str in model_name:
            params, hidden_dim, num_layers = config
            break
    
    bytes_per_param = 2  # BF16
    
    # 1. 模型权重 (frozen, 只占用推理内存)
    model_mem = params * bytes_per_param
    
    # 2. LoRA 参数 + 梯度 + 优化器状态
    # 每层: q,k,v,o,gate,up,down = 7个目标模块
    # 每个模块: 2个矩阵 (A: hidden_dim x r, B: r x hidden_dim)
    lora_params_per_layer = 7 * 2 * (hidden_dim * lora_r)
    total_lora_params = lora_params_per_layer * num_layers
    # 参数(BF16) + 梯度(BF16) + Adam状态(FP32 m和v)
    lora_mem = total_lora_params * (2 + 2 + 8)
    
    # 3. 激活值 (with gradient checkpointing)
    # 粗略估算: batch_size=1, seq_len=max_length
    # 每层保存: attention scores + 中间激活
    activation_per_layer = max_length * hidden_dim * 4 * bytes_per_param
    activation_mem = activation_per_layer * (num_layers // 4)  # checkpointing 减少到 1/4
    
    # 4. 额外开销 (CUDA kernels, workspace, fragmentation等)
    # 实际训练中通常需要 1.5-2x 的理论值
    overhead = 4 * (1024**3)  # ~4GB base overhead
    
    theoretical = model_mem + lora_mem + activation_mem + overhead
    # 添加安全边际 (1.5x) 因为 PyTorch/CUDA 内存碎片化
    practical = theoretical * 1.5
    practical_gb = practical / (1024**3)
    
    return {
        "model_params": f"{params/1e9:.0f}B",
        "model_memory_gb": model_mem / (1024**3),
        "lora_memory_gb": lora_mem / (1024**3),
        "activation_memory_gb": activation_mem / (1024**3),
        "total_estimated_gb": practical_gb,
        "recommended_gpu": (
            "A100 80GB x1" if practical_gb < 70 else
            "H100 96GB x1" if practical_gb < 85 else
            "H200 192GB x1" if practical_gb < 170 else
            "Multi-GPU required"
        )
    }


def setup_model(config: TrainingConfig):
    """加载并配置模型"""
    print("\n🔧 Setting up model...")
    print(f"  Model: {config.model_name}")
    
    # 检查可用 GPU
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name} ({total_mem:.0f} GB)")
    
    # 加载模型
    print("\n  Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if config.fp16 else torch.bfloat16,
        device_map="auto",
    )
    
    # 启用 gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    
    # 配置 LoRA
    print(f"\n  Configuring LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if config.fp16:
        # Trainer's FP16 AMP needs the trainable (LoRA) parameters in FP32; the frozen base
        # weights stay FP16.
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    model.config.use_cache = False
    model.enable_input_require_grads()

    return model


def estimate_training_time(
    num_samples: int,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    gpu_type: str = "A100"
) -> Dict:
    """估算训练时间"""
    total_steps = math.ceil(num_samples / (batch_size * grad_accum)) * epochs
    
    # 每步时间估算 (秒)
    step_times = {
        "A100": 4.0,
        "H100": 2.5,
        "H200": 1.5,
    }
    
    step_time = step_times.get(gpu_type, 4.0)
    total_seconds = total_steps * step_time
    
    return {
        "total_steps": total_steps,
        "step_time_seconds": step_time,
        "total_hours": total_seconds / 3600,
        "gpu_type": gpu_type,
    }


def main():
    parser = argparse.ArgumentParser(description="Train RedBlackBench model")
    
    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=None)  # 默认 2*r
    
    # Data
    parser.add_argument("--train", type=str, default="data/train.jsonl")
    parser.add_argument("--val", type=str, default="data/val.jsonl")
    parser.add_argument("--max_length", type=int, default=4096)
    
    # Training
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    
    # Logging
    parser.add_argument("--wandb", type=str, default=None, help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None)
    
    # Other
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--estimate_only", action="store_true", help="Only estimate, don't train")
    
    args = parser.parse_args()
    
    # 构建配置
    config = TrainingConfig(
        model_name=args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha or args.lora_r * 2,
        train_path=args.train,
        val_path=args.val,
        max_length=args.max_length,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
    )
    
    # 打印配置
    print("=" * 70)
    print("🎮 RedBlackBench SFT Training")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch: {config.batch_size} x {config.grad_accum} = {config.batch_size * config.grad_accum}")
    print(f"  Max length: {config.max_length}")
    print(f"  Output: {config.output_dir}")
    
    # GPU 显存估算 (不需要加载模型)
    mem_est = estimate_gpu_memory(config.model_name, config.lora_r, config.max_length)
    print(f"\n📊 GPU Memory Estimation:")
    print(f"  Model ({mem_est['model_params']}): {mem_est['model_memory_gb']:.1f} GB")
    print(f"  LoRA (r={config.lora_r}): {mem_est['lora_memory_gb']:.1f} GB")
    print(f"  Activations: {mem_est['activation_memory_gb']:.1f} GB")
    print(f"  Total: ~{mem_est['total_estimated_gb']:.0f} GB")
    print(f"  Recommended: {mem_est['recommended_gpu']}")
    
    # 检查数据文件
    if not os.path.exists(config.train_path):
        print(f"\n❌ Training data not found: {config.train_path}")
        print("   Run prepare_data.py first!")
        return
    
    # 估算训练时间 (基于文件行数)
    with open(config.train_path) as f:
        num_train_samples = sum(1 for _ in f)
    
    print(f"\n📈 Training Data: {num_train_samples:,} samples")
    
    for gpu_type in ["A100", "H100", "H200"]:
        est = estimate_training_time(
            num_train_samples, config.epochs, 
            config.batch_size, config.grad_accum, gpu_type
        )
        print(f"\n⏱️  Training time estimate ({gpu_type}):")
        print(f"    Steps: {est['total_steps']:,}")
        print(f"    Time: ~{est['total_hours']:.1f} hours")
    
    if args.estimate_only:
        print("\n✓ Estimation complete (--estimate_only)")
        return
    
    # 加载 tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据
    print("\n📥 Loading datasets...")
    train_dataset = RedBlackDataset(config.train_path, tokenizer, config.max_length)
    val_dataset = RedBlackDataset(config.val_path, tokenizer, config.max_length)
    
    # 设置模型
    model = setup_model(config)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        lr_scheduler_type="cosine",
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=False,  # already enabled on the model
        dataloader_num_workers=0,
        report_to="wandb" if args.wandb else "none",
        run_name=args.run_name or f"redblack-r{config.lora_r}-lr{config.lr}",
        remove_unused_columns=False,
        logging_first_step=True,
    )
    
    # Callbacks
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=config.patience,
            early_stopping_threshold=config.threshold,
        )
    ]
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda b: collate_fn(b, tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    
    # 训练
    print("\n" + "=" * 70)
    print("🚀 Starting Training")
    print("=" * 70)
    
    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()
    
    # 保存模型
    final_path = os.path.join(config.output_dir, "final_model")
    print(f"\n💾 Saving final model to: {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    # 保存训练配置
    config_dict = {
        "model_name": config.model_name,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "learning_rate": config.lr,
        "epochs": config.epochs,
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    with open(os.path.join(final_path, "training_config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print(f"   Model saved to: {final_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

