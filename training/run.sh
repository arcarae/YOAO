#!/bin/bash
# RedBlackBench SFT 训练脚本
# 
# 使用方法:
#   ./run.sh prepare           # 准备数据
#   ./run.sh train             # 训练 (default Qwen3-14B)
#   ./run.sh train --model Qwen/Qwen3-14B  # 训练 14B
#   ./run.sh estimate          # 只估算资源需求

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
MODEL="${MODEL:-Qwen/Qwen3-14B}"
EPOCHS="${EPOCHS:-3}"
LORA_R="${LORA_R:-128}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

show_help() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  prepare     Prepare training data from HuggingFace dataset"
    echo "  train       Train the model"
    echo "  estimate    Estimate GPU requirements without training"
    echo ""
    echo "Environment Variables:"
    echo "  MODEL       Model name (default: Qwen/Qwen3-8B)"
    echo "  EPOCHS      Number of epochs (default: 3)"
    echo "  LORA_R      LoRA rank (default: 64)"
    echo "  LR          Learning rate (default: 2e-5)"
    echo "  BATCH_SIZE  Per-device batch size (default: 1)"
    echo "  GRAD_ACCUM  Gradient accumulation steps (default: 16)"
    echo "  OUTPUT_DIR  Output directory (default: outputs)"
    echo ""
    echo "Examples:"
    echo "  $0 prepare"
    echo "  $0 train"
    echo "  MODEL=Qwen/Qwen3-14B EPOCHS=5 $0 train"
    echo "  $0 estimate --model Qwen/Qwen3-14B"
}

prepare_data() {
    echo "=========================================="
    echo "📊 Preparing RedBlackBench SFT Data"
    echo "=========================================="
    
    python prepare_data.py \
        --data_dir hf_dataset \
        --output_dir data \
        --save_by_scenario \
        "$@"
}

train_model() {
    echo "=========================================="
    echo "🚀 Training RedBlackBench Model"
    echo "=========================================="
    echo "Model: $MODEL"
    echo "Epochs: $EPOCHS"
    echo "LoRA r: $LORA_R"
    echo "LR: $LR"
    echo "Batch: $BATCH_SIZE x $GRAD_ACCUM"
    echo "=========================================="
    
    python train.py \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --lora_r "$LORA_R" \
        --lr "$LR" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --max_length "$MAX_LENGTH" \
        --output "$OUTPUT_DIR" \
        "$@"
}

estimate_resources() {
    echo "=========================================="
    echo "📊 Estimating GPU Requirements"
    echo "=========================================="
    
    python train.py \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --lora_r "$LORA_R" \
        --estimate_only \
        "$@"
}

# 主入口
case "${1:-help}" in
    prepare)
        shift
        prepare_data "$@"
        ;;
    train)
        shift
        train_model "$@"
        ;;
    estimate)
        shift
        estimate_resources "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac

