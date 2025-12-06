#!/bin/bash
# 快速训练脚本

# 设置默认参数
MODEL_NAME="cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_DIR="./outputs"
BATCH_SIZE=16
LEARNING_RATE=2e-5
NUM_EPOCHS=5
ALPHA=0.5
BETA=0.5

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
            ;;
        --use_features)
            USE_FEATURES="--use_features"
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "开始训练模型..."
echo "模型: $MODEL_NAME"
echo "输出目录: $OUTPUT_DIR"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "训练轮数: $NUM_EPOCHS"

python train.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --alpha "$ALPHA" \
    --beta "$BETA" \
    $USE_FEATURES

echo "训练完成！模型保存在: $OUTPUT_DIR/final_model"

