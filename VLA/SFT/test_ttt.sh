#!/bin/bash

# 井字棋VLA模型测试脚本
# 使用方法: bash test_ttt.sh [model_path]

# 默认模型路径
DEFAULT_MODEL="/pub_data/Codes/minimind-v/out/sft_vlm_ttt_768.pth"
MODEL_PATH=${1:-$DEFAULT_MODEL}

echo "开始测试井字棋VLA模型..."
echo "模型路径: $MODEL_PATH"

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "请先训练模型或指定正确的模型路径"
    echo "使用方法: bash test_ttt.sh /path/to/your/model.pth"
    exit 1
fi

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 测试参数
TEST_DATA="/pub_data/Codes/minimind-v/VLA/data/splits/ttt_test.jsonl"
IMAGES_ROOT="/pub_data/Codes/minimind-v/VLA/data/"
BATCH_SIZE=32
OUTPUT_DIR="/pub_data/Codes/minimind-v/VLA/test_results"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行测试
conda run --no-capture-output -n minimind-v python test_ttt_model.py \
    --model_path $MODEL_PATH \
    --test_data $TEST_DATA \
    --images_root $IMAGES_ROOT \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR

echo "测试完成! 结果保存在: $OUTPUT_DIR"
