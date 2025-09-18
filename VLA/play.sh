#!/bin/bash

# 井字棋VLA对战启动脚本
# Disable ALSA warnings
export ALSA_CONFIG_PATH=/dev/null
# 默认模型路径 - 请根据实际情况修改
DEFAULT_MODEL_PATH="/pub_data/Codes/minimind-v/out/sft_vlm_ttt_768.pth"

# 使用conda环境运行
CONDA_ENV="minimind-v"

echo "🎮 启动井字棋VLA对战..."
echo "================================"


# 使用conda运行脚本
conda run --no-capture-output -n $CONDA_ENV python /pub_data/Codes/minimind-v/VLA/play.py --model_path "$DEFAULT_MODEL_PATH"
