#!/bin/bash

# 井字棋VLA模型训练脚本
# 使用方法: bash train_ttt.sh

echo "开始训练井字棋VLA模型..."

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 训练参数
EPOCHS=5
BATCH_SIZE=16
LEARNING_RATE=1e-5
VAL_INTERVAL=50          # 每50步验证一次
MAX_VAL_STEPS=50         # 每次验证最多50步
EARLY_STOP_WINDOW=20     # 早停滑动窗口
EARLY_STOP_THRESHOLD=0.95 # 早停阈值95%
EARLY_STOP_PATIENCE=3    # 连续3次满足阈值才早停

# 数据路径
TRAIN_DATA="/pub_data/Codes/minimind-v/VLA/data/splits/ttt_train.jsonl"
VAL_DATA="/pub_data/Codes/minimind-v/VLA/data/splits/ttt_val.jsonl"
IMAGES_ROOT="/pub_data/Codes/minimind-v/VLA/data/"

# 输出目录
OUT_DIR="/pub_data/Codes/minimind-v/out"

# 运行训练
conda run --no-capture-output -n minimind-v python train_ttt_sft.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --data_path $TRAIN_DATA \
    --val_data_path $VAL_DATA \
    --images_root $IMAGES_ROOT \
    --out_dir $OUT_DIR \
    --val_interval $VAL_INTERVAL \
    --max_val_steps $MAX_VAL_STEPS \
    --early_stop_window $EARLY_STOP_WINDOW \
    --early_stop_threshold $EARLY_STOP_THRESHOLD \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --use_wandb \
    --wandb_project "MiniMind-V-TTT-EarlyStop" \
    --log_interval 10 \
    --save_interval 100

echo "训练完成!"
