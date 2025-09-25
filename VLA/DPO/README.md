# DPO: Direct Preference Optimization（井字棋｜MiniMind-V）

本目录提供多种DPO训练方法，包括传统的离线DPO和更先进的在线偏好学习方法：

## 方法对比

### 1. 传统方法 (generate_dpo_dataset.py)
- **问题**: 简单地从SFT数据集构造偏好对，chosen来自ground truth，rejected是随机动作
- **缺点**: 偏好对质量差，缺乏真实的策略分布信息

### 2. 采样式DPO (generate_dpo_from_sampling.py)  
- **改进**: 从VLA模型采样多个回答，基于启发式评估选择偏好对
- **优势**: 偏好对来自模型实际分布，更符合DPO理论假设

### 3. 环境反馈DPO (generate_dpo_with_env_feedback.py)
- **改进**: 结合井字棋环境规则评估动作质量，构造更准确的偏好对
- **优势**: 基于真实任务反馈，偏好对质量更高

### 4. 在线DPO (train_online_dpo.py)
- **最佳方法**: 训练过程中动态采样和构造偏好对，实时更新策略
- **优势**: 避免分布偏移，始终使用当前策略的采样数据

## 数据格式

JSONL，一行一条偏好对：

```
## 数据格式

JSONL，一行一条偏好对：

```json
{
  "instruction": "...",                // 同 SFT
  "input": "...",                      // 同 SFT，可为空
  "image_path": "VLA/data/images/...", // 单图路径
  "chosen": "{\"thinking\":..., \"action\":[r,c]}",   // 偏好答案(JSON字符串)
  "rejected": "{\"thinking\":..., \"action\":[r,c]}"  // 非偏好答案(JSON字符串)
}
```

## 推荐使用方法

### 方法1: 采样式DPO（推荐用于离线训练）

```bash
# 1. 从已训练的VLA模型采样构造偏好对
conda run --no-capture-output -n minimind-v python VLA/DPO/generate_dpo_from_sampling.py 
  --model_path VLA/models/sft_vlm_ttt_768.pth 
  --sft_jsonl VLA/data/ttt_train_alpaca.jsonl 
  --out_jsonl VLA/data/ttt_train_dpo_sampling.jsonl 
  --num_samples 4 
  --temperature 1.0

# 2. 使用生成的偏好对训练DPO
conda run --no-capture-output -n minimind-v python VLA/DPO/train_ttt_dpo.py 
  --train_jsonl VLA/data/ttt_train_dpo_sampling.jsonl 
  --val_jsonl VLA/data/ttt_val_alpaca.jsonl 
  --epochs 1 --batch_size 16 --learning_rate 1e-5 --use_reference
```

### 方法2: 环境反馈DPO（推荐用于更高质量的偏好对）

```bash
# 1. 基于环境反馈构造偏好对
conda run --no-capture-output -n minimind-v python VLA/DPO/generate_dpo_with_env_feedback.py 
  --model_path VLA/models/sft_vlm_ttt_768.pth 
  --sft_jsonl VLA/data/ttt_train_alpaca.jsonl 
  --out_jsonl VLA/data/ttt_train_dpo_env.jsonl 
  --num_samples 6 
  --temperature 1.2 
  --preference_margin 0.1

# 2. 训练DPO
conda run --no-capture-output -n minimind-v python VLA/DPO/train_ttt_dpo.py 
  --train_jsonl VLA/data/ttt_train_dpo_env.jsonl 
  --val_jsonl VLA/data/ttt_val_alpaca.jsonl 
  --epochs 2 --batch_size 12 --learning_rate 5e-6 --use_reference
```

### 方法3: 在线DPO（最佳方法，推荐用于最终训练）

```bash
# 直接进行在线DPO训练，无需预先生成偏好对
conda run --no-capture-output -n minimind-v python VLA/DPO/train_online_dpo.py 
  --sft_jsonl VLA/data/ttt_train_alpaca.jsonl 
  --val_jsonl VLA/data/ttt_val_alpaca.jsonl 
  --policy_ckpt VLA/models/sft_vlm_ttt_768.pth 
  --reference_ckpt MiniMind2-V/pytorch_model.bin 
  --epochs 2 
  --batch_size 8 
  --learning_rate 5e-6 
  --beta 0.1 
  --temperature 1.0 
  --num_samples 3
```

## 传统方法（不推荐，仅用于对比）

```bash
# 生成简单的随机偏好对（质量较差）
conda run --no-capture-output -n minimind-v python VLA/DPO/generate_dpo_dataset.py 
  --sft-jsonl VLA/data/ttt_train_alpaca.jsonl 
  --out-jsonl VLA/data/ttt_train_dpo_simple.jsonl
```

```

- 模型会把 instruction/input 打包为 user 消息，<image> 替换成视觉 token 占位符，与 SFT 一致。
- chosen 与 rejected 将分别拼接为 assistant 回复，形成两个完整样本；再计算两者的对数概率差构成 DPO 损失。

## 快速开始

1) 生成偏好对（基于现有 SFT 数据自动构造一个“正确vs随机”对）：

```bash
conda run --no-capture-output -n minimind-v python VLA/DPO/generate_dpo_dataset.py \
  --sft-jsonl VLA/data/ttt_train_alpaca.jsonl \
  --out-jsonl VLA/data/ttt_train_dpo.jsonl
```

也可同样生成 val/test：

```bash
conda run --no-capture-output -n minimind-v python VLA/DPO/generate_dpo_dataset.py \
  --sft-jsonl VLA/data/ttt_val_alpaca.jsonl \
  --out-jsonl VLA/data/ttt_val_dpo.jsonl
```

2) 训练 DPO：

```bash
conda run --no-capture-output -n minimind-v python VLA/DPO/train_ttt_dpo.py \
  --train_jsonl VLA/data/ttt_train_dpo.jsonl \
  --val_jsonl VLA/data/ttt_val_dpo.jsonl \
  --epochs 1 --batch_size 16 --learning_rate 1e-5
## 评估

```bash
conda run --no-capture-output -n minimind-v python VLA/DPO/evaluate.py \
  --ckpt VLA/models/online_dpo_vlm_final.pth \
  --val_jsonl VLA/data/ttt_val_alpaca.jsonl
```

## 核心改进说明

### 为什么不应该重复利用SFT数据集？

1. **分布偏移问题**: SFT数据集反映的是人类标注的偏好，而DPO需要的是当前策略模型的输出分布
2. **偏好对质量差**: 简单地用ground truth作为chosen，随机动作作为rejected，无法体现真实的偏好差异
3. **训练效果有限**: 模型学习的是如何区分human-labeled vs random，而不是如何改进自身策略

### 正确的DPO方法应该：

1. **从当前策略采样**: 使用当前训练的模型生成多个候选回答
2. **基于真实反馈排序**: 使用环境奖励或启发式评估对候选回答排序
3. **动态更新偏好对**: 随着训练进行，不断用新的策略采样更新偏好对
4. **保持分布一致性**: 确保训练数据来自当前策略分布，避免分布偏移

## 实验建议

1. **对比实验**: 可以对比传统方法和新方法的效果差异
2. **参数调优**: 重点调节temperature、num_samples、preference_margin等参数  
3. **评估指标**: 不仅看动作准确率，还要观察策略改进的趋势
4. **可视化分析**: 观察偏好对的质量分布和模型采样的多样性

## 参考文献
- DPO论文：Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- 在线学习相关：Constitutional AI, RLHF等
- VLA相关：RT-1, RT-2等视觉-语言-动作模型
