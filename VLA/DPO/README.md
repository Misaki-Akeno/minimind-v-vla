# DPO: Direct Preference Optimization（井字棋｜MiniMind-V）

本目录提供一个最小可运行的 DPO 训练流程，复用已有的 VLM 封装与井字棋环境，支持：
- 偏好数据（chosen/rejected）格式
- 从 SFT 数据与环境生成简单偏好对
- 训练时计算 DPO 损失并优化策略模型（policy）

注意：本实现旨在教学/验证流程，未对齐工业级高效实现；默认仅微调语言侧与动作头，不动视觉编码器权重。

## 数据格式

JSONL，一行一条偏好对：

```
{
  "instruction": "...",                // 同 SFT
  "input": "...",                      // 同 SFT，可为空
  "image_path": "VLA/data/images/...", // 单图路径
  "chosen": "{\"thinking\":..., \"action\":[r,c]}",   // 偏好答案(JSON字符串)
  "rejected": "{\"thinking\":..., \"action\":[r,c]}"
}
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
```

3) 评估（动作准确率与语言损失，与 SFT 评估相同接口）：

```bash
conda run --no-capture-output -n minimind-v python VLA/DPO/evaluate.py \
  --ckpt VLA/models/dpo_vlm_ttt_768.pth \
  --val_jsonl VLA/data/ttt_val_alpaca.jsonl
```

## 参考
- DPO 论文：Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- 代码以 VLA/SFT 下训练与评估脚本为基础做了最小修改。
