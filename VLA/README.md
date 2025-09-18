# VLA井字棋智能体

基于MiniMind-V的视觉语言动作(VLA)智能体，能够玩井字棋并展示详细的思考过程。

## 🚀 快速开始

### 环境配置
```bash
conda activate minimind-v
cd VLA
```

### 立即体验人机对战
```bash
# 人机对战（需要预训练模型）
conda run --no-capture-output -n minimind-v python play.py \
    --model_path ../out/sft_vlm_ttt_768.pth
```

### 训练自己的模型
```bash
# 1. 生成训练数据
python envs/generate_dataset.py

# 2. 开始训练
bash train_ttt.sh

# 3. 测试模型
bash test_ttt.sh
```

## 🎮 游戏说明

- **你是 X**，VLA模型是 **O**
- **鼠标单击**空格放置棋子
- 观察控制台中VLA的**详细思考过程**
- 关闭窗口或按Ctrl+C退出

## 🧠 VLA思考过程示例

```
🤖 VLA模型开始思考...
📋 当前棋盘状态:
X 空位 空位 
空位 O 空位 
空位 空位 空位 

🔍 战术分析: O在中心位置有优势

📊 各位置选择概率:
   位置1 (1,2): 0.156 
   位置7 (3,2): 0.201 ⭐

🎯 最终决策: 位置7 (3,2)
   置信度: 0.201
   理由: 中等置信度选择
```

## 🔧 技术栈

- **模型**: MiniMind-V + Action Head
- **环境**: Gymnasium + Pygame  
- **训练**: PyTorch + Transformers
- **可视化**: Weights & Biases

## 📊 性能指标

- **动作准确率**: 85%+
- **平均推理时间**: <200ms
- **支持分辨率**: 300x300 RGB
- **动作空间**: 9个位置 (3x3棋盘)

## 🎯 特色功能

- ✅ 端到端视觉-语言-动作学习
- ✅ 实时思考过程可视化  
- ✅ 流畅的人机交互界面
- ✅ 完整的训练评估pipeline
- ✅ 模块化可扩展架构

---

## 📁 主要文件

- `play.py` - **人机对战游戏** ⭐
- `train_ttt_sft.py` - 训练脚本
- `test_ttt_model.py` - 模型测试
- `model_wrapper.py` - VLA模型封装
- `envs/tic_tac_toe_env.py` - 游戏环境

详细技术报告: [VLA_FINETUNE_REPORT.md](VLA_FINETUNE_REPORT.md)

---

# 原项目文档

## VLA: 井字棋 VLM SFT 微调

本目录提供基于 MiniMind-V 的井字棋（TicTacToe）视觉-语言微调（SFT）脚本与数据集适配器，避免修改原工程文件。

## 数据格式

输入采用 JSONL（每行一个 JSON 对象），示例：

```
{"instruction": "你是一个井字棋智能体。给定一张棋盘图像，请推理当前局面并仅输出一个 JSON:{\"thinking\": string, \"action\": [row, col]}；行列取值 0-2，且只输出 JSON。", "input": "", "output": "{\"thinking\": \"我能了解到现在的棋盘是[[X,  ,  ], [ ,  ,  ], [ ,  ,  ]]，我是其中的O，下一步轮到我下。中心位置通常更强，可优先考虑中心。因此我的选择是落子到[1,1]。\", \"action\": [1, 1]}", "image_path": "../data/out/images/ttt_000000.png", "meta": {"board": [["X", " ", " "], [" ", " ", " "], [" ", " ", " "]], "current_player": "O", "valid_actions": [1, 2, 3, 4, 5, 6, 7, 8]}}
```

字段说明：
- instruction: 指令文本。
- input: 额外输入（可为空）。
- output: 期望模型输出的文本（建议严格是 JSON 字符串）。
- image_path: 棋盘图像路径；支持绝对路径或相对路径（可配合 `--images_root`）。
- meta: 可选，训练中不使用。

## 组件说明

- `ttt_dataset.py`: 自定义 `TicTacToeVLMDataset` 数据集类，构造两轮对话（user: `<image>+instruction(+input)`；assistant: `output`），并对 `<image>` 替换为视觉占位 token 序列。
- `train_ttt_sft.py`: 训练脚本，复用现有 MiniMind-V 模型与视觉编码器，保存权重到 `--out_dir`。

## 运行

基础单卡运行：

```bash
python VLA/train_ttt_sft.py \
  --data_path /path/to/ttt.jsonl \
  --images_root /path/to/images_root \
  --out_dir ./out \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 1e-5
```

说明：
- 若有预训练权重，可指定 `--pretrained_path` 指向 `pretrain_vlm_*.pth`；否则会从头开始 SFT。
- 默认视觉编码器路径为 `../model/vision_model/clip-vit-base-patch16`（相对本脚本所在位置）。

多卡（DDP）运行（示例）：

```bash
torchrun --nproc_per_node=2 VLA/train_ttt_sft.py \
  --ddp \
  --data_path /path/to/ttt.jsonl \
  --images_root /path/to/images_root \
  --out_dir ./out \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 1e-5
```

## 常见问题

1. 找不到图片：检查 `--images_root` 与 `image_path` 的组合路径是否存在；`image_path` 也可为绝对路径。
2. 显存不足：减小 `--batch_size` 或使用梯度累积 `--accumulation_steps`。
3. 训练过慢：开启 bfloat16/float16（默认自动），或减少 `--hidden_size`/`--num_hidden_layers`，仅用于实验验证。
