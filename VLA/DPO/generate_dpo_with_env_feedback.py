import argparse
import json
import os
import random
from typing import Dict, Any, Tuple, List
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np

from model.model_vlm import VLMConfig
from VLA.envs.model_wrapper import MiniMindVLMWithAction


class TicTacToeEnvironment:
    """
    简化的井字棋环境，用于评估动作质量
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)  # 0: empty, 1: X, -1: O
        self.current_player = 1  # X先手
        return self.board.copy()
    
    def is_valid_action(self, row: int, col: int) -> bool:
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row, col] == 0
    
    def make_move(self, row: int, col: int, player: int = None) -> Tuple[np.ndarray, bool, int]:
        """
        执行动作
        返回: (new_board, game_over, reward)
        reward: 1表示当前玩家胜利, -1表示失败, 0表示平局或游戏继续
        """
        if player is None:
            player = self.current_player
            
        if not self.is_valid_action(row, col):
            return self.board.copy(), True, -1  # 非法动作，惩罚
        
        self.board[row, col] = player
        
        # 检查游戏是否结束
        winner = self.check_winner()
        if winner != 0:
            reward = 1 if winner == player else -1
            return self.board.copy(), True, reward
        
        if self.is_board_full():
            return self.board.copy(), True, 0  # 平局
        
        # 切换玩家
        self.current_player = -self.current_player
        return self.board.copy(), False, 0
    
    def check_winner(self) -> int:
        """检查获胜者, 返回1(X胜), -1(O胜), 0(无胜者)"""
        # 检查行
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                return self.board[i, 0]
        
        # 检查列
        for j in range(3):
            if abs(sum(self.board[:, j])) == 3:
                return self.board[0, j]
        
        # 检查对角线
        if abs(sum([self.board[i, i] for i in range(3)])) == 3:
            return self.board[0, 0]
        
        if abs(sum([self.board[i, 2-i] for i in range(3)])) == 3:
            return self.board[0, 2]
        
        return 0
    
    def is_board_full(self) -> bool:
        return np.all(self.board != 0)
    
    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """获取所有有效动作"""
        valid_actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    valid_actions.append((i, j))
        return valid_actions
    
    def evaluate_board_state(self, player: int) -> float:
        """
        评估当前棋盘状态对指定玩家的价值
        返回-1到1之间的值
        """
        winner = self.check_winner()
        if winner == player:
            return 1.0
        elif winner == -player:
            return -1.0
        elif self.is_board_full():
            return 0.0
        
        # 简单的启发式评估
        score = 0.0
        
        # 评估每行、每列、每对角线
        lines = []
        # 行
        for i in range(3):
            lines.append(self.board[i, :])
        # 列
        for j in range(3):
            lines.append(self.board[:, j])
        # 对角线
        lines.append([self.board[i, i] for i in range(3)])
        lines.append([self.board[i, 2-i] for i in range(3)])
        
        for line in lines:
            line_score = self._evaluate_line(line, player)
            score += line_score
        
        return np.clip(score / 8.0, -1.0, 1.0)
    
    def _evaluate_line(self, line: List[int], player: int) -> float:
        """评估单条线(行/列/对角线)的价值"""
        my_count = sum(1 for x in line if x == player)
        opp_count = sum(1 for x in line if x == -player)
        
        if opp_count > 0 and my_count > 0:
            return 0  # 被阻断
        elif my_count == 3:
            return 10  # 获胜
        elif opp_count == 3:
            return -10  # 失败
        elif my_count == 2:
            return 5  # 即将获胜
        elif opp_count == 2:
            return -5  # 需要防守
        elif my_count == 1:
            return 1
        elif opp_count == 1:
            return -1
        else:
            return 0


def parse_board_state_from_image_path(image_path: str) -> np.ndarray:
    """
    从图像路径推断棋盘状态
    这是一个简化的方法，实际中需要图像识别
    """
    # 从文件名中提取信息（假设文件名包含状态信息）
    filename = os.path.basename(image_path)
    
    # 创建一个随机的棋盘状态用于演示
    # 实际应用中需要真正的图像识别
    board = np.zeros((3, 3), dtype=int)
    
    # 这里可以根据文件名或其他信息来推断棋盘状态
    # 现在先返回一个示例状态
    return board


def evaluate_action_with_environment(
    action_text: str, 
    image_path: str, 
    player: int = 1
) -> float:
    """
    使用环境反馈评估动作质量
    """
    try:
        # 解析动作
        obj = json.loads(action_text)
        if not isinstance(obj, dict) or 'action' not in obj:
            return 0.0
        
        row, col = int(obj['action'][0]), int(obj['action'][1])
        
        # 创建环境并设置棋盘状态
        env = TicTacToeEnvironment()
        # 这里应该根据图像解析出真实的棋盘状态
        # 现在使用简化的方法
        board_state = parse_board_state_from_image_path(image_path)
        env.board = board_state
        env.current_player = player
        
        # 检查动作是否有效
        if not env.is_valid_action(row, col):
            return -0.5  # 非法动作
        
        # 执行动作并评估结果
        new_board, game_over, reward = env.make_move(row, col, player)
        
        if game_over:
            return float(reward)  # 直接获胜/失败/平局
        
        # 游戏继续，评估新状态
        env.board = new_board
        state_value = env.evaluate_board_state(player)
        
        # 考虑位置的战略价值
        position_bonus = 0.0
        if row == 1 and col == 1:  # 中心位置
            position_bonus = 0.2
        elif (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]:  # 角落
            position_bonus = 0.1
        
        return state_value + position_bonus
        
    except Exception as e:
        return 0.0  # 解析错误


def sample_from_model(
    model: MiniMindVLMWithAction,
    tokenizer,
    instruction: str,
    user_input: str,
    image_path: str,
    device: str,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    max_new_tokens: int = 100,
    num_samples: int = 4,
) -> List[str]:
    """从模型采样多个回答"""
    model.eval()
    
    # 构造prompt
    content_parts = ["<image>", instruction.strip() if instruction else ""]
    if user_input:
        content_parts.append(user_input.strip())
    content = "\n".join([p for p in content_parts if p])
    
    # 替换image token
    image_special_token = '@' * 196
    messages = [
        {"role": "user", "content": content.replace('<image>', image_special_token)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # 加载图像
    image = Image.open(image_path)
    image_tensor = model.vlm.image2tensor(image, model.processor)
    pixel_values = torch.stack([image_tensor], dim=0).to(device)
    
    # tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            generated_ids = input_ids.clone()
            
            for _ in range(max_new_tokens):
                outputs = model(
                    input_ids=generated_ids,
                    pixel_values=pixel_values,
                )
                
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                # top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            generated_text = tokenizer.decode(
                generated_ids[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            samples.append(generated_text)
    
    return samples


def create_environment_based_preference_pairs(
    samples: List[str], 
    image_path: str,
    preference_margin: float = 0.1
) -> List[Tuple[str, str, float]]:
    """
    基于环境反馈创建偏好对
    返回 (chosen, rejected, score_diff) 的列表
    """
    # 评估每个样本
    scored_samples = []
    for sample in samples:
        score = evaluate_action_with_environment(sample, image_path)
        scored_samples.append((sample, score))
    
    # 按分数排序
    scored_samples.sort(key=lambda x: x[1], reverse=True)
    
    # 创建偏好对
    pairs = []
    for i, (sample_i, score_i) in enumerate(scored_samples):
        for j, (sample_j, score_j) in enumerate(scored_samples):
            if i < j and score_i - score_j > preference_margin:
                pairs.append((sample_i, sample_j, score_i - score_j))
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate DPO dataset with environment-based preferences")
    parser.add_argument("--model_path", type=str, required=True, help="VLA模型检查点路径")
    parser.add_argument("--sft_jsonl", type=str, required=True, help="SFT数据集路径")
    parser.add_argument("--out_jsonl", type=str, required=True, help="输出DPO数据集路径")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=6, help="每个prompt采样的回答数量")
    parser.add_argument("--temperature", type=float, default=1.2, help="采样温度")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--preference_margin", type=float, default=0.1, help="偏好对的最小分数差")
    parser.add_argument("--max_samples", type=int, default=500, help="最大处理样本数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = args.device
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
    
    # 加载模型配置
    config = VLMConfig(
        vocab_size=6400,
        hidden_size=768,
        intermediate_size=1536,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000,
        use_moe=False,
    )
    
    model = MiniMindVLMWithAction(config)
    
    # 加载检查点
    if args.model_path.endswith('.pth'):
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
    else:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.vlm.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    # 读取SFT数据
    sft_samples = []
    with open(args.sft_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sft_samples.append(json.loads(line))
    
    if args.max_samples > 0:
        sft_samples = sft_samples[:args.max_samples]
    
    print(f"Processing {len(sft_samples)} samples...")
    
    dpo_pairs = []
    for i, sample in enumerate(sft_samples):
        if i % 20 == 0:
            print(f"Processing sample {i}/{len(sft_samples)}, generated {len(dpo_pairs)} pairs so far")
        
        instruction = sample.get('instruction', '')
        user_input = sample.get('input', '')
        image_path = sample.get('image_path', '')
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}, skipping...")
            continue
        
        try:
            # 从模型采样
            samples = sample_from_model(
                model=model,
                tokenizer=tokenizer,
                instruction=instruction,
                user_input=user_input,
                image_path=image_path,
                device=device,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_samples=args.num_samples,
            )
            
            # 基于环境创建偏好对
            pairs = create_environment_based_preference_pairs(
                samples, 
                image_path, 
                args.preference_margin
            )
            
            # 创建数据项
            for chosen, rejected, score_diff in pairs:
                dpo_item = {
                    "instruction": instruction,
                    "input": user_input,
                    "image_path": image_path,
                    "chosen": chosen,
                    "rejected": rejected,
                    "score_diff": float(score_diff),  # 记录分数差用于分析
                }
                dpo_pairs.append(dpo_item)
        
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # 写入输出文件
    with open(args.out_jsonl, 'w', encoding='utf-8') as f:
        for item in dpo_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(dpo_pairs)} environment-based DPO pairs -> {args.out_jsonl}")
    
    # 统计信息
    if dpo_pairs:
        score_diffs = [item['score_diff'] for item in dpo_pairs]
        print(f"Score difference stats: min={min(score_diffs):.3f}, max={max(score_diffs):.3f}, mean={np.mean(score_diffs):.3f}")


if __name__ == "__main__":
    main()
