#!/usr/bin/env python3
"""
井字棋VLA模型对战脚本

让训练好的VLA模型与玩家进行井字棋对战，展示模型的思考过程。

使用方法:
    python play.py --model_path /path/to/model.pth
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
import numpy as np
import pygame
from PIL import Image

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_vlm import VLMConfig
from VLA.model_wrapper import MiniMindVLMWithAction
from VLA.envs.tic_tac_toe_env import TicTacToeEnv, Player
from transformers import AutoTokenizer

# 固定配置（与训练时保持一致）
HIDDEN_SIZE = 768
NUM_LAYERS = 16
MAX_SEQ_LEN = 512
USE_MOE = False


class VLAPlayer:
    """VLA模型玩家"""
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        print(f"🚀 初始化VLA模型玩家...")
        print(f"   设备: {self.device}")
        print(f"   模型路径: {model_path}")
        
        # 初始化模型配置
        self.model_config = VLMConfig(
            hidden_size=HIDDEN_SIZE, 
            num_hidden_layers=NUM_LAYERS,
            max_seq_len=MAX_SEQ_LEN, 
            use_moe=USE_MOE
        )
        
        # 初始化tokenizer
        tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "model")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        # 加载模型
        self._load_model()
        
    def _load_model(self):
        """加载训练好的模型"""
        try:
            print("📦 正在加载模型...")
            
            # 初始化模型
            self.model = MiniMindVLMWithAction(
                config=self.model_config, 
                vision_model_path=os.path.join(os.path.dirname(__file__), "..", "model", "vision_model", "clip-vit-base-patch16")
            )
            
            # 加载权重
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
                
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # 使用strict=False来忽略不匹配的键（与测试脚本保持一致）
            self.model.load_state_dict(state_dict, strict=False)
            print("✅ 已加载模型权重")
                
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ 模型加载完成，设备: {self.device}")
            print(f"   可训练参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.3f} 百万")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 转换为PIL图像
        if image.dtype == np.uint8:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
        
        # 使用模型的processor处理图像
        pixel_values = self.model.processor(images=pil_image, return_tensors="pt").pixel_values
        
        # 添加num_images维度: (B, C, H, W) -> (B, 1, C, H, W)
        pixel_values = pixel_values.unsqueeze(1)
        
        return pixel_values.to(self.device)
    
    def _format_board_description(self, board: np.ndarray) -> str:
        """格式化棋盘描述"""
        board_str = ""
        for i in range(3):
            for j in range(3):
                if board[i, j] == 0:
                    board_str += "空位 "
                elif board[i, j] == 1:
                    board_str += "X "
                else:
                    board_str += "O "
            if i < 2:
                board_str += "\n"
        return board_str
    
    def _analyze_board_state(self, board: np.ndarray) -> str:
        """分析当前棋盘状态"""
        analysis = []
        
        # 检查是否有即将获胜的机会
        for player in [2, 1]:  # 先检查AI(O=2)，再检查玩家(X=1)
            symbol = "O" if player == 2 else "X"
            
            # 检查行
            for i in range(3):
                if np.sum(board[i, :] == player) == 2 and np.sum(board[i, :] == 0) == 1:
                    empty_pos = np.where(board[i, :] == 0)[0][0]
                    analysis.append(f"{symbol}在第{i+1}行有两子，位置({i+1},{empty_pos+1})可获胜")
            
            # 检查列
            for j in range(3):
                if np.sum(board[:, j] == player) == 2 and np.sum(board[:, j] == 0) == 1:
                    empty_pos = np.where(board[:, j] == 0)[0][0]
                    analysis.append(f"{symbol}在第{j+1}列有两子，位置({empty_pos+1},{j+1})可获胜")
            
            # 检查对角线
            if np.sum([board[i, i] for i in range(3)] == player) == 2:
                for i in range(3):
                    if board[i, i] == 0:
                        analysis.append(f"{symbol}在主对角线有两子，位置({i+1},{i+1})可获胜")
                        break
            
            if np.sum([board[i, 2-i] for i in range(3)] == player) == 2:
                for i in range(3):
                    if board[i, 2-i] == 0:
                        analysis.append(f"{symbol}在副对角线有两子，位置({i+1},{3-i})可获胜")
                        break
        
        return " ".join(analysis) if analysis else "当前无明显威胁或机会"
    
    def predict_action(self, board: np.ndarray, observation: np.ndarray) -> int:
        """预测下一步动作并显示思考过程"""
        print("\n🤖 VLA模型开始思考...")
        print("=" * 50)
        
        # 显示当前棋盘状态
        print("📋 当前棋盘状态:")
        print(self._format_board_description(board))
        
        # 显示战术分析
        print("\n🔍 战术分析:")
        analysis = self._analyze_board_state(board)
        print(f"   {analysis}")
        
        # 预处理图像
        print("\n🖼️ 正在处理视觉输入...")
        pixel_values = self._preprocess_image(observation)
        print(f"   图像尺寸: {pixel_values.shape}")
        
        # 构建文本提示
        prompt = "分析当前井字棋局面，我是O，你需要帮我选择最佳位置(0-8):"
        print(f"\n💭 文本提示: {prompt}")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        print(f"   Token长度: {input_ids.shape[1]}")
        
        # 模型推理
        print("\n⚡ 模型推理中...")
        start_time = time.time()
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )
                
                # 获取动作概率
                action_logits = outputs.action_logits
                action_probs = F.softmax(action_logits, dim=-1)
                
        except Exception as e:
            print(f"❌ 模型推理错误: {e}")
            import traceback
            traceback.print_exc()
            return 0
            
        inference_time = time.time() - start_time
        print(f"   推理耗时: {inference_time:.3f}秒")
        
        # 显示每个位置的概率
        print("\n📊 各位置选择概率:")
        probs_np = action_probs.cpu().numpy()[0]
        for i in range(9):
            row, col = divmod(i, 3)
            if board[row, col] == 0:  # 只显示空位的概率
                print(f"   位置{i} ({row+1},{col+1}): {probs_np[i]:.3f} {'⭐' if probs_np[i] > 0.2 else ''}")
            else:
                print(f"   位置{i} ({row+1},{col+1}): 已占用")
        
        # 筛选有效动作
        valid_actions = []
        valid_probs = []
        for i in range(9):
            row, col = divmod(i, 3)
            if board[row, col] == 0:
                valid_actions.append(i)
                valid_probs.append(probs_np[i])
        
        if not valid_actions:
            print("❌ 没有有效动作!")
            return 0
        
        # 选择概率最高的有效动作
        best_idx = np.argmax(valid_probs)
        action = valid_actions[best_idx]
        confidence = valid_probs[best_idx]
        
        row, col = divmod(action, 3)
        print(f"\n🎯 最终决策: 位置{action} ({row+1},{col+1})")
        print(f"   置信度: {confidence:.3f}")
        print(f"   理由: {'高置信度选择' if confidence > 0.3 else '中等置信度选择' if confidence > 0.15 else '低置信度选择，可能在探索'}")
        
        print("=" * 50)
        return action


class TicTacToeGame:
    """井字棋游戏控制器"""
    
    def __init__(self, model_path: str):
        self.vla_player = VLAPlayer(model_path)
        self.env = TicTacToeEnv(render_mode="human", human_player=True, use_internal_ai=False)
        self.game_count = 0
        self.player_wins = 0
        self.vla_wins = 0
        self.draws = 0
        
    def print_instructions(self):
        """打印游戏说明"""
        print("\n" + "=" * 60)
        print("🎮 井字棋 VLA 对战")
        print("=" * 60)
        print("🎯 游戏规则:")
        print("   • 你是 X，VLA模型是 O")
        print("   • 鼠标单击空格放置你的棋子")
        print("   • 先连成三子者获胜")
        print("   • 按 ESC 退出游戏")
        print("   • 关闭窗口结束游戏")
        print("\n🤖 VLA模型会在控制台显示详细的思考过程")
        print("=" * 60)
    
    def print_game_result(self, winner: Optional[Player]):
        """打印游戏结果"""
        print("\n" + "🏁" * 20)
        if winner == Player.X:
            print("🎉 恭喜！你获胜了！")
            self.player_wins += 1
        elif winner == Player.O:
            print("🤖 VLA模型获胜！")
            self.vla_wins += 1
        else:
            print("🤝 平局！")
            self.draws += 1
        
        print(f"\n📊 战绩统计:")
        print(f"   玩家获胜: {self.player_wins}")
        print(f"   VLA获胜: {self.vla_wins}")
        print(f"   平局: {self.draws}")
        print(f"   总局数: {self.game_count}")
        if self.game_count > 0:
            print(f"   玩家胜率: {self.player_wins/self.game_count*100:.1f}%")
        print("🏁" * 20)
    
    def play_game(self):
        """进行一局游戏"""
        self.game_count += 1
        print(f"\n🎮 第 {self.game_count} 局开始!")
        
        try:
            observation, info = self.env.reset()
        except Exception as e:
            print(f"❌ 环境reset错误: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        human_turn_message_shown = False  # 标记是否已显示玩家回合消息
        
        while True:
            # 渲染环境
            try:
                self.env.render()
            except Exception as e:
                print(f"\n❌ 渲染错误: {e}")
                return False
            
            # 获取当前状态
            current_player = self.env.current_player
            board = self.env.board.copy()
            
            if self.env.game_over:
                self.print_game_result(self.env.winner)
                break
            
            if current_player == Player.X:
                # 玩家回合 - 只显示一次提示消息
                if not human_turn_message_shown:
                    print(f"\n👤 轮到你了 (X)，请在游戏窗口中点击选择位置...")
                    human_turn_message_shown = True
                
                try:
                    observation, reward, terminated, truncated, info = self.env.step(None)
                except Exception as e:
                    print(f"❌ 玩家回合step错误: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
                
                # 如果玩家已经行动，重置消息标记
                if reward != 0.0 or current_player != self.env.current_player:
                    human_turn_message_shown = False
                
                # 检查窗口是否被关闭（通过检查pygame状态）
                if not pygame.get_init():
                    print("\n👋 游戏窗口已关闭")
                    return False
                    
            else:
                # VLA模型回合
                human_turn_message_shown = False  # 重置消息标记
                print(f"\n🤖 轮到VLA模型 (O)...")
                
                # 获取模型动作
                action = self.vla_player.predict_action(board, observation)
                
                # 执行动作
                try:
                    observation, reward, terminated, truncated, info = self.env.step(action)
                except Exception as e:
                    print(f"❌ VLA回合step错误: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
                
                row, col = divmod(action, 3)
                print(f"✅ VLA模型选择了位置 ({row+1},{col+1})")
            
            # 短暂暂停让玩家看清状态
            time.sleep(0.1)
        
        # 等待用户输入继续下一局
        print("\n按任意键继续下一局，或关闭窗口退出...")
        input()
        return True
    
    def run(self):
        """运行游戏循环"""
        self.print_instructions()
        
        try:
            while True:
                if not self.play_game():
                    break
        except KeyboardInterrupt:
            print("\n\n👋 游戏被用户中断")
        except Exception as e:
            print(f"\n❌ 游戏出现错误: {e}")
        finally:
            self.env.close()
            print("\n🔚 游戏结束，感谢游玩！")


def main():
    parser = argparse.ArgumentParser(description="井字棋VLA对战")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="运行设备")
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"❌ 模型文件不存在: {args.model_path}")
        return
    
    # 创建并运行游戏
    game = TicTacToeGame(args.model_path)
    game.run()


if __name__ == "__main__":
    main()
