#!/usr/bin/env python3
"""
独立的井字棋模型测试脚本

使用方法:
    python test_ttt_model.py --model_path /path/to/model.pth --test_data /path/to/test.jsonl
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_vlm import VLMConfig
from VLA.envs.model_wrapper import MiniMindVLMWithAction
from VLA.envs.ttt_dataset import TicTacToeVLMDataset
from transformers import AutoTokenizer

# 固定配置（与训练时保持一致）
HIDDEN_SIZE = 768
NUM_LAYERS = 16
MAX_SEQ_LEN = 512
USE_MOE = False


class TicTacToeTester:
    """井字棋模型测试器"""
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # 初始化模型配置
        self.model_config = VLMConfig(
            hidden_size=HIDDEN_SIZE, 
            num_hidden_layers=NUM_LAYERS,
            max_seq_len=MAX_SEQ_LEN, 
            use_moe=USE_MOE
        )
        
        # 加载模型和tokenizer
        self.model, self.tokenizer, self.preprocess = self._load_model()
        
    def _load_model(self):
        """加载模型和相关组件"""
        print(f"从 {self.model_path} 加载模型...")
        
        # 初始化tokenizer
        tokenizer = AutoTokenizer.from_pretrained('model')
        
        # 初始化模型
        model = MiniMindVLMWithAction(
            self.model_config, 
            vision_model_path="model/vision_model/clip-vit-base-patch16"
        )
        
        # 加载权重
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        
        print(f"模型加载完成，设备: {self.device}")
        print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万")
        
        _, preprocess = model.vision_encoder, model.processor
        return model, tokenizer, preprocess
    
    def test_on_dataset(self, test_data_path: str, 
                       batch_size: int = 32, max_samples: int = None) -> Dict[str, Any]:
        """在测试集上评估模型"""
        print(f"开始在测试集 {test_data_path} 上评估...")
        
        # 创建测试数据集
        test_ds = TicTacToeVLMDataset(
            test_data_path,
            self.tokenizer,
            preprocess=self.preprocess,
            image_special_token=self.model_config.image_special_token,
            max_length=MAX_SEQ_LEN,
        )
        
        if max_samples:
            # 限制测试样本数量
            test_ds.samples = test_ds.samples[:max_samples]
        
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=0,  # 测试时用单线程，避免问题
        )
        
        print(f"测试集大小: {len(test_ds)}")
        
        # 评估指标
        total_samples = 0
        correct_actions = 0
        total_lm_loss = 0.0
        total_action_loss = 0.0
        action_predictions = []
        action_targets = []
        
        # 详细结果记录
        detailed_results = []
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="测试中")):
                X = batch['input_ids'].to(self.device)
                Y = batch['labels'].to(self.device)
                loss_mask = batch['loss_mask'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)
                action = batch['action'].to(self.device)
                
                # 前向传播
                res = self.model(X, pixel_values=pixel_values)
                
                # 语言建模损失
                loss_lm_tok = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                loss_lm = (loss_lm_tok * loss_mask).sum() / (loss_mask.sum() + 1e-8)
                
                # 动作预测
                last_hidden = res.last_hidden_state[:, -1, :]  # [B, H]
                action_logits = self.model.action_head(last_hidden)  # [B, 9]
                loss_action = F.cross_entropy(action_logits, action)
                
                # 预测动作
                pred_actions = action_logits.argmax(dim=-1)
                
                # 统计
                batch_correct = (pred_actions == action).sum().item()
                correct_actions += batch_correct
                total_samples += action.size(0)
                
                total_lm_loss += loss_lm.item()
                total_action_loss += loss_action.item()
                
                # 保存预测结果用于详细分析
                action_predictions.extend(pred_actions.cpu().numpy())
                action_targets.extend(action.cpu().numpy())
                
                # 记录详细结果（可选，用于错误分析）
                for i in range(action.size(0)):
                    sample_idx = batch_idx * batch_size + i
                    if sample_idx < len(test_ds.samples):
                        detailed_results.append({
                            'sample_idx': sample_idx,
                            'predicted_action': pred_actions[i].item(),
                            'target_action': action[i].item(),
                            'correct': pred_actions[i].item() == action[i].item(),
                            'confidence': F.softmax(action_logits[i], dim=0).max().item(),
                        })
        
        # 计算最终指标
        test_time = time.time() - start_time
        action_accuracy = correct_actions / total_samples
        avg_lm_loss = total_lm_loss / len(test_loader)
        avg_action_loss = total_action_loss / len(test_loader)
        
        # 计算每个动作类别的准确率
        action_predictions = np.array(action_predictions)
        action_targets = np.array(action_targets)
        
        per_action_stats = {}
        for action_idx in range(9):
            mask = action_targets == action_idx
            if mask.sum() > 0:
                acc = (action_predictions[mask] == action_idx).mean()
                count = mask.sum()
                per_action_stats[action_idx] = {
                    'accuracy': float(acc),
                    'count': int(count),
                    'position': f"({action_idx//3}, {action_idx%3})"
                }
        
        results = {
            'overall_metrics': {
                'action_accuracy': action_accuracy,
                'avg_lm_loss': avg_lm_loss,
                'avg_action_loss': avg_action_loss,
                'total_samples': total_samples,
                'correct_actions': correct_actions,
                'test_time': test_time,
                'samples_per_second': total_samples / test_time,
            },
            'per_action_stats': per_action_stats,
            'detailed_results': detailed_results,
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """保存测试结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"测试结果已保存到: {output_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """打印测试摘要"""
        overall = results['overall_metrics']
        per_action = results['per_action_stats']
        
        print("\n" + "="*60)
        print("测试结果摘要")
        print("="*60)
        print(f"总样本数: {overall['total_samples']}")
        print(f"动作准确率: {overall['action_accuracy']:.4f} ({overall['correct_actions']}/{overall['total_samples']})")
        print(f"语言建模损失: {overall['avg_lm_loss']:.4f}")
        print(f"动作损失: {overall['avg_action_loss']:.4f}")
        print(f"测试用时: {overall['test_time']:.2f}秒")
        print(f"处理速度: {overall['samples_per_second']:.2f} 样本/秒")
        
        print("\n各位置动作准确率:")
        print("-"*40)
        for action_idx in range(9):
            if action_idx in per_action:
                stats = per_action[action_idx]
                print(f"位置 {stats['position']}: {stats['accuracy']:.4f} ({stats['count']} 样本)")
            else:
                row, col = action_idx // 3, action_idx % 3
                print(f"位置 ({row}, {col}): 无测试样本")
        
        # 错误分析
        detailed = results['detailed_results']
        errors = [r for r in detailed if not r['correct']]
        if errors:
            print(f"\n错误样本数: {len(errors)}")
            print("低置信度错误 (置信度 < 0.7):")
            low_conf_errors = [e for e in errors if e['confidence'] < 0.7]
            for error in low_conf_errors[:5]:  # 只显示前5个
                pred_pos = (error['predicted_action']//3, error['predicted_action']%3)
                target_pos = (error['target_action']//3, error['target_action']%3)
                print(f"  样本 {error['sample_idx']}: 预测{pred_pos} vs 目标{target_pos}, 置信度={error['confidence']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="井字棋模型测试")
    parser.add_argument("--model_path", type=str,default='VLA/models/sft_vlm_ttt_768.pth', help="模型权重路径")
    parser.add_argument("--test_data", type=str, 
                       default="VLA/data/ttt_test_alpaca.jsonl", 
                       help="测试数据路径")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--device", type=str, 
                       default="cuda:0" if torch.cuda.is_available() else "cpu", 
                       help="设备")
    parser.add_argument("--max_samples", type=int, default=None, 
                       help="最大测试样本数，None表示测试全部")
    parser.add_argument("--output_dir", type=str, default="VLA/result", 
                       help="结果输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化测试器
    tester = TicTacToeTester(args.model_path, args.device)
    
    # 运行测试
    results = tester.test_on_dataset(
        test_data_path=args.test_data,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # 保存结果
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    output_path = os.path.join(args.output_dir, f"test_results_{model_name}.json")
    tester.save_results(results, output_path)
    
    # 打印摘要
    tester.print_summary(results)


if __name__ == "__main__":
    main()
