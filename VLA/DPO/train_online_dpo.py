import argparse
import math
import os
import time
import json
import random
from contextlib import nullcontext
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
from PIL import Image

from model.model_vlm import VLMConfig
from VLA.envs.model_wrapper import MiniMindVLMWithAction
from VLA.SFT.evaluate import evaluate_model


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class OnlineDPODataset(Dataset):
    """
    在线DPO数据集，在训练过程中动态生成偏好对
    """
    
    def __init__(
        self,
        sft_jsonl_path: str,
        policy_model: MiniMindVLMWithAction,
        reference_model: MiniMindVLMWithAction,
        tokenizer,
        preprocess=None,
        max_length: int = 512,
        image_special_token: str = '@' * 196,
        num_samples_per_prompt: int = 4,
        temperature: float = 1.0,
        device: str = "cuda:0",
    ):
        super().__init__()
        
        # 加载基础数据
        self.sft_samples = []
        with open(sft_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.sft_samples.append(json.loads(line))
        
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.max_length = max_length
        self.image_token = image_special_token
        self.num_samples_per_prompt = num_samples_per_prompt
        self.temperature = temperature
        self.device = device
        
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids
        
        # 缓存生成的偏好对
        self.cached_pairs = {}
        self.cache_update_interval = 100  # 每100步更新一次缓存
        self.last_cache_update = 0
    
    def __len__(self):
        return len(self.sft_samples)
    
    def sample_responses(
        self, 
        instruction: str, 
        user_input: str, 
        image_path: str,
        model: MiniMindVLMWithAction
    ) -> List[str]:
        """从给定模型采样回答"""
        content_parts = ["<image>", instruction.strip() if instruction else ""]
        if user_input:
            content_parts.append(user_input.strip())
        content = "\n".join([p for p in content_parts if p])
        
        messages = [
            {"role": "user", "content": content.replace('<image>', self.image_token)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # 加载图像
        image = Image.open(image_path)
        image_tensor = model.vlm.image2tensor(image, model.processor)
        pixel_values = torch.stack([image_tensor], dim=0).to(self.device)
        
        # tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        samples = []
        model.eval()
        with torch.no_grad():
            for _ in range(self.num_samples_per_prompt):
                generated_ids = input_ids.clone()
                
                for _ in range(100):  # max_new_tokens
                    outputs = model(
                        input_ids=generated_ids,
                        pixel_values=pixel_values,
                    )
                    
                    next_token_logits = outputs.logits[:, -1, :] / self.temperature
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                
                generated_text = self.tokenizer.decode(
                    generated_ids[0][input_ids.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                samples.append(generated_text)
        
        return samples
    
    def evaluate_response_quality(self, response: str) -> float:
        """评估回答质量的简单启发式方法"""
        try:
            obj = json.loads(response)
            if not isinstance(obj, dict) or 'action' not in obj:
                return 0.1  # 格式不正确
            
            action = obj['action']
            if not isinstance(action, list) or len(action) != 2:
                return 0.2
            
            row, col = action
            if not (0 <= row <= 2 and 0 <= col <= 2):
                return 0.3  # 动作超出范围
            
            # 简单的位置价值评估
            if row == 1 and col == 1:  # 中心
                base_score = 0.8
            elif (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]:  # 角落
                base_score = 0.7
            else:  # 边缘
                base_score = 0.6
            
            # 考虑thinking的质量
            thinking = obj.get('thinking', '')
            if len(thinking) > 10:  # 有思考过程
                base_score += 0.1
            
            return min(1.0, base_score + random.uniform(-0.05, 0.05))
        
        except Exception:
            return 0.1
    
    def create_preference_pairs(
        self, 
        policy_samples: List[str], 
        reference_samples: List[str]
    ) -> List[Tuple[str, str]]:
        """创建偏好对"""
        all_samples = policy_samples + reference_samples
        
        # 评估所有样本
        scored_samples = []
        for sample in all_samples:
            score = self.evaluate_response_quality(sample)
            scored_samples.append((sample, score))
        
        # 排序
        scored_samples.sort(key=lambda x: x[1], reverse=True)
        
        # 创建偏好对
        pairs = []
        n = len(scored_samples)
        for i in range(n):
            for j in range(i + 1, n):
                chosen, chosen_score = scored_samples[i]
                rejected, rejected_score = scored_samples[j]
                
                # 只有当分数差足够大时才创建偏好对
                if chosen_score - rejected_score > 0.1:
                    pairs.append((chosen, rejected))
        
        return pairs[:3]  # 限制偏好对数量
    
    def _create_prompt(self, instruction: str, user_input: str) -> str:
        content_parts = ["<image>", instruction.strip() if instruction else ""]
        if user_input:
            content_parts.append(user_input.strip())
        content = "\n".join([p for p in content_parts if p])
        messages = [
            {"role": "user", "content": content.replace('<image>', self.image_token)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt
    
    def _generate_loss_mask(self, input_ids: List[int]) -> List[int]:
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
    
    def _build_pair(self, prompt: str, answer: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text = prompt + answer
        tokenized = self.tokenizer(text)
        ids = tokenized.input_ids[:self.max_length]
        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids = ids + [self.tokenizer.pad_token_id] * pad_len
        loss_mask_list = self._generate_loss_mask(ids)
        X = torch.tensor(ids[:-1], dtype=torch.long)
        Y = torch.tensor(ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask_list[1:], dtype=torch.float)
        return X, Y, loss_mask
    
    def __getitem__(self, index: int):
        sample = self.sft_samples[index]
        instruction = sample.get('instruction', '')
        user_input = sample.get('input', '')
        image_path = sample.get('image_path', '')
        
        # 检查缓存是否需要更新
        cache_key = f"{index}_{self.last_cache_update // self.cache_update_interval}"
        
        if cache_key not in self.cached_pairs:
            # 生成新的偏好对
            try:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                
                # 从policy模型和reference模型采样
                policy_samples = self.sample_responses(
                    instruction, user_input, image_path, self.policy_model
                )
                reference_samples = self.sample_responses(
                    instruction, user_input, image_path, self.reference_model
                )
                
                # 创建偏好对
                pairs = self.create_preference_pairs(policy_samples, reference_samples)
                
                if pairs:
                    self.cached_pairs[cache_key] = {
                        'pairs': pairs,
                        'image_path': image_path,
                        'prompt': self._create_prompt(instruction, user_input)
                    }
                else:
                    # 如果没有生成偏好对，使用随机的fallback
                    all_samples = policy_samples + reference_samples
                    if len(all_samples) >= 2:
                        chosen = random.choice(all_samples)
                        rejected = random.choice([s for s in all_samples if s != chosen])
                        self.cached_pairs[cache_key] = {
                            'pairs': [(chosen, rejected)],
                            'image_path': image_path,
                            'prompt': self._create_prompt(instruction, user_input)
                        }
            
            except Exception as e:
                print(f"Error generating preference pair for index {index}: {e}")
                # 使用简单的fallback
                default_chosen = '{"thinking": "我选择中心位置", "action": [1, 1]}'
                default_rejected = '{"thinking": "随机选择", "action": [0, 0]}'
                self.cached_pairs[cache_key] = {
                    'pairs': [(default_chosen, default_rejected)],
                    'image_path': image_path if os.path.exists(image_path) else "VLA/data/images/train/ttt_train_O_000000.png",
                    'prompt': self._create_prompt(instruction, user_input)
                }
        
        # 从缓存中获取数据
        cached_data = self.cached_pairs[cache_key]
        chosen, rejected = random.choice(cached_data['pairs'])
        prompt = cached_data['prompt']
        resolved_image_path = cached_data['image_path']
        
        # 构建训练数据
        Xc, Yc, Mc = self._build_pair(prompt, chosen)
        Xr, Yr, Mr = self._build_pair(prompt, rejected)
        
        # 加载图像
        image = Image.open(resolved_image_path)
        image_tensor = self.policy_model.vlm.image2tensor(image, self.preprocess)
        pixel_values = torch.stack([image_tensor], dim=0)
        
        return {
            'input_ids_chosen': Xc,
            'labels_chosen': Yc,
            'loss_mask_chosen': Mc,
            'input_ids_rejected': Xr,
            'labels_rejected': Yr,
            'loss_mask_rejected': Mr,
            'pixel_values': pixel_values,
        }
    
    def update_cache_step(self):
        """更新缓存步数，用于控制何时重新生成偏好对"""
        self.last_cache_update += 1
        
        # 每隔一定步数清空缓存，强制重新生成
        if self.last_cache_update % (self.cache_update_interval * 5) == 0:
            self.cached_pairs.clear()
            print(f"Cache cleared at step {self.last_cache_update}")


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    lp = torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return lp


def dpo_loss(
    model,
    batch,
    beta: float = 0.1,
    reference_model=None,
    device: str = "cuda:0",
):
    """计算DPO损失"""
    # Chosen forward
    out_c = model(batch['input_ids_chosen'].to(device), pixel_values=batch['pixel_values'].to(device))
    lp_c = log_probs_from_logits(out_c.logits, batch['labels_chosen'].to(device))
    lp_c = (lp_c * batch['loss_mask_chosen'].to(device)).sum(dim=-1) / (batch['loss_mask_chosen'].to(device).sum(dim=-1) + 1e-8)

    # Rejected forward
    out_r = model(batch['input_ids_rejected'].to(device), pixel_values=batch['pixel_values'].to(device))
    lp_r = log_probs_from_logits(out_r.logits, batch['labels_rejected'].to(device))
    lp_r = (lp_r * batch['loss_mask_rejected'].to(device)).sum(dim=-1) / (batch['loss_mask_rejected'].to(device).sum(dim=-1) + 1e-8)

    if reference_model is not None:
        with torch.no_grad():
            ref_c = reference_model(batch['input_ids_chosen'].to(device), pixel_values=batch['pixel_values'].to(device))
            ref_lp_c = log_probs_from_logits(ref_c.logits, batch['labels_chosen'].to(device))
            ref_lp_c = (ref_lp_c * batch['loss_mask_chosen'].to(device)).sum(dim=-1) / (batch['loss_mask_chosen'].to(device).sum(dim=-1) + 1e-8)

            ref_r = reference_model(batch['input_ids_rejected'].to(device), pixel_values=batch['pixel_values'].to(device))
            ref_lp_r = log_probs_from_logits(ref_r.logits, batch['labels_rejected'].to(device))
            ref_lp_r = (ref_lp_r * batch['loss_mask_rejected'].to(device)).sum(dim=-1) / (batch['loss_mask_rejected'].to(device).sum(dim=-1) + 1e-8)
        adv = (lp_c - ref_lp_c) - (lp_r - ref_lp_r)
    else:
        adv = lp_c - lp_r

    loss = -torch.nn.functional.logsigmoid(beta * adv).mean()
    return loss


def main():
    parser = argparse.ArgumentParser(description="Online DPO Training for MiniMind-V")
    parser.add_argument("--sft_jsonl", type=str, default="VLA/data/ttt_train_alpaca.jsonl")
    parser.add_argument("--val_jsonl", type=str, default="VLA/data/ttt_val_alpaca.jsonl")
    parser.add_argument("--out_dir", type=str, default="VLA/models")
    parser.add_argument("--policy_ckpt", type=str, default="MiniMind2-V/pytorch_model.bin")
    parser.add_argument("--reference_ckpt", type=str, default="MiniMind2-V/pytorch_model.bin")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=100)
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
    
    # 模型配置
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
    
    # 加载policy模型
    policy_model = MiniMindVLMWithAction(config)
    checkpoint = torch.load(args.policy_ckpt, map_location='cpu')
    policy_model.vlm.load_state_dict(checkpoint, strict=False)
    policy_model.to(device)
    
    # 加载reference模型
    reference_model = MiniMindVLMWithAction(config)
    ref_checkpoint = torch.load(args.reference_ckpt, map_location='cpu')
    reference_model.vlm.load_state_dict(ref_checkpoint, strict=False)
    reference_model.to(device)
    reference_model.eval()  # reference模型始终处于eval模式
    
    # 创建在线数据集
    train_dataset = OnlineDPODataset(
        sft_jsonl_path=args.sft_jsonl,
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        preprocess=policy_model.processor,
        num_samples_per_prompt=args.num_samples,
        temperature=args.temperature,
        device=device,
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    # 优化器
    optimizer = optim.AdamW(policy_model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    
    # 训练循环
    policy_model.train()
    total_steps = args.epochs * len(train_loader)
    step = 0
    
    print(f"Starting Online DPO training...")
    print(f"Total steps: {total_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Beta: {args.beta}")
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 更新数据集的缓存步数
            train_dataset.update_cache_step()
            
            optimizer.zero_grad()
            
            # 计算DPO损失
            loss = dpo_loss(
                model=policy_model,
                batch=batch,
                beta=args.beta,
                reference_model=reference_model,
                device=device,
            )
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            
            # 学习率调度
            current_lr = get_lr(step, total_steps, args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            optimizer.step()
            
            epoch_loss += loss.item()
            step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.2e}',
                'Step': step
            })
            
            # 定期保存
            if step % args.save_interval == 0:
                save_path = os.path.join(args.out_dir, f"online_dpo_vlm_step_{step}.pth")
                torch.save(policy_model.state_dict(), save_path)
                print(f"\nModel saved to {save_path}")
            
            # 定期评估
            if step % args.eval_interval == 0:
                try:
                    print(f"\nEvaluating at step {step}...")
                    acc = evaluate_model(
                        policy_model, 
                        args.val_jsonl, 
                        device=device, 
                        max_samples=50
                    )
                    print(f"Validation accuracy: {acc:.3f}")
                except Exception as e:
                    print(f"Evaluation failed: {e}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    # 保存最终模型
    final_save_path = os.path.join(args.out_dir, "online_dpo_vlm_final.pth")
    torch.save(policy_model.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")


if __name__ == "__main__":
    main()
