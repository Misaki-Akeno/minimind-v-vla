import argparse
import json
import os
import random
from typing import Dict, Any, Tuple, List
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model.model_vlm import VLMConfig
from VLA.envs.model_wrapper import MiniMindVLMWithAction


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
    """
    从模型采样多个回答
    """
    model.eval()
    
    # 构造prompt
    content_parts = ["<image>", instruction.strip() if instruction else ""]
    if user_input:
        content_parts.append(user_input.strip())
    content = "\n".join([p for p in content_parts if p])
    
    # 替换image token (假设使用196个@符号)
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
    pixel_values = torch.stack([image_tensor], dim=0).to(device)  # [1,1,3,H,W]
    
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
                
                # 检查是否生成结束token
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            # 提取生成的文本
            generated_text = tokenizer.decode(
                generated_ids[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            ).strip()
            samples.append(generated_text)
    
    return samples


def parse_action_from_output(output_text: str) -> Tuple[int, int]:
    """解析动作，返回(row, col)"""
    try:
        obj = json.loads(output_text)
        if isinstance(obj, dict) and 'action' in obj:
            r, c = int(obj['action'][0]), int(obj['action'][1])
            r = max(0, min(2, r))
            c = max(0, min(2, c))
            return r, c
    except Exception:
        pass
    return 1, 1  # fallback to center


def evaluate_action_quality(action_row: int, action_col: int, board_state: str = None) -> float:
    """
    简单的动作质量评估函数
    这里可以根据井字棋规则来评估动作的好坏
    返回0-1之间的分数，越高越好
    """
    # 这是一个简化的评估，实际应该基于棋盘状态
    # 中心位置通常比较好
    center_bonus = 0.3 if (action_row == 1 and action_col == 1) else 0
    # 角落位置也不错
    corner_bonus = 0.2 if (action_row, action_col) in [(0,0), (0,2), (2,0), (2,2)] else 0
    
    base_score = 0.5  # 基础分数
    return min(1.0, base_score + center_bonus + corner_bonus + random.uniform(-0.1, 0.1))


def create_preference_pairs_from_samples(samples: List[str], k: int = 2) -> List[Tuple[str, str]]:
    """
    从采样结果中创建偏好对
    选择质量最好的k个作为chosen，质量最差的k个作为rejected
    """
    if len(samples) < 2:
        return []
    
    # 评估每个样本的质量
    scored_samples = []
    for sample in samples:
        try:
            r, c = parse_action_from_output(sample)
            quality = evaluate_action_quality(r, c)
            scored_samples.append((sample, quality))
        except:
            scored_samples.append((sample, 0.0))  # 解析失败的样本质量最低
    
    # 按质量排序
    scored_samples.sort(key=lambda x: x[1], reverse=True)
    
    # 生成偏好对
    pairs = []
    good_samples = scored_samples[:k]  # 最好的k个
    bad_samples = scored_samples[-k:]  # 最差的k个
    
    for good_sample, _ in good_samples:
        for bad_sample, _ in bad_samples:
            if good_sample != bad_sample:
                pairs.append((good_sample, bad_sample))
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate DPO dataset by sampling from VLA model")
    parser.add_argument("--model_path", type=str, required=True, help="VLA模型检查点路径")
    parser.add_argument("--sft_jsonl", type=str, required=True, help="SFT数据集路径（用于获取prompt和图像）")
    parser.add_argument("--out_jsonl", type=str, required=True, help="输出DPO数据集路径")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=4, help="每个prompt采样的回答数量")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top_k", type=int, default=50, help="top-k采样")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样")
    parser.add_argument("--max_samples", type=int, default=1000, help="最大处理样本数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = args.device
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
    
    # 加载模型
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
        # 假设是原始的pytorch_model.bin
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
    
    # 限制处理数量
    if args.max_samples > 0:
        sft_samples = sft_samples[:args.max_samples]
    
    print(f"Processing {len(sft_samples)} samples...")
    
    dpo_pairs = []
    for i, sample in enumerate(sft_samples):
        if i % 50 == 0:
            print(f"Processing sample {i}/{len(sft_samples)}")
        
        instruction = sample.get('instruction', '')
        user_input = sample.get('input', '')
        image_path = sample.get('image_path', '')
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}, skipping...")
            continue
        
        try:
            # 从模型采样多个回答
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
            
            # 创建偏好对
            pairs = create_preference_pairs_from_samples(samples)
            
            # 为每个偏好对创建数据项
            for chosen, rejected in pairs:
                dpo_item = {
                    "instruction": instruction,
                    "input": user_input,
                    "image_path": image_path,
                    "chosen": chosen,
                    "rejected": rejected,
                }
                dpo_pairs.append(dpo_item)
        
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # 写入输出文件
    with open(args.out_jsonl, 'w', encoding='utf-8') as f:
        for item in dpo_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(dpo_pairs)} DPO pairs -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
