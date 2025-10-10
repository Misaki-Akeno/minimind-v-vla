#!/usr/bin/env python3
"""
基于 PPO 的井字棋在线强化学习训练脚本。

默认使用 MiniMindVLM 作为视觉-语言骨干，仅更新动作头与价值头。
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam
from tqdm import trange
from transformers import AutoTokenizer

from VLA.envs.model_wrapper import MiniMindVLMWithAction
from VLA.envs.tic_tac_toe_env import Difficulty, Player, TicTacToeEnv
from model.model_vlm import VLMConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def board_to_image(board_image: np.ndarray) -> Image.Image:
    return Image.fromarray(board_image)


def board_valid_mask(board: np.ndarray) -> torch.Tensor:
    flat = board.flatten()
    mask = torch.tensor([flat[idx] == Player.EMPTY.value for idx in range(9)], dtype=torch.bool)
    return mask


def build_prompt(tokenizer, image_token: str, board: np.ndarray) -> torch.Tensor:
    mapping = {0: " ", 1: "X", 2: "O"}
    rows = ["[" + ", ".join(mapping[int(board[r, c])] for c in range(3)) + "]" for r in range(3)]
    description = "[" + ", ".join(rows) + "]"
    instruction = (
        "你是一名井字棋智能体。根据图像与棋盘描述判断当前局面，"
        "仅输出 JSON {\"thinking\": string, \"action\": [row, col]}。"
    )
    content = "\n".join([image_token, instruction, f"当前棋盘: {description}。请为 O 选择最优落子位置。"])
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt, return_tensors="pt").input_ids


def mask_logits(logits: torch.Tensor, valid_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-5)
    masked = logits.masked_fill(~valid_mask, -1e9)
    return masked


@dataclass
class PPOConfig:
    rollout_steps: int = 256
    batch_size: int = 64
    minibatch_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    total_updates: int = 500
    temperature: float = 1.0
    seed: int = 42
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    max_seq_len: int = 512
    pad_token_id: int = 0
    save_every: int = 100
    save_path: str = "out/ppo_vlm_ttt.pth"


class PPOBuffer:
    def __init__(self) -> None:
        self.input_ids: List[torch.Tensor] = []
        self.pixel_values: List[torch.Tensor] = []
        self.valid_masks: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def add(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        valid_mask: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
    ) -> None:
        self.input_ids.append(input_ids.cpu())
        self.pixel_values.append(pixel_values.cpu())
        self.valid_masks.append(valid_mask.cpu())
        self.actions.append(action.cpu())
        self.log_probs.append(log_prob.cpu())
        self.values.append(value.cpu())
        self.rewards.append(float(reward))
        self.dones.append(float(done))

    def compute_returns(self, last_value: torch.Tensor, last_done: float, cfg: PPOConfig) -> None:
        if len(self.rewards) == 0:
            self.advantages = torch.tensor([])
            self.returns = torch.tensor([])
            return

        values = torch.stack(self.values).squeeze(-1)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        advantages = torch.zeros_like(rewards)
        lastgaelam = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value.squeeze().cpu()
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            delta = rewards[t] + cfg.gamma * next_value * next_non_terminal - values[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        self.advantages = advantages
        self.returns = returns

    def iter_minibatches(self, batch_size: int, device: torch.device):
        assert self.advantages is not None and self.returns is not None
        idxs = torch.randperm(len(self.input_ids))
        for start in range(0, len(idxs), batch_size):
            mb_idx = idxs[start : start + batch_size]
            yield self._gather_minibatch(mb_idx, device)

    def _gather_minibatch(self, indices: torch.Tensor, device: torch.device):
        input_ids = torch.stack([self.input_ids[i] for i in indices], dim=0).to(device)
        pixel_values = torch.stack([self.pixel_values[i] for i in indices], dim=0).to(device)
        valid_masks = torch.stack([self.valid_masks[i] for i in indices], dim=0).to(device)
        actions = torch.stack([self.actions[i] for i in indices], dim=0).to(device)
        old_log_probs = torch.stack([self.log_probs[i] for i in indices], dim=0).to(device)
        values = torch.stack([self.values[i] for i in indices], dim=0).squeeze(-1).to(device)
        advantages = self.advantages[indices].to(device)
        returns = self.returns[indices].to(device)

        pixel_values = pixel_values.unsqueeze(1) if pixel_values.ndim == 4 else pixel_values
        if pixel_values.ndim == 5 and pixel_values.size(1) != 1:
            pixel_values = pixel_values[:, :1]

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "valid_masks": valid_masks,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "old_values": values,
            "advantages": advantages,
            "returns": returns,
        }


class PPOPolicy(nn.Module):
    def __init__(self, config: VLMConfig, vision_model_path: str = "model/vision_model/clip-vit-base-patch16"):
        super().__init__()
        self.backbone = MiniMindVLMWithAction(config, vision_model_path=vision_model_path)
        self.value_head = nn.Linear(config.hidden_size, 1)

        for param in self.backbone.vlm.parameters():
            param.requires_grad = False
        if self.backbone.vision_encoder is not None:
            for param in self.backbone.vision_encoder.parameters():
                param.requires_grad = False
            self.backbone.vision_encoder.eval()

        self.backbone.vlm.eval()

    @property
    def processor(self):
        return self.backbone.processor

    @property
    def config(self):
        return self.backbone.config

    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor):
        outputs = self.backbone(input_ids=input_ids, pixel_values=pixel_values)
        logits = outputs.action_logits
        last_hidden = outputs.last_hidden_state[:, -1, :]
        value = self.value_head(last_hidden)
        return logits, value


def pad_to_length(input_ids: torch.Tensor, pad_token_id: int, max_length: int) -> torch.Tensor:
    length = input_ids.size(-1)
    if length > max_length:
        return input_ids[..., :max_length]
    if length < max_length:
        pad_shape = list(input_ids.shape[:-1]) + [max_length - length]
        pad = torch.full(pad_shape, pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([input_ids, pad], dim=-1)
    return input_ids


def collect_rollout(
    env: TicTacToeEnv,
    policy: PPOPolicy,
    tokenizer,
    image_token: str,
    device: torch.device,
    cfg: PPOConfig,
) -> Tuple[PPOBuffer, torch.Tensor, float]:
    policy.eval()
    buffer = PPOBuffer()

    obs, info = env.reset()
    last_done = 0.0

    for _ in range(cfg.rollout_steps):
        board = env.board.copy()
        input_ids = build_prompt(tokenizer, image_token, board).to(device)
        input_ids = pad_to_length(input_ids, cfg.pad_token_id, cfg.max_seq_len)
        pixel_values = policy.processor(images=board_to_image(obs), return_tensors="pt").pixel_values.to(device)
        pixel_values = pixel_values.unsqueeze(1)
        valid_mask = board_valid_mask(board).to(device)

        with torch.no_grad():
            logits, value = policy(input_ids=input_ids, pixel_values=pixel_values)
        masked_logits = mask_logits(logits.squeeze(0), valid_mask, cfg.temperature)
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
        done = terminated or truncated

        buffer.add(
            input_ids.squeeze(0),
            pixel_values.squeeze(0),
            valid_mask,
            action,
            log_prob,
            value.squeeze(0),
            reward,
            done,
        )

        obs = next_obs
        last_done = float(done)

        if done:
            obs, info = env.reset()

    if last_done:
        last_value = torch.zeros(1, device=device)
    else:
        board = env.board.copy()
        input_ids = build_prompt(tokenizer, image_token, board).to(device)
        input_ids = pad_to_length(input_ids, cfg.pad_token_id, cfg.max_seq_len)
        pixel_values = policy.processor(images=board_to_image(obs), return_tensors="pt").pixel_values.to(device)
        pixel_values = pixel_values.unsqueeze(1)
        with torch.no_grad():
            _, last_value = policy(input_ids=input_ids, pixel_values=pixel_values)

    buffer.compute_returns(last_value.detach(), last_done, cfg)
    return buffer, last_value.detach(), last_done


def ppo_update(buffer: PPOBuffer, policy: PPOPolicy, optimizer: Adam, cfg: PPOConfig, device: torch.device) -> dict:
    policy.train()
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0

    for _ in range(cfg.minibatch_epochs):
        for batch in buffer.iter_minibatches(cfg.batch_size, device):
            logits, values = policy(batch["input_ids"], batch["pixel_values"])
            masked_logits = mask_logits(logits, batch["valid_masks"], cfg.temperature)
            dist = Categorical(logits=masked_logits)

            new_log_probs = dist.log_prob(batch["actions"])
            entropy = dist.entropy().mean()

            values = values.squeeze(-1)
            advantages = batch["advantages"]
            returns = batch["returns"]
            old_log_probs = batch["old_log_probs"]

            ratios = torch.exp(new_log_probs - old_log_probs)
            clipped = torch.clamp(ratios, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
            policy_loss = -(torch.min(ratios * advantages, clipped * advantages)).mean()
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            updates += 1

    for key in stats:
        stats[key] /= max(updates, 1)
    return stats


def save_checkpoint(policy: PPOPolicy, path: str) -> None:
    payload = {
        "backbone": policy.backbone.state_dict(),
        "value_head": policy.value_head.state_dict(),
    }
    torch.save(payload, path)


def load_initial_weights(policy: PPOPolicy, path: str, device: torch.device) -> None:
    state_dict = torch.load(path, map_location=device)
    if isinstance(state_dict, dict) and "backbone" in state_dict:
        policy.backbone.load_state_dict(state_dict["backbone"], strict=False)
        if "value_head" in state_dict:
            policy.value_head.load_state_dict(state_dict["value_head"])
    else:
        policy.backbone.load_state_dict(state_dict, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO training for MiniMind TicTacToe agent")
    parser.add_argument("--model-path", type=str, default="out/sft_vlm_ttt_768.pth", help="初始化权重路径")
    parser.add_argument("--vision-model", type=str, default="model/vision_model/clip-vit-base-patch16")
    parser.add_argument("--total-updates", type=int, default=None, help="覆盖默认的 total_updates")
    parser.add_argument("--save-path", type=str, default=None, help="训练权重保存路径")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = PPOConfig()
    if args.total_updates is not None:
        cfg.total_updates = args.total_updates
    if args.save_path is not None:
        cfg.save_path = args.save_path
    if args.seed is not None:
        cfg.seed = args.seed

    device = torch.device(cfg.device)
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg.pad_token_id = tokenizer.pad_token_id
    config = VLMConfig(hidden_size=768, num_hidden_layers=16, max_seq_len=512, use_moe=False)
    policy = PPOPolicy(config, vision_model_path=args.vision_model).to(device)

    cfg.max_seq_len = config.max_seq_len

    load_initial_weights(policy, args.model_path, device)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    env = TicTacToeEnv(
        render_mode="rgb_array",
        difficulty=Difficulty.HARD,
        use_internal_ai=False,
    )

    image_token = policy.config.image_special_token

    for update in trange(cfg.total_updates, desc="PPO"):
        rollout_buffer, last_value, last_done = collect_rollout(env, policy, tokenizer, image_token, device, cfg)
        if rollout_buffer.advantages is None or len(rollout_buffer.advantages) == 0:
            continue
        stats = ppo_update(rollout_buffer, policy, optimizer, cfg, device)

        if (update + 1) % 10 == 0:
            avg_reward = np.mean(rollout_buffer.rewards) if rollout_buffer.rewards else 0.0
            print(
                f"[Update {update + 1}] reward={avg_reward:.3f} "
                f"policy_loss={stats['policy_loss']:.4f} "
                f"value_loss={stats['value_loss']:.4f} "
                f"entropy={stats['entropy']:.4f}"
            )

        if (update + 1) % cfg.save_every == 0:
            save_checkpoint(policy, cfg.save_path)
            print(f"保存模型到 {cfg.save_path}")

    save_checkpoint(policy, cfg.save_path)
    print(f"训练完成，最终模型已保存到 {cfg.save_path}")


if __name__ == "__main__":
    main()
