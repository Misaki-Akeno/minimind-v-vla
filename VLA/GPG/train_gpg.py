from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import optim
from torch.distributions import Categorical
from tqdm import trange
from transformers import AutoTokenizer

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

THIS_DIR = Path(__file__).resolve()
VLA_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR.parents[2]
for candidate in (VLA_ROOT, PROJECT_ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.append(path_str)

from model.model_vlm import VLMConfig
from VLA.envs.model_wrapper import MiniMindVLMWithAction
from VLA.envs.tic_tac_toe_env import Difficulty, Player, TicTacToeEnv


CKPT_DEFAULT = "VLA/models/sft_vlm_ttt_768.pth"
HIDDEN_SIZE = 768
NUM_LAYERS = 16
MAX_SEQ_LEN = 512
USE_MOE = False


@dataclass
class PromptState:
    board: np.ndarray
    image: np.ndarray
    valid_actions: List[int]


def board_tokens(board: np.ndarray) -> List[List[str]]:
    mapping = {0: " ", 1: "X", 2: "O"}
    return [[mapping[int(board[r, c])] for c in range(3)] for r in range(3)]


def format_board_text(board: np.ndarray) -> str:
    rows = ["[" + ", ".join(row) + "]" for row in board_tokens(board)]
    return "[" + ", ".join(rows) + "]"


def _check_winner_local(board: np.ndarray) -> Player | None:
    for row in range(3):
        if board[row, 0] == board[row, 1] == board[row, 2] != Player.EMPTY.value:
            return Player(int(board[row, 0]))
    for col in range(3):
        if board[0, col] == board[1, col] == board[2, col] != Player.EMPTY.value:
            return Player(int(board[0, col]))
    if board[0, 0] == board[1, 1] == board[2, 2] != Player.EMPTY.value:
        return Player(int(board[0, 0]))
    if board[0, 2] == board[1, 1] == board[2, 0] != Player.EMPTY.value:
        return Player(int(board[0, 2]))
    return None


def _board_full(board: np.ndarray) -> bool:
    return np.all(board != Player.EMPTY.value)


def _minimax(board: np.ndarray, current: Player, depth: int) -> float:
    winner = _check_winner_local(board)
    if winner == Player.O:
        return 1.0 - depth * 0.1
    if winner == Player.X:
        return -1.0 + depth * 0.1
    if _board_full(board):
        return 0.0

    if current == Player.O:
        best = -math.inf
        for idx in range(9):
            r, c = divmod(idx, 3)
            if board[r, c] != Player.EMPTY.value:
                continue
            board[r, c] = Player.O.value
            score = _minimax(board, Player.X, depth + 1)
            board[r, c] = Player.EMPTY.value
            best = max(best, score)
        return best

    best = math.inf
    for idx in range(9):
        r, c = divmod(idx, 3)
        if board[r, c] != Player.EMPTY.value:
            continue
        board[r, c] = Player.X.value
        score = _minimax(board, Player.O, depth + 1)
        board[r, c] = Player.EMPTY.value
        best = min(best, score)
    return best


def evaluate_action(board: np.ndarray, action: int) -> float:
    r, c = divmod(action, 3)
    if board[r, c] != Player.EMPTY.value:
        return -1.5  # strong penalty for invalid move
    board[r, c] = Player.O.value
    reward = _minimax(board, Player.X, 0)
    board[r, c] = Player.EMPTY.value
    return reward


def random_state_for_o(
    env: TicTacToeEnv,
    rng: random.Random,
    max_pairs: int,
    tries: int,
) -> PromptState:
    for _ in range(tries):
        env.board[:, :] = Player.EMPTY.value
        env.winner = None
        env.game_over = False
        env.current_player = Player.X

        rounds = rng.randint(0, max_pairs)
        ok = True

        for _ in range(rounds):
            valid = [a for a in range(9) if env._is_valid_action(a)]
            if not valid:
                ok = False
                break
            act_x = rng.choice(valid)
            rx, cx = divmod(act_x, 3)
            env.board[rx, cx] = Player.X.value
            if env._check_winner():
                env.winner = None
                ok = False
                break

            valid = [a for a in range(9) if env._is_valid_action(a)]
            if not valid:
                ok = False
                break
            act_o = rng.choice(valid)
            ro, co = divmod(act_o, 3)
            env.board[ro, co] = Player.O.value
            if env._check_winner():
                env.winner = None
                ok = False
                break

        if not ok:
            continue

        valid = [a for a in range(9) if env._is_valid_action(a)]
        if not valid:
            continue
        act_x = rng.choice(valid)
        rx, cx = divmod(act_x, 3)
        env.board[rx, cx] = Player.X.value
        if env._check_winner():
            env.winner = None
            continue

        x_count = int((env.board == Player.X.value).sum())
        o_count = int((env.board == Player.O.value).sum())
        if x_count != o_count + 1:
            continue
        if env._check_winner():
            env.winner = None
            continue

        env.current_player = Player.O
        image = env._get_observation()
        valid_actions = [a for a in range(9) if env._is_valid_action(a)]
        if not valid_actions:
            continue
        return PromptState(board=env.board.copy(), image=image, valid_actions=valid_actions)

    raise RuntimeError("Failed to sample a valid state for O to play")


def pixel_values_from_image(model: MiniMindVLMWithAction, image: np.ndarray) -> torch.Tensor:
    pixel_values = model.processor(images=image, return_tensors="pt").pixel_values  # [1, C, H, W]
    return pixel_values.unsqueeze(1)  # [1, 1, C, H, W]


def build_prompt(tokenizer, image_token: str, board: np.ndarray) -> torch.Tensor:
    board_text = format_board_text(board)
    instruction = (
        "你是一名井字棋智能体。根据图像与棋盘描述判断最优落子，"
        "返回 JSON {\"thinking\": string, \"action\": [row, col]}。"
    )
    description = f"当前棋盘: {board_text}。请帮我选择最优的 O 落子位置。"
    content = "\n".join(["<image>", instruction, description])
    messages = [{"role": "user", "content": content.replace("<image>", image_token)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(prompt, return_tensors="pt")
    return tokens.input_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Group Policy Gradient fine-tuning for MiniMind TicTacToe")
    parser.add_argument("--steps", type=int, default=1000, help="优化步数")
    parser.add_argument("--batch-size", type=int, default=4, help="每步采样的 prompt 数量")
    parser.add_argument("--group-size", type=int, default=4, help="每个 prompt 采样的响应数量")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epsilon", type=float, default=1e-6, help="奖励归一化平滑项")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--board-size", type=int, default=300)
    parser.add_argument("--max-pairs", type=int, default=3)
    parser.add_argument("--tries", type=int, default=100)
    parser.add_argument("--save-path", type=str, default="VLA/models/gpg_vlm_ttt.pth")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument(
        "--log-path",
        type=str,
        default="VLA/GPG/logs/gpg_train_log.jsonl",
        help="训练过程中写入指标的 JSONL 日志文件；为空则不落库",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained('model', trust_remote_code=True)
    config = VLMConfig(
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        use_moe=USE_MOE,
    )
    model = MiniMindVLMWithAction(config, vision_model_path="model/vision_model/clip-vit-base-patch16")

    if not os.path.exists(CKPT_DEFAULT):
        raise FileNotFoundError(f"Missing pretrain checkpoint at {CKPT_DEFAULT}")
    state = torch.load(CKPT_DEFAULT, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    env = TicTacToeEnv(
        render_mode="rgb_array",
        difficulty=Difficulty.HARD,
        board_size=args.board_size,
        use_internal_ai=False,
    )

    image_token = config.image_special_token

    log_file = None
    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")

    progress = trange(1, args.steps + 1, desc="GPG Training")
    for step in progress:
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        batch_rewards = []
        batch_advantages = []
        valid_sample_count = 0
        skipped_groups = 0

        for _ in range(args.batch_size):
            state = random_state_for_o(env, rng, args.max_pairs, args.tries)
            input_ids = build_prompt(tokenizer, image_token, state.board).to(device)
            pixel_values = pixel_values_from_image(model, state.image).to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values)
            logits = outputs.action_logits.squeeze(0)  # [9]
            logits = logits / max(args.temperature, 1e-5)
            dist = Categorical(logits=logits)

            rewards = []
            log_probs = []

            for _ in range(args.group_size):
                action = dist.sample()
                log_prob = dist.log_prob(action)
                reward = evaluate_action(state.board.copy(), int(action.item()))
                rewards.append(reward)
                log_probs.append(log_prob)

            rewards_tensor = torch.tensor(rewards, device=device)
            mean_reward = rewards_tensor.mean()
            std_reward = rewards_tensor.std(unbiased=False)

            if torch.isclose(std_reward, torch.tensor(0.0, device=device)):
                skipped_groups += 1
                continue

            advantages = (rewards_tensor - mean_reward) / (std_reward + args.epsilon)

            for advantage, log_prob in zip(advantages, log_probs):
                total_loss = total_loss - advantage.detach() * log_prob
            valid_sample_count += len(advantages)

            batch_rewards.append(mean_reward.item())
            batch_advantages.extend(advantages.tolist())

        if valid_sample_count == 0:
            progress.set_postfix(skipped=skipped_groups)
            continue

        total_loss = total_loss / valid_sample_count
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_interval == 0:
            avg_reward = float(sum(batch_rewards) / len(batch_rewards)) if batch_rewards else 0.0
            avg_adv = float(sum(batch_advantages) / len(batch_advantages)) if batch_advantages else 0.0
            metrics = {
                "step": step,
                "loss": float(total_loss.item()),
                "avg_group_reward": avg_reward,
                "avg_advantage": avg_adv,
                "skipped_groups": skipped_groups,
            }
            if log_file is not None:
                log_file.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                log_file.flush()
            progress.set_postfix({
                "loss": f"{metrics['loss']:.3f}",
                "reward": f"{avg_reward:.3f}",
                "skip": skipped_groups,
            })

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    state_dict = model.state_dict()
    clean_state = {k: v for k, v in state_dict.items() if not k.startswith('vision_encoder.')}
    clean_state = {k: v.half() for k, v in clean_state.items()}
    torch.save(clean_state, args.save_path)

    if log_file is not None:
        log_file.write(
            json.dumps(
                {
                    "event": "checkpoint_saved",
                    "path": os.path.abspath(args.save_path),
                    "steps": args.steps,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        log_file.close()


if __name__ == "__main__":
    main()
