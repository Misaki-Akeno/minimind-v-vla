from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import trange
from transformers import AutoTokenizer

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

THIS_DIR = Path(__file__).resolve()
VLA_ROOT = THIS_DIR.parent
PROJECT_ROOT = THIS_DIR.parent.parent
for candidate in (VLA_ROOT, PROJECT_ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.append(path_str)

from model.model_vlm import VLMConfig
from VLA.envs.model_wrapper import MiniMindVLMWithAction
from VLA.envs.tic_tac_toe_env import Difficulty, Player, TicTacToeEnv


CKPT_DEFAULT = "VLA/models/gpg_vlm_ttt.pth"
HIDDEN_SIZE = 768
NUM_LAYERS = 16
MAX_SEQ_LEN = 512
USE_MOE = False


@dataclass
class GameStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.total if self.total else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.total if self.total else 0.0


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


def optimal_action(board: np.ndarray, player: Player) -> int:
    best_action = None
    best_score = -math.inf if player == Player.O else math.inf

    for idx in range(9):
        r, c = divmod(idx, 3)
        if board[r, c] != Player.EMPTY.value:
            continue
        board[r, c] = player.value
        score = _minimax(board, Player.X if player == Player.O else Player.O, 0)
        board[r, c] = Player.EMPTY.value

        if player == Player.O:
            if score > best_score:
                best_score = score
                best_action = idx
        else:
            if score < best_score:
                best_score = score
                best_action = idx

    if best_action is None:
        empties = np.where(board.flatten() == Player.EMPTY.value)[0]
        if len(empties) == 0:
            return 0
        return int(random.choice(empties))
    return int(best_action)


def build_prompt(tokenizer, image_token: str, board: np.ndarray) -> torch.Tensor:
    instruction = (
        "你是一名井字棋智能体。根据图像与棋盘描述判断当前局面，"
        "仅输出 JSON {\"thinking\": string, \"action\": [row, col]}。"
    )
    description = f"当前棋盘: {format_board_text(board)}。请为 O 选择最优落子位置。"
    content = "\n".join(["<image>", instruction, description])
    messages = [{"role": "user", "content": content.replace("<image>", image_token)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt, return_tensors="pt").input_ids


def board_to_image(board_image: np.ndarray) -> Image.Image:
    return Image.fromarray(board_image)


def mask_invalid(logits: torch.Tensor, board: np.ndarray) -> torch.Tensor:
    masked = logits.clone()
    for idx in range(9):
        r, c = divmod(idx, 3)
        if board[r, c] != Player.EMPTY.value:
            masked[idx] = -1e9
    return masked


def run_episode(
    env: TicTacToeEnv,
    model: MiniMindVLMWithAction,
    tokenizer,
    image_token: str,
    device: torch.device,
    temperature: float,
) -> str:
    obs, _ = env.reset()
    env.use_internal_ai = False

    while True:
        current_player = env.current_player
        board_copy = env.board.copy()

        if current_player == Player.X:
            action = optimal_action(board_copy, Player.X)
        else:
            input_ids = build_prompt(tokenizer, image_token, board_copy).to(device)
            pil_image = board_to_image(obs)
            pixel_values = model.processor(images=pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.unsqueeze(1).to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, pixel_values=pixel_values)
            logits = outputs.action_logits.squeeze(0)
            logits = logits / max(temperature, 1e-5)
            masked_logits = mask_invalid(logits, env.board)
            action = int(masked_logits.argmax().item())
            r, c = divmod(action, 3)
            if env.board[r, c] != Player.EMPTY.value:
                valid = np.where(env.board.flatten() == Player.EMPTY.value)[0]
                if len(valid) == 0:
                    action = 0
                else:
                    action = int(random.choice(valid))

        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            if env.winner == Player.O:
                return "win"
            if env.winner == Player.X:
                return "loss"
            return "draw"


def main() -> None:
    parser = argparse.ArgumentParser(description="Close-loop evaluation: model vs. pattern-based opponent")
    parser.add_argument("--model-path", type=str, default=CKPT_DEFAULT)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--board-size", type=int, default=300)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
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
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    env = TicTacToeEnv(
        render_mode="rgb_array",
        difficulty=Difficulty.HARD,
        board_size=args.board_size,
        use_internal_ai=False,
    )

    stats = GameStats()
    image_token = config.image_special_token

    for _ in trange(args.games, desc="Evaluating"):
        result = run_episode(env, model, tokenizer, image_token, device, args.temperature)
        if result == "win":
            stats.wins += 1
        elif result == "loss":
            stats.losses += 1
        else:
            stats.draws += 1

    print("\n=== Evaluation Summary ===")
    print(f"Games       : {stats.total}")
    print(f"Wins        : {stats.wins} ({stats.win_rate:.2%})")
    print(f"Draws       : {stats.draws} ({stats.draw_rate:.2%})")
    print(f"Losses      : {stats.losses} ({stats.loss_rate:.2%})")


if __name__ == "__main__":
    main()
