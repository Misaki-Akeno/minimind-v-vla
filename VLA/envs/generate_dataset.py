"""
Tic-Tac-Toe dataset generator (Alpaca-style SFT)

功能：
- 采样可达且未终局、轮到 O 下子的棋局
- 使用极大极小算法为 O 标注最优动作
- 渲染棋盘为图片；导出 Alpaca 风格 JSONL，每行为：
  {
    "instruction": string,
    "input": string,
    "output": string(JSON字符串：{"thinking": str, "action": [row,col]}),
    "image_path": string,
    "meta": {board, current_player, valid_actions}
  }

默认输出到 VLA/data/out/，可用 --num-samples 控制条数。
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# 为无显示环境设置 dummy driver，需在导入 pygame 之前生效；
# 但我们的环境模块在 import 时会导入 pygame，因此先设置环境变量。
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.append("/pub_data/Codes/minimind-v/VLA")

import numpy as np

try:
    # pillow 优先
    from PIL import Image  # type: ignore
    _SAVE_BACKEND = "PIL"
except Exception:  # pragma: no cover - 兜底到 imageio
    try:
        import imageio.v2 as imageio  # type: ignore
        _SAVE_BACKEND = "IMAGEIO"
    except Exception:
        _SAVE_BACKEND = "NONE"

from envs.tic_tac_toe_env import TicTacToeEnv, Difficulty, Player


@dataclass
class Sample:
    board: np.ndarray  # 3x3 int array
    image: np.ndarray  # HxWx3 uint8
    action: Tuple[int, int]  # (row, col)
    valid_actions: List[int]


def save_image(arr: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if _SAVE_BACKEND == "PIL":
        Image.fromarray(arr).save(path)
    elif _SAVE_BACKEND == "IMAGEIO":
        imageio.imwrite(path, arr)
    else:
        raise RuntimeError("No image backend available. Please install pillow or imageio.")


def board_tokens(board: np.ndarray) -> List[List[str]]:
    mapping = {0: " ", 1: "X", 2: "O"}
    return [[mapping[int(board[r, c])] for c in range(3)] for r in range(3)]


def format_board_text(board: np.ndarray) -> str:
    toks = board_tokens(board)
    # 形如 [[O, X,  ], [ ,  , X], [ , O,  ]]
    rows = ["[" + ", ".join(tok_row) + "]" for tok_row in toks]
    return "[" + ", ".join(rows) + "]"


def find_winning_moves(env: TicTacToeEnv, player: Player) -> List[Tuple[int, int]]:
    wins: List[Tuple[int, int]] = []
    valid = [a for a in range(9) if env._is_valid_action(a)]
    for a in valid:
        r, c = divmod(a, 3)
        env.board[r, c] = player.value
        if env._check_winner() and env.winner == player:
            wins.append((r, c))
        # 回滚
        env.board[r, c] = Player.EMPTY.value
        env.winner = None
    return wins


def gen_thinking_text(env: TicTacToeEnv, action: Tuple[int, int]) -> str:
    btxt = format_board_text(env.board)
    # 检查直接取胜与需要拦截
    o_wins = find_winning_moves(env, Player.O)
    x_wins = find_winning_moves(env, Player.X)

    parts = [
        f"我能了解到现在的棋盘是{btxt}，我是其中的O，下一步轮到我下。"
    ]
    if o_wins:
        list_str = ", ".join([f"[{r},{c}]" for r, c in o_wins])
        parts.append(f"我有直接取胜的落点：{list_str}。")
    if x_wins:
        list_str = ", ".join([f"[{r},{c}]" for r, c in x_wins])
        parts.append(f"注意到对手X存在威胁（一步取胜点）：{list_str}，需要优先拦截。")
    if not o_wins and not x_wins:
        # 简单偏好说明
        r, c = action
        if (r, c) == (1, 1):
            parts.append("中心位置通常更强，可优先考虑中心。")
        elif (r, c) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            parts.append("角落位置有利于形成两路威胁。")
        else:
            parts.append("当前没有直接威胁，选择稳健走法。")

    parts.append(f"因此我的选择是落子到[{action[0]},{action[1]}]。")
    return "".join(parts)


def sample_state_o_to_move(env: TicTacToeEnv, rng: random.Random, max_pairs: int = 3, tries: int = 50) -> bool:
    """
    通过随机自博弈构造一个可达且未终局、轮到 O 落子的局面：
    - 从空棋盘出发，进行 m 个 (X,O) 完整回合，然后再让 X 下 1 步；此时轮到 O。
    - 若任一过程终局，则重试。
    返回是否成功。
    """
    for _ in range(tries):
        env.board[:, :] = Player.EMPTY.value
        env.winner = None
        # m in [0, max_pairs]
        m = rng.randint(0, max_pairs)
        ok = True

        # m 个完整回合 (X->O)
        for _round in range(m):
            # X 随机
            valid = [a for a in range(9) if env._is_valid_action(a)]
            if not valid:
                ok = False
                break
            a_x = rng.choice(valid)
            rx, cx = divmod(a_x, 3)
            env.board[rx, cx] = Player.X.value
            if env._check_winner():
                ok = False
                env.winner = None
                break

            # O 随机
            valid = [a for a in range(9) if env._is_valid_action(a)]
            if not valid:
                ok = False
                break
            a_o = rng.choice(valid)
            ro, co = divmod(a_o, 3)
            env.board[ro, co] = Player.O.value
            if env._check_winner():
                ok = False
                env.winner = None
                break

        if not ok:
            continue

        # 额外让 X 再下一步，此时轮到 O
        valid = [a for a in range(9) if env._is_valid_action(a)]
        if not valid:
            continue
        a_x = rng.choice(valid)
        rx, cx = divmod(a_x, 3)
        env.board[rx, cx] = Player.X.value
        if env._check_winner():
            env.winner = None
            continue

        # 校验：轮到 O、非终局
        x_cnt = int((env.board == Player.X.value).sum())
        o_cnt = int((env.board == Player.O.value).sum())
        if x_cnt == o_cnt + 1 and not env._check_winner():
            env.winner = None
            return True
        env.winner = None

    return False


def make_sample(env: TicTacToeEnv) -> Sample:
    # 保证是 O 落子局面
    action_idx = env._get_minimax_action()
    r, c = divmod(action_idx, 3)
    image = env._get_observation()
    return Sample(
        board=env.board.copy(),
        image=image,
        action=(r, c),
        valid_actions=[a for a in range(9) if env._is_valid_action(a)],
    )


def sample_and_label(env: TicTacToeEnv, rng: random.Random) -> Sample:
    ok = sample_state_o_to_move(env, rng)
    if not ok:
        raise RuntimeError("Failed to sample a valid O-to-move state")
    # 标注 + 渲染
    return make_sample(env)


def build_alpaca_record(image_path: str, sample: Sample, thinking: str) -> Dict[str, Any]:
    output_obj = {
        "thinking": thinking,
        "action": [int(sample.action[0]), int(sample.action[1])],  # 0-based [row, col]
    }
    record = {
        "instruction": (
            "你是一个井字棋智能体。给定一张棋盘图像，请推理当前局面并仅输出一个 JSON："
            '{"thinking": string, "action": [row, col]}；行列取值 0-2，且只输出 JSON。'
        ),
        "input": "",
        "output": json.dumps(output_obj, ensure_ascii=False),
        "image_path": image_path,
        "meta": {
            "board": board_tokens(sample.board),
            "current_player": "O",
            "valid_actions": sample.valid_actions,
        },
    }
    return record


def main():
    parser = argparse.ArgumentParser(description="Generate TicTacToe Alpaca-style dataset")
    parser.add_argument("--num-samples", type=int, default=10000, help="生成样本数量")
    parser.add_argument("--out-dir", type=str, default="/pub_data/Codes/minimind-v/VLA/data", help="输出目录")
    parser.add_argument("--prefix", type=str, default="ttt", help="图片与文件前缀")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--board-size", type=int, default=300, help="棋盘图像尺寸")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 初始化环境（rgb_array 渲染；AI 难度不影响我们直接调用的极大极小）
    env = TicTacToeEnv(render_mode="rgb_array", difficulty=Difficulty.HARD, board_size=args.board_size)

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    jsonl_path = os.path.join(args.out_dir, f"{args.prefix}_alpaca.jsonl")
    count = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i in range(args.num_samples):
            sample = sample_and_label(env, rng)
            thinking = gen_thinking_text(env, sample.action)
            # 保存图像
            img_name = f"{args.prefix}_{i:06d}.png"
            img_path = os.path.join(img_dir, img_name)
            save_image(sample.image, img_path)

            # 记录相对路径，便于跨机器移动
            rel_img_path = os.path.relpath(img_path)
            rec = build_alpaca_record(rel_img_path, sample, thinking)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"Generated {count} samples -> {jsonl_path}")


if __name__ == "__main__":
    main()

