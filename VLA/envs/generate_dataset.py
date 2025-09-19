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
from tqdm import trange

# 为无显示环境设置 dummy driver，需在导入 pygame 之前生效；
# 但我们的环境模块在 import 时会导入 pygame，因此先设置环境变量。
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.append("/pub_data/Codes/minimind-v/VLA")

import numpy as np
from PIL import Image  # type: ignore
from envs.tic_tac_toe_env import TicTacToeEnv, Difficulty, Player


@dataclass
class Sample:
    board: np.ndarray  # 3x3 int array
    image: np.ndarray  # HxWx3 uint8
    action: Tuple[int, int]  # (row, col)
    valid_actions: List[int]


def save_image(arr: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)


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


def gen_thinking_text(env: TicTacToeEnv, action: Tuple[int, int], player: Player) -> str:
    btxt = format_board_text(env.board)
    # 检查直接取胜与需要拦截
    player_wins = find_winning_moves(env, player)
    opp = Player.X if player == Player.O else Player.O
    opp_wins = find_winning_moves(env, opp)

    parts = [
        f"我能了解到现在的棋盘是{btxt}，我是其中的{player.name}，下一步轮到我下。"
    ]
    if player_wins:
        list_str = ", ".join([f"[{r},{c}]" for r, c in player_wins])
        parts.append(f"我有直接取胜的落点：{list_str}。")
    if opp_wins:
        list_str = ", ".join([f"[{r},{c}]" for r, c in opp_wins])
        parts.append(f"注意到对手{opp.name}存在威胁（一步取胜点）：{list_str}，需要优先拦截。")
    if not player_wins and not opp_wins:
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


def sample_state_to_move(env: TicTacToeEnv, rng: random.Random, player: Player, max_pairs: int = 3, tries: int = 50) -> bool:
    """
    通过随机自博弈构造一个可达且未终局、轮到指定玩家（X 或 O）落子的局面。
    - 从空棋盘出发，进行 m 个 (X,O) 完整回合；若需要让 O 下，则额外让 X 再下一步。
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

        if player == Player.O:
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

        # 校验：指定玩家轮到、非终局
        x_cnt = int((env.board == Player.X.value).sum())
        o_cnt = int((env.board == Player.O.value).sum())
        if player == Player.O:
            cond = (x_cnt == o_cnt + 1)
        else:
            cond = (x_cnt == o_cnt)

        if cond and not env._check_winner():
            env.winner = None
            return True
        env.winner = None

    return False


def _local_check_winner(board_arr: np.ndarray) -> Optional[Player]:
    # 不改变 env 状态的局部胜负判断
    for row in range(3):
        if (board_arr[row, 0] == board_arr[row, 1] == board_arr[row, 2] != Player.EMPTY.value):
            return Player(int(board_arr[row, 0]))
    for col in range(3):
        if (board_arr[0, col] == board_arr[1, col] == board_arr[2, col] != Player.EMPTY.value):
            return Player(int(board_arr[0, col]))
    if (board_arr[0, 0] == board_arr[1, 1] == board_arr[2, 2] != Player.EMPTY.value):
        return Player(int(board_arr[0, 0]))
    if (board_arr[0, 2] == board_arr[1, 1] == board_arr[2, 0] != Player.EMPTY.value):
        return Player(int(board_arr[0, 2]))
    return None


def _local_is_full(board_arr: np.ndarray) -> bool:
    return np.all(board_arr != Player.EMPTY.value)


def _local_minimax(board_arr: np.ndarray, current: Player, depth: int) -> float:
    winner = _local_check_winner(board_arr)
    if winner == Player.O:
        return 1.0 - depth * 0.1
    if winner == Player.X:
        return -1.0 + depth * 0.1
    if _local_is_full(board_arr):
        return 0.0

    if current == Player.O:
        best = float('-inf')
        for a in range(9):
            r, c = divmod(a, 3)
            if board_arr[r, c] != Player.EMPTY.value:
                continue
            board_arr[r, c] = Player.O.value
            score = _local_minimax(board_arr, Player.X, depth + 1)
            board_arr[r, c] = Player.EMPTY.value
            best = max(best, score)
        return best
    else:
        best = float('inf')
        for a in range(9):
            r, c = divmod(a, 3)
            if board_arr[r, c] != Player.EMPTY.value:
                continue
            board_arr[r, c] = Player.X.value
            score = _local_minimax(board_arr, Player.O, depth + 1)
            board_arr[r, c] = Player.EMPTY.value
            best = min(best, score)
        return best


def get_best_action_for_player(board_arr: np.ndarray, player: Player) -> int:
    valid = [a for a in range(9) if board_arr.flatten()[a] == Player.EMPTY.value]
    if not valid:
        raise RuntimeError("No valid actions")
    best_action = None
    if player == Player.O:
        best_score = float('-inf')
        for a in valid:
            r, c = divmod(a, 3)
            board_arr[r, c] = Player.O.value
            score = _local_minimax(board_arr, Player.X, 0)
            board_arr[r, c] = Player.EMPTY.value
            if score > best_score:
                best_score = score
                best_action = a
    else:
        best_score = float('inf')
        for a in valid:
            r, c = divmod(a, 3)
            board_arr[r, c] = Player.X.value
            score = _local_minimax(board_arr, Player.O, 0)
            board_arr[r, c] = Player.EMPTY.value
            if score < best_score:
                best_score = score
                best_action = a

    return best_action if best_action is not None else valid[0]


def make_sample(env: TicTacToeEnv, player: Player) -> Sample:
    # 保证指定玩家落子局面
    env.current_player = player
    board_copy = env.board.copy()
    action_idx = get_best_action_for_player(board_copy, player)
    r, c = divmod(action_idx, 3)
    image = env._get_observation()
    return Sample(
        board=env.board.copy(),
        image=image,
        action=(r, c),
        valid_actions=[a for a in range(9) if env._is_valid_action(a)],
    )


def sample_and_label(env: TicTacToeEnv, rng: random.Random, player: Player) -> Sample:
    ok = sample_state_to_move(env, rng, player)
    if not ok:
        raise RuntimeError(f"Failed to sample a valid {player.name}-to-move state")
    # 标注 + 渲染
    return make_sample(env, player)


def build_alpaca_record(image_path: str, sample: Sample, thinking: str, current_player: str) -> Dict[str, Any]:
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
            "current_player": current_player,
            "valid_actions": sample.valid_actions,
        },
    }
    return record


def main():
    parser = argparse.ArgumentParser(description="Generate TicTacToe Alpaca-style dataset")
    parser.add_argument("--num-samples", type=int, default=10000, help="生成样本数量")
    parser.add_argument("--out-dir", type=str, default="VLA/data", help="输出目录")
    parser.add_argument("--prefix", type=str, default="ttt", help="图片与文件前缀")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--board-size", type=int, default=300, help="棋盘图像尺寸")
    parser.add_argument("--split-ratios", type=str, default="0.8,0.1,0.1", help="train,val,test 比例，逗号分隔，和为1")
    parser.add_argument("--players", type=str, default="O", help="要生成的当前执子方：O, X, 或 both")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 初始化环境（rgb_array 渲染；AI 难度不影响我们直接调用的极大极小）
    env = TicTacToeEnv(render_mode="rgb_array", difficulty=Difficulty.HARD, board_size=args.board_size)

    os.makedirs(args.out_dir, exist_ok=True)

    # parse split ratios
    ratios = [float(x) for x in args.split_ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise RuntimeError("--split-ratios must be three comma-separated numbers that sum to 1")

    players_opt = args.players.lower()
    players: List[Player] = []
    if players_opt == "o":
        players = [Player.O]
    elif players_opt == "x":
        players = [Player.X]
    elif players_opt == "both":
        players = [Player.O, Player.X]
    else:
        raise RuntimeError("--players must be one of O, X, both")

    # allocate counts per split
    n_total = args.num_samples
    n_train = int(n_total * ratios[0])
    n_val = int(n_total * ratios[1])
    n_test = n_total - n_train - n_val
    split_counts = {"train": n_train, "val": n_val, "test": n_test}

    # prepare files and dirs
    writers = {}
    img_dirs = {}
    for split in ["train", "val", "test"]:
        jsonl_path = os.path.join(args.out_dir, f"{args.prefix}_{split}_alpaca.jsonl")
        f = open(jsonl_path, "w", encoding="utf-8")
        writers[split] = f
        d = os.path.join(args.out_dir, "images", split)
        os.makedirs(d, exist_ok=True)
        img_dirs[split] = d

    count = 0
    # iterate splits and players round-robin to fill counts
    for split in ["train", "val", "test"]:
        need = split_counts[split]
        w = writers[split]
        d = img_dirs[split]
        
        for i in trange(need, desc=f"Generating {split}"):
            # choose player in round-robin from requested players
            player = players[i % len(players)]
            sample = sample_and_label(env, rng, player)
            thinking = gen_thinking_text(env, sample.action, player)

            # 保存图像，文件名包含 split 与 player
            img_name = f"{args.prefix}_{split}_{player.name}_{i:06d}.png"
            img_path = os.path.join(d, img_name)
            save_image(sample.image, img_path)

            rel_img_path = os.path.relpath(img_path)
            rec = build_alpaca_record(rel_img_path, sample, thinking, current_player=player.name)
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    for f in writers.values():
        f.close()

    print(f"Generated {count} samples -> {args.out_dir}")


if __name__ == "__main__":
    main()

