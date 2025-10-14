
"""
Enhanced Tic-Tac-Toe dataset generator (Alpaca-style SFT).

Key features:
- Mixed-policy self-play sampler with adjustable depth and exploration for richer states.
- Optional rotational / reflection augmentations with synchronized label remapping.
- Cached minimax search for fast optimal-action labels and reasoning text generation.

CLI options configure sample counts, player mix, symmetry augmentation, and difficulty balance.
Outputs Alpaca-style JSONL splits plus rendered board images under the chosen directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import trange

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.append("/pub_data/Codes/minimind-v/VLA")

import numpy as np
from PIL import Image  # type: ignore

from envs.tic_tac_toe_env import Difficulty, Player, TicTacToeEnv


@dataclass
class Sample:
    board: np.ndarray
    image: np.ndarray
    action: Tuple[int, int]
    valid_actions: List[int]


TRANSFORM_FUNCS = {
    "identity": lambda arr: arr.copy(),
    "rot90": lambda arr: np.rot90(arr, k=1),
    "rot180": lambda arr: np.rot90(arr, k=2),
    "rot270": lambda arr: np.rot90(arr, k=3),
    "flip_h": lambda arr: np.fliplr(arr),
    "flip_v": lambda arr: np.flipud(arr),
    "flip_diag": lambda arr: np.transpose(arr),
    "flip_anti": lambda arr: np.rot90(np.transpose(arr), k=2),
}

DEFAULT_INSTRUCTION = (
    "你是一个井字棋智能体。给定一张棋盘图像，请推理当前局面并仅输出一个 JSON："
    '{"thinking": string, "action": [row, col]}；行列取值 0-2，且只输出 JSON。'
)


def save_image(arr: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)


def board_tokens(board: np.ndarray) -> List[List[str]]:
    mapping = {0: " ", 1: "X", 2: "O"}
    return [[mapping[int(board[r, c])] for c in range(3)] for r in range(3)]


def format_board_text(board: np.ndarray) -> str:
    toks = board_tokens(board)
    rows = ["[" + ", ".join(row) + "]" for row in toks]
    return "[" + ", ".join(rows) + "]"


def transform_board(board: np.ndarray, transform: str) -> np.ndarray:
    if transform not in TRANSFORM_FUNCS:
        raise ValueError(f"Unsupported transform: {transform}")
    return np.array(TRANSFORM_FUNCS[transform](board), dtype=board.dtype)


def _transform_coords(row: int, col: int, transform: str) -> Tuple[int, int]:
    if transform == "identity":
        return row, col
    if transform == "rot90":
        return col, 2 - row
    if transform == "rot180":
        return 2 - row, 2 - col
    if transform == "rot270":
        return 2 - col, row
    if transform == "flip_h":
        return row, 2 - col
    if transform == "flip_v":
        return 2 - row, col
    if transform == "flip_diag":
        return col, row
    if transform == "flip_anti":
        return 2 - col, 2 - row
    raise ValueError(f"Unsupported transform: {transform}")


def transform_action_indices(indices: Iterable[int], transform: str) -> List[int]:
    transformed: List[int] = []
    for idx in indices:
        r, c = divmod(idx, 3)
        nr, nc = _transform_coords(r, c, transform)
        transformed.append(nr * 3 + nc)
    return transformed


def transform_action_tuple(action: Tuple[int, int], transform: str) -> Tuple[int, int]:
    return _transform_coords(action[0], action[1], transform)


def _local_check_winner(board_arr: np.ndarray) -> Optional[Player]:
    for row in range(3):
        if board_arr[row, 0] == board_arr[row, 1] == board_arr[row, 2] != Player.EMPTY.value:
            return Player(int(board_arr[row, 0]))
    for col in range(3):
        if board_arr[0, col] == board_arr[1, col] == board_arr[2, col] != Player.EMPTY.value:
            return Player(int(board_arr[0, col]))
    if board_arr[0, 0] == board_arr[1, 1] == board_arr[2, 2] != Player.EMPTY.value:
        return Player(int(board_arr[0, 0]))
    if board_arr[0, 2] == board_arr[1, 1] == board_arr[2, 0] != Player.EMPTY.value:
        return Player(int(board_arr[0, 2]))
    return None


def _local_is_full(board_arr: np.ndarray) -> bool:
    return np.all(board_arr != Player.EMPTY.value)


def board_to_key(board_arr: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(v) for v in board_arr.reshape(-1))


def _key_to_board(board_key: Tuple[int, ...]) -> np.ndarray:
    return np.array(board_key, dtype=int).reshape(3, 3)


@lru_cache(maxsize=20000)
def _minimax_cached(board_key: Tuple[int, ...], current_value: int, depth: int) -> float:
    board_arr = _key_to_board(board_key)
    winner = _local_check_winner(board_arr)
    if winner == Player.O:
        return 1.0 - depth * 0.1
    if winner == Player.X:
        return -1.0 + depth * 0.1
    if _local_is_full(board_arr):
        return 0.0

    if Player(current_value) == Player.O:
        best = float("-inf")
        for idx in range(9):
            if board_arr.flatten()[idx] != Player.EMPTY.value:
                continue
            next_board = board_arr.copy()
            r, c = divmod(idx, 3)
            next_board[r, c] = Player.O.value
            score = _minimax_cached(board_to_key(next_board), Player.X.value, depth + 1)
            best = max(best, score)
        return best

    best = float("inf")
    for idx in range(9):
        if board_arr.flatten()[idx] != Player.EMPTY.value:
            continue
        next_board = board_arr.copy()
        r, c = divmod(idx, 3)
        next_board[r, c] = Player.X.value
        score = _minimax_cached(board_to_key(next_board), Player.O.value, depth + 1)
        best = min(best, score)
    return best


@lru_cache(maxsize=20000)
def _best_action_cached(board_key: Tuple[int, ...], player_value: int) -> int:
    board_arr = _key_to_board(board_key)
    board_flat = board_arr.flatten()
    valid = [idx for idx in range(9) if board_flat[idx] == Player.EMPTY.value]
    if not valid:
        raise RuntimeError("No valid actions")

    player = Player(player_value)
    best_action = valid[0]
    if player == Player.O:
        best_score = float("-inf")
        for idx in valid:
            next_board = board_arr.copy()
            r, c = divmod(idx, 3)
            next_board[r, c] = Player.O.value
            score = _minimax_cached(board_to_key(next_board), Player.X.value, 1)
            if score > best_score:
                best_score = score
                best_action = idx
    else:
        best_score = float("inf")
        for idx in valid:
            next_board = board_arr.copy()
            r, c = divmod(idx, 3)
            next_board[r, c] = Player.X.value
            score = _minimax_cached(board_to_key(next_board), Player.O.value, 1)
            if score < best_score:
                best_score = score
                best_action = idx
    return best_action


def get_best_action_for_player(board_arr: np.ndarray, player: Player) -> int:
    board_key = board_to_key(board_arr)
    return _best_action_cached(board_key, player.value)


def find_winning_moves(board: np.ndarray, player: Player) -> List[Tuple[int, int]]:
    wins: List[Tuple[int, int]] = []
    for idx in range(9):
        r, c = divmod(idx, 3)
        if board[r, c] != Player.EMPTY.value:
            continue
        board[r, c] = player.value
        if _local_check_winner(board) == player:
            wins.append((r, c))
        board[r, c] = Player.EMPTY.value
    return wins


def gen_thinking_text(board: np.ndarray, action: Tuple[int, int], player: Player) -> str:
    board_copy = board.copy()
    text = format_board_text(board_copy)
    player_wins = find_winning_moves(board_copy.copy(), player)
    opp = Player.X if player == Player.O else Player.O
    opp_wins = find_winning_moves(board_copy.copy(), opp)

    parts = [f"我能了解到现在的棋盘是{text}，我是其中的{player.name}，下一步轮到我下。"]
    if player_wins:
        wins_str = ", ".join(f"[{r},{c}]" for r, c in player_wins)
        parts.append(f"我有直接取胜的落点：{wins_str}。")
    if opp_wins:
        threat_str = ", ".join(f"[{r},{c}]" for r, c in opp_wins)
        parts.append(f"注意到对手{opp.name}存在威胁（一步取胜点）：{threat_str}，需要优先拦截。")
    if not player_wins and not opp_wins:
        r, c = action
        if (r, c) == (1, 1):
            parts.append("中心位置通常更强，可优先考虑中心。")
        elif (r, c) in {(0, 0), (0, 2), (2, 0), (2, 2)}:
            parts.append("角落位置有利于形成两路威胁。")
        else:
            parts.append("当前没有直接威胁，选择稳健走法。")
    parts.append(f"因此我的选择是落子到[{action[0]},{action[1]}]。")
    return "".join(parts)


def render_board_image(env: TicTacToeEnv, board: np.ndarray, player: Player) -> np.ndarray:
    prev_board = env.board.copy()
    prev_player = env.current_player
    env.board[:, :] = board
    env.current_player = player
    image = env._get_observation()
    env.board[:, :] = prev_board
    env.current_player = prev_player
    return image


def choose_action(env: TicTacToeEnv, rng: random.Random, player: Player, explore_prob: float) -> Optional[int]:
    valid = [idx for idx in range(9) if env._is_valid_action(idx)]
    if not valid:
        return None
    if explore_prob > 0 and rng.random() < explore_prob:
        return rng.choice(valid)
    board_copy = env.board.copy()
    return get_best_action_for_player(board_copy, player)


def sample_state_to_move(
    env: TicTacToeEnv,
    rng: random.Random,
    player: Player,
    *,
    min_pairs: int,
    max_pairs: int,
    tries: int,
    explore_prob: float,
) -> bool:
    max_pairs = max(min_pairs, max_pairs)
    for _ in range(tries):
        env.board[:, :] = Player.EMPTY.value
        env.winner = None
        env.current_player = Player.X

        rounds = rng.randint(min_pairs, max_pairs)
        ok = True

        for _ in range(rounds):
            action_x = choose_action(env, rng, Player.X, explore_prob)
            if action_x is None:
                ok = False
                break
            rx, cx = divmod(action_x, 3)
            env.board[rx, cx] = Player.X.value
            if env._check_winner():
                ok = False
                env.winner = None
                break

            action_o = choose_action(env, rng, Player.O, explore_prob)
            if action_o is None:
                ok = False
                break
            ro, co = divmod(action_o, 3)
            env.board[ro, co] = Player.O.value
            if env._check_winner():
                ok = False
                env.winner = None
                break

        if not ok:
            continue

        if player == Player.O:
            action_x = choose_action(env, rng, Player.X, explore_prob)
            if action_x is None:
                continue
            rx, cx = divmod(action_x, 3)
            env.board[rx, cx] = Player.X.value
            if env._check_winner():
                env.winner = None
                continue

        if _local_is_full(env.board):
            env.winner = None
            continue

        x_cnt = int((env.board == Player.X.value).sum())
        o_cnt = int((env.board == Player.O.value).sum())
        cond = (x_cnt == o_cnt + 1) if player == Player.O else (x_cnt == o_cnt)
        if cond and not env._check_winner():
            env.winner = None
            env.current_player = player
            return True
        env.winner = None
    return False


def make_sample(env: TicTacToeEnv, player: Player) -> Sample:
    env.current_player = player
    board_copy = env.board.copy()
    action_idx = get_best_action_for_player(board_copy, player)
    r, c = divmod(action_idx, 3)
    image = render_board_image(env, board_copy, player)
    valid_actions = [idx for idx in range(9) if env._is_valid_action(idx)]
    if not valid_actions:
        raise RuntimeError("No valid actions")
    return Sample(board=board_copy, image=image, action=(r, c), valid_actions=valid_actions)


def sample_and_label(
    env: TicTacToeEnv,
    rng: random.Random,
    player: Player,
    sample_cfg: Dict[str, Any],
) -> Sample:
    ok = sample_state_to_move(
        env,
        rng,
        player,
        min_pairs=sample_cfg["min_pairs"],
        max_pairs=sample_cfg["max_pairs"],
        tries=sample_cfg["tries"],
        explore_prob=sample_cfg["explore_prob"],
    )
    if not ok:
        raise RuntimeError(f"Failed to sample a valid {player.name}-to-move state")
    return make_sample(env, player)


def build_alpaca_record(
    image_path: str,
    sample: Sample,
    thinking: str,
    current_player: Player,
    *,
    source: str,
    transform: str,
) -> Dict[str, Any]:
    output_obj = {
        "thinking": thinking,
        "action": [int(sample.action[0]), int(sample.action[1])],
    }
    return {
        "instruction": DEFAULT_INSTRUCTION,
        "input": "",
        "output": json.dumps(output_obj, ensure_ascii=False),
        "image_path": image_path,
        "meta": {
            "board": board_tokens(sample.board),
            "current_player": current_player.name,
            "valid_actions": sample.valid_actions,
            "source": source,
            "transform": transform,
        },
    }


def augment_symmetry(
    env: TicTacToeEnv,
    base_sample: Sample,
    player: Player,
    transforms: Sequence[str],
) -> List[Tuple[str, Sample]]:
    augmented: List[Tuple[str, Sample]] = []
    for transform in transforms:
        tname = transform.strip()
        if not tname or tname == "identity":
            continue
        if tname not in TRANSFORM_FUNCS:
            continue
        aug_board = transform_board(base_sample.board, tname)
        aug_action = transform_action_tuple(base_sample.action, tname)
        aug_valid = transform_action_indices(base_sample.valid_actions, tname)
        image = render_board_image(env, aug_board, player)
        augmented.append(
            (
                tname,
                Sample(board=aug_board, image=image, action=aug_action, valid_actions=aug_valid),
            )
        )
    return augmented


def parse_transforms(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    if not parts:
        return ["identity"]
    for part in parts:
        if part not in TRANSFORM_FUNCS:
            raise ValueError(f"Unknown transform: {part}")
    return parts


def parse_players(opt: str) -> List[Player]:
    val = opt.lower()
    if val == "o":
        return [Player.O]
    if val == "x":
        return [Player.X]
    if val == "both":
        return [Player.O, Player.X]
    raise RuntimeError("--players must be one of O, X, both")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TicTacToe Alpaca-style dataset (enhanced)")
    parser.add_argument("--num-samples", type=int, default=5000, help="基样本数量（不含增强）")
    parser.add_argument("--out-dir", type=str, default="VLA/data", help="输出根目录")
    parser.add_argument("--prefix", type=str, default="ttt", help="文件名前缀")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--board-size", type=int, default=300, help="棋盘图像尺寸")
    parser.add_argument("--split-ratios", type=str, default="0.8,0.1,0.1", help="train,val,test 比例")
    parser.add_argument("--players", type=str, default="both", help="执子：O, X, both")
    parser.add_argument("--min-pairs", type=int, default=1, help="随机自博弈最少完整回合数")
    parser.add_argument("--max-pairs", type=int, default=8, help="随机自博弈最多完整回合数")
    parser.add_argument("--explore-prob", type=float, default=0.35, help="随机动作探索概率")
    parser.add_argument("--sample-tries", type=int, default=80, help="单局采样最大重试次数")
    parser.add_argument("--augment-symmetry", action="store_true", help="启用旋转/翻转增强")
    parser.add_argument(
        "--augment-transforms",
        type=str,
        default="rot90,rot180,rot270,flip_h,flip_v",
        help="增强类型，逗号分隔，可选 identity,rot90,rot180,rot270,flip_h,flip_v,flip_diag,flip_anti",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = TicTacToeEnv(render_mode="rgb_array", difficulty=Difficulty.HARD, board_size=args.board_size)

    os.makedirs(args.out_dir, exist_ok=True)
    ratios = [float(x) for x in args.split_ratios.split(',')]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise RuntimeError("--split-ratios must contain three numbers summing to 1")

    players = parse_players(args.players)
    base_counts = {
        "train": int(args.num_samples * ratios[0]),
        "val": int(args.num_samples * ratios[1]),
        "test": args.num_samples - int(args.num_samples * ratios[0]) - int(args.num_samples * ratios[1]),
    }

    writers: Dict[str, Any] = {}
    img_dirs: Dict[str, str] = {}
    for split in ["train", "val", "test"]:
        jsonl_path = os.path.join(args.out_dir, f"{args.prefix}_{split}_alpaca.jsonl")
        writers[split] = open(jsonl_path, "w", encoding="utf-8")
        img_dir = os.path.join(args.out_dir, "images", split)
        os.makedirs(img_dir, exist_ok=True)
        img_dirs[split] = img_dir

    sample_cfg = {
        "min_pairs": max(0, args.min_pairs),
        "max_pairs": max(args.min_pairs, args.max_pairs),
        "tries": max(1, args.sample_tries),
        "explore_prob": min(1.0, max(0.0, args.explore_prob)),
    }

    transforms = parse_transforms(args.augment_transforms)
    total_records = 0

    try:
        for split in ["train", "val", "test"]:
            need = base_counts[split]
            writer = writers[split]
            img_dir = img_dirs[split]
            for i in trange(need, desc=f"Generating {split}"):
                player = players[i % len(players)]
                sample = sample_and_label(env, rng, player, sample_cfg)
                thinking = gen_thinking_text(sample.board.copy(), sample.action, player)

                img_name = f"{args.prefix}_{split}_{player.name}_{i:06d}.png"
                img_path = os.path.join(img_dir, img_name)
                save_image(sample.image, img_path)
                rel_img_path = os.path.relpath(img_path)
                record = build_alpaca_record(rel_img_path, sample, thinking, player, source="base", transform="identity")
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_records += 1

                if args.augment_symmetry:
                    augmented = augment_symmetry(env, sample, player, transforms)
                    for aug_idx, (transform, aug_sample) in enumerate(augmented, start=1):
                        aug_thinking = gen_thinking_text(aug_sample.board.copy(), aug_sample.action, player)
                        aug_img_name = f"{args.prefix}_{split}_{player.name}_{i:06d}_aug{aug_idx}.png"
                        aug_img_path = os.path.join(img_dir, aug_img_name)
                        save_image(aug_sample.image, aug_img_path)
                        aug_rel = os.path.relpath(aug_img_path)
                        aug_record = build_alpaca_record(aug_rel, aug_sample, aug_thinking, player, source="augmented", transform=transform)
                        writer.write(json.dumps(aug_record, ensure_ascii=False) + "\n")
                        total_records += 1
    finally:
        for f in writers.values():
            f.close()

    print(f"Generated {total_records} records -> {args.out_dir}")


if __name__ == "__main__":
    main()
