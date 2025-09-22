import argparse
import json
import os
import random
from typing import Dict, Any, Tuple


def parse_action_from_output(output_text: str) -> Tuple[int, int]:
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


def make_json_text(r: int, c: int, thinking_hint: str = "") -> str:
    obj = {
        "thinking": thinking_hint or f"我分析当前局面后，选择走[{r},{c}]。",
        "action": [int(r), int(c)],
    }
    return json.dumps(obj, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Build simple DPO pairs from SFT jsonl")
    parser.add_argument("--sft-jsonl", type=str, required=True, help="输入的 SFT 样本（alpaca风格）")
    parser.add_argument("--out-jsonl", type=str, required=True, help="输出的 DPO 偏好对 jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n = 0
    with open(args.sft_jsonl, 'r', encoding='utf-8') as fin, open(args.out_jsonl, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec: Dict[str, Any] = json.loads(line)
            inst = rec.get('instruction', '')
            inp = rec.get('input', '')
            img = rec.get('image_path', '')
            output = rec.get('output', '{}')
            r, c = parse_action_from_output(output)

            # 随机一个不同的动作作为 rejected
            valid = [(rr, cc) for rr in range(3) for cc in range(3) if not (rr == r and cc == c)]
            rr, cc = rng.choice(valid)

            chosen = make_json_text(r, c, "优先选择能赢或能阻止对手的落点。")
            rejected = make_json_text(rr, cc, "这个落点可能无法最大化胜率。")

            out = {
                "instruction": inst,
                "input": inp,
                "image_path": img,
                "chosen": chosen,
                "rejected": rejected,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} DPO pairs -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
