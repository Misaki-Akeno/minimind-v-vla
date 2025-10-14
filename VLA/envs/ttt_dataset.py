import json
import os
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import json as _json

from model.model_vlm import MiniMindVLM


class TicTacToeVLMDataset(Dataset):
    """
    适配井字棋样本的 SFT 数据集。

    期望每行一个 JSON 对象，示例字段：
    {
      "instruction": str,
      "input": str,
      "output": str,  # 训练目标，严格为模型应输出的 JSON 文本
      "image_path": str,  # 图片路径，可为绝对或相对路径
      "meta": {...}  # 可选，不参与训练
    }

    训练时我们构造一个两轮对话：
    - user: <image> + instruction (+ input)
    - assistant: output

    然后使用 tokenizer.apply_chat_template 包装，并将 <image> 替换成图像占位符 token 序列（例如 196 个 '@'）。
    返回 dict：{input_ids, labels, loss_mask, pixel_values, action}
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        preprocess=None,
        max_length: int = 512,
        image_special_token: str = '@' * 196,
        images_root: Optional[str] = None,
    ):
        super().__init__()
        self.jsonl_path = os.path.abspath(jsonl_path)
        self.samples = self._load_jsonl(self.jsonl_path)
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.max_length = max_length
        self.image_token = image_special_token
        self.images_root = images_root

        # ensure padding token exists for mask truncation
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id

        # special token ids used to构造loss mask（与现有实现一致）
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, image_path: str, jsonl_path: Optional[str] = None) -> str:
        # 如果是绝对路径或本身存在，直接返回
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path
        if os.path.exists(image_path):
            return image_path

        # 若提供 images_root，优先从中拼接
        if self.images_root:
            candidate = os.path.join(self.images_root, image_path)
            if os.path.exists(candidate):
                return candidate

        # 若是相对路径，尝试相对于 jsonl 文件位置
        if (jsonl_path or self.jsonl_path) and not os.path.isabs(image_path):
            base_dir = os.path.dirname(os.path.abspath(jsonl_path or self.jsonl_path))
            candidate = os.path.normpath(os.path.join(base_dir, image_path))
            if os.path.exists(candidate):
                return candidate

        # 最后返回原始（可能不存在），由上层失败时提示
        return image_path

    def _create_prompt(self, instruction: str, user_input: str, current_player: str) -> str:
        player_line = self._format_player_line(current_player)
        # 将占位 <image> 放在最前，紧接指令和可选输入，便于模型对齐
        # 注意：后续会把 <image> 替换为图像特殊 token 序列
        content_parts = ["<image>", instruction.strip() if instruction else ""]
        if player_line:
            content_parts.append(player_line)
        if user_input:
            content_parts.append(user_input.strip())
        content = "\n".join([p for p in content_parts if p])
        # 使用 chat 模板
        messages = [
            {"role": "user", "content": content.replace('<image>', self.image_token)},
        ]
        # tokenizer.apply_chat_template 返回完整聊天串（包含 <|im_start|>/assistant 等）
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # 为 assistant 留生成提示
        )
        return prompt

    def _format_player_line(self, current_player: str) -> str:
        player = (current_player or "").strip().upper()
        if player not in {"X", "O"}:
            return "当前轮到你落子，请选择最佳位置。"
        return f"当前轮到 {player} 落子，请为 {player} 选择最佳落子位置。"

    def _generate_loss_mask(self, input_ids: List[int]) -> List[int]:
        # 与 dataset/lm_dataset.py 中逻辑一致：仅在 assistant 段计算 loss
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    if self.pad_id is not None and input_ids[end] == self.pad_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id), len(input_ids))):
                    if self.pad_id is not None and input_ids[j] == self.pad_id:
                        break
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int):
        sample = self.samples[index]
        instruction: str = sample.get('instruction', '')
        user_input: str = sample.get('input', '')
        target: str = sample.get('output', '')
        image_path: str = sample.get('image_path', '')

        meta: Dict[str, Any] = sample.get('meta') or {}
        current_player = meta.get('current_player', 'O')

        # 1) prompt（仅包含 user 段，assistant 留生成提示）
        prompt = self._create_prompt(instruction, user_input, current_player)

        # 2) 将 assistant 回复（目标）拼接到 prompt 后，使用 chat 模板的一致前后缀
        # 这里直接接上 target，确保训练时标签覆盖 assistant 段
        target_text = target if isinstance(target, str) else json.dumps(target, ensure_ascii=False)
        target_text = target_text.strip()
        full_text = prompt + target_text + '<|im_end|>'

        tokenized = self.tokenizer(full_text)
        input_ids = tokenized.input_ids[: self.max_length]
        # padding
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len

        loss_mask_list = self._generate_loss_mask(input_ids)

        # 语言建模的输入与标签（shift by 1）
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask_list[1:], dtype=torch.float)

        # 3) 加载图像，按现有实现返回形状 [num_images, 1, C, H, W]
        resolved_path = self._resolve_image_path(image_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Image not found: {resolved_path}")
        image = Image.open(resolved_path)
        image_tensor = MiniMindVLM.image2tensor(image, self.preprocess)  # [1, C, H, W]
        pixel_values = torch.stack([image_tensor], dim=0)  # [1, 1, C, H, W]

        # 4) 提取当前局面的最优动作标签（0-8）
        # 来自生成数据时 output 字段中的 JSON：{"thinking": str, "action": [row, col]}
        action_idx = -1
        try:
            out_obj = _json.loads(target) if isinstance(target, str) else target
            if isinstance(out_obj, dict) and 'action' in out_obj:
                rc: Tuple[int, int] = tuple(out_obj['action'])  # type: ignore
                r, c = int(rc[0]), int(rc[1])
                if 0 <= r <= 2 and 0 <= c <= 2:
                    action_idx = r * 3 + c
        except Exception:
            action_idx = -1
        if action_idx == -1:
            # 兜底：若解析失败，置为 0（不理想但避免崩溃）；
            # 更好的做法是抛错或从 meta.valid_actions 选择一个，但训练更稳妥的是跳过该样本。
            action_idx = 0

        sample_dict = {
            'input_ids': X,
            'labels': Y,  # 与原来一致
            'loss_mask': loss_mask,
            'pixel_values': pixel_values,  # 图像
            'action': torch.tensor(action_idx, dtype=torch.long),  # 新增：0-8 的整型动作
        }

        return sample_dict
