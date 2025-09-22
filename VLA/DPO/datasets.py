import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import json as _json

from model.model_vlm import MiniMindVLM


class TicTacToeVLMDPODataset(Dataset):
    """
    偏好数据集：每条样本包含 chosen / rejected 两个回答，用于 DPO。

    JSONL schema:
    {
      "instruction": str,
      "input": str,
      "image_path": str,
      "chosen": str,    # JSON字符串 {"thinking":..., "action":[r,c]}
      "rejected": str   # JSON字符串
    }

    输出：
    {
      'input_ids_chosen': LongTensor[L-1],
      'labels_chosen': LongTensor[L-1],
      'loss_mask_chosen': FloatTensor[L-1],
      'input_ids_rejected': LongTensor[L-1],
      'labels_rejected': LongTensor[L-1],
      'loss_mask_rejected': FloatTensor[L-1],
      'pixel_values': FloatTensor[1,1,3,H,W],
      'action': LongTensor[]  # 0..8，从chosen中提取，供评估
    }
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
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path
        if os.path.exists(image_path):
            return image_path
        if self.images_root:
            candidate = os.path.join(self.images_root, image_path)
            if os.path.exists(candidate):
                return candidate
        if (jsonl_path or self.jsonl_path) and not os.path.isabs(image_path):
            base_dir = os.path.dirname(os.path.abspath(jsonl_path or self.jsonl_path))
            candidate = os.path.normpath(os.path.join(base_dir, image_path))
            if os.path.exists(candidate):
                return candidate
        return image_path

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

    @staticmethod
    def _parse_action_from_json(answer_text: str) -> int:
        idx = 0
        try:
            obj = _json.loads(answer_text) if isinstance(answer_text, str) else answer_text
            if isinstance(obj, dict) and 'action' in obj:
                rc: Tuple[int, int] = tuple(obj['action'])  # type: ignore
                r, c = int(rc[0]), int(rc[1])
                if 0 <= r <= 2 and 0 <= c <= 2:
                    idx = r * 3 + c
                    return idx
        except Exception:
            pass
        return 0

    def _build_pair(self, prompt: str, answer: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text = prompt + answer
        tokenized = self.tokenizer(text)
        ids = tokenized.input_ids[: self.max_length]
        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids = ids + [self.tokenizer.pad_token_id] * pad_len
        loss_mask_list = self._generate_loss_mask(ids)
        X = torch.tensor(ids[:-1], dtype=torch.long)
        Y = torch.tensor(ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask_list[1:], dtype=torch.float)
        return X, Y, loss_mask

    def __getitem__(self, index: int):
        s = self.samples[index]
        instruction: str = s.get('instruction', '')
        user_input: str = s.get('input', '')
        image_path: str = s.get('image_path', '')
        chosen: str = s.get('chosen', '')
        rejected: str = s.get('rejected', '')

        prompt = self._create_prompt(instruction, user_input)
        Xc, Yc, Mc = self._build_pair(prompt, chosen)
        Xr, Yr, Mr = self._build_pair(prompt, rejected)

        resolved_path = self._resolve_image_path(image_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Image not found: {resolved_path}")
        image = Image.open(resolved_path)
        image_tensor = MiniMindVLM.image2tensor(image, self.preprocess)
        pixel_values = torch.stack([image_tensor], dim=0)  # [1,1,3,H,W]

        action_idx = self._parse_action_from_json(chosen)

        return {
            'input_ids_chosen': Xc,
            'labels_chosen': Yc,
            'loss_mask_chosen': Mc,
            'input_ids_rejected': Xr,
            'labels_rejected': Yr,
            'loss_mask_rejected': Mr,
            'pixel_values': pixel_values,
            'action': torch.tensor(action_idx, dtype=torch.long),
        }
