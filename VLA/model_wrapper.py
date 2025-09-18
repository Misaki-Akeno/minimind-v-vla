from typing import Optional, Tuple, List, Union

import torch
from torch import nn

from model.model_vlm import MiniMindVLM, VLMConfig


class ActionHead(nn.Module):
    def __init__(self, hidden_size: int, action_dim: int = 9):
        super().__init__()
        self.fc = nn.Linear(hidden_size, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H]
        return self.fc(x)


class MiniMindVLMWithAction(nn.Module):
    """
    轻量封装：包含一个 MiniMindVLM 作为骨干 + 动作分类头。
    - 前向返回原有输出，并追加 res['action_logits']，使用最后一个 token 的隐状态做决策。
    - 透传 vision_encoder 与 processor，便于外部取预处理器。
    """

    def __init__(self, config: VLMConfig, vision_model_path: str = "../model/vision_model/clip-vit-base-patch16"):
        super().__init__()
        self.vlm = MiniMindVLM(config, vision_model_path=vision_model_path)
        self.action_head = ActionHead(hidden_size=config.hidden_size, action_dim=9)

        # 方便外部访问
        self.vision_encoder = self.vlm.vision_encoder
        self.processor = self.vlm.processor
        self.config = self.vlm.config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        pixel_values: Optional[torch.FloatTensor] = None,
        **args,
    ):
        res = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            pixel_values=pixel_values,
            **args,
        )

        # 追加动作 logits
        last_hidden = res.last_hidden_state[:, -1, :]
        action_logits = self.action_head(last_hidden)
        try:
            res.__setitem__('action_logits', action_logits)
        except Exception:
            # 兜底：若输出为命名元组，直接赋属性
            setattr(res, 'action_logits', action_logits)
        return res
