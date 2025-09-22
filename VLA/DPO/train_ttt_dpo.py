import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from model.model_vlm import VLMConfig
from VLA.envs.model_wrapper import MiniMindVLMWithAction
from VLA.DPO.datasets import TicTacToeVLMDPODataset
from VLA.SFT.evaluate import evaluate_model


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CKPT_DEFAULT = "MiniMind2-V/pytorch_model.bin"
HIDDEN_SIZE = 768
NUM_LAYERS = 16
MAX_SEQ_LEN = 512
USE_MOE = False


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute log p(y|x) for each token label given logits. Shape-preserving.
    logits: [B, T, V], labels: [B, T]
    Returns: [B, T]
    """
    logp = F.log_softmax(logits, dim=-1)
    lp = torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return lp


def dpo_loss(
    model,
    batch,
    beta: float = 0.1,
    reference_model=None,
    device: str = "cuda:0",
):
    """Compute DPO loss for a batch.
    Optionally subtract reference log-probs if reference_model is provided.
    """
    # Chosen forward
    out_c = model(batch['input_ids_chosen'].to(device), pixel_values=batch['pixel_values'].to(device))
    lp_c = log_probs_from_logits(out_c.logits, batch['labels_chosen'].to(device))
    # mask仅统计assistant段
    lp_c = (lp_c * batch['loss_mask_chosen'].to(device)).sum(dim=-1) / (batch['loss_mask_chosen'].to(device).sum(dim=-1) + 1e-8)

    # Rejected forward
    out_r = model(batch['input_ids_rejected'].to(device), pixel_values=batch['pixel_values'].to(device))
    lp_r = log_probs_from_logits(out_r.logits, batch['labels_rejected'].to(device))
    lp_r = (lp_r * batch['loss_mask_rejected'].to(device)).sum(dim=-1) / (batch['loss_mask_rejected'].to(device).sum(dim=-1) + 1e-8)

    if reference_model is not None:
        with torch.no_grad():
            ref_c = reference_model(batch['input_ids_chosen'].to(device), pixel_values=batch['pixel_values'].to(device))
            ref_lp_c = log_probs_from_logits(ref_c.logits, batch['labels_chosen'].to(device))
            ref_lp_c = (ref_lp_c * batch['loss_mask_chosen'].to(device)).sum(dim=-1) / (batch['loss_mask_chosen'].to(device).sum(dim=-1) + 1e-8)

            ref_r = reference_model(batch['input_ids_rejected'].to(device), pixel_values=batch['pixel_values'].to(device))
            ref_lp_r = log_probs_from_logits(ref_r.logits, batch['labels_rejected'].to(device))
            ref_lp_r = (ref_lp_r * batch['loss_mask_rejected'].to(device)).sum(dim=-1) / (batch['loss_mask_rejected'].to(device).sum(dim=-1) + 1e-8)
        adv = (lp_c - ref_lp_c) - (lp_r - ref_lp_r)
    else:
        adv = lp_c - lp_r

    loss = -torch.nn.functional.logsigmoid(beta * adv).mean()
    return loss


def main():
    parser = argparse.ArgumentParser(description="MiniMind-V TicTacToe DPO")
    parser.add_argument("--train_jsonl", type=str, default="VLA/data/ttt_train_dpo.jsonl")
    parser.add_argument("--val_jsonl", type=str, default="VLA/data/ttt_val_dpo.jsonl")
    parser.add_argument("--out_dir", type=str, default="VLA/models")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--use_reference", action="store_true")
    args = parser.parse_args()

    device = args.device
    device_type = "cuda" if "cuda" in device else "cpu"
    if device_type == "cpu":
        ctx = nullcontext()
    else:
        amp_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
        ctx = torch.cuda.amp.autocast(dtype=amp_dtype)

    tokenizer = AutoTokenizer.from_pretrained('model')
    config = VLMConfig(hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_LAYERS, max_seq_len=MAX_SEQ_LEN, use_moe=USE_MOE)
    model = MiniMindVLMWithAction(config, vision_model_path="model/vision_model/clip-vit-base-patch16").to(device)

    # 加载固定的预训练权重（语言骨干），strict=False 允许视觉键缺失
    if not os.path.exists(CKPT_DEFAULT):
        raise FileNotFoundError(f"Missing pretrain ckpt: {CKPT_DEFAULT}")
    state = torch.load(CKPT_DEFAULT, map_location=device)
    model.load_state_dict(state, strict=False)

    reference_model = None
    if args.use_reference:
        reference_model = MiniMindVLMWithAction(config, vision_model_path="model/vision_model/clip-vit-base-patch16").to(device)
        reference_model.load_state_dict(state, strict=False)
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad = False

    train_ds = TicTacToeVLMDPODataset(
        args.train_jsonl, tokenizer, preprocess=model.processor,
        image_special_token=config.image_special_token, max_length=args.max_seq_len,
    )
    val_raw_ds = None
    if os.path.exists("VLA/data/ttt_val_alpaca.jsonl"):
        from VLA.envs.ttt_dataset import TicTacToeVLMDataset
        val_raw_ds = TicTacToeVLMDataset(
            "VLA/data/ttt_val_alpaca.jsonl", tokenizer, preprocess=model.processor,
            image_special_token=config.image_special_token, max_length=args.max_seq_len,
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    if val_raw_ds is not None:
        val_loader = DataLoader(val_raw_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        val_loader = None

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    os.makedirs(args.out_dir, exist_ok=True)
    total_steps = max(1, args.epochs * len(train_loader))
    step_count = 0
    pbar = None

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"DPO Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            step_count += 1
            lr = get_lr(step_count, total_steps, args.learning_rate)
            for g in optimizer.param_groups:
                g['lr'] = lr

            with ctx:
                loss = dpo_loss(model, batch, beta=args.beta, reference_model=reference_model, device=device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({
                'loss': f"{float(loss):.3f}",
                'lr': f"{lr:.2e}",
            })

            if (step_count % args.eval_interval == 0) and (val_loader is not None):
                model.eval()
                metrics = evaluate_model(model, val_loader, device)
                print({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()})
                model.train()

            if step_count % args.save_interval == 0:
                model.eval()
                moe_path = '_moe' if config.use_moe else ''
                ckp = f"{args.out_dir}/dpo_vlm_ttt_{config.hidden_size}{moe_path}.pth"
                state_dict = model.state_dict()
                clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vision_encoder.')}
                clean_state_dict = {k: v.half() for k, v in clean_state_dict.items()}
                torch.save(clean_state_dict, ckp)
                model.train()

        # epoch end save
        model.eval()
        moe_path = '_moe' if config.use_moe else ''
        ckp = f"{args.out_dir}/dpo_vlm_ttt_{config.hidden_size}{moe_path}.pth"
        state_dict = model.state_dict()
        clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vision_encoder.')}
        clean_state_dict = {k: v.half() for k, v in clean_state_dict.items()}
        torch.save(clean_state_dict, ckp)


if __name__ == "__main__":
    main()
