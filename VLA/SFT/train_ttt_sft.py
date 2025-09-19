import argparse
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from tqdm import tqdm
from contextlib import nullcontext
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from model.model_vlm import VLMConfig
from VLA.envs.model_wrapper import MiniMindVLMWithAction
from VLA.envs.ttt_dataset import TicTacToeVLMDataset
from VLA.SFT.early_stopping import EarlyStopping
from VLA.SFT.evaluate import evaluate_model

# 减少 fork 后 tokenizers 的并行警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 减少碎片导致的 OOM（见 PyTorch 文档）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 固定与 MiniMind2-V 匹配的模型与权重配置（写死，保证最大匹配）
CKPT_DEFAULT = "MiniMind2-V/pytorch_model.bin"
HIDDEN_SIZE = 768
NUM_LAYERS = 16
MAX_SEQ_LEN = 512
USE_MOE = False

def Logger(content):
    if (not ddp) or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb, val_loader=None, early_stopping=None):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for step, batch in enumerate(pbar):
        # 兼容 Dataset 的 dict 返回
        X = batch['input_ids'].to(args.device)
        Y = batch['labels'].to(args.device)
        loss_mask = batch['loss_mask'].to(args.device)
        pixel_values = batch['pixel_values'].to(args.device)
        action = batch['action'].to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X, pixel_values=pixel_values)
            # 语言建模损失
            loss_lm_tok = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss_lm = (loss_lm_tok * loss_mask).sum() / (loss_mask.sum() + 1e-8)

            # 动作头前向（仅最后一个 token 的隐状态）
            last_hidden = res.last_hidden_state[:, -1, :]  # [B, H]
            ah = model.module.action_head if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.action_head
            action_logits = ah(last_hidden)       # [B, 9]
            loss_action = F.cross_entropy(action_logits, action)

            # 联合损失
            loss = loss_lm + res.aux_loss + 0.5 * loss_action
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # 梯度裁剪（模型 + 动作头）
            params_to_clip = list(model.parameters())
            torch.nn.utils.clip_grad_norm_(params_to_clip, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            # 动作准确率
            with torch.no_grad():
                acc = (action_logits.argmax(dim=-1) == action).float().mean().item()
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'act_acc': f'{acc:.2f}',
            })

            if (wandb is not None) and ((not ddp) or dist.get_rank() == 0):
                wandb.log({"loss": loss,
                           "loss_lm": loss_lm,
                           "loss_action": loss_action,
                           "action_acc": acc,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 验证和早停检查
        if val_loader is not None and early_stopping is not None and (step + 1) % args.val_interval == 0:
            val_metrics = evaluate_model(model, val_loader, args.device, max_eval_steps=args.max_val_steps)
            
            # 记录到wandb
            if (wandb is not None) and ((not ddp) or dist.get_rank() == 0):
                wandb.log({
                    "val_action_accuracy": val_metrics['action_accuracy'],
                    "val_lm_loss": val_metrics['avg_lm_loss'],
                    "val_action_loss": val_metrics['avg_action_loss'],
                })
            
            # 早停检查
            if early_stopping.update(val_metrics['action_accuracy']):
                return True  # 返回True表示早停
            
            model.train()  # 恢复训练模式

        if (step + 1) % args.save_interval == 0 and ((not ddp) or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if model_config.use_moe else ''
            ckp = f'{args.save_dir}/sft_vlm_ttt_{model_config.hidden_size}{moe_path}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            clean_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith('vision_encoder.')
            }
            clean_state_dict = {k: v.half() for k, v in clean_state_dict.items()}
            torch.save(clean_state_dict, ckp)
            # 额外保存动作头（可选）
            # 动作头已包含在统一模型中，无需单独保存
            model.train()
    
    pbar.close()
    return False  # 返回False表示正常完成


def init_model(model_config: VLMConfig):
    tokenizer = AutoTokenizer.from_pretrained('model')
    pre_ckp = CKPT_DEFAULT

    if not os.path.exists(pre_ckp):
        raise FileNotFoundError(f"未找到固定预训练权重：{pre_ckp}，请确认路径。")

    model = MiniMindVLMWithAction(model_config, vision_model_path="model/vision_model/clip-vit-base-patch16")
    state_dict = torch.load(pre_ckp, map_location=args.device)
    # 严格匹配当前固定配置；允许缺少 vision_encoder.* 等视觉相关键（strict=False）
    model.load_state_dict(state_dict, strict=False)

    Logger(f'VLM可训练参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    _, preprocess = model.vision_encoder, model.processor
    return model.to(args.device), tokenizer, preprocess


def init_distributed_mode():
    if not ddp:
        return
    global ddp_local_rank, DEVICE
    dist.init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V TicTacToe SFT")
    parser.add_argument("--out_dir", type=str, default="VLA/models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", default=True, action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-TTT")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="VLA/data/ttt_train_alpaca.jsonl", help="训练集jsonl文件路径")
    parser.add_argument("--val_data_path", type=str, default="VLA/data/ttt_val_alpaca.jsonl", help="验证集jsonl文件路径")
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN, help="训练时最大序列长度（截断），建议 512~2048 之间")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=50, help="验证间隔步数")
    parser.add_argument("--max_val_steps", type=int, default=50, help="每次验证的最大步数，None表示验证整个验证集")
    parser.add_argument("--early_stop_window", type=int, default=20, help="早停滑动窗口大小")
    parser.add_argument("--early_stop_threshold", type=float, default=0.95, help="早停准确率阈值")
    parser.add_argument("--early_stop_patience", type=int, default=3, help="早停耐心值（连续满足阈值次数）")
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    model_config = VLMConfig(hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_LAYERS,
                             max_seq_len=MAX_SEQ_LEN, use_moe=USE_MOE)
    max_seq_len = args.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = (
        f"Epoch-{args.epochs} BatchSize-{args.batch_size} LR-{args.learning_rate}"
    )

    # 根据 dtype 设定 autocast 精度（bfloat16/float16），CPU 则禁用
    if device_type == "cpu":
        ctx = nullcontext()
    else:
        amp_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
        ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
    ddp = int(os.environ.get("RANK", -1)) != -1 or args.ddp
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and ((not ddp) or ddp_local_rank == 0):
        import wandb
        import glob
        import os
        # 筛选 VLA/ 下的 .py 文件，排除 VLA/data
        wandb.init(project=args.wandb_project, 
               name=args.wandb_run_name, 
               save_code=True,
               )
    else:
        wandb = None

    model, tokenizer, preprocess = init_model(model_config)

    # 使用封装内置的动作头

    train_ds = TicTacToeVLMDataset(
        args.data_path,
        tokenizer,
        preprocess=preprocess,
        image_special_token=model_config.image_special_token,
        max_length=max_seq_len,
    )
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )

    # 验证集
    val_ds = TicTacToeVLMDataset(
        args.val_data_path,
        tokenizer,
        preprocess=preprocess,
        image_special_token=model_config.image_special_token,
        max_length=max_seq_len,
    )
    val_sampler = DistributedSampler(val_ds) if ddp else None
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=val_sampler,
    )

    # 仅在 float16 下使用 GradScaler；bfloat16 不需要
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 直接优化封装模型的所有参数
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 初始化早停
    early_stopping = EarlyStopping(
        window_size=args.early_stop_window,
        threshold=args.early_stop_threshold,
        patience=args.early_stop_patience
    )

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        if ddp and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        
        # 训练一个epoch，如果早停则退出
        should_stop = train_epoch(epoch, wandb, val_loader, early_stopping)
        if should_stop:
            break
