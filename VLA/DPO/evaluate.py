import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.model_vlm import VLMConfig
from VLA.envs.model_wrapper import MiniMindVLMWithAction
from VLA.envs.ttt_dataset import TicTacToeVLMDataset
from VLA.SFT.evaluate import evaluate_model


def load_model(ckpt: str, device: str, hidden_size: int = 768, num_layers: int = 16, max_seq_len: int = 512, use_moe: bool = False):
    config = VLMConfig(hidden_size=hidden_size, num_hidden_layers=num_layers, max_seq_len=max_seq_len, use_moe=use_moe)
    model = MiniMindVLMWithAction(config, vision_model_path="model/vision_model/clip-vit-base-patch16").to(device)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPO-trained VLM on TicTacToe")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, default="VLA/data/ttt_val_alpaca.jsonl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_seq_len", type=int, default=512)
    args = parser.parse_args()

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained('model')
    model = load_model(args.ckpt, device, max_seq_len=args.max_seq_len)

    ds = TicTacToeVLMDataset(
        args.val_jsonl,
        tokenizer,
        preprocess=model.processor,
        image_special_token=model.config.image_special_token,
        max_length=args.max_seq_len,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    metrics = evaluate_model(model, dl, device)
    print(metrics)


if __name__ == "__main__":
    main()
