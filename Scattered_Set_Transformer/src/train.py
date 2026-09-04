"""Train Set Transformer on scattered kinetic groups (M1-M20).

Train noise is drawn uniformly from {0, 0.5, 1, 2}%; validation uses 1%.
Architecture default: hidden 256, 8 heads, 6 ISAB layers, 16 inducing points, dropout 0.1.

Usage (from repo root):
    python src/train.py --data-dir data/default --save-path checkpoints/default_member_0.pth --gpu 0
    python src/train.py --data-dir data/combined --save-path checkpoints/combined_member_0.pth --gpu 0
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.chdir(PROJECT_ROOT)

from models.base import get_model
from utils.scattered_dataset import AugmentedScatteredDataset, get_collate_fn

DEFAULT_FEATURES = list(range(20))


def parse_args():
    p = argparse.ArgumentParser(description="Train Set Transformer on scattered kinetics")
    p.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data", "default"))
    p.add_argument(
        "--save-path",
        default=os.path.join(PROJECT_ROOT, "checkpoints", "default_member_0.pth"),
    )
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--n-min", type=int, default=6)
    p.add_argument("--n-max", type=int, default=15)
    p.add_argument("--samples-per-group", type=int, default=10)
    p.add_argument("--error-range", default="0,0.005,0.01,0.02")
    p.add_argument("--dim-hidden", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-inducing", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


def setup_logging():
    log_dir = os.path.join(PROJECT_ROOT, "logs", "train")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("train_scattered")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"train_{ts}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger, ts


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    logger, ts = setup_logging()
    ckpt_path = args.save_path if os.path.isabs(args.save_path) else os.path.join(PROJECT_ROOT, args.save_path)
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    n_workers = args.num_workers if args.num_workers is not None else min(multiprocessing.cpu_count() // 4, 8)
    feat_idx = DEFAULT_FEATURES

    logger.info(f"data_dir={args.data_dir}")
    logger.info(f"save_path={ckpt_path}")
    logger.info(f"device={device} workers={n_workers}")
    logger.info(f"error_range={args.error_range} n=[{args.n_min},{args.n_max}]")

    model_params = {
        "in_dim": len(feat_idx),
        "num_classes": 20,
        "dim_hidden": args.dim_hidden,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "num_inducing": args.num_inducing,
        "dropout": args.dropout,
    }
    logger.info(f"model_params={model_params}")

    collate = get_collate_fn(args.n_min, args.n_max, feat_idx)
    train_ds = AugmentedScatteredDataset(
        os.path.join(args.data_dir, "train.parquet"),
        repeat_factor=args.samples_per_group,
        error_range=args.error_range,
        is_train=True,
        noise_std=0.01,
    )
    val_ds = AugmentedScatteredDataset(
        os.path.join(args.data_dir, "val.parquet"),
        repeat_factor=1,
        is_train=False,
        noise_std=0.01,
    )
    logger.info(f"train groups={len(train_ds) // args.samples_per_group} val groups={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=n_workers,
        collate_fn=collate, pin_memory=device.type == "cuda",
        persistent_workers=n_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=n_workers,
        collate_fn=collate, pin_memory=device.type == "cuda",
        persistent_workers=n_workers > 0,
    )

    model = get_model("settransformer", **model_params).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"parameters={n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=device.type == "cuda")
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for features, labels in pbar:
            optimizer.zero_grad(set_to_none=True)
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(features)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(enabled=device.type == "cuda"):
                    logits = model(features)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(val_total, 1)
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"TrLoss={train_loss / max(total, 1):.4f} | "
            f"ValLoss={val_loss / max(val_total, 1):.4f} ValAcc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "model_type": "settransformer",
                    "model_params": model_params,
                    "sampling_cfg": {
                        "n_min": args.n_min,
                        "n_max": args.n_max,
                        "error_range": args.error_range,
                    },
                },
                ckpt_path,
            )
            logger.info(f"  saved best model -> {ckpt_path}")

    logger.info(f"Done. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
