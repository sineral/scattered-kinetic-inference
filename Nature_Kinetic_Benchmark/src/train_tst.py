"""Train Time-Series Transformer (early-fuse, Nature-aligned).

Default architecture matches the paper comparison:
  d=128, 3 layers, 4 heads, attention readout, force include t=0.
Train: tps in {3..10, 15, 20}, noise {0, 0.5, 1, 2}%, batch 2048, Adam 1e-4,
early stopping patience 300, up to 3000 epochs.

Usage (from repo root):
    python src/train_tst.py
    python src/train_tst.py --readout attn --dim-hidden 128 --num-layers 3 --gpu 0
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import pickle
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from models.base import get_model
from models.ts_transformer import READOUT_CHOICES


def parse_args():
    p = argparse.ArgumentParser(description="Train TST on Nature data")
    p.add_argument("--readout", default="attn", choices=list(READOUT_CHOICES))
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=300)
    p.add_argument("--dim-hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--no-force-t0", action="store_true")
    return p.parse_args()


def setup_logging(log_dir, prefix):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"train_tst_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(os.path.join(log_dir, f"{prefix}_{timestamp}.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger, timestamp


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class AugmentedTSTDataset(Dataset):
    def __init__(
        self,
        x1,
        x2,
        y,
        augment=True,
        tps=None,
        error_range=(0, 0.5, 1, 2),
        force_include_t0=True,
        mean_x2=None,
        std_x2=None,
    ):
        self.x1 = torch.tensor(x1, dtype=torch.float32)
        self.x2 = torch.tensor(x2, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long).squeeze()
        self.N, self.T_total, self.feat_dim = self.x2.shape
        self.augment = augment
        self.tps = tps or [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.error_range = list(error_range)
        self.force_include_t0 = force_include_t0
        if mean_x2 is not None:
            self.mean_x2, self.std_x2 = mean_x2, std_x2
        else:
            flat = self.x2.view(-1, self.feat_dim)
            self.mean_x2 = flat.mean(dim=0)
            self.std_x2 = flat.std(dim=0)

    def __len__(self):
        return self.N

    def _sample_indices(self, T_sample):
        T_sample = min(T_sample, self.T_total)
        if self.force_include_t0 and T_sample >= 1:
            if T_sample == 1:
                return torch.tensor([0], dtype=torch.long)
            rest = torch.randperm(self.T_total - 1)[: T_sample - 1] + 1
            indices = torch.cat([torch.tensor([0], dtype=torch.long), rest])
            indices, _ = torch.sort(indices)
            return indices
        indices = torch.randperm(self.T_total)[:T_sample]
        indices, _ = torch.sort(indices)
        return indices

    def __getitem__(self, idx):
        x1_i, x2_i = self.x1[idx], self.x2[idx]
        if self.augment:
            T_sample = min(random.choice(self.tps), self.T_total)
            x2_i = x2_i[self._sample_indices(T_sample)]
            x2_i = (x2_i - self.mean_x2) / (self.std_x2 + 1e-8)
            err = random.choice(self.error_range)
            if err > 0:
                noise = torch.randn_like(x2_i) * (err / 100.0)
                noise[:, [0, 3, 6, 9]] = 0.0  # time channels
                x2_i = x2_i + noise
        else:
            x2_i = (x2_i - self.mean_x2) / (self.std_x2 + 1e-8)
        return x1_i, x2_i, self.y[idx]


def tst_collate(batch):
    x1_list, x2_list, labels = zip(*batch)
    max_len = max(p.shape[0] for p in x2_list)
    B, feat = len(batch), x2_list[0].shape[-1]
    padded = torch.zeros(B, max_len, feat)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, pts in enumerate(x2_list):
        L = pts.shape[0]
        padded[i, :L] = pts
        mask[i, :L] = True
    return (torch.stack(x1_list), padded), mask, torch.tensor(labels, dtype=torch.long)


def train(args):
    if args.gpu is not None:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    force_t0 = not args.no_force_t0
    data_dir = os.path.join(PROJECT_ROOT, "data", "nature_data")
    run = "tst"
    out_dir = os.path.join(PROJECT_ROOT, "checkpoints", run)
    log_dir = os.path.join(PROJECT_ROOT, "logs", run)
    os.makedirs(out_dir, exist_ok=True)
    logger, timestamp = setup_logging(log_dir, "train_tst")

    tps_list = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    error_range = [0, 0.5, 1, 2]
    logger.info("=" * 72)
    logger.info(
        f"TST early-fuse | readout={args.readout} | d={args.dim_hidden} "
        f"L={args.num_layers} | force_t0={force_t0}"
    )
    logger.info("=" * 72)

    x1_tr = load_pkl(os.path.join(data_dir, "x1_train_M1_M20_train_val_test_set.pkl"))
    x2_tr = load_pkl(os.path.join(data_dir, "x2_train_M1_M20_train_val_test_set.pkl"))
    y_tr = load_pkl(os.path.join(data_dir, "y_train_M1_M20_train_val_test_set.pkl"))
    x1_val = load_pkl(os.path.join(data_dir, "x1_val_M1_M20_train_val_test_set.pkl"))
    x2_val = load_pkl(os.path.join(data_dir, "x2_val_M1_M20_train_val_test_set.pkl"))
    y_val = load_pkl(os.path.join(data_dir, "y_val_M1_M20_train_val_test_set.pkl"))

    train_ds = AugmentedTSTDataset(
        x1_tr, x2_tr, y_tr, augment=True, tps=tps_list,
        error_range=error_range, force_include_t0=force_t0,
    )
    val_ds = AugmentedTSTDataset(
        x1_val, x2_val, y_val, augment=True, tps=tps_list, error_range=[1],
        force_include_t0=force_t0, mean_x2=train_ds.mean_x2, std_x2=train_ds.std_x2,
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=tst_collate,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=tst_collate,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        "tst",
        in_dim=0,
        dim_x1=4,
        dim_x2=12,
        num_classes=20,
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=0.1,
        readout=args.readout,
    ).to(device)
    if args.gpu is None and torch.cuda.device_count() > 1:
        logger.info(f"DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    logger.info(f"device={device} params={sum(p.numel() for p in model.parameters()):,}")

    config = {
        "model": {
            "name": "tst",
            "dim_x1": 4,
            "dim_x2": 12,
            "dim_hidden": args.dim_hidden,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "num_classes": 20,
            "dropout": 0.1,
            "readout": args.readout,
        }
    }

    best_acc, patience_c = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for inputs, mask, y in train_dl:
            x1, x2 = inputs
            inputs_d = (x1.to(device, non_blocking=True), x2.to(device, non_blocking=True))
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs_d, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            n += y.size(0)

        model.eval()
        val_preds, val_y = [], []
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for inputs, mask, y in val_dl:
                x1, x2 = inputs
                inputs_d = (x1.to(device, non_blocking=True), x2.to(device, non_blocking=True))
                mask = mask.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(inputs_d, mask)
                val_loss += criterion(logits, y).item() * y.size(0)
                val_preds.extend(logits.argmax(-1).cpu().numpy())
                val_y.extend(y.cpu().numpy())
                vn += y.size(0)
        val_acc = accuracy_score(val_y, val_preds)
        val_f1 = f1_score(val_y, val_preds, average="macro")
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"TrLoss={total_loss / n:.4f} TrAcc={correct / n:.4f} | "
            f"ValLoss={val_loss / vn:.4f} ValAcc={val_acc:.4f} ValF1={val_f1:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            patience_c = 0
            path = os.path.join(out_dir, "best_model.pth")
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(
                {
                    "model_state_dict": state,
                    "epoch": epoch,
                    "model_type": "tst",
                    "best_val_accuracy": val_acc,
                    "mean_x2": train_ds.mean_x2,
                    "std_x2": train_ds.std_x2,
                    "config": config,
                },
                path,
            )
            logger.info(f"  saved best model -> {path}")
        else:
            patience_c += 1
            if patience_c >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"Done. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    train(parse_args())
