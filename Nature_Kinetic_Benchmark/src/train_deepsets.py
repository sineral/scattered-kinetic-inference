"""Train DeepSets baseline on Nature M1-M20 data.

Input: unordered set of [cat0, t, S, P] points (in_dim=4).
Train: random sequence length 3-20, discrete noise {0, 0.5, 1, 2}%.
Val: same length range, noise fixed at 1%.

Usage (from repo root):
    python src/train_deepsets.py
    python src/train_deepsets.py --gpu 0 --epochs 500
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


def parse_args():
    p = argparse.ArgumentParser(description="Train DeepSets on Nature data")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--min-pts", type=int, default=3)
    p.add_argument("--max-pts", type=int, default=20)
    return p.parse_args()


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_deepsets_{timestamp}.log")
    logger = logging.getLogger("train_deepsets")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger, timestamp


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class AugmentedFlatDeepSetsDataset(Dataset):
    """Flatten 4 kinetic traces into an unordered set of [cat0, t, S, P]."""

    def __init__(
        self,
        x1_data,
        x2_data,
        y_data,
        mean=None,
        std=None,
        min_pts=3,
        max_pts=20,
        error_range=(0, 0.5, 1, 2),
        augment=True,
    ):
        self.x1 = torch.tensor(x1_data, dtype=torch.float32)
        self.x2 = torch.tensor(x2_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long).squeeze()
        self.N, self.T_total, _ = self.x2.shape
        self.augment = augment
        self.error_range = list(error_range)
        self.min_pts = max(1, min_pts)
        self.max_pts = min(max_pts, self.T_total)
        if mean is not None and std is not None:
            self.mean, self.std = mean, std
        else:
            all_pts = self._build_flat_sets(self.x1, self.x2, self.T_total)
            flat = all_pts.view(-1, 4)
            self.mean = flat.mean(dim=0)
            self.std = flat.std(dim=0)

    def _build_flat_sets(self, x1, x2, T):
        if x1.dim() == 1:
            x2_r = x2.view(T, 4, 3)
            x1_r = x1.unsqueeze(0).unsqueeze(-1).expand(T, 4, 1)
            return torch.cat([x1_r, x2_r], dim=-1).view(T * 4, 4)
        B = x2.shape[0]
        x2_r = x2.view(B, T, 4, 3)
        x1_r = x1.unsqueeze(1).unsqueeze(-1).expand(-1, T, 4, 1)
        return torch.cat([x1_r, x2_r], dim=-1).view(B, T * 4, 4)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x1_i, x2_i = self.x1[idx], self.x2[idx]
        if self.augment:
            T_sample = torch.randint(self.min_pts, self.max_pts + 1, (1,)).item()
            indices = torch.randperm(self.T_total)[:T_sample]
            x2_i = x2_i[indices]
            points = self._build_flat_sets(x1_i, x2_i, T_sample)
            points = (points - self.mean) / (self.std + 1e-8)
            err = random.choice(self.error_range)
            if err > 0:
                noise = torch.randn_like(points) * (err / 100.0)
                noise[:, 0:2] = 0.0  # keep cat0 and t clean
                points = points + noise
        else:
            points = self._build_flat_sets(x1_i, x2_i, self.T_total)
            points = (points - self.mean) / (self.std + 1e-8)
        return points, self.y[idx]


def deepsets_collate(batch):
    points_list, labels = zip(*batch)
    max_len = max(pts.shape[0] for pts in points_list)
    B = len(batch)
    padded = torch.zeros(B, max_len, 4)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, pts in enumerate(points_list):
        L = pts.shape[0]
        padded[i, :L] = pts
        mask[i, :L] = True
    return padded, mask, torch.tensor(labels, dtype=torch.long)


def train(args):
    if args.gpu is not None:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    data_dir = os.path.join(PROJECT_ROOT, "data", "nature_data")
    out_dir = os.path.join(PROJECT_ROOT, "checkpoints", "deepsets")
    log_dir = os.path.join(PROJECT_ROOT, "logs", "deepsets")
    os.makedirs(out_dir, exist_ok=True)
    logger, timestamp = setup_logging(log_dir)

    error_range = [0, 0.5, 1, 2]
    logger.info("=" * 72)
    logger.info(
        f"DeepSets | pts={args.min_pts}-{args.max_pts} | "
        f"train noise={error_range}% | val noise=1%"
    )
    logger.info("=" * 72)

    x1_tr = load_pkl(os.path.join(data_dir, "x1_train_M1_M20_train_val_test_set.pkl"))
    x2_tr = load_pkl(os.path.join(data_dir, "x2_train_M1_M20_train_val_test_set.pkl"))
    y_tr = load_pkl(os.path.join(data_dir, "y_train_M1_M20_train_val_test_set.pkl"))
    x1_val = load_pkl(os.path.join(data_dir, "x1_val_M1_M20_train_val_test_set.pkl"))
    x2_val = load_pkl(os.path.join(data_dir, "x2_val_M1_M20_train_val_test_set.pkl"))
    y_val = load_pkl(os.path.join(data_dir, "y_val_M1_M20_train_val_test_set.pkl"))
    logger.info(f"Train={len(x1_tr):,}  Val={len(x1_val):,}")

    train_ds = AugmentedFlatDeepSetsDataset(
        x1_tr, x2_tr, y_tr, augment=True,
        min_pts=args.min_pts, max_pts=args.max_pts, error_range=error_range,
    )
    val_ds = AugmentedFlatDeepSetsDataset(
        x1_val, x2_val, y_val, augment=True,
        min_pts=args.min_pts, max_pts=args.max_pts, error_range=[1],
        mean=train_ds.mean, std=train_ds.std,
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=deepsets_collate, num_workers=args.num_workers,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=deepsets_collate, num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        "deepsets", in_dim=4, num_classes=20,
        dim_hidden=128, latent_dim=128, dropout=0.1,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    logger.info(f"device={device}  params={sum(p.numel() for p in model.parameters()):,}")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, mask, y in train_dl:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            n += y.size(0)
        train_acc = correct / n

        model.eval()
        val_preds, val_y = [], []
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for x, mask, y in val_dl:
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                logits = model(x, mask)
                val_loss += criterion(logits, y).item() * y.size(0)
                val_preds.extend(logits.argmax(-1).cpu().numpy())
                val_y.extend(y.cpu().numpy())
                vn += y.size(0)
        val_acc = accuracy_score(val_y, val_preds)
        val_f1 = f1_score(val_y, val_preds, average="macro")
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"TrLoss={total_loss / n:.4f} TrAcc={train_acc:.4f} | "
            f"ValLoss={val_loss / vn:.4f} ValAcc={val_acc:.4f} ValF1={val_f1:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            path = os.path.join(out_dir, "best_model.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "in_dim": 4,
                    "num_classes": 20,
                    "model_type": "deepsets",
                    "best_val_accuracy": val_acc,
                    "mean": train_ds.mean,
                    "std": train_ds.std,
                },
                path,
            )
            logger.info(f"  saved best model -> {path}")

    logger.info(f"Done. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    train(parse_args())
