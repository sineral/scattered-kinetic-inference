"""Evaluate a trained DeepSets checkpoint (force-include t=0).

EVAL_TPS is the number of extra non-zero timesteps besides t=0.
Total timesteps = EVAL_TPS + 1; flat set size = (EVAL_TPS + 1) * 4.

Usage (from repo root):
    python src/eval_deepsets.py
    python src/eval_deepsets.py --eval-tps 2 --manual-noise 0.01
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from eval_utils import (
    confusion_and_per_class,
    noise_tag,
    save_analysis_plot,
    save_confusion_csv,
    topk_accuracy,
)
from models.base import get_model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DeepSets")
    DEFAULT_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "deepsets", "best_model.pth")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--eval-tps", type=int, default=2, help="Extra non-zero timesteps besides t=0")
    p.add_argument("--dataset-tps", type=int, default=6)
    p.add_argument("--dataset-noise", type=float, default=0)
    p.add_argument("--manual-noise", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"eval_ds_{ts}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(os.path.join(log_dir, f"eval_deepsets_{ts}.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger, ts


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_x2_slice(x2_dict, tps, noise):
    tp_key = tps if tps in x2_dict else str(tps)
    if tp_key not in x2_dict:
        raise KeyError(f"tps={tps} not in keys {list(x2_dict.keys())}")
    noise_dict = x2_dict[tp_key]
    cands = []
    for cast in (lambda x: x, int, float, str):
        try:
            cands.append(cast(noise))
        except (TypeError, ValueError):
            pass
    try:
        cands.append(int(round(float(noise))))
    except (TypeError, ValueError):
        pass
    matched = next((k for k in cands if k in noise_dict), None)
    if matched is None:
        raise KeyError(f"noise={noise} not in keys {list(noise_dict.keys())}")
    return tp_key, matched, noise_dict[matched]


class TestFlatDeepSetsDataset(Dataset):
    def __init__(self, x1, x2, y, mean, std, extra_tps=2, noise_std=0.0):
        self.x1 = torch.tensor(x1, dtype=torch.float32)
        self.x2 = torch.tensor(x2, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long).squeeze()
        self.N, self.T_total, _ = self.x2.shape
        self.mean = mean.cpu() if isinstance(mean, torch.Tensor) else torch.tensor(mean)
        self.std = std.cpu() if isinstance(std, torch.Tensor) else torch.tensor(std)
        self.extra_tps = min(extra_tps, max(self.T_total - 1, 0))
        self.noise_std = noise_std

    def _build(self, x1, x2, T):
        x2_r = x2.view(T, 4, 3)
        x1_r = x1.unsqueeze(0).unsqueeze(-1).expand(T, 4, 1)
        return torch.cat([x1_r, x2_r], dim=-1).view(T * 4, 4)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x1_i, x2_i = self.x1[idx], self.x2[idx]
        g = torch.Generator().manual_seed(idx)
        if self.T_total <= 1 or self.extra_tps <= 0:
            indices = torch.tensor([0], dtype=torch.long)
        else:
            rest = torch.randperm(self.T_total - 1, generator=g)[: self.extra_tps] + 1
            indices = torch.cat([torch.tensor([0], dtype=torch.long), rest])
        indices, _ = torch.sort(indices)
        x2_i = x2_i[indices]
        points = self._build(x1_i, x2_i, indices.numel())
        points = (points - self.mean) / (self.std + 1e-8)
        if self.noise_std > 0:
            noise = torch.randn_like(points) * self.noise_std
            noise[:, 0:2] = 0.0
            points = points + noise
        return points, self.y[idx]


def collate(batch):
    pts, labels = zip(*batch)
    max_len = max(p.shape[0] for p in pts)
    B = len(batch)
    padded = torch.zeros(B, max_len, 4)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, p in enumerate(pts):
        padded[i, : p.shape[0]] = p
        mask[i, : p.shape[0]] = True
    return padded, mask, torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    logits_all, preds, labels = [], [], []
    for x, mask, y in loader:
        x, mask = x.to(device), mask.to(device)
        logits = model(x, mask)
        logits_all.append(logits.cpu().numpy())
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    logits = np.vstack(logits_all)
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    stats = confusion_and_per_class(preds, labels)
    stats["accuracy"] = float((preds == labels).mean())
    stats["top3_accuracy"] = topk_accuracy(logits, labels, 3)
    return stats


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(PROJECT_ROOT, args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "results", "deepsets")
    logger, ts = setup_logging(os.path.join(PROJECT_ROOT, "logs", "eval"))
    logger.info(f"ckpt={ckpt_path}  eval_tps={args.eval_tps}  noise={args.manual_noise}")

    data_dir = os.path.join(PROJECT_ROOT, "data", "nature_data")
    x1 = load_pkl(os.path.join(data_dir, "x1_test_M1_M20_train_val_test_set.pkl"))
    x2_dict = load_pkl(os.path.join(data_dir, "x2_test_M1_M20_train_val_test_set.pkl"))
    y = load_pkl(os.path.join(data_dir, "y_test_M1_M20_train_val_test_set.pkl"))
    tp_key, err_key, x2 = resolve_x2_slice(x2_dict, args.dataset_tps, args.dataset_noise)

    ds = TestFlatDeepSetsDataset(
        x1, x2, y, mean=ckpt["mean"], std=ckpt["std"],
        extra_tps=args.eval_tps, noise_std=args.manual_noise,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers,
    )

    model = get_model(
        "deepsets", in_dim=4, num_classes=20,
        dim_hidden=128, latent_dim=128, dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    results = run_eval(model, loader, device)
    logger.info(f"Top-1={results['accuracy']:.4f}  Top-3={results['top3_accuracy']:.4f}")

    tag = noise_tag(err_key, args.manual_noise)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"confusion_deepsets_{args.eval_tps}pts_{tag}_{ts}.csv")
    save_confusion_csv(
        results["confusion_matrix"], results["mechs"], csv_path,
        extra={"Top-1 Accuracy": results["accuracy"], "Top-3 Accuracy": results["top3_accuracy"]},
    )
    save_analysis_plot(
        results["confusion_matrix"], results["mechs"], results["per_class_accuracy"],
        title=f"DeepSets  extra_tps={args.eval_tps}  {tag}",
        path=os.path.join(out_dir, f"plot_deepsets_{args.eval_tps}pts_{tag}_{ts}.png"),
    )
    logger.info(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
