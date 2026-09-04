"""Evaluate a trained TST checkpoint (force-include t=0).

EVAL_TPS is the number of extra non-zero timesteps besides t=0.

Usage (from repo root):
    python src/eval_tst.py
    python src/eval_tst.py --eval-tps 6 --manual-noise 0.01
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
    p = argparse.ArgumentParser(description="Evaluate TST")
    DEFAULT_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "tst", "best_model.pth")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--eval-tps", type=int, default=6)
    p.add_argument("--dataset-tps", type=int, default=6)
    p.add_argument("--dataset-noise", type=float, default=0)
    p.add_argument("--manual-noise", type=float, default=0.0)
    p.add_argument("--no-force-t0", action="store_true")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"eval_tst_{ts}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(os.path.join(log_dir, f"eval_tst_{ts}.log"))
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


class TSTTestDataset(Dataset):
    def __init__(self, x1, x2, y, mean_x2, std_x2, extra_tps=6, force_t0=True, noise_std=0.0):
        self.x1 = torch.tensor(x1, dtype=torch.float32)
        self.x2 = torch.tensor(x2, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long).squeeze()
        self.N, self.T_total, _ = self.x2.shape
        self.mean_x2 = mean_x2.cpu() if isinstance(mean_x2, torch.Tensor) else torch.tensor(mean_x2)
        self.std_x2 = std_x2.cpu() if isinstance(std_x2, torch.Tensor) else torch.tensor(std_x2)
        self.force_t0 = force_t0
        self.noise_std = noise_std
        self.extra_tps = min(extra_tps, max(self.T_total - 1, 0) if force_t0 else self.T_total)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x1_i, x2_i = self.x1[idx], self.x2[idx]
        g = torch.Generator().manual_seed(idx)
        if self.force_t0:
            if self.T_total <= 1 or self.extra_tps <= 0:
                indices = torch.tensor([0], dtype=torch.long)
            else:
                rest = torch.randperm(self.T_total - 1, generator=g)[: self.extra_tps] + 1
                indices = torch.cat([torch.tensor([0], dtype=torch.long), rest])
        else:
            indices = torch.randperm(self.T_total, generator=g)[: self.extra_tps]
        indices, _ = torch.sort(indices)
        x2_i = x2_i[indices]
        x2_i = (x2_i - self.mean_x2) / (self.std_x2 + 1e-8)
        if self.noise_std > 0:
            noise = torch.randn_like(x2_i) * self.noise_std
            noise[:, [0, 3, 6, 9]] = 0.0
            x2_i = x2_i + noise
        return x1_i, x2_i, self.y[idx]


def tst_collate(batch):
    x1_list, x2_list, labels = zip(*batch)
    max_len = max(p.shape[0] for p in x2_list)
    B, feat = len(batch), x2_list[0].shape[-1]
    padded = torch.zeros(B, max_len, feat)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, pts in enumerate(x2_list):
        padded[i, : pts.shape[0]] = pts
        mask[i, : pts.shape[0]] = True
    return (torch.stack(x1_list), padded), mask, torch.tensor(labels, dtype=torch.long)


def build_model(ckpt, device):
    cfg = ckpt.get("config", {}).get("model", {})
    name = ckpt.get("model_type", cfg.get("name", "tst"))
    kwargs = {
        "dim_x1": cfg.get("dim_x1", 4),
        "dim_x2": cfg.get("dim_x2", 12),
        "dim_hidden": cfg.get("dim_hidden", 128),
        "num_heads": cfg.get("num_heads", 4),
        "num_layers": cfg.get("num_layers", 3),
        "dropout": cfg.get("dropout", 0.1),
        "readout": cfg.get("readout", "attn"),
        "num_classes": cfg.get("num_classes", 20),
        "in_dim": 0,
    }
    model = get_model(name, **kwargs).to(device)
    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in ckpt["model_state_dict"].items()
    }
    model.load_state_dict(state)
    model.eval()
    return model, kwargs


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    logits_all, preds, labels = [], [], []
    for inputs, mask, y in loader:
        x1, x2 = inputs
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model((x1, x2), mask)
        logits_all.append(logits.cpu().numpy())
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    logits = np.concatenate(logits_all, axis=0)
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
    model, kwargs = build_model(ckpt, device)

    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "results", "tst")
    logger, ts = setup_logging(os.path.join(PROJECT_ROOT, "logs", "eval"))
    logger.info(f"ckpt={ckpt_path}  readout={kwargs['readout']}  eval_tps={args.eval_tps}")

    data_dir = os.path.join(PROJECT_ROOT, "data", "nature_data")
    x1 = load_pkl(os.path.join(data_dir, "x1_test_M1_M20_train_val_test_set.pkl"))
    x2_dict = load_pkl(os.path.join(data_dir, "x2_test_M1_M20_train_val_test_set.pkl"))
    y = load_pkl(os.path.join(data_dir, "y_test_M1_M20_train_val_test_set.pkl"))
    tp_key, err_key, x2 = resolve_x2_slice(x2_dict, args.dataset_tps, args.dataset_noise)

    ds = TSTTestDataset(
        x1, x2, y, mean_x2=ckpt["mean_x2"], std_x2=ckpt["std_x2"],
        extra_tps=args.eval_tps, force_t0=not args.no_force_t0,
        noise_std=args.manual_noise,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=tst_collate, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    results = run_eval(model, loader, device)
    logger.info(f"Top-1={results['accuracy']:.4f}  Top-3={results['top3_accuracy']:.4f}")

    tag = noise_tag(err_key, args.manual_noise)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"confusion_tst_{args.eval_tps}pts_{tag}_{ts}.csv")
    save_confusion_csv(
        results["confusion_matrix"], results["mechs"], csv_path,
        extra={"Top-1 Accuracy": results["accuracy"], "Top-3 Accuracy": results["top3_accuracy"]},
    )
    save_analysis_plot(
        results["confusion_matrix"], results["mechs"], results["per_class_accuracy"],
        title=f"TST  extra_tps={args.eval_tps}  {tag}",
        path=os.path.join(out_dir, f"plot_tst_{args.eval_tps}pts_{tag}_{ts}.png"),
    )
    logger.info(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
