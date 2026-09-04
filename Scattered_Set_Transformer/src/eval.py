"""Evaluate a Set Transformer checkpoint or a 5-member ensemble directory.

Usage (from repo root):
    python src/eval.py --ensemble default --n-points 10
    python src/eval.py --ensemble combined --n-points 10
    python src/eval.py --data-dir data/default --ckpt checkpoints/default_member_0.pth
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.chdir(PROJECT_ROOT)

from models.base import get_model
from utils.scattered_dataset import AugmentedScatteredDataset, get_collate_fn


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate scattered Set Transformer")
    p.add_argument("--data-dir", default=None, help="Dataset dir (default: data/<ensemble>)")
    p.add_argument("--ckpt", default=None, help="Single .pth file")
    p.add_argument(
        "--ckpt-dir",
        default=os.path.join(PROJECT_ROOT, "checkpoints"),
        help="Directory containing {ensemble}_member_*.pth",
    )
    p.add_argument(
        "--ensemble",
        default="default",
        choices=["default", "combined"],
        help="Which named ensemble to evaluate",
    )
    p.add_argument("--n-points", type=int, default=10)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def setup_logging(out_dir: str):
    log_dir = os.path.join(PROJECT_ROOT, "logs", "eval")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"eval_{ts}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"eval_{ts}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger, ts


def infer_hparams(state_dict):
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
    in_w = sd["input_processing.1.weight"]
    layers = sorted(
        {int(k.split(".")[1]) for k in sd if k.startswith("encoder.") and k.endswith(".inducing")}
    )
    return {
        "in_dim": int(in_w.shape[1]),
        "num_classes": 20,
        "dim_hidden": int(in_w.shape[0]),
        "num_heads": 8,
        "num_layers": len(layers),
        "num_inducing": int(sd["encoder.0.inducing"].shape[0]),
        "dropout": 0.0,
    }


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in ckpt["model_state_dict"].items()
    }
    params = ckpt.get("model_params") or infer_hparams(ckpt["model_state_dict"])
    params = dict(params)
    params["dropout"] = 0.0
    model = get_model("settransformer", **params).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, params, ckpt


@torch.no_grad()
def run_eval(model, loader, device):
    labels, preds, top3 = [], [], []
    for features, y in tqdm(loader, desc="eval"):
        features = features.to(device)
        logits = model(features)
        preds.extend(logits.argmax(1).cpu().numpy())
        _, t3 = logits.topk(3, 1, True, True)
        top3.extend(t3.cpu().numpy())
        labels.extend(y.numpy())
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    top3 = np.asarray(top3)
    return {
        "top1": float((labels == preds).mean()),
        "top3": float(np.mean([t in p for t, p in zip(labels, top3)])),
        "n": int(len(labels)),
    }


def collect_ckpts(args):
    if args.ckpt:
        path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(PROJECT_ROOT, args.ckpt)
        return [(os.path.splitext(os.path.basename(path))[0], path)]
    root = args.ckpt_dir if os.path.isabs(args.ckpt_dir) else os.path.join(PROJECT_ROOT, args.ckpt_dir)
    members = sorted(glob.glob(os.path.join(root, f"{args.ensemble}_member_*.pth")))
    if members:
        return [(os.path.splitext(os.path.basename(p))[0], p) for p in members]
    raise FileNotFoundError(f"No {args.ensemble}_member_*.pth under {root}")


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    data_dir = args.data_dir or os.path.join(PROJECT_ROOT, "data", args.ensemble)
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(PROJECT_ROOT, data_dir)
    test_file = os.path.join(data_dir, "test.parquet")
    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "results", os.path.basename(data_dir.rstrip("/")))
    logger, ts = setup_logging(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}  test={test_file}  n={args.n_points}  noise={args.noise_std}")

    ckpts = collect_ckpts(args)
    feat_idx = list(range(20))
    collate = get_collate_fn(args.n_points, args.n_points, feat_idx)
    ds = AugmentedScatteredDataset(test_file, repeat_factor=1, is_train=False, noise_std=args.noise_std)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    rows = []
    for name, path in ckpts:
        logger.info(f"[{name}] {path}")
        model, params, ckpt = load_model(path, device)
        logger.info(f"  params={params}  ckpt_best_acc={ckpt.get('best_acc')}")
        stats = run_eval(model, loader, device)
        logger.info(f"  Top-1={stats['top1']:.4f}  Top-3={stats['top3']:.4f}  n={stats['n']}")
        rows.append(
            {
                "member": name,
                "top1_acc": stats["top1"],
                "top3_acc": stats["top3"],
                "n_samples": stats["n"],
                "n_points": args.n_points,
                "noise_std": args.noise_std,
                "ckpt_best_acc": ckpt.get("best_acc"),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    if len(df) > 1:
        df.loc[len(df)] = {
            "member": "mean",
            "top1_acc": df["top1_acc"].mean(),
            "top3_acc": df["top3_acc"].mean(),
            "n_samples": df["n_samples"].iloc[0],
            "n_points": args.n_points,
            "noise_std": args.noise_std,
            "ckpt_best_acc": np.nan,
        }
    csv_path = os.path.join(out_dir, f"summary_n{args.n_points}_noise{args.noise_std}_{ts}.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Wrote {csv_path}")
    logger.info("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
