"""Evaluate a trained Set Transformer checkpoint on the Nature test set.

Usage (from repo root):
    python src/eval_set_transformer.py
    python src/eval_set_transformer.py --n-points 10 --manual-noise 0.01
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.chdir(PROJECT_ROOT)

from eval_utils import (
    confusion_and_per_class,
    group_accuracy,
    noise_tag,
    save_analysis_plot,
    save_confusion_csv,
    topk_accuracy,
)
from models.base import get_model
from utils.config_loader import get_model_params
from utils.scatter_set_loader import create_dataloader


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Set Transformer")
    DEFAULT_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "set_transformer", "best_model.pth")
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to .pth checkpoint")
    p.add_argument("--config", default=None, help="YAML config (default: saved next to ckpt)")
    p.add_argument("--n-points", type=int, default=10, help="Fixed set size at test time")
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"eval_st_{ts}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(os.path.join(log_dir, f"eval_st_{ts}.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger, ts


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_logits, all_preds, all_labels = [], [], []
    total_loss, n = 0.0, 0
    for features, mask, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        logits = model(features, mask)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * features.size(0)
        n += features.size(0)
        all_logits.append(logits.cpu().numpy())
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    logits = np.vstack(all_logits)
    preds = np.asarray(all_preds)
    labels = np.asarray(all_labels)
    stats = confusion_and_per_class(preds, labels)
    stats.update(
        {
            "loss": total_loss / max(n, 1),
            "accuracy": float((preds == labels).mean()),
            "top3_accuracy": topk_accuracy(logits, labels, 3),
            "group_accuracy": group_accuracy(logits, labels, 0.99),
        }
    )
    return stats


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(PROJECT_ROOT, args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg_path = args.config
    if cfg_path is None:
        ckpt_dir = os.path.dirname(ckpt_path)
        candidates = [
            os.path.join(ckpt_dir, "config.yaml"),
            os.path.join(PROJECT_ROOT, "config", "train_set_transformer.yaml"),
        ]
        cfg_path = next((c for c in candidates if os.path.isfile(c)), candidates[-1])
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(PROJECT_ROOT, cfg_path)
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "results", "set_transformer")
    logger, ts = setup_logging(os.path.join(PROJECT_ROOT, "logs", "eval"))
    logger.info(f"ckpt={ckpt_path}")
    logger.info(f"config={cfg_path}  n={args.n_points}  noise={args.manual_noise}")

    data_dir = config.get("data_dir", "data/nature_data")
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(PROJECT_ROOT, data_dir)
    feature_config = config["set_generation"].get("features")

    loader, meta = create_dataloader(
        data_dir=data_dir,
        mechanisms=config["mechanisms"],
        split="test",
        n_min=args.n_points,
        n_max=args.n_points,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_nature_test=True,
        nature_test_timepoints=args.dataset_tps,
        nature_test_error=int(args.dataset_noise),
        noise_enabled=True,
        noise_std_dev=args.manual_noise,
        feature_config=feature_config,
        shuffle=False,
    )

    in_dim = meta["feature_dim"]
    model_params = get_model_params(config)
    if "hyperparams" in ckpt:
        try:
            model_params = get_model_params(ckpt["hyperparams"])
        except Exception:
            pass
    model = get_model(
        name=ckpt.get("model_type", "settransformer"),
        in_dim=ckpt.get("in_dim", in_dim),
        num_classes=ckpt.get("num_classes", 20),
        **model_params,
    ).to(device)
    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in ckpt["model_state_dict"].items()
    }
    model.load_state_dict(state)

    results = run_eval(model, loader, device)
    logger.info(
        f"Top-1={results['accuracy']:.4f}  Top-3={results['top3_accuracy']:.4f}  "
        f"Group={results['group_accuracy']:.4f}"
    )

    tag = noise_tag(args.dataset_noise, args.manual_noise)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"confusion_st_n{args.n_points}_{tag}_{ts}.csv")
    save_confusion_csv(
        results["confusion_matrix"],
        results["mechs"],
        csv_path,
        extra={
            "Top-1 Accuracy": results["accuracy"],
            "Top-3 Accuracy": results["top3_accuracy"],
            "Group Accuracy (p=0.99)": results["group_accuracy"],
        },
    )
    save_analysis_plot(
        results["confusion_matrix"],
        results["mechs"],
        results["per_class_accuracy"],
        title=f"Set Transformer  n={args.n_points}  {tag}",
        path=os.path.join(out_dir, f"plot_st_n{args.n_points}_{tag}_{ts}.png"),
    )
    logger.info(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
