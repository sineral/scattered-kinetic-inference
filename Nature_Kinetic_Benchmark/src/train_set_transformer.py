"""Train Set Transformer on Nature M1-M20 kinetic profiles.

Samples unordered 20-D set elements on the fly. Training noise is drawn
uniformly from {0, 0.5, 1, 2}%; validation uses a fixed 1% relative noise.

Usage (from repo root):
    python src/train_set_transformer.py
    python src/train_set_transformer.py --config config/train_set_transformer.yaml --gpu 0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.chdir(PROJECT_ROOT)

from models.base import get_model
from utils.config_loader import get_model_params
from utils.scatter_set_loader import create_dataloader, load_config


def parse_args():
    p = argparse.ArgumentParser(description="Train Set Transformer on Nature data")
    p.add_argument("--config", default="config/train_set_transformer.yaml")
    p.add_argument("--gpu", type=int, default=None, help="Physical GPU index")
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def setup_logging(log_dir: str, log_file: str):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{timestamp}_{log_file}")
    logger = logging.getLogger("train_set_transformer")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    logger.info(f"Log file: {log_path}")
    return logger, timestamp


def train_epoch(model, loader, optimizer, device, epoch, total_epochs, logger) -> float:
    model.train()
    total_loss, n = 0.0, 0
    num_batches = len(loader)
    for batch_idx, (features, mask, labels) in enumerate(loader):
        features = features.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(features, mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
        n += features.size(0)
        if (batch_idx + 1) % 100 == 0 or batch_idx == num_batches - 1:
            print(
                f"\rEpoch {epoch:03d}/{total_epochs}  "
                f"{batch_idx + 1}/{num_batches}  loss={loss.item():.4f}",
                end="",
                flush=True,
            )
    print()
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for features, mask, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        logits = model(features, mask)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * features.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += features.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    config = load_config(config_path)

    # Paper protocol: dynamic train noise, fixed 1% val noise.
    train_noise = [0.0, 0.005, 0.01, 0.02]
    val_noise = [0.01]

    out_dir = os.path.join(PROJECT_ROOT, "checkpoints", "set_transformer")
    log_dir = os.path.join(PROJECT_ROOT, "logs", "set_transformer")
    config["training"]["output"]["output_dir"] = out_dir
    config["training"]["output"]["log_dir"] = log_dir
    os.makedirs(out_dir, exist_ok=True)

    logger, timestamp = setup_logging(log_dir, "training.log")
    logger.info("=" * 72)
    logger.info(f"Set Transformer | train noise={train_noise} | val noise={val_noise}")
    logger.info("=" * 72)

    data_dir = config.get("data_dir", "data/nature_data")
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(PROJECT_ROOT, data_dir)
    mechanisms = config["mechanisms"]
    n_min = config["set_generation"]["n_min"]
    n_max = config["set_generation"]["n_max"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    samples_per_epoch = config["training"]["samples_per_epoch"]
    feature_config = config["set_generation"].get("features")
    epochs = args.epochs or config["training"]["epochs"]
    lr = config["training"]["learning_rate"]

    train_loader, train_meta = create_dataloader(
        data_dir=data_dir,
        mechanisms=mechanisms,
        split="train",
        n_min=n_min,
        n_max=n_max,
        batch_size=batch_size,
        num_workers=num_workers,
        samples_per_epoch=samples_per_epoch,
        noise_enabled=True,
        noise_std_dev=train_noise,
        feature_config=feature_config,
    )
    val_loader, _ = create_dataloader(
        data_dir=data_dir,
        mechanisms=mechanisms,
        split="val",
        n_min=n_min,
        n_max=n_max,
        batch_size=batch_size,
        num_workers=num_workers,
        noise_enabled=True,
        noise_std_dev=val_noise,
        feature_config=feature_config,
    )

    in_dim = train_meta["feature_dim"]
    num_classes = len(mechanisms)
    logger.info(f"in_dim={in_dim}  classes={num_classes}  n=[{n_min},{n_max}]")

    if torch.cuda.is_available() and not config["training"]["gpu"]["no_cuda"]:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"device={device}")

    model_params = get_model_params(config)
    model = get_model(
        name=config["model"]["name"],
        in_dim=in_dim,
        num_classes=num_classes,
        **model_params,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"parameters={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.empty_cache()
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, epochs, logger
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        logger.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(out_dir, "best_model.pth")
            cfg_path = os.path.join(out_dir, "config.yaml")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mechanisms": mechanisms,
                    "label_mapping": train_meta["label_mapping"],
                    "in_dim": in_dim,
                    "num_classes": num_classes,
                    "model_type": config["model"]["name"],
                    "hyperparams": config,
                    "epoch": epoch,
                    "best_val_accuracy": best_acc,
                },
                ckpt_path,
            )
            with open(cfg_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"  saved best model (val_acc={val_acc:.4f}) -> {ckpt_path}")

    logger.info(f"Done. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
