"""Shared evaluation helpers: metrics, confusion-matrix CSV, and plots."""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    topk = np.argsort(logits, axis=1)[:, -k:]
    return float(np.mean([labels[i] in topk[i] for i in range(len(labels))]))


def group_accuracy(logits: np.ndarray, labels: np.ndarray, p: float = 0.99) -> float:
    """Accuracy when the true class is inside the smallest prefix whose
    softmax mass reaches probability p.
    """
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_logits)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    shifted = np.pad(cumsum[:, :-1], ((0, 0), (1, 0)), constant_values=0)
    included = shifted < p
    true_rank = np.argmax(order == labels[:, None], axis=1)
    return float(np.mean([included[i, r] for i, r in enumerate(true_rank)]))


def confusion_and_per_class(
    preds: np.ndarray, labels: np.ndarray, num_classes: int = 20
) -> Dict:
    mechs = [f"M{i + 1}" for i in range(num_classes)]
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    per_acc, per_n = {}, {}
    for c in range(num_classes):
        mask = labels == c
        n = int(mask.sum())
        per_n[mechs[c]] = n
        per_acc[mechs[c]] = float((preds[mask] == labels[mask]).mean()) if n else 0.0
    return {
        "confusion_matrix": cm,
        "mechs": mechs,
        "per_class_accuracy": per_acc,
        "per_class_counts": per_n,
    }


def save_confusion_csv(cm: np.ndarray, mechs, path: str, extra: Optional[Dict] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(cm, index=mechs, columns=mechs)
    df.loc["Total"] = df.sum(axis=0)
    df["Total"] = df.sum(axis=1)
    df.to_csv(path)
    if extra:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n")
            for k, v in extra.items():
                f.write(f"{k},{v}\n")


def save_analysis_plot(cm, mechs, per_class_acc, title: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 11), gridspec_kw={"width_ratios": [3, 1]}
    )
    fig.suptitle(title, fontsize=22, fontweight="bold")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=mechs, yticklabels=mechs, ax=ax1,
        cbar=True, cbar_kws={"shrink": 0.8},
        linewidths=0.5, linecolor="lightgray", annot_kws={"size": 12},
    )
    ax1.set_title("Confusion Matrix", fontsize=18, fontweight="bold")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.tick_params(axis="x", rotation=45)

    accs = [per_class_acc[m] for m in mechs]
    bars = ax2.barh(mechs, accs, color="skyblue", edgecolor="black")
    ax2.set_xlim(0, 1.1)
    ax2.set_title("Per-Class Accuracy", fontsize=18, fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{acc:.3f}", va="center", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()


def noise_tag(dataset_noise, manual_noise: float) -> str:
    base = int(round(float(dataset_noise)))
    if manual_noise and manual_noise > 0:
        return f"base{base}err_plus{int(round(manual_noise * 100))}err"
    return f"base{base}err"
