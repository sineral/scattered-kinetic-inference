"""Run a small evaluation grid over sampled-point counts and noise levels.

Usage (from repo root):
    python src/run_eval_grid.py --model st
    python src/run_eval_grid.py --model deepsets --gpu 0
    python src/run_eval_grid.py --model tst
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set Transformer: n = set cardinality.
ST_N_LIST = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
# DeepSets / TST: extra non-zero timesteps besides t=0 (Pts table uses 4, 8, ..., 24).
SEQ_TPS_LIST = [1, 2, 3, 4, 5, 6]
NOISE_LIST = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]


DEFAULT_CKPT = {
    "st": os.path.join("checkpoints", "set_transformer", "best_model.pth"),
    "deepsets": os.path.join("checkpoints", "deepsets", "best_model.pth"),
    "tst": os.path.join("checkpoints", "tst", "best_model.pth"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Batch evaluation grid")
    p.add_argument("--model", required=True, choices=["st", "deepsets", "tst"])
    p.add_argument("--ckpt", default=None, help="Checkpoint path (default: shipped best model)")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def run(cmd):
    print("\n" + "=" * 60)
    print(" ".join(cmd))
    print("=" * 60)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


def main():
    args = parse_args()
    py = sys.executable
    gpu = [] if args.gpu is None else ["--gpu", str(args.gpu)]
    ckpt = args.ckpt or DEFAULT_CKPT[args.model]

    if args.model == "st":
        script = os.path.join(PROJECT_ROOT, "src", "eval_set_transformer.py")
        out = args.out_dir or os.path.join(PROJECT_ROOT, "results", "set_transformer", "grid")
        for n in ST_N_LIST:
            for noise in NOISE_LIST:
                run(
                    [py, script, "--ckpt", ckpt, "--n-points", str(n),
                     "--manual-noise", str(noise), "--out-dir", out] + gpu
                )
    else:
        script = os.path.join(
            PROJECT_ROOT, "src",
            "eval_deepsets.py" if args.model == "deepsets" else "eval_tst.py",
        )
        out = args.out_dir or os.path.join(PROJECT_ROOT, "results", args.model, "grid")
        for tps in SEQ_TPS_LIST:
            for noise in NOISE_LIST:
                run(
                    [py, script, "--ckpt", ckpt, "--eval-tps", str(tps),
                     "--manual-noise", str(noise), "--out-dir", out] + gpu
                )
    print("Grid finished.")


if __name__ == "__main__":
    main()
