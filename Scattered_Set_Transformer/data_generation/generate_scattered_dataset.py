#!/usr/bin/env python3
"""Build a scattered kinetic parquet dataset from screened rate-constant CSVs.

For each passing ``k`` vector the sampler draws a fixed number of observations
(default 80) with two coupled quotas:

* **cat0 layers** — five catalyst-loading bands covering 1–10 mol%, with a
  minority of "same-excess" initial conditions (S0 + P0 = 1).
* **RP bins** — six reaction-progress intervals so early, mid, and late
  conversion are all represented.

Each observation is obtained by integrating the mechanism ODE (BDF) from a
drawn initial condition and interpolating onto a feasible RP that still has
quota remaining. Failed / unreachable draws are retried up to a cap.

Usage (from the Scattered_Set_Transformer root), after screening CSVs exist:

    python data_generation/generate_scattered_dataset.py
    python data_generation/generate_scattered_dataset.py --n-pools 20 --mechanisms M1,M2

Writes ``train/val/test`` parquet files (70/15/15 split on ``k_hash``) and
``k_lookup.csv``. The default layout matches the Zenodo ``data/default``
schema described in ``data/README``.
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import multiprocessing as mp
import numpy as np
import pandas as pd
import yaml
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from common import (  # noqa: E402
    DEFAULT_GEN_YAML,
    DEFAULT_MECH_YAML,
    PROJECT_ROOT,
    TimeoutCheck,
    build_ode_fun,
    extra_ic_from_k_row,
    filter_unique_x,
    merge_mechanism,
)

SOLVER_KWARGS = {
    "method": "BDF",
    "dense_output": True,
    "rtol": 1e-6,
    "atol": 1e-8,
}
T_MAX = 5000.0
STEADY_STATE_TOLERANCE = 1e-8
MAX_TRAJ_ATTEMPTS = 500
SOLVER_TIMEOUT_SEC = 20.0
NORM_TIMEOUT_SEC = 30.0

# Default 80-point design used for the published default split (1–10 mol% cat0).
GLOBAL_RP_BINS = [
    ([0.00, 0.05], 10),
    ([0.05, 0.20], 10),
    ([0.20, 0.50], 20),
    ([0.50, 0.80], 20),
    ([0.80, 0.98], 10),
    ([0.98, 1.00], 10),
]
# (cat0_low, cat0_high), n_points, same-excess fraction
CAT0_LAYERS = [
    ([0.010, 0.025], 18, 0.25),
    ([0.025, 0.045], 22, 0.25),
    ([0.045, 0.065], 18, 0.25),
    ([0.065, 0.085], 12, 0.30),
    ([0.085, 0.100], 10, 0.30),
]
TOTAL_POINTS_PER_K = sum(n for _, n, _ in CAT0_LAYERS)
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}

assert TOTAL_POINTS_PER_K == sum(n for _, n in GLOBAL_RP_BINS)


def get_norm_t(mech: dict, k_vals: dict) -> float:
    """Reference time t_ss (minutes): 1.1 × time to 99% of observed conversion at 1 mol% cat0."""
    ode, species = build_ode_fun(mech, k_vals)
    idx = {s: i for i, s in enumerate(species)}
    c0 = np.zeros(len(species))
    if "S" in idx:
        c0[idx["S"]] = 1.0
    if "cat" in idx:
        c0[idx["cat"]] = 0.01

    def ss_event(t, C):
        return np.max(np.abs(ode(t, C))) - STEADY_STATE_TOLERANCE

    ss_event.terminal = True
    timeout_event = TimeoutCheck(NORM_TIMEOUT_SEC)
    try:
        sol = solve_ivp(ode, (0, T_MAX), c0, events=[ss_event, timeout_event], **SOLVER_KWARGS)
        if "S" not in idx or len(sol.t) < 2:
            return T_MAX
        x_t = 1.0 - sol.y[idx["S"]]
        x_max = float(x_t[-1])
        if x_max < 0.2:
            return T_MAX
        x_f, t_f = filter_unique_x(x_t, sol.t)
        if len(x_f) < 2:
            return T_MAX
        f_interp = interp1d(x_f, t_f, fill_value="extrapolate")
        return float(f_interp(0.99 * x_max)) * 1.1
    except Exception:
        return T_MAX


class QuotaSampler:
    """Fill RP-bin and cat0-layer quotas for one kinetic instance."""

    def __init__(self, mech_def: dict, k_vals: dict, t_norm: float, extra_params: dict):
        self.mech_def = mech_def
        self.k_vals = k_vals
        self.t_norm = t_norm
        self.extra_params = extra_params
        self.species = mech_def["species"]
        self.sp_map = {s: i for i, s in enumerate(self.species)}
        self.ode_func, _ = build_ode_fun(mech_def, k_vals)

        self.ic_plan = []
        for range_val, count, se_ratio in CAT0_LAYERS:
            se_count = int(count * se_ratio)
            for i in range(count):
                self.ic_plan.append({"cat_range": range_val, "is_same_excess": i < se_count})
        np.random.shuffle(self.ic_plan)
        self.rp_bins = GLOBAL_RP_BINS
        self.rp_quotas = [count for _, count in self.rp_bins]
        self.pool: list[dict] = []

    def sample_all(self, k_hash: str, mech_id: str):
        attempts = 0
        while len(self.pool) < TOTAL_POINTS_PER_K and self.ic_plan and attempts < MAX_TRAJ_ATTEMPTS:
            attempts += 1
            ic_cfg = self.ic_plan.pop(0)
            cat0 = float(np.random.uniform(*ic_cfg["cat_range"]))
            if ic_cfg["is_same_excess"]:
                s0 = float(np.random.uniform(0.4, 0.8))
                p0 = 1.0 - s0
            else:
                s0, p0 = 1.0, 0.0

            ic = {"S": s0, "P": p0, "cat": cat0}
            for key, val in self.extra_params.items():
                if key in self.sp_map:
                    ic[key] = val
            ranges = self.mech_def.get("initial_ranges", {})
            for s in self.species:
                if s not in ic:
                    lo, hi = ranges.get(s, [0.0, 0.0])
                    ic[s] = float(np.random.uniform(lo, hi))

            try:
                c0 = [ic.get(s, 0.0) for s in self.species]
                tout = TimeoutCheck(SOLVER_TIMEOUT_SEC)
                sol = solve_ivp(self.ode_func, (0, self.t_norm), c0, events=[tout], **SOLVER_KWARGS)
                if sol.t_events and len(sol.t_events) > 0 and np.size(sol.t_events[0]) > 0:
                    continue

                s_idx = self.sp_map["S"]
                p_idx = self.sp_map.get("P", -1)
                x_abs = (s0 - sol.y[s_idx]) / s0 if s0 > 1e-6 else np.zeros_like(sol.t)
                x_limit = max(0.01, float(x_abs[-1]))
                rp_traj = np.maximum.accumulate(np.clip(x_abs / x_limit, 0, 1))

                avail = [
                    i
                    for i, (br, _) in enumerate(self.rp_bins)
                    if self.rp_quotas[i] > 0 and rp_traj[0] <= br[1] and rp_traj[-1] >= br[0]
                ]
                if not avail:
                    self.ic_plan.append(ic_cfg)
                    continue

                tidx = int(np.random.choice(avail))
                br = self.rp_bins[tidx][0]
                rp_f, tau_f = filter_unique_x(rp_traj, sol.t / self.t_norm)
                if len(rp_f) < 2:
                    self.ic_plan.append(ic_cfg)
                    continue

                lo = max(float(rp_f[0]), br[0])
                hi = min(float(rp_f[-1]), br[1])
                if hi <= lo:
                    self.ic_plan.append(ic_cfg)
                    continue
                trp = float(np.random.uniform(lo, hi))
                f_tau = interp1d(rp_f, tau_f, fill_value="extrapolate")
                tau = max(0.0, min(1.0, float(f_tau(trp))))
                conc = sol.sol(tau * self.t_norm)
                st = float(conc[s_idx])
                pt = float(conc[p_idx]) if p_idx >= 0 else 0.0
                self.pool.append(
                    {
                        "mech_id": mech_id,
                        "k_hash": k_hash,
                        "t": tau * self.t_norm,
                        "tau": tau,
                        "X": (s0 - st) / s0,
                        "RP": trp,
                        "cat0": cat0,
                        "S0": s0,
                        "P0": p0,
                        "St": st,
                        "Pt": pt,
                        "t_ss": self.t_norm,
                    }
                )
                self.rp_quotas[tidx] -= 1
            except Exception:
                self.ic_plan.append(ic_cfg)
        return self.pool if len(self.pool) == TOTAL_POINTS_PER_K else None


def generate_pool_worker(args):
    try:
        mech_name, mech_def, row = args
        k_vals = {c: row[c] for c in row.keys() if str(c).startswith("k")}
        k_hash = hashlib.sha1(json.dumps(k_vals, sort_keys=True).encode()).hexdigest()[:8]
        t_norm = get_norm_t(mech_def, k_vals)
        if t_norm >= T_MAX:
            return None
        extra = extra_ic_from_k_row(row, mech_def["species"])
        sampler = QuotaSampler(mech_def, k_vals, t_norm, extra)
        res = sampler.sample_all(k_hash, mech_name)
        if res:
            lookup = {**k_vals, "k_hash": k_hash, "mech_id": mech_name, "t_norm": t_norm}
            lookup.update(extra)
            return res, lookup
        return None
    except Exception:
        return None


def find_k_csv(k_root: str, mech_name: str) -> str | None:
    patterns = [
        os.path.join(k_root, mech_name, f"{mech_name}_*.csv"),
        os.path.join(k_root, f"{mech_name}_k_screening", f"{mech_name}_*.csv"),
        os.path.join(k_root, f"{mech_name}_*.csv"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = [p for p in files if os.path.isfile(p)]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def setup_logging(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logger = logging.getLogger("scattered_gen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def parse_args():
    p = argparse.ArgumentParser(description="Generate scattered kinetic parquet datasets")
    p.add_argument("--mech-config", default=DEFAULT_MECH_YAML)
    p.add_argument("--gen-config", default=DEFAULT_GEN_YAML)
    p.add_argument(
        "--k-root",
        default=os.path.join(PROJECT_ROOT, "data", "k_screening"),
        help="Directory that contains per-mechanism screening CSVs",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join(PROJECT_ROOT, "data", "generated"),
    )
    p.add_argument("--n-pools", type=int, default=10000, help="Successful k instances per mechanism")
    p.add_argument("--mechanisms", default="all", help="Comma-separated ids, or 'all'")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--no-concat",
        action="store_true",
        help="Keep only per-mechanism parquet files (skip combined train/val/test.parquet)",
    )
    return p.parse_args()


def assign_splits(lookup_rows: list[dict]) -> dict[str, str]:
    hashes = [m["k_hash"] for m in lookup_rows]
    rng = np.random.default_rng()
    rng.shuffle(hashes)
    n = len(hashes)
    n_train = int(n * SPLIT_RATIO["train"])
    n_val = int(n * SPLIT_RATIO["val"])
    mapping = {}
    for i, h in enumerate(hashes):
        if i < n_train:
            mapping[h] = "train"
        elif i < n_train + n_val:
            mapping[h] = "val"
        else:
            mapping[h] = "test"
    return mapping


def main():
    args = parse_args()
    if args.mechanisms.strip().lower() == "all":
        mech_list = [f"M{i}" for i in range(1, 21)]
    else:
        mech_list = [m.strip() for m in args.mechanisms.split(",") if m.strip()]

    out_dir = args.out_dir
    log_file = os.path.join(out_dir, "generation.log")
    lookup_path = os.path.join(out_dir, "k_lookup.csv")
    if args.overwrite:
        if os.path.exists(lookup_path):
            os.remove(lookup_path)
        if os.path.exists(log_file):
            os.remove(log_file)
    os.makedirs(out_dir, exist_ok=True)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)

    logger = setup_logging(log_file)
    n_workers = args.workers or min(mp.cpu_count(), 64)
    logger.info("Scattered generation started (%s workers, %s pools/mech)", n_workers, args.n_pools)

    with open(args.mech_config, encoding="utf-8") as f:
        ode_cfg = yaml.safe_load(f)
    all_k_keys = set()
    for m in ode_cfg["mechanisms"].values():
        for r in m.get("reactions", {}).values():
            all_k_keys.update(r["kinetic_constant"].keys())
    standard_cols = ["k_hash", "mech_id", "t_norm"] + sorted(all_k_keys)

    for mech_name in mech_list:
        try:
            mech_def, _ = merge_mechanism(mech_name, args.mech_config, args.gen_config)
        except KeyError as exc:
            logger.warning("%s: skip (%s)", mech_name, exc)
            continue

        k_csv = find_k_csv(args.k_root, mech_name)
        if k_csv is None:
            logger.warning("%s: no screening CSV under %s", mech_name, args.k_root)
            continue

        df = pd.read_csv(k_csv)
        if "passed" not in df.columns:
            logger.warning("%s: CSV missing 'passed' column: %s", mech_name, k_csv)
            continue
        passed = df[df["passed"].astype(bool)].to_dict("records")
        np.random.shuffle(passed)
        if not passed:
            logger.warning("%s: no passing k vectors in %s", mech_name, k_csv)
            continue
        logger.info("[%s] %s passing k from %s", mech_name, len(passed), os.path.basename(k_csv))

        mech_pts, mech_lookup = [], []
        k_ptr = 0
        attempted = 0
        pbar = tqdm(total=args.n_pools, desc=f"  {mech_name}")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            active = {}
            while len(active) < n_workers * 2 and k_ptr < len(passed):
                fut = executor.submit(generate_pool_worker, (mech_name, mech_def, passed[k_ptr]))
                active[fut] = k_ptr
                k_ptr += 1
            while active and len(mech_lookup) < args.n_pools:
                for fut in as_completed(active):
                    res = fut.result()
                    del active[fut]
                    attempted += 1
                    if res:
                        pts, lk = res
                        mech_pts.extend(pts)
                        mech_lookup.append(lk)
                        pbar.update(1)
                    if len(mech_lookup) + len(active) < args.n_pools and k_ptr < len(passed):
                        new_fut = executor.submit(
                            generate_pool_worker, (mech_name, mech_def, passed[k_ptr])
                        )
                        active[new_fut] = k_ptr
                        k_ptr += 1
                    if len(mech_lookup) >= args.n_pools:
                        break
        pbar.close()

        if not mech_lookup:
            logger.warning("[%s] no successful pools", mech_name)
            continue

        df_lk = pd.DataFrame(mech_lookup).reindex(columns=standard_cols)
        header = not os.path.exists(lookup_path)
        df_lk.to_csv(lookup_path, mode="a", index=False, header=header)

        split_of = assign_splits(mech_lookup)
        df_mech = pd.DataFrame(mech_pts)
        df_mech["split"] = df_mech["k_hash"].map(split_of)
        counts = {}
        for split in ("train", "val", "test"):
            sub = df_mech[df_mech["split"] == split].drop(columns=["split"])
            counts[split] = len(sub)
            if not sub.empty:
                sub.to_parquet(os.path.join(out_dir, split, f"{mech_name}.parquet"), index=False)
        logger.info(
            "[%s] ok=%s attempted=%s discarded=%s pts=%s train/val/test=%s/%s/%s",
            mech_name,
            len(mech_lookup),
            attempted,
            attempted - len(mech_lookup),
            len(mech_pts),
            counts["train"],
            counts["val"],
            counts["test"],
        )
        del df_mech, mech_pts, mech_lookup
        gc.collect()

    if not args.no_concat:
        for split in ("train", "val", "test"):
            parts = sorted(glob.glob(os.path.join(out_dir, split, "M*.parquet")))
            if not parts:
                continue
            combined = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
            dest = os.path.join(out_dir, f"{split}.parquet")
            combined.to_parquet(dest, index=False)
            logger.info("Wrote %s (%s rows)", dest, len(combined))

    logger.info("Done.")


if __name__ == "__main__":
    main()
