#!/usr/bin/env python3
"""Screen log-uniform rate-constant vectors for one mechanism (M1-M20).

Pipeline per candidate ``k``:
  1. Draw each kinetic constant from a log-uniform prior (see
     ``config/k_generation.yaml`` → ``global_k_sampler``).
  2. Global filter: at 1 mol% catalyst, final conversion must be ≥ 50%.
  3. Mechanism-specific screening rules at a conversion window (optional).
     Mechanisms without a ``screening`` block pass step 3 automatically.

Usage (from the Scattered_Set_Transformer root):

    python data_generation/screen_k.py M8
    python data_generation/screen_k.py M8 --n-accept 50 --max-attempts 20000 --jobs 8

Outputs (timestamped) under ``data/k_screening/<mech>/``:
    CSV of accepted + a capped set of rejected vectors, and a log10(k) plot
    of a subset of accepted draws.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Limit BLAS threads so process-level parallelism is not oversubscribed.
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
    merge_mechanism,
    sample_k_vec,
    sanitize_expr,
    safe_var_name,
)

T_MAX = 5000.0
DCDT_TOL = 1e-5
SOLVER_TIMEOUT_SEC = 10.0
SOLVER_KWARGS = {"method": "BDF", "rtol": 1e-6, "atol": 1e-8, "max_step": 0.2}

# Set in worker initializer (avoids reloading YAML on every seed).
_WORKER_MECH: dict | None = None
_WORKER_SAMPLER: dict | None = None


def run_to_steady(mech: dict, k_dict: dict, y0: np.ndarray):
    """Integrate until |dC/dt| < DCDT_TOL or t_max / wall-clock timeout."""
    ode, species = build_ode_fun(mech, k_dict)

    def event_steady(t, C):
        return np.max(np.abs(ode(t, C))) - DCDT_TOL

    event_steady.terminal = True
    event_steady.direction = -1
    timeout_event = TimeoutCheck(SOLVER_TIMEOUT_SEC)

    try:
        sol = solve_ivp(
            ode,
            (0.0, T_MAX),
            y0,
            events=[event_steady, timeout_event],
            **SOLVER_KWARGS,
        )
    except Exception:
        return None, 0.0, False, species

    if not sol.success or sol.t.size == 0:
        t_last = float(sol.t[-1]) if sol.t.size > 0 else 0.0
        return sol, t_last, False, species

    timed_out = sol.t_events is not None and len(sol.t_events) > 1 and np.size(sol.t_events[1]) > 0
    if timed_out:
        return sol, float(sol.t[-1]), False, species

    reached = sol.t_events is not None and len(sol.t_events) > 0 and np.size(sol.t_events[0]) > 0
    t_final = float(sol.t_events[0][0]) if reached else float(sol.t[-1])
    return sol, t_final, reached, species


def check_global_conversion(mech: dict, k_dict: dict, rng: np.random.Generator,
                            cat0_val: float = 0.01, min_conv: float = 0.5) -> bool:
    """Require ≥50% conversion at 1 mol% cat0 (standard S0=1, P0=0).

    For mechanisms with an ``inhibitor`` species the initial inhibitor loading
    is drawn from ``initial_ranges`` and stored on ``k_dict`` when the check
    passes, so later dataset generation can reuse that value.
    """
    species = mech["species"]
    idx = {s: i for i, s in enumerate(species)}
    if "cat" not in idx or "S" not in idx:
        return True

    sc = mech.get("constraints", {}).get("screening")
    y0 = np.zeros(len(species))
    if sc and "initial_override" in sc:
        for sp, val in sc["initial_override"].items():
            if sp in idx:
                y0[idx[sp]] = val
    else:
        y0[idx["S"]] = 1.0
        if "P" in idx:
            y0[idx["P"]] = 0.0
    y0[idx["cat"]] = cat0_val

    if "inhibitor" in idx:
        ranges = mech.get("initial_ranges", {})
        if "inhibitor" in ranges:
            low, high = ranges["inhibitor"]
            inhib = float(rng.uniform(low, high))
            y0[idx["inhibitor"]] = inhib
        else:
            inhib = 0.0
            y0[idx["inhibitor"]] = 0.0
    else:
        inhib = None

    sol, _, _, _ = run_to_steady(mech, k_dict, y0)
    if sol is None or sol.t.size < 2:
        return False

    s0 = sol.y[idx["S"], 0]
    if s0 <= 0:
        return False
    conv = (s0 - sol.y[idx["S"], -1]) / s0
    if conv >= min_conv:
        if inhib is not None:
            k_dict["inhibitor_initial_check"] = inhib
        return True
    return False


def check_screening_for_mech(mech: dict, k_dict: dict, rng: np.random.Generator) -> bool:
    """Evaluate conversion-window concentration rules, scanning cat0 if needed.

    ``existential: true`` means *any* cat0 on the grid may pass.
    ``existential: false`` (default) means *every* cat0 must pass.
    """
    sc = mech.get("constraints", {}).get("screening")
    if sc is None:
        return True

    species = mech["species"]
    idx = {s: i for i, s in enumerate(species)}
    if "cat" not in idx or "S" not in idx:
        raise KeyError(f"Screening requires species 'cat' and 'S' in {mech.get('description', '')}")

    scan_cfg = sc["cat0_scan"]
    cat0_vals = np.linspace(scan_cfg["low"], scan_cfg["high"], scan_cfg["n"])
    existential = bool(scan_cfg.get("existential", False))
    x_low, x_high = float(sc["at_conv"]["X"][0]), float(sc["at_conv"]["X"][1])
    if np.isclose(x_low, x_high):
        x_low, x_high = max(0.0, x_low - 0.01), min(1.0, x_high + 0.01)

    rule_exprs = [
        sanitize_expr(r["expr"])
        for r in sc["at_conv"]["rules"]
        if isinstance(r, dict) and "expr" in r
    ]

    override = sc.get("initial_override", {})
    ranges = mech.get("initial_ranges", {})

    for cat0 in cat0_vals:
        y0 = np.zeros(len(species))
        for sp, val in override.items():
            if sp in idx:
                y0[idx[sp]] = val
        y0[idx["cat"]] = cat0
        for sp in species:
            if sp not in override and sp != "cat" and sp in ranges:
                low, high = ranges[sp]
                y0[idx[sp]] = rng.uniform(low, high)

        sol, _, _, _ = run_to_steady(mech, k_dict, y0)
        if sol is None or sol.t.size == 0:
            if not existential:
                return False
            continue

        s0 = sol.y[idx["S"], 0]
        if s0 <= 0:
            if not existential:
                return False
            continue
        x_traj = (s0 - sol.y[idx["S"], :]) / s0
        mask = (x_traj >= x_low) & (x_traj <= x_high)
        if not mask.any():
            if not existential:
                return False
            continue

        found = False
        for j in np.where(mask)[0]:
            env = {safe_var_name(sp): float(sol.y[idx[sp], j]) for sp in species}
            # Expressions come from the trusted local YAML, not user input.
            if all(eval(expr, {"__builtins__": {}}, env) for expr in rule_exprs):
                found = True
                break

        if found:
            if existential:
                return True
        elif not existential:
            return False

    return False if existential else True


def _init_worker(mech_yaml: str, gen_yaml: str, mech_name: str):
    global _WORKER_MECH, _WORKER_SAMPLER
    _WORKER_MECH, _WORKER_SAMPLER = merge_mechanism(mech_name, mech_yaml, gen_yaml)


def worker(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    k_dict = sample_k_vec(_WORKER_MECH, _WORKER_SAMPLER, rng)
    passed_global = check_global_conversion(_WORKER_MECH, k_dict, rng)
    passed_screening = False
    if passed_global:
        try:
            passed_screening = check_screening_for_mech(_WORKER_MECH, k_dict, rng)
        except Exception:
            passed_screening = False
    record = {
        "seed": seed,
        "passed": bool(passed_screening),
        "passed_global_check": bool(passed_global),
    }
    record.update(k_dict)
    return record


def parse_args():
    p = argparse.ArgumentParser(description="Screen kinetic constants for one M1-M20 mechanism")
    p.add_argument("mechanism", nargs="?", default="M8", help="Mechanism id, e.g. M8")
    p.add_argument("--n-accept", type=int, default=20000, help="Stop after this many passing vectors")
    p.add_argument("--max-attempts", type=int, default=50_000_000)
    p.add_argument("--jobs", type=int, default=None, help="Worker processes (default: CPU count)")
    p.add_argument("--mech-config", default=DEFAULT_MECH_YAML)
    p.add_argument("--gen-config", default=DEFAULT_GEN_YAML)
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: data/k_screening/<mechanism>)",
    )
    p.add_argument("--plot-max", type=int, default=200, help="Accepted vectors to draw in the PNG")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--reject-cap", type=int, default=None,
                   help="Rejected rows to keep in the CSV (default: same as --n-accept)")
    return p.parse_args()


def main():
    args = parse_args()
    mech_name = args.mechanism
    n_jobs = args.jobs or mp.cpu_count()
    reject_cap = args.n_accept if args.reject_cap is None else args.reject_cap
    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "data", "k_screening", mech_name)
    os.makedirs(out_dir, exist_ok=True)

    merge_mechanism(mech_name, args.mech_config, args.gen_config)  # fail fast
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{mech_name}_N{args.max_attempts}-{args.n_accept}_{ts}"
    out_csv = os.path.join(out_dir, f"{prefix}.csv")
    out_png = os.path.join(out_dir, f"{prefix}.png")

    print(f"Mechanism: {mech_name}")
    print(f"Workers:   {n_jobs}")
    print(f"Target:    {args.n_accept} accepted / {args.max_attempts} max attempts")
    print(f"Output:    {out_dir}")

    accepted, rejected = [], []
    processed = 0
    t0 = time.time()

    ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
    with ctx.Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(args.mech_config, args.gen_config, mech_name),
    ) as pool:
        for record in pool.imap_unordered(worker, range(args.max_attempts), chunksize=16):
            processed += 1
            if record["passed"]:
                accepted.append(record)
                print(
                    f"Attempt {processed} (seed={record['seed']}): PASSED. "
                    f"Accepted {len(accepted)}/{args.n_accept}"
                )
            else:
                if len(rejected) < reject_cap:
                    rejected.append(record)
                if processed % 50 == 0:
                    print(
                        f"Attempt {processed} (seed={record['seed']}): failed "
                        f"(global={record['passed_global_check']})"
                    )
            if len(accepted) >= args.n_accept:
                pool.terminate()
                break

    elapsed = time.time() - t0
    print(f"\n=== Summary for {mech_name} ===")
    print(f"Elapsed:   {elapsed:.1f} s")
    print(f"Processed: {processed}")
    print(f"Accepted:  {len(accepted)}")
    print(f"Rejected recorded: {len(rejected)}")

    if not accepted and not rejected:
        print("Nothing to save.")
        return

    df = pd.DataFrame(accepted + rejected)
    df.to_csv(out_csv, index=False)
    print(f"CSV: {out_csv}")

    if args.no_plot:
        return
    k_cols = [c for c in df.columns if str(c).startswith("k")]
    passed_df = df[df["passed"] == True]  # noqa: E712
    if passed_df.empty or not k_cols:
        return

    import matplotlib.pyplot as plt

    plot_df = passed_df.sample(n=min(args.plot_max, len(passed_df)), random_state=0)
    fig, ax = plt.subplots(figsize=(12, 7))
    for _, row in plot_df.iterrows():
        logk = [np.log10(row[k]) for k in k_cols if k in row and row[k] > 0]
        ax.plot(k_cols[: len(logk)], logk, marker="o", alpha=0.35, lw=0.8)
    ax.set_xlabel("Rate constant")
    ax.set_ylabel("log10(k)")
    ax.set_title(f"Accepted k vectors ({mech_name}, n={len(plot_df)} shown)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"PNG: {out_png}")


if __name__ == "__main__":
    main()
