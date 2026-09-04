"""Shared oracle, ensemble evaluator, and I/O for active sampling."""

from __future__ import annotations

import glob
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import entropy as scipy_entropy

from eval import load_model
from utils.scattered_dataset import ScatteredKineticDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOLVER_KWARGS = {"method": "BDF", "dense_output": True, "rtol": 1e-6, "atol": 1e-8}
T_MAX_SEARCH = 5000
STEADY_STATE_TOLERANCE = 1e-8
SOLVER_TIMEOUT_SEC = 5.0

CAT0_MIN, CAT0_MAX = 0.01, 0.10
T_MIN = 0.1  # minutes (same clock as t_ss / t_norm)


class TimeoutCheck:
    def __init__(self, max_seconds):
        self.start_time = time.time()
        self.max_seconds = max_seconds
        self.terminal = True

    def __call__(self, t, y):
        return (time.time() - self.start_time) - self.max_seconds


def filter_unique_x(x, y):
    _, idx = np.unique(x, return_index=True)
    sort_idx = np.sort(idx)
    return x[sort_idx], y[sort_idx]


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_samples(path):
    data = load_yaml(path)
    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    out = []
    for s in samples:
        mech_id = s["mech_id"]
        out.append(
            {
                "mech_id": mech_id,
                "mech_idx": int(str(mech_id)[1:]) - 1,
                "instance_id": s.get("instance_id", s.get("k_hash", mech_id)),
                "k": {str(k): float(v) for k, v in s["k"].items()},
            }
        )
    return out


def collect_ensemble_ckpts(ckpt_dir, ensemble="default"):
    members = sorted(glob.glob(os.path.join(ckpt_dir, f"{ensemble}_member_*.pth")))
    if not members:
        raise FileNotFoundError(f"No {ensemble}_member_*.pth under {ckpt_dir}")
    return members


def load_ensemble(ckpt_paths, device):
    models, feat_idx = [], list(range(20))
    for path in ckpt_paths:
        model, params, _ckpt = load_model(path, device)
        models.append(model)
        feat_idx = list(range(int(params.get("in_dim", 20))))
    return models, feat_idx


def build_ode(mech, k_dict):
    idx = {s: i for i, s in enumerate(mech["species"])}
    rxn_list = []
    for _, rinfo in mech["reactions"].items():
        k_name = list(rinfo["kinetic_constant"].keys())[0]
        if k_name in k_dict:
            rxn_list.append(
                {
                    "k": float(k_dict[k_name]),
                    "react": rinfo.get("reactants", {}),
                    "prod": rinfo.get("products", {}),
                }
            )

    def ode(t, C):
        dCdt = np.zeros_like(C)
        for rxn in rxn_list:
            rate = rxn["k"]
            for sp, sto in rxn["react"].items():
                rate *= C[idx[sp]] ** sto
            for sp, sto in rxn["react"].items():
                dCdt[idx[sp]] -= sto * rate
            for sp, sto in rxn["prod"].items():
                dCdt[idx[sp]] += sto * rate
        return dCdt

    return ode, idx


def initial_state(mech, idx, cat0, k_vals):
    y0 = np.zeros(len(mech["species"]))
    if "S" in idx:
        y0[idx["S"]] = 1.0
    if "cat" in idx:
        y0[idx["cat"]] = cat0
    if "inhibitor" in idx:
        y0[idx["inhibitor"]] = k_vals.get("inhibitor", 0.0)
    return y0


def compute_t_norm(mech, k_vals, cat0=0.01):
    """Reference time t_ss (minutes): 1.1 × time to 99% of observed conversion at 1 mol% cat0."""
    ode, idx = build_ode(mech, k_vals)
    y0 = initial_state(mech, idx, cat0, k_vals)

    def ss_event(t, C):
        return np.max(np.abs(ode(t, C))) - STEADY_STATE_TOLERANCE

    ss_event.terminal = True
    try:
        sol = solve_ivp(
            ode,
            (0, T_MAX_SEARCH),
            y0,
            events=[ss_event, TimeoutCheck(SOLVER_TIMEOUT_SEC)],
            **SOLVER_KWARGS,
        )
        X_t = 1.0 - sol.y[idx["S"]]
        final_x = X_t[-1]
        target_x = 0.99 * final_x if final_x > 0.01 else final_x
        X_f, t_f = filter_unique_x(X_t, sol.t)
        f_interp = interp1d(X_f, t_f, fill_value="extrapolate")
        return max(float(f_interp(target_x)) * 1.1, 1.0)
    except Exception:
        return float(T_MAX_SEARCH)


class KineticOracle:
    """Integrate the ground-truth ODE for one (mechanism, k) instance."""

    def __init__(self, mech_def, k_vals, mech_id, instance_id):
        self.mech_def = mech_def
        self.k_vals = k_vals
        self.mech_id = mech_id
        self.instance_id = instance_id
        self.ode_func, self.sp_id = build_ode(mech_def, k_vals)
        self.t_norm = compute_t_norm(mech_def, k_vals, cat0=0.01)

    def observe(self, t, cat0):
        t = float(np.clip(t, T_MIN, self.t_norm))
        cat0 = float(np.clip(cat0, CAT0_MIN, CAT0_MAX))
        y0 = initial_state(self.mech_def, self.sp_id, cat0, self.k_vals)
        sol = solve_ivp(self.ode_func, (0, max(t, self.t_norm) * 1.05), y0, **SOLVER_KWARGS)
        c_t = sol.sol(t)
        st = float(c_t[self.sp_id["S"]])
        pt = float(c_t[self.sp_id["P"]]) if "P" in self.sp_id else 0.0
        s_end = float(sol.sol(self.t_norm)[self.sp_id["S"]])
        rp = np.clip((1.0 - st) / max(0.01, 1.0 - s_end), 0, 1)
        return {
            "mech_id": self.mech_id,
            "k_hash": self.instance_id,
            "t": t,
            "tau": t / self.t_norm,
            "cat0": cat0,
            "S0": 1.0,
            "P0": 0.0,
            "St": st,
            "Pt": pt,
            "RP": float(rp),
        }


class EnsembleEvaluator:
    """Deep ensemble used as the shared classifier after each new point."""

    def __init__(self, models, feat_idx, device, n_max_input=20, noise_std=0.01):
        self.models = models
        self.feat_idx = feat_idx
        self.device = device
        self.n_max_input = n_max_input
        self.noise_std = noise_std
        self.member_probs = None

    def predict(self, samples):
        if not samples:
            uniform = np.ones(20) / 20.0
            self.member_probs = np.tile(uniform, (len(self.models), 1))
            return uniform
        df = pd.DataFrame(samples)
        ds = ScatteredKineticDataset(df, use_tau=False, noise_std=self.noise_std)
        feat, _ = ds[0]
        inp = torch.zeros((1, self.n_max_input, feat.shape[1]), device=self.device)
        n = min(len(df), self.n_max_input)
        inp[0, :n, :] = feat[:n, self.feat_idx]
        member_probs = []
        with torch.no_grad():
            for model in self.models:
                logits = model(inp)
                member_probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
        self.member_probs = np.asarray(member_probs)
        return self.member_probs.mean(axis=0)


def uncertainty(ens_probs, member_probs):
    pred_entropy = float(scipy_entropy(ens_probs))
    if member_probs is None or len(member_probs) == 0:
        return pred_entropy, 0.0
    mean_member = float(np.mean([scipy_entropy(mp) for mp in member_probs]))
    return pred_entropy, pred_entropy - mean_member


def make_run_dir(out_root, method, noise_std, mech_id, instance_id, extra=None):
    parts = [out_root, f"{method}_{noise_std}"]
    if extra:
        parts.append(extra)
    parts.append(f"{mech_id}_{instance_id}")
    parts.append("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def append_log(path, msg):
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def save_campaign(save_dir, history, member_records):
    df = pd.DataFrame(history)
    csv_path = os.path.join(save_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    if member_records:
        pd.DataFrame(member_records).to_csv(os.path.join(save_dir, "member_probs.csv"), index=False)
    return df


def save_summary_plot(save_dir, oracle, df):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for c0 in df["cat0"].unique():
        y0 = initial_state(oracle.mech_def, oracle.sp_id, c0, oracle.k_vals)
        sol = solve_ivp(oracle.ode_func, (0, oracle.t_norm * 1.05), y0, **SOLVER_KWARGS)
        ax1.plot(sol.t, sol.y[oracle.sp_id["S"]], color="#4c72b0", alpha=0.4, lw=2)
    sc = ax1.scatter(
        df["t"], df["St"], c=df["RP"], cmap="plasma", s=120, edgecolors="black", lw=1.5, zorder=10
    )
    for _, row in df.iterrows():
        ax1.annotate(
            str(int(row["step"])),
            (row["t"], row["St"]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=12,
        )
    plt.colorbar(sc, ax=ax1, label="Reaction Progress (RP)")
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("[S]")
    ax1.set_xlim(0, oracle.t_norm * 1.05)
    ax1.set_ylim(-0.05, 1.05)

    ax2.plot(df["step"], df["entropy"], "o-", color="orange", label="Entropy", lw=2)
    if "mutual_info" in df.columns:
        ax2.plot(df["step"], df["mutual_info"], "^--", color="purple", label="Mutual Info", lw=2)
    ax2_p = ax2.twinx()
    ax2_p.plot(df["step"], df["game_prob_gt"], "s-", color="green", label="GT Probability", lw=2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Entropy")
    ax2_p.set_ylabel("Ground Truth Confidence")
    ax2_p.set_ylim(0, 1.05)
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_p.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")
    fig.savefig(os.path.join(save_dir, "summary_plot.png"), dpi=300)
    plt.close(fig)
