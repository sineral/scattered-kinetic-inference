"""Shared ODE helpers for k-screening and scattered-dataset generation.

Mechanism ODEs come from ``config/mechanisms.yaml``. Screening ranges and
rate-constant priors come from ``config/k_generation.yaml``.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import yaml

# Species names that are invalid Python identifiers in screening expressions.
SAFE_NAME_MAP = {"cat*": "cat_star", "cat*S": "cat_starS"}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MECH_YAML = os.path.join(PROJECT_ROOT, "config", "mechanisms.yaml")
DEFAULT_GEN_YAML = os.path.join(PROJECT_ROOT, "config", "k_generation.yaml")


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_mechanism(name: str, mech_yaml: str, gen_yaml: str) -> tuple[dict, dict]:
    """Return (mechanism dict with screening fields, global_k_sampler)."""
    ode_cfg = load_yaml(mech_yaml)
    gen_cfg = load_yaml(gen_yaml)
    if name not in ode_cfg["mechanisms"]:
        raise KeyError(f"Unknown mechanism {name!r} in {mech_yaml}")
    mech = dict(ode_cfg["mechanisms"][name])
    extra = gen_cfg.get("mechanisms", {}).get(name, {})
    if "initial_ranges" in extra:
        mech["initial_ranges"] = extra["initial_ranges"]
    if "constraints" in extra:
        mech["constraints"] = extra["constraints"]
    sampler = gen_cfg.get("global_k_sampler")
    if sampler is None:
        raise KeyError(f"global_k_sampler missing from {gen_yaml}")
    return mech, sampler


def filter_unique_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-increasing x so interp1d does not divide by zero."""
    mask = np.concatenate(([True], np.diff(x) > 1e-11))
    return x[mask], y[mask]


class TimeoutCheck:
    """solve_ivp event: abort if wall time exceeds ``max_seconds``.

    More portable than SIGALRM and safe to use inside worker processes.
    """

    def __init__(self, max_seconds: float):
        self.start_time = time.time()
        self.max_seconds = max_seconds
        self.terminal = True
        self.direction = 1

    def __call__(self, t, y):
        return (time.time() - self.start_time) - self.max_seconds


def safe_var_name(species_name: str) -> str:
    return SAFE_NAME_MAP.get(species_name, species_name)


def sanitize_expr(expr: str) -> str:
    ordered = sorted(SAFE_NAME_MAP.items(), key=lambda kv: -len(kv[0]))
    for old, new in ordered:
        expr = expr.replace(old, new)
    return expr


def round_sigfigs(x: float, sigfigs: int) -> float:
    if x == 0 or not np.isfinite(x):
        return x
    return float(f"{x:.{sigfigs}g}")


def build_ode_fun(mech: dict, k_dict: dict):
    """Mass-action ODE. Returns (ode(t, C), species list)."""
    species = mech["species"]
    idx = {s: i for i, s in enumerate(species)}
    rxn_list = []
    for rinfo in mech["reactions"].values():
        k_name = list(rinfo["kinetic_constant"].keys())[0]
        if k_name not in k_dict:
            continue
        rxn_list.append(
            {
                "k": k_dict[k_name],
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

    return ode, species


def sample_k_vec(mech: dict, global_k_sampler: dict, rng: np.random.Generator) -> dict[str, float]:
    reactions = mech["reactions"]
    k_names = sorted({k for r in reactions.values() for k in r["kinetic_constant"]})
    low = float(global_k_sampler["low"])
    high = float(global_k_sampler["high"])
    sigfigs = int(global_k_sampler.get("round_sigfigs", 3))
    log_low, log_high = np.log10(low), np.log10(high)
    u = rng.uniform(log_low, log_high, size=len(k_names))
    vals = [round_sigfigs(float(v), sigfigs) for v in 10**u]
    return dict(zip(k_names, vals))


def extra_ic_from_k_row(row: dict[str, Any], species: list[str]) -> dict[str, float]:
    """Map screening CSV extras (e.g. inhibitor_initial_check) onto species names."""
    extra = {}
    for key, val in row.items():
        key_l = str(key).lower()
        if "inhibitor" in key_l and "inhibitor" in species:
            extra["inhibitor"] = val
    return extra
