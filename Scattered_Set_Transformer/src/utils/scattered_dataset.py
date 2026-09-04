"""Load scattered kinetic parquet files and build 20-D set features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Channel order is fixed (matches shipped checkpoints). Names follow the paper table.
FEATURE_NAMES = [
    "Catalyst loading",       # 0  [cat]_0
    "Initial substrate",      # 1  [S]_0
    "Initial product",        # 2  [P]_0
    "Reaction time",          # 3  t
    "Residual substrate",     # 4  [S]_t
    "Generated product",      # 5  [P]_t
    "Substrate rate",         # 6  ([S]_t - [S]_0) / (t + eps)
    "Product rate",           # 7  ([P]_t - [P]_0) / (t + eps)
    "Substrate fraction",     # 8  [S]_t / ([S]_0 + eps)
    "Conversion",             # 9  ([S]_0 - [S]_t) / ([S]_0 + eps)
    "Total material",         # 10 [S]_t + [P]_t
    "Mass deviation",         # 11 ([S]_0 + [P]_0) - ([S]_t + [P]_t)
    "Product fraction",       # 12 [P]_t / ([S]_0 + eps)
    "Substrate ratio",        # 13 ln((|[S]_t| + eps) / (|[S]_0| + eps))
    "Product ratio",          # 14 ln((|[P]_t| + eps) / (|[P]_0| + eps))
    "Specific productivity",  # 15 ([P]_t - [P]_0) / ((t + eps)([cat]_0 + eps))
    "Substrate/catalyst",     # 16 [S]_t / ([cat]_0 + eps)
    "Product/catalyst",       # 17 [P]_t / ([cat]_0 + eps)
    "Substrate/product",      # 18 [S]_t / ([P]_t + eps)
    "Turnover number",        # 19 ([P]_t - [P]_0) / ([cat]_0 + eps)
]

_TIME_COLS = ("Time (min)", "Time", "Time (s)")


def time_values_minutes(df: pd.DataFrame) -> pd.Series:
    """Return reaction times from a CSV-like frame."""
    for col in _TIME_COLS:
        if col in df.columns:
            return df[col].astype(float)
    raise KeyError("Need a time column: Time or Time (min).")


def time_value_minutes(record: dict) -> float:
    """Return one reaction time from a CSV row dict."""
    for col in _TIME_COLS:
        if col in record and record[col] is not None and pd.notna(record[col]):
            return float(record[col])
    raise KeyError("Need a time column: Time or Time (min).")


class ScatteredKineticDataset(Dataset):
    """One sample is one (mech_id, k_hash) group: a variable-length set of points."""

    def __init__(
        self,
        data_source,
        noise_std: float = 0.0,
        use_tau: bool = False,
        keep_raw: bool = False,
        only_standard_ic: bool = False,
    ):
        self.noise_std = noise_std
        self.use_tau = use_tau
        self.keep_raw = keep_raw

        if isinstance(data_source, str):
            df = pd.read_parquet(data_source)
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        else:
            raise ValueError("data_source must be a parquet path or a DataFrame")

        if only_standard_ic:
            mask = (np.abs(df["S0"] - 1.0) < 1e-4) & (np.abs(df["P0"] - 0.0) < 1e-4)
            df = df[mask].reset_index(drop=True)

        if "k_hash" in df.columns:
            sort_cols = ["mech_id", "k_hash", "t"] if "mech_id" in df.columns else ["k_hash", "t"]
            df = df.sort_values(sort_cols).reset_index(drop=True)

        if self.use_tau:
            df = df[df["tau"] <= 1.01].copy()

        if self.keep_raw:
            self.raw_df = df.copy()

        t = df["tau"].values if use_tau else df["t"].values
        self.features = self._compute_feature_matrix(
            df["cat0"].values, df["S0"].values, df["P0"].values, t, df["St"].values, df["Pt"].values
        )

        if "k_hash" in df.columns and "mech_id" in df.columns:
            group_ids = df["mech_id"].astype(str) + "_" + df["k_hash"].astype(str)
            keys = group_ids.values
            _, start_indices = np.unique(keys, return_index=True)
            sort_idx = np.argsort(start_indices)
            self.group_starts = start_indices[sort_idx]
            self.group_keys = df["k_hash"].values[self.group_starts]
        elif "k_hash" in df.columns:
            keys = df["k_hash"].values
            unique_ks, start_indices = np.unique(keys, return_index=True)
            sort_idx = np.argsort(start_indices)
            self.group_keys = unique_ks[sort_idx]
            self.group_starts = start_indices[sort_idx]
        else:
            self.group_keys = np.array(["single_group"])
            self.group_starts = np.array([0])

        self.group_ends = np.append(self.group_starts[1:], len(df))

        if "mech_id" in df.columns:
            mech_ids = df["mech_id"].values[self.group_starts]
            self.labels = np.array([int(str(m)[1:]) - 1 for m in mech_ids], dtype=np.int64)
        else:
            self.labels = np.full(len(self.group_starts), -1, dtype=np.int64)

        if not self.keep_raw:
            del df

    @staticmethod
    def _compute_feature_matrix(cat0, S0, P0, t, St, Pt):
        eps = 1e-9
        cat_loading = cat0
        initial_substrate = S0
        initial_product = P0
        reaction_time = t
        residual_substrate = St
        generated_product = Pt
        substrate_rate = (St - S0) / (t + eps)
        product_rate = (Pt - P0) / (t + eps)
        substrate_fraction = St / (S0 + eps)
        conversion = (S0 - St) / (S0 + eps)
        total_material = St + Pt
        mass_deviation = (S0 + P0) - (St + Pt)
        product_fraction = Pt / (S0 + eps)
        substrate_ratio = np.log((np.abs(St) + eps) / (np.abs(S0) + eps))
        product_ratio = np.log((np.abs(Pt) + eps) / (np.abs(P0) + eps))
        specific_productivity = product_rate / (cat0 + eps)
        substrate_catalyst = St / (cat0 + eps)
        product_catalyst = Pt / (cat0 + eps)
        substrate_product = St / (Pt + eps)
        turnover_number = (Pt - P0) / (cat0 + eps)
        return np.stack(
            [
                cat_loading,
                initial_substrate,
                initial_product,
                reaction_time,
                residual_substrate,
                generated_product,
                substrate_rate,
                product_rate,
                substrate_fraction,
                conversion,
                total_material,
                mass_deviation,
                product_fraction,
                substrate_ratio,
                product_ratio,
                specific_productivity,
                substrate_catalyst,
                product_catalyst,
                substrate_product,
                turnover_number,
            ],
            axis=1,
        ).astype(np.float32)

    def __len__(self):
        return len(self.group_keys)

    def __getitem__(self, idx):
        start, end = self.group_starts[idx], self.group_ends[idx]
        label = self.labels[idx]
        if self.noise_std > 0:
            base = self.features[start:end, :6].copy()
            cat0, S0, P0, t = base[:, 0], base[:, 1], base[:, 2], base[:, 3]
            St, Pt = base[:, 4], base[:, 5]
            St_n = np.maximum(0, St * (1 + np.random.normal(0, self.noise_std, St.shape)))
            Pt_n = np.maximum(0, Pt * (1 + np.random.normal(0, self.noise_std, Pt.shape)))
            feats = self._compute_feature_matrix(cat0, S0, P0, t, St_n, Pt_n)
        else:
            feats = self.features[start:end].copy()
        return torch.from_numpy(feats), torch.tensor(label, dtype=torch.long)


class AugmentedScatteredDataset(ScatteredKineticDataset):
    """Logical repeat of groups, with optional per-sample noise drawn from a list."""

    def __init__(
        self,
        parquet_path,
        repeat_factor: int = 1,
        error_range=None,
        is_train: bool = True,
        **kwargs,
    ):
        super().__init__(parquet_path, **kwargs)
        self.repeat_factor = repeat_factor
        self.is_train = is_train
        if error_range and isinstance(error_range, str):
            self.error_range = [float(x) for x in error_range.split(",")]
        else:
            self.error_range = error_range

    def __len__(self):
        return super().__len__() * self.repeat_factor

    def __getitem__(self, idx):
        actual_idx = idx % super().__len__()
        if self.is_train and self.error_range:
            self.noise_std = float(np.random.choice(self.error_range))
        return super().__getitem__(actual_idx)


def get_collate_fn(n_min: int, n_max: int, feat_idx):
    def collate_fn(batch):
        processed, labels = [], []
        for feats, label in batch:
            feats = feats[:, feat_idx]
            n_avail = feats.shape[0]
            target_n = np.random.randint(n_min, min(n_max, n_avail) + 1)
            indices = torch.randperm(n_avail)[:target_n]
            indices, _ = torch.sort(indices)
            processed.append(feats[indices])
            labels.append(label)
        max_len = max(f.shape[0] for f in processed)
        padded = torch.stack(
            [torch.nn.functional.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in processed]
        )
        return padded, torch.stack(labels)

    return collate_fn
