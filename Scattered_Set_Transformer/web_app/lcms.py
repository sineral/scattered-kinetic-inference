"""Parse LC-MS / PDA JSON into a chromatogram, peak table, and [S]/[P].

The JSON schema matches the lab LC-MS service used in the original Windows
deployment (nested ``Data.LSSPDAData.ListPDAData[0]``). Mock hardware writes
the same schema so the parser is shared.
"""

from __future__ import annotations

import json
import traceback
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

LogFn = Optional[Callable[[str], None]]


def process_lcms_json(json_path: str, prm: dict, log_cb: LogFn = None):
    """Return (chromatogram figure, peak table, St, Pt).

    Concentrations use internal-standard calibration::

        St = (area_S / area_IS) * factor_s
        Pt = (area_P / area_IS) * factor_p

    Each chromatographic peak is assigned to at most one of [S], [P], [IS]
    (nearest retention time within ``rt_tol``).
    """
    if log_cb:
        log_cb(f"[LCMS] Parsing {json_path}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)

        if isinstance(full_data, str):
            full_data = json.loads(full_data)
        if "Data" in full_data and isinstance(full_data["Data"], str):
            full_data["Data"] = json.loads(full_data["Data"])

        pda_data = full_data["Data"]["LSSPDAData"]["ListPDAData"][0]
        chroma = pda_data["PDAChroma"]
        intensity_mau = chroma["ChromList"]
        # Lab instrument reports StartRT / EndRT in milliseconds.
        time_min = np.linspace(chroma["StartRT"], chroma["EndRT"], len(intensity_mau)) / 60_000.0

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_min,
                y=intensity_mau,
                fill="tozeroy",
                line=dict(color="#1A5276", width=1.2),
                name="PDA",
            )
        )
        fig.update_layout(
            title="Chromatogram (PDA)",
            xaxis_title="RT (min)",
            yaxis_title="Abs (mAU)",
            height=300,
            template="plotly_white",
        )

        peaks = pda_data.get("PDAPeakData", [])
        if log_cb:
            log_cb(f"[LCMS] Detected {len(peaks)} component peak(s)")

        peak_list = []
        area_s, area_p, area_is = 0.0, 0.0, 0.0
        tol = prm["rt_tol"]
        targets = [
            ("[S]", float(prm["rt_s"]), "s"),
            ("[P]", float(prm["rt_p"]), "p"),
            ("[IS]", float(prm["rt_is"]), "is"),
        ]

        for p in peaks:
            rt = p["Rt"]
            area = p.get("Area", 0.0)
            pct = p.get("AreaPct", 0.0)
            best_label, best_key, best_dist = None, None, None
            for label, rt_tgt, key in targets:
                dist = abs(rt - rt_tgt)
                if dist <= tol and (best_dist is None or dist < best_dist):
                    best_label, best_key, best_dist = label, key, dist
            matched = ""
            if best_key == "s":
                area_s = max(area_s, area)
                matched = best_label
            elif best_key == "p":
                area_p = max(area_p, area)
                matched = best_label
            elif best_key == "is":
                area_is = max(area_is, area)
                matched = best_label
            if matched and log_cb:
                log_cb(f"      captured {matched}: RT={rt:.2f}, Area={area}, %={pct:.1f}")
            if pct > 0.5:
                peak_list.append({"RT (min)": rt, "Area": area, "AreaPct (%)": pct})

        df_peaks = pd.DataFrame(peak_list).sort_values(by="RT (min)").reset_index(drop=True)
        df_peaks.index += 1

        if area_is == 0:
            if log_cb:
                log_cb("[LCMS] Internal standard [IS] not found; St=Pt=0")
            st, pt = 0.0, 0.0
        else:
            st = (area_s / area_is) * prm["factor_s"]
            pt = (area_p / area_is) * prm["factor_p"]
            if log_cb:
                log_cb(f"[LCMS] St = ({area_s:.1f}/{area_is:.1f})*{prm['factor_s']} = {st:.4f}")
                log_cb(f"[LCMS] Pt = ({area_p:.1f}/{area_is:.1f})*{prm['factor_p']} = {pt:.4f}")

        return fig, df_peaks, min(1.0, float(st)), min(1.0, float(pt))
    except Exception as e:
        if log_cb:
            log_cb(f"[LCMS] Parse failed: {e}\n{traceback.format_exc()}")
        return go.Figure(), pd.DataFrame(), 0.0, 0.0


def to_done_flag(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(v) == 1
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "done", "yes", "y"}
    return False


def parse_status_payload(status_obj):
    """Normalize GetStatus JSON to (result_file, done_flag, raw_df, inner_dict)."""
    if not isinstance(status_obj, dict):
        return "", False, pd.DataFrame(), {}

    inner = status_obj.get("Status", status_obj)
    if not isinstance(inner, dict):
        return "", False, pd.DataFrame(), {}

    result_file = str(inner.get("result_file", "") or "").strip()
    done_flag = to_done_flag(inner.get("done"))
    raw_df = pd.DataFrame()
    raw_data_2d = inner.get("raw_data", [])

    if isinstance(raw_data_2d, list) and len(raw_data_2d) > 0:
        try:
            cols = raw_data_2d[0]
            data = raw_data_2d[1:] if len(raw_data_2d) > 1 else []
            if isinstance(cols, list):
                raw_df = pd.DataFrame(data, columns=cols)
                if (not done_flag) and ("done" in raw_df.columns):
                    for v in raw_df["done"][::-1]:
                        if str(v).strip() != "":
                            done_flag = to_done_flag(v)
                            break
        except Exception:
            raw_df = pd.DataFrame()

    return result_file, done_flag, raw_df, inner
