"""Autosave, resume, and zip export for web-app runs."""

from __future__ import annotations

import glob
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from utils.scattered_dataset import time_value_minutes
from web_app.config import PROJECT_ROOT, RESULTS_DIR
from web_app.plots import plot_mechanism_probs


def display_path(p, base=PROJECT_ROOT) -> str:
    """Short path relative to the repo root (do not leak full home directories)."""
    try:
        rel = os.path.relpath(str(p), base)
        return rel if not rel.startswith("..") else os.path.basename(str(p))
    except Exception:
        return os.path.basename(str(p))


def autosave_loop_snapshot(
    run_dir,
    loop,
    history_records,
    reason,
    learner,
    current_raw_df=None,
    df_peaks=None,
    fig_lcms=None,
    fig_learn=None,
    fig_probs=None,
    status_payload=None,
    prediction_history=None,
    member_probs=None,
    llm_history=None,
    log_text="",
):
    loop_dir = os.path.join(run_dir, f"loop_{int(loop):03d}")
    os.makedirs(loop_dir, exist_ok=True)

    pd.DataFrame(history_records).to_csv(
        os.path.join(loop_dir, "history_records.csv"), index=False, encoding="utf-8-sig"
    )

    probs = learner.current_probs.tolist() if learner.current_probs is not None else []
    top_idx = int(np.argmax(probs)) + 1 if len(probs) > 0 else None
    last_metric = learner.history_metrics[-1] if learner.history_metrics else {}
    entropy_val = float(last_metric.get("entropy")) if last_metric.get("entropy") is not None else None

    snapshot = {
        "loop": int(loop),
        "reasoning": reason,
        "top_mechanism": f"M{top_idx}" if top_idx else None,
        "entropy": entropy_val,
        "observed_samples_count": len(learner.observed_samples),
        "probabilities": probs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(loop_dir, "snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    if status_payload is not None:
        with open(os.path.join(loop_dir, "status_payload.json"), "w", encoding="utf-8") as f:
            json.dump(status_payload, f, ensure_ascii=False, indent=2, default=str)

    if isinstance(df_peaks, pd.DataFrame) and (not df_peaks.empty):
        df_peaks.to_csv(os.path.join(loop_dir, "peaks_integration.csv"), index=False, encoding="utf-8-sig")
    if isinstance(current_raw_df, pd.DataFrame) and (not current_raw_df.empty):
        current_raw_df.to_csv(os.path.join(loop_dir, "raw_sensor_data.csv"), index=False, encoding="utf-8-sig")

    try:
        if fig_probs is not None:
            fig_probs.write_html(os.path.join(loop_dir, "mechanism_probs.html"))
        if fig_learn is not None:
            fig_learn.write_html(os.path.join(loop_dir, "learning_entropy.html"))
        if fig_lcms is not None:
            fig_lcms.write_html(os.path.join(loop_dir, "lcms_chromatogram.html"))
    except Exception:
        pass

    with open(os.path.join(loop_dir, "llm_trace.json"), "w", encoding="utf-8") as f:
        json.dump(llm_history or [], f, ensure_ascii=False, indent=2)
    with open(os.path.join(loop_dir, "terminal_logs.txt"), "w", encoding="utf-8") as f:
        f.write(log_text or "")

    if prediction_history:
        with open(os.path.join(loop_dir, "prediction_history.json"), "w", encoding="utf-8") as f:
            json.dump(prediction_history, f, ensure_ascii=False, indent=2)
        pd.DataFrame(prediction_history).to_csv(
            os.path.join(loop_dir, "prediction_history.csv"), index=False, encoding="utf-8-sig"
        )
    if member_probs:
        pd.DataFrame(member_probs).to_csv(
            os.path.join(loop_dir, "member_probs.csv"), index=False, encoding="utf-8-sig"
        )


def find_latest_autosave_dir(results_dir=RESULTS_DIR):
    dirs = [d for d in glob.glob(os.path.join(results_dir, "autosave_*")) if os.path.isdir(d)]
    return sorted(dirs)[-1] if dirs else None


def load_resume_state(resume_dir, learner, state, log=None):
    """Restore learner + UI globals from an autosave directory.

    ``state`` is the mutable dict of lists/dicts used by ``app.py``.
    Returns (history_records, start_loop).
    """

    def _log(m):
        if log:
            log(m)

    loop_dirs = sorted(glob.glob(os.path.join(resume_dir, "loop_*")))
    if not loop_dirs:
        _log(f"[Resume] No loop_* directories in {display_path(resume_dir)}")
        return [], 0

    last_loop_dir = loop_dirs[-1]
    history_records = []
    hist_csv = os.path.join(last_loop_dir, "history_records.csv")
    if os.path.exists(hist_csv):
        try:
            df = pd.read_csv(hist_csv)
            history_records = df.to_dict("records")
            for r in history_records:
                learner.observed_samples.append(
                    {
                        "t": time_value_minutes(r),
                        "cat0": float(r["Cat0"]),
                        "S0": 1.0,
                        "P0": 0.0,
                        "St": float(r["[S]"]),
                        "Pt": float(r["[P]"]),
                    }
                )
        except Exception as e:
            _log(f"[Resume] Failed to parse history_records.csv: {e}")

    pred_json = os.path.join(last_loop_dir, "prediction_history.json")
    if os.path.exists(pred_json):
        try:
            with open(pred_json, "r", encoding="utf-8") as f:
                preds = json.load(f)
            state["prediction_history"].extend(preds)
            for p in preds:
                learner.history_metrics.append(
                    {"step": p.get("loop"), "entropy": p.get("entropy", p.get("total_entropy"))}
                )
        except Exception as e:
            _log(f"[Resume] Failed to read prediction_history.json: {e}")

    member_csv = os.path.join(last_loop_dir, "member_probs.csv")
    if os.path.exists(member_csv):
        try:
            state["member_probs"].extend(pd.read_csv(member_csv).to_dict("records"))
        except Exception as e:
            _log(f"[Resume] Failed to read member_probs.csv: {e}")

    llm_json = os.path.join(last_loop_dir, "llm_trace.json")
    if os.path.exists(llm_json):
        try:
            with open(llm_json, "r", encoding="utf-8") as f:
                state["llm_history"].extend(json.load(f))
        except Exception as e:
            _log(f"[Resume] Failed to read llm_trace.json: {e}")

    for d in loop_dirs:
        try:
            loop_id = int(os.path.basename(d).split("_")[1])
        except Exception:
            continue
        reasoning = ""
        snap = os.path.join(d, "snapshot.json")
        if os.path.exists(snap):
            try:
                with open(snap, "r", encoding="utf-8") as f:
                    reasoning = json.load(f).get("reasoning", "")
            except Exception:
                pass
        df_peaks = pd.DataFrame()
        pk = os.path.join(d, "peaks_integration.csv")
        if os.path.exists(pk):
            try:
                df_peaks = pd.read_csv(pk)
            except Exception:
                pass
        df_raw = pd.DataFrame()
        rw = os.path.join(d, "raw_sensor_data.csv")
        if os.path.exists(rw):
            try:
                df_raw = pd.read_csv(rw)
            except Exception:
                pass
        probs_for_loop = None
        for p in state["prediction_history"]:
            if p.get("loop") == loop_id:
                probs_for_loop = p.get("probabilities")
                break
        state["history"][loop_id] = {
            "reasoning": reasoning,
            "fig_probs": plot_mechanism_probs(np.array(probs_for_loop)) if probs_for_loop else None,
            "fig_learn": None,
            "fig_lcms": None,
            "df_peaks": df_peaks,
            "df_raw": df_raw,
        }

    if learner.observed_samples:
        try:
            learner.current_probs = learner.predict(learner.observed_samples)
        except Exception as e:
            _log(f"[Resume] predict() failed: {e}")

    state["history_records"][:] = history_records
    start_loop = len(history_records)
    _log(f"[Resume] Restored {start_loop} completed loop(s) from {os.path.basename(resume_dir)}")
    return history_records, start_loop


def export_all_data(state, log_text, results_dir=RESULTS_DIR):
    """Write a timestamped zip under results/web_app/. Returns the zip path."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(results_dir, f"export_{timestamp}")
    os.makedirs(export_dir, exist_ok=True)

    full_terminal_log = "\n".join(state["terminal_log"]) + ("\n" if state["terminal_log"] else "")
    terminal_log_to_save = full_terminal_log if full_terminal_log else (log_text or "")
    if terminal_log_to_save:
        with open(os.path.join(export_dir, "terminal_logs.txt"), "w", encoding="utf-8") as f:
            f.write(terminal_log_to_save)

    if state["llm_history"]:
        with open(os.path.join(export_dir, "llm_trace.json"), "w", encoding="utf-8") as f:
            json.dump(state["llm_history"], f, indent=2, ensure_ascii=False)
    if state["prediction_history"]:
        with open(os.path.join(export_dir, "prediction_history.json"), "w", encoding="utf-8") as f:
            json.dump(state["prediction_history"], f, indent=2, ensure_ascii=False)
        pd.DataFrame(state["prediction_history"]).to_csv(
            os.path.join(export_dir, "prediction_history.csv"), index=False, encoding="utf-8-sig"
        )
    if state["member_probs"]:
        pd.DataFrame(state["member_probs"]).to_csv(
            os.path.join(export_dir, "member_probs.csv"), index=False, encoding="utf-8-sig"
        )
    if state["history_records"]:
        pd.DataFrame(state["history_records"]).to_csv(
            os.path.join(export_dir, "history_records.csv"), index=False, encoding="utf-8-sig"
        )

    for step, data in state["history"].items():
        step_dir = os.path.join(export_dir, f"loop_{step}")
        os.makedirs(step_dir, exist_ok=True)
        if data.get("df_peaks") is not None and not data["df_peaks"].empty:
            data["df_peaks"].to_csv(os.path.join(step_dir, "peaks_integration.csv"), index=False)
        if data.get("df_raw") is not None and not data["df_raw"].empty:
            data["df_raw"].to_csv(os.path.join(step_dir, "raw_sensor_data.csv"), index=False)
        try:
            if data.get("fig_probs") is not None:
                data["fig_probs"].write_html(os.path.join(step_dir, "probs.html"))
            if data.get("fig_learn") is not None:
                data["fig_learn"].write_html(os.path.join(step_dir, "learning_entropy.html"))
            if data.get("fig_lcms") is not None:
                data["fig_lcms"].write_html(os.path.join(step_dir, "lcms_chromatogram.html"))
        except Exception as e:
            print(f"Error exporting figures for loop {step}: {e}")

    zip_path = os.path.join(results_dir, f"SDL_Export_{timestamp}.zip")
    shutil.make_archive(zip_path.replace(".zip", ""), "zip", export_dir)
    return zip_path
