"""Self-Driving Lab Gradio app: LLM designs (t, cat0), ensemble classifies M1–M20.

Run from the Scattered_Set_Transformer repository root::

    python web_app/app.py

Default hardware is a mock device (no instruments). Point the UI at "Lab
instruments" and set LAB_HUB_URL / LAB_LCMS_URL when you have a real hub.
LLM keys: UI field, or DASHSCOPE_API_KEY / OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.chdir(PROJECT_ROOT)

from web_app.config import (
    CKPT_DIR,
    DEFAULT_ENSEMBLE,
    DEFAULT_HARDWARE_BACKEND,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_T_MAX,
    DEFAULT_T_MIN,
    DEFAULT_T_NORM,
    ENSEMBLE_CAT0_BOUNDS,
    MECH_CONFIG_PATH,
    N_MAX_INPUT,
    RESULTS_DIR,
    collect_ensemble_ckpts,
    resolve_api_key,
)
from web_app.hardware import make_hardware
from web_app.io_utils import (
    autosave_loop_snapshot,
    display_path,
    export_all_data,
    find_latest_autosave_dir,
    load_resume_state,
)
from web_app.lcms import parse_status_payload, process_lcms_json
from web_app.learner import RealWorldLearner, build_member_records
from web_app.plots import plot_learning_progress, plot_mechanism_probs

# Mutable run state shared by the Gradio callbacks (single-user loop).
STATE = {
    "stop": False,
    "history": {},
    "llm_history": [],
    "terminal_log": [],
    "prediction_history": [],
    "member_probs": [],
    "history_records": [],
}

CUSTOM_CSS = """
.huge-scroll-box {max-height:800px; overflow-y:auto;}
.huge-scroll-box textarea {
    font-family: 'Consolas', 'Courier New', monospace !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
}
.reasoning-box {
    background-color:#f0f8ff; padding:10px; border-radius:5px;
    margin-bottom:10px; font-style:italic; border-left: 4px solid #1A5276;
}
.wrap-text pre, .wrap-text code {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    word-break: break-all !important;
    overflow-x: auto !important;
    background-color: #f8f9fa !important;
}
"""


def _reset_state() -> None:
    STATE["stop"] = False
    STATE["history"].clear()
    STATE["llm_history"].clear()
    STATE["terminal_log"].clear()
    STATE["prediction_history"].clear()
    STATE["member_probs"].clear()
    STATE["history_records"].clear()


def start_automated_loop(
    api_key,
    base_url,
    model_name,
    ensemble_name,
    hardware_name,
    t_norm,
    t_min,
    t_max,
    cat_min,
    cat_max,
    max_loops,
    rts,
    rtp,
    rtis,
    factor_s,
    factor_p,
    tol,
    resume_enabled=False,
    resume_dir_input="",
):
    _reset_state()
    max_log_lines = 800
    log_lines = []

    def get_log_text():
        return ("\n".join(log_lines) + "\n") if log_lines else ""

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        log_lines.append(line)
        STATE["terminal_log"].append(line)
        if len(log_lines) > max_log_lines:
            del log_lines[:-max_log_lines]
        print(line, flush=True)

    def get_full_log_text():
        return ("\n".join(STATE["terminal_log"]) + "\n") if STATE["terminal_log"] else ""

    key = resolve_api_key(api_key)
    if not key:
        log("No LLM API key. Set the UI field or DASHSCOPE_API_KEY / OPENAI_API_KEY.")
        yield pack_yield_placeholder("### Workflow: stopped", "Missing API key")
        return

    ensemble_name = (ensemble_name or DEFAULT_ENSEMBLE).strip().lower()
    try:
        ckpt_paths = collect_ensemble_ckpts(CKPT_DIR, ensemble_name)
    except FileNotFoundError as e:
        log(str(e))
        yield pack_yield_placeholder("### Workflow: stopped", str(e))
        return

    lcms_params = {
        "rt_s": float(rts),
        "rt_p": float(rtp),
        "rt_is": float(rtis),
        "factor_s": float(factor_s),
        "factor_p": float(factor_p),
        "rt_tol": float(tol),
    }
    hardware = make_hardware(hardware_name, lcms_params=lcms_params)
    poll_sleep = 0.5 if hardware_name.strip().lower() in {"mock", ""} else 3.0

    cfg = {
        "api_key": key,
        "base_url": base_url or DEFAULT_LLM_BASE_URL,
        "model_name": model_name or DEFAULT_LLM_MODEL,
        "t_norm": float(t_norm),
        "t_min": float(t_min),
        "t_max": float(t_max),
        "cat_min": float(cat_min),
        "cat_max": float(cat_max),
        "n_max_input": N_MAX_INPUT,
        "ckpt_paths": ckpt_paths,
        "mech_config_path": MECH_CONFIG_PATH,
    }
    learner = RealWorldLearner(cfg, log_callback=log)
    if not learner.models:
        yield pack_yield_placeholder("### Workflow: stopped", "Failed to load ensemble weights")
        return

    history_records = []
    current_raw_df = pd.DataFrame()

    start_loop = 0
    resume_dir = None
    if resume_enabled:
        resume_dir = (resume_dir_input or "").strip() or find_latest_autosave_dir()
        if resume_dir and os.path.isdir(resume_dir):
            log(f"[Resume] Loading state from: {display_path(resume_dir)}")
            history_records, start_loop = load_resume_state(resume_dir, learner, STATE, log=log)
            learner.llm_history = list(STATE["llm_history"])
        else:
            log("[Resume] Requested but no valid autosave dir found. Starting fresh.")
            resume_dir = None

    if resume_enabled and resume_dir and start_loop > 0:
        autosave_run_dir = resume_dir
    else:
        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        autosave_run_dir = os.path.join(RESULTS_DIR, f"autosave_{run_stamp}")
    os.makedirs(autosave_run_dir, exist_ok=True)

    def generate_flowchart(max_l, curr_l, phase_desc):
        steps = []
        if curr_l == 0:
            steps.append("**Init: Network**")
        else:
            steps.append("Init")
        for i in range(1, max_l + 1):
            if i < curr_l:
                steps.append(f"L{i}")
            elif i == curr_l:
                steps.append(f"**L{i}: {phase_desc}**")
            else:
                steps.append(f"L{i}")
        return "### Workflow:  " + " -> ".join(steps)

    def format_llm_history():
        if not STATE["llm_history"]:
            return "*No prompts recorded yet...*"
        lines = []
        sys_prompt = STATE["llm_history"][0].get("system", "")
        lines.append(f"### System Prompt (Global)\n```text\n{sys_prompt}\n```\n---")
        for h in STATE["llm_history"]:
            lines.append(f"### Step {h['step'] + 1}")
            lines.append(f"**User content:**\n```text\n{h.get('user', '')}\n```")
            lines.append(f"**Response:**\n```json\n{h.get('response', '')}\n```")
            lines.append("---")
        return "\n".join(lines)

    llm_trace_cache = format_llm_history()

    def get_raw_preview(df, max_rows=60):
        if isinstance(df, pd.DataFrame) and (not df.empty):
            return df.tail(max_rows).reset_index(drop=True)
        return pd.DataFrame()

    def pack_yield(
        flow_text,
        status_text,
        reason_text=None,
        fig_lcms=None,
        df_peaks=None,
        fig_learn=None,
        fig_probs=None,
        raw_df=None,
        llm_trace_text=None,
    ):
        df_hist = (
            pd.DataFrame(history_records)
            if history_records
            else pd.DataFrame(columns=["Step", "Time (min)", "Cat0", "[S]", "[P]"])
        )
        return (
            flow_text,
            status_text,
            get_log_text(),
            reason_text if reason_text is not None else gr.update(),
            df_hist,
            fig_probs if fig_probs is not None else gr.update(),
            fig_learn if fig_learn is not None else gr.update(),
            fig_lcms if fig_lcms is not None else gr.update(),
            df_peaks if df_peaks is not None else gr.update(),
            raw_df if raw_df is not None else gr.update(),
            llm_trace_text if llm_trace_text is not None else gr.update(),
        )

    log("Experimental loop started.")
    log(f"Ensemble={ensemble_name}  hardware={hardware_name}  autosave={display_path(autosave_run_dir)}")
    yield pack_yield(generate_flowchart(max_loops, 0, "Initializing"), "Initializing...", llm_trace_text=llm_trace_cache)

    init_status = hardware.check_status(log_cb=log)
    if init_status is None:
        log("Warning: handshake (GetStatus) failed. Continuing anyway.")

    if start_loop > 0:
        log(f"[Resume] Continuing from loop {start_loop + 1} (already completed {start_loop}).")
        if start_loop >= max_loops:
            log(f"[Resume] Planned iterations ({max_loops}) <= completed loops ({start_loop}). Increase the slider.")
            yield pack_yield(
                generate_flowchart(max_loops, start_loop, "Already completed"),
                f"Nothing to run: target {max_loops} <= completed {start_loop}",
                llm_trace_text=llm_trace_cache,
            )

    for loop in range(start_loop + 1, max_loops + 1):
        if STATE["stop"]:
            log("Stop flag set; aborting.")
            yield pack_yield(generate_flowchart(max_loops, loop, "Stopped"), "Workflow stopped manually")
            break

        log(f"======== LOOP {loop}/{max_loops} ========")
        yield pack_yield(generate_flowchart(max_loops, loop, "Requesting LLM"), f"[L{loop}] Requesting LLM decision...")

        sugg = learner.get_llm_suggestion()
        STATE["llm_history"][:] = learner.llm_history
        raw_t = float(sugg.get("t", learner.t_min))
        raw_cat = float(sugg.get("cat0", learner.cat_min))
        cat_t = float(np.clip(raw_t, learner.t_min, learner.t_max))
        cat_val = float(np.clip(raw_cat, learner.cat_min, learner.cat_max))
        reason = sugg.get("reasoning", "API returned no reasoning.")
        if (raw_t != cat_t) or (raw_cat != cat_val):
            log(f"Out-of-range proposal (t={raw_t:.2f}, cat0={raw_cat:.4f}) clamped to (t={cat_t:.2f}, cat0={cat_val:.4f})")
        llm_trace_cache = format_llm_history()
        log(f"LLM decision: t={cat_t:.1f} min, cat0={cat_val:.3f}")

        yield pack_yield(
            generate_flowchart(max_loops, loop, "Dispatching"),
            f"[L{loop}] Dispatching parameters...",
            reason,
            llm_trace_text=llm_trace_cache,
        )
        success = hardware.send_conditions(loop, cat_val, cat_t, log_cb=log)
        if not success:
            log("Fatal: failed to send hardware parameters.")
            break

        yield pack_yield(generate_flowchart(max_loops, loop, "Monitoring"), f"[L{loop}] Monitoring device...", reason)
        target_res_file = None
        last_status_payload = None
        current_raw_df = pd.DataFrame()
        error_count = 0
        done_wait_logged = False

        for i in range(20000):
            if STATE["stop"]:
                break
            status = hardware.check_status(log_cb=None)
            if status and isinstance(status, dict):
                error_count = 0
                last_status_payload = status
                res_file, done_flag, parsed_raw_df, _inner = parse_status_payload(status)
                if isinstance(parsed_raw_df, pd.DataFrame) and (not parsed_raw_df.empty):
                    current_raw_df = parsed_raw_df
                if res_file:
                    target_res_file = res_file
                    log(f"[Device] Result path: {target_res_file}")
                    log(f"Raw 2D data rows: {len(current_raw_df)}")
                    break
                if done_flag and (not done_wait_logged):
                    log("Device done=1 but result_file is still empty, waiting...")
                    done_wait_logged = True
                if i % 10 == 0:
                    yield pack_yield(
                        generate_flowchart(max_loops, loop, f"Polled {i * poll_sleep:.0f}s"),
                        f"Monitoring device... polled for {i * poll_sleep:.0f}s",
                        reason,
                        raw_df=get_raw_preview(current_raw_df),
                    )
            else:
                error_count += 1
                if error_count % 5 == 0:
                    log(f"[Offline] Disconnected {error_count} times, last returning: {status}")
                    yield pack_yield(
                        generate_flowchart(max_loops, loop, "Retrying"),
                        f"Network fluctuation retrying ({error_count})",
                        reason,
                        raw_df=get_raw_preview(current_raw_df),
                    )
            time.sleep(poll_sleep)

        if STATE["stop"]:
            log("Aborted by user.")
            break

        def _save_failure_snapshot():
            autosave_loop_snapshot(
                run_dir=autosave_run_dir,
                loop=loop,
                history_records=history_records,
                reason=reason,
                learner=learner,
                current_raw_df=current_raw_df,
                df_peaks=pd.DataFrame(),
                fig_lcms=None,
                fig_learn=plot_learning_progress(learner),
                fig_probs=plot_mechanism_probs(learner.current_probs) if learner.current_probs is not None else None,
                status_payload=last_status_payload,
                prediction_history=STATE["prediction_history"],
                member_probs=STATE["member_probs"],
                llm_history=STATE["llm_history"],
                log_text=get_full_log_text(),
            )

        if not target_res_file:
            log("Polling timeout (never received result path).")
            _save_failure_snapshot()
            break

        yield pack_yield(
            generate_flowchart(max_loops, loop, "Fetching LCMS"),
            f"[L{loop}] Fetching LC-MS file...",
            reason,
            raw_df=get_raw_preview(current_raw_df),
        )

        lcms_json_path = None
        if not str(target_res_file).lower().endswith(".lcd"):
            target_res_file = str(target_res_file) + ".lcd"
            log(f"Added missing .lcd suffix: {target_res_file}")

        max_retries = 10
        for fetch_attempt in range(max_retries):
            if STATE["stop"]:
                break
            temp_path = hardware.fetch_lcms(target_res_file, log_cb=log)
            if not temp_path:
                time.sleep(3)
                continue
            with open(temp_path, "r", encoding="utf-8") as f:
                check_data = json.load(f)
            if isinstance(check_data, dict) and check_data.get("Data") == "NG":
                err_m = check_data.get("Msg", "")
                log(f"[Fetch] API Data=NG ({err_m}). Retry {fetch_attempt + 1}/{max_retries}")
                time.sleep(5)
                continue
            lcms_json_path = temp_path
            break

        if STATE["stop"]:
            break
        if not lcms_json_path:
            log("LC-MS fetch failed permanently.")
            _save_failure_snapshot()
            break

        fig_lcms, df_peaks, st, pt = process_lcms_json(lcms_json_path, lcms_params, log_cb=log)
        history_records.append(
            {
                "Step": loop,
                "Time (min)": round(cat_t, 1),
                "Cat0": round(cat_val, 3),
                "[S]": round(st, 4),
                "[P]": round(pt, 4),
            }
        )
        STATE["history_records"][:] = history_records

        learner.observed_samples.append(
            {"t": cat_t, "cat0": cat_val, "S0": 1.0, "P0": 0.0, "St": st, "Pt": pt}
        )
        unc = learner.predict_ensemble(learner.observed_samples)
        learner.current_uncertainty = unc
        probs = unc["mean_probs"]
        learner.current_probs = probs
        ent = unc["total_entropy"]
        learner.history_metrics.append({"step": loop, "entropy": ent})

        top_idx = np.argsort(probs)[-1]
        log(f"[Ensemble] M{top_idx + 1}: {probs[top_idx]:.2%} | entropy={ent:.3f}")
        STATE["prediction_history"].append(
            {
                "loop": int(loop),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "top_mechanism": f"M{int(top_idx) + 1}",
                "top_probability": float(probs[top_idx]),
                "entropy": float(ent),
                "probabilities": [float(p) for p in probs],
                "per_model_probabilities": [[float(x) for x in pm] for pm in unc["per_model_probs"]],
            }
        )
        STATE["member_probs"].extend(build_member_records(loop, unc["per_model_probs"]))

        fig_learn = plot_learning_progress(learner)
        fig_probs = plot_mechanism_probs(probs)
        STATE["history"][loop] = {
            "reasoning": reason,
            "fig_probs": fig_probs,
            "fig_learn": fig_learn,
            "fig_lcms": fig_lcms,
            "df_peaks": df_peaks,
            "df_raw": current_raw_df,
        }

        autosave_loop_snapshot(
            run_dir=autosave_run_dir,
            loop=loop,
            history_records=history_records,
            reason=reason,
            learner=learner,
            current_raw_df=current_raw_df,
            df_peaks=df_peaks,
            fig_lcms=fig_lcms,
            fig_learn=fig_learn,
            fig_probs=fig_probs,
            status_payload=last_status_payload,
            prediction_history=STATE["prediction_history"],
            member_probs=STATE["member_probs"],
            llm_history=STATE["llm_history"],
            log_text=get_full_log_text(),
        )
        log(f"Loop {loop} autosaved.")

        yield pack_yield(
            generate_flowchart(max_loops, loop, "Loop finished"),
            f"[L{loop}] Round finished, preparing next loop",
            reason,
            fig_lcms,
            df_peaks,
            fig_learn,
            fig_probs,
            current_raw_df,
            llm_trace_text=llm_trace_cache,
        )

    log("Workflow routine terminated.")
    log(f"Autosave root: {display_path(autosave_run_dir)}")
    yield pack_yield(generate_flowchart(max_loops, max_loops, "Completed"), "Workflow stopped", llm_trace_text=llm_trace_cache)


def pack_yield_placeholder(flow, status):
    empty = pd.DataFrame()
    return (
        flow,
        status,
        ("\n".join(STATE["terminal_log"]) + "\n") if STATE["terminal_log"] else "",
        gr.update(),
        empty,
        go.Figure(),
        go.Figure(),
        go.Figure(),
        empty,
        empty,
        "*stopped*",
    )


def stop_automated_loop():
    STATE["stop"] = True
    return "Stop requested. Waiting for the current poll / API call to return..."


def query_history(step_idx):
    idx = int(step_idx)
    if idx in STATE["history"]:
        h = STATE["history"][idx]
        return (
            h["reasoning"],
            h["fig_probs"] if h["fig_probs"] else go.Figure(),
            h["fig_lcms"] if h["fig_lcms"] else go.Figure(),
            h["df_peaks"] if h["df_peaks"] is not None else pd.DataFrame(),
            h["df_raw"] if h["df_raw"] is not None else pd.DataFrame(),
        )
    return ("*No data for this step.*", go.Figure(), go.Figure(), pd.DataFrame(), pd.DataFrame())


def export_clicked(log_text):
    zip_path = export_all_data(STATE, log_text)
    return gr.update(value=zip_path, label="Download ready (click to save)", interactive=True)


def _cat0_defaults(ensemble: str):
    return ENSEMBLE_CAT0_BOUNDS.get(ensemble, ENSEMBLE_CAT0_BOUNDS["combined"])


def build_demo() -> gr.Blocks:
    cat_lo, cat_hi = _cat0_defaults(DEFAULT_ENSEMBLE)
    with gr.Blocks(title="Self-Driving Lab (Deep Ensemble)") as demo:
        gr.Markdown(
            "# Self-Driving Lab — Deep Ensemble (entropy)\n"
            "LLM proposes the next `(t, cat0)`; a 5-member Set Transformer ensemble "
            "updates P(M1–M20). Default hardware is **mock** (no lab instruments). "
            "See `web_app/README.md`."
        )
        workflow_flowchart = gr.Markdown("### Workflow: waiting to start...", elem_classes="reasoning-box")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("LLM and model configuration", open=True):
                    api_key = gr.Textbox(
                        label="API key (leave blank to use DASHSCOPE_API_KEY / OPENAI_API_KEY)",
                        type="password",
                        value="",
                    )
                    base_url = gr.Textbox(label="Base URL", value=DEFAULT_LLM_BASE_URL)
                    model_name = gr.Textbox(label="Model name", value=DEFAULT_LLM_MODEL)
                    ensemble_name = gr.Dropdown(
                        choices=["combined", "default"],
                        value=DEFAULT_ENSEMBLE,
                        label="Ensemble checkpoints (combined covers 0.1–20 mol% cat0)",
                    )
                    hardware_name = gr.Radio(
                        choices=["mock", "lab"],
                        value=DEFAULT_HARDWARE_BACKEND if DEFAULT_HARDWARE_BACKEND in {"mock", "lab"} else "mock",
                        label="Hardware backend",
                        info="mock: synthetic chromatogram. lab: HTTP hub + LC-MS (set LAB_HUB_URL / LAB_LCMS_URL).",
                    )
                    t_norm = gr.Number(label="Time-domain normalization (min)", value=DEFAULT_T_NORM)
                    t_min = gr.Number(label="t min (min)", value=DEFAULT_T_MIN)
                    t_max = gr.Number(label="t max (min)", value=DEFAULT_T_MAX)
                    cat_min = gr.Number(label="cat0 min (mole fraction)", value=cat_lo)
                    cat_max = gr.Number(label="cat0 max (mole fraction)", value=cat_hi)
                    max_loops = gr.Slider(minimum=1, maximum=20, step=1, value=5, label="Planned iterations")
                    resume_enabled = gr.Checkbox(label="Resume from autosave", value=False)
                    resume_dir_input = gr.Textbox(
                        label="Resume dir (blank = latest autosave_*)",
                        value="",
                        placeholder="results/web_app/autosave_YYYYMMDD_HHMMSS",
                    )

                with gr.Group():
                    gr.Markdown("### LC-MS integration parameters")
                    rts = gr.Number(label="[S] retention target (min)", value=8.3)
                    rtp = gr.Number(label="[P] retention target (min)", value=9.1)
                    rtis = gr.Number(label="[IS] retention target (min)", value=11.0)
                    factor_s = gr.Number(label="[S] response factor / IS conc.", value=1.43)
                    factor_p = gr.Number(label="[P] response factor / IS conc.", value=1.16)
                    tol = gr.Number(label="Peak-match tolerance (min)", value=0.8)

                start_btn = gr.Button("Start automated loop", variant="primary")
                stop_btn = gr.Button("Emergency stop", variant="stop")
                export_btn = gr.DownloadButton("Pack and download run data", variant="secondary")
                ui_status = gr.Textbox(label="Main status", value="Waiting for command...", interactive=False)

            with gr.Column(scale=3):
                gr.Markdown("### LLM agent reasoning")
                llm_reasoning = gr.Markdown("Waiting for initial reasoning...", elem_classes="reasoning-box")
                with gr.Tabs():
                    with gr.TabItem("AI inference"):
                        with gr.Row():
                            with gr.Column():
                                prob_plot = gr.Plot(label="Ensemble-averaged mechanism probabilities", value=go.Figure())
                            with gr.Column():
                                history_plot = gr.Plot(label="Ensemble entropy", value=go.Figure())
                        hist_table = gr.Dataframe(label="Experiment history", interactive=True)

                    with gr.TabItem("Hardware monitor"):
                        live_lcms_plot = gr.Plot(label="PDA chromatogram", value=go.Figure())
                        live_peak_tbl = gr.Dataframe(label="Integrated peaks", interactive=True)

                    with gr.TabItem("Raw device data"):
                        raw_data_tbl = gr.Dataframe(label="Hub raw_data table", interactive=True)

                    with gr.TabItem("History viewer"):
                        gr.Markdown("Review outputs from completed loops.")
                        with gr.Row():
                            hist_idx = gr.Number(label="Step index", value=1, step=1, precision=0)
                            hist_load_btn = gr.Button("Load step")
                        hist_reason_cache = gr.Markdown("*Select a step and load...*", elem_classes="reasoning-box")
                        with gr.Row():
                            hist_prob_plot = gr.Plot(label="Cached probabilities", value=go.Figure())
                            hist_lcms_plot = gr.Plot(label="Cached LC-MS figure", value=go.Figure())
                        with gr.Row():
                            hist_peaks_data = gr.Dataframe(label="Cached peaks", interactive=True)
                            hist_raw_data = gr.Dataframe(label="Cached raw data", interactive=True)

                    with gr.TabItem("LLM prompt trace"):
                        llm_trace_md = gr.Markdown(
                            "*Waiting for interaction to begin...*",
                            elem_classes=["huge-scroll-box", "wrap-text"],
                        )

                    with gr.TabItem("Terminal logs"):
                        log_board = gr.Textbox(
                            label="Backend log",
                            lines=35,
                            elem_classes="huge-scroll-box",
                            interactive=False,
                        )

        def _on_ensemble_change(name):
            lo, hi = _cat0_defaults(name)
            return gr.update(value=lo), gr.update(value=hi)

        ensemble_name.change(fn=_on_ensemble_change, inputs=[ensemble_name], outputs=[cat_min, cat_max])

        start_btn.click(
            fn=lambda: gr.update(value="Initialize...", interactive=False),
            inputs=[],
            outputs=[start_btn],
        ).then(
            fn=start_automated_loop,
            inputs=[
                api_key,
                base_url,
                model_name,
                ensemble_name,
                hardware_name,
                t_norm,
                t_min,
                t_max,
                cat_min,
                cat_max,
                max_loops,
                rts,
                rtp,
                rtis,
                factor_s,
                factor_p,
                tol,
                resume_enabled,
                resume_dir_input,
            ],
            outputs=[
                workflow_flowchart,
                ui_status,
                log_board,
                llm_reasoning,
                hist_table,
                prob_plot,
                history_plot,
                live_lcms_plot,
                live_peak_tbl,
                raw_data_tbl,
                llm_trace_md,
            ],
        ).then(
            fn=lambda: gr.update(value="Start automated loop", interactive=True),
            inputs=[],
            outputs=[start_btn],
        )

        stop_btn.click(
            fn=lambda: gr.update(value="Stop sent...", interactive=False),
            inputs=[],
            outputs=[stop_btn],
        ).then(
            fn=stop_automated_loop,
            inputs=[],
            outputs=[ui_status],
        ).then(
            fn=lambda: gr.update(value="Emergency stop", interactive=True),
            inputs=[],
            outputs=[stop_btn],
        )

        hist_load_btn.click(
            fn=lambda: gr.update(value="Fetching...", interactive=False),
            inputs=[],
            outputs=[hist_load_btn],
        ).then(
            fn=query_history,
            inputs=[hist_idx],
            outputs=[hist_reason_cache, hist_prob_plot, hist_lcms_plot, hist_peaks_data, hist_raw_data],
        ).then(
            fn=lambda: gr.update(value="Load step", interactive=True),
            inputs=[],
            outputs=[hist_load_btn],
        )

        export_btn.click(
            fn=lambda: gr.update(label="Packing... please wait", interactive=False),
            inputs=[],
            outputs=[export_btn],
        ).then(
            fn=export_clicked,
            inputs=[log_board],
            outputs=[export_btn],
        )

    return demo


def parse_args():
    p = argparse.ArgumentParser(description="Self-Driving Lab Gradio app")
    p.add_argument("--server-name", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7863)
    p.add_argument("--share", action="store_true", help="Create a temporary Gradio public URL")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = build_demo()
    queued = demo.queue(default_concurrency_limit=4)
    launch_kwargs = dict(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        css=CUSTOM_CSS,
    )
    try:
        queued.launch(**launch_kwargs)
    except TypeError:
        launch_kwargs.pop("css", None)
        queued.launch(**launch_kwargs)
