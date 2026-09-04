"""Deep-ensemble Set Transformer + LLM experiment designer."""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from openai import OpenAI
from scipy.stats import entropy

from eval import load_model
from utils.scattered_dataset import ScatteredKineticDataset
from web_app.config import MECH_CONFIG_PATH, N_MAX_INPUT, N_MECHANISMS

LogFn = Optional[Callable[[str], None]]


def ensemble_uncertainty(probs_list):
    """Decompose ensemble uncertainty (nats, same as ``scipy.stats.entropy``).

    Returns mean_probs, total_entropy (predictive), expected_entropy (aleatoric),
    mutual_information (epistemic / BALD).
    """
    eps = 1e-12
    p = np.stack(probs_list, axis=0)
    mean_p = p.mean(axis=0)
    total_entropy = float(-np.sum(mean_p * np.log(mean_p + eps)))
    per_model_entropy = -np.sum(p * np.log(p + eps), axis=1)
    expected_entropy = float(per_model_entropy.mean())
    mutual_info = float(max(0.0, total_entropy - expected_entropy))
    return mean_p, total_entropy, expected_entropy, mutual_info


def build_member_records(loop, per_model_probs, timestamp=None):
    """One CSV row per ensemble member, with P_M1 ... P_M20."""
    ts = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")
    records = []
    for m_idx, mp in enumerate(per_model_probs):
        mp = np.asarray(mp)
        rec = {
            "loop": int(loop),
            "timestamp": ts,
            "member_id": f"model_{m_idx + 1}",
            "pred_label": f"M{int(np.argmax(mp)) + 1}",
        }
        rec.update({f"P_M{c + 1}": float(mp[c]) for c in range(len(mp))})
        records.append(rec)
    return records


class RealWorldLearner:
    """Load shipped ensemble weights and ask an LLM for the next (t, cat0)."""

    def __init__(self, config: dict, log_callback: LogFn = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        self.log_callback = log_callback

        self.observed_samples = []
        self.history_metrics = []
        self.current_probs = None
        self.current_uncertainty = None
        self.llm_history = []

        self.t_norm = float(self.config.get("t_norm", 300.0))
        self.t_min = float(self.config.get("t_min", 15.0))
        self.t_max = float(self.config.get("t_max", self.t_norm))
        self.cat_min = float(self.config.get("cat_min", 0.01))
        self.cat_max = float(self.config.get("cat_max", 0.10))
        self.n_max_input = int(self.config.get("n_max_input", N_MAX_INPUT))

        self._load_models()
        self._load_mechanism_text()

    def _log(self, msg: str) -> None:
        if self.log_callback:
            self.log_callback(msg)

    def _load_mechanism_text(self) -> None:
        path = self.config.get("mech_config_path", MECH_CONFIG_PATH)
        try:
            with open(path, "r", encoding="utf-8") as f:
                all_mechs = yaml.safe_load(f)["mechanisms"]
            self.config["mech_descriptions"] = {
                name: m.get("description", "")[:150] for name, m in all_mechs.items()
            }
        except Exception as e:
            self._log(f"[Agent] Warning: failed to read mechanism config: {e}")
            self.config["mech_descriptions"] = {}

    def _load_models(self) -> None:
        self.feat_idx = list(range(N_MECHANISMS))
        self.models = []
        ckpt_paths = self.config.get("ckpt_paths") or [self.config.get("ckpt_path")]
        for p in ckpt_paths:
            try:
                model, params, _ckpt = load_model(p, self.device)
                self.models.append(model)
                self.feat_idx = list(range(int(params.get("in_dim", N_MECHANISMS))))
                self._log(f"[Agent] Ensemble member loaded: {os.path.basename(p)}")
            except Exception as e:
                self._log(f"[Agent] Failed to load {os.path.basename(str(p))}: {e}")
        self.model = self.models[0] if self.models else None
        if not self.models:
            self._log("[Agent] Fatal: no ensemble member could be loaded.")
        else:
            self._log(f"[Agent] Deep ensemble ready with {len(self.models)} model(s).")

    def _build_model_input(self, samples):
        df = pd.DataFrame(samples)
        ds = ScatteredKineticDataset(df, use_tau=False, noise_std=0.0)
        feat, _ = ds[0]
        n_real = min(len(df), self.n_max_input)
        inp = torch.zeros((1, self.n_max_input, feat.shape[1]), device=self.device)
        inp[0, :n_real, :] = feat[:n_real, self.feat_idx]
        # True = real observation; False = padding (model uses key_padding_mask = ~mask).
        mask = torch.zeros((1, self.n_max_input), dtype=torch.bool, device=self.device)
        mask[0, :n_real] = True
        return inp, mask

    def predict_ensemble(self, samples):
        if not samples or not self.models:
            uniform = np.ones(N_MECHANISMS) / float(N_MECHANISMS)
            h0 = float(np.log(N_MECHANISMS))
            return {
                "mean_probs": uniform,
                "per_model_probs": [uniform],
                "total_entropy": h0,
                "expected_entropy": h0,
            }

        inp, mask = self._build_model_input(samples)
        per_model = []
        with torch.no_grad():
            for m in self.models:
                logits = m(inp, mask=mask)
                p = torch.softmax(logits, dim=1).cpu().numpy()[0]
                per_model.append(p)
        mean_p, h_tot, h_exp, _ = ensemble_uncertainty(per_model)
        return {
            "mean_probs": mean_p,
            "per_model_probs": per_model,
            "total_entropy": h_tot,
            "expected_entropy": h_exp,
        }

    def predict(self, samples):
        res = self.predict_ensemble(samples)
        self.current_uncertainty = res
        return res["mean_probs"]

    def get_llm_suggestion(self):
        is_initial = len(self.observed_samples) == 0

        system_prompt = f"""You are an expert Active Learning Scientist in Chemical Kinetics.

Your Goal:
Identify the true mechanism (M1-M20) by sequentially designing the next experiment to minimize prediction entropy as efficiently as possible.

System Constraints (HARD LIMITS — must be strictly obeyed):
- Time (t): {self.t_min:.1f} min to {self.t_max:.1f} min (inclusive)
- Catalyst loading (cat0): {self.cat_min} to {self.cat_max} (inclusive)
- Standard initial condition: S0 = 1.0, P0 = 0.0 (unitless concentrations)
- Any proposal with t or cat0 outside these inclusive ranges is INVALID and will be REJECTED, and you will be asked to re-propose. NEVER output values outside these ranges.

Important Real-Experiment Principle:
You do NOT know reaction progress (RP) in advance when proposing the next experiment.
Therefore, you must choose conditions based only on:
1. previous experimental observations,
2. current model uncertainty,
3. mechanistic discrimination value,
4. practical kinetic sampling rules.

Your task is to propose the next single experiment BEFORE seeing its outcome.

STRATEGY GUIDANCE:

1. Uncertainty-Driven Design
- The uncertainty signal is provided by a Deep Ensemble of {len(self.models)} independently trained Set Transformer models.
- Prediction Entropy is calculated from the ensemble-averaged prediction and represents the total uncertainty of the current mechanism assignment.
- Use the current entropy and top candidate mechanisms to design experiments that most strongly separate their predicted behaviors.
- Prioritize experiments that can discriminate specific mechanistic features, such as catalyst order dependence, reversible catalyst association/dimerization, resting-state accumulation, catalyst inhibition/deactivation, auxiliary-species-mediated pathways, reversibility / approach to equilibrium.

2. Practical Kinetic Sampling Rules
- In real kinetic experiments, the most informative observations usually lie in transitional regions of the trajectory rather than in flat regions.
- Avoid proposing conditions that are likely to place the system either too early in the trajectory, where conversion is negligible, or in a plateau region.

3. How to Interpret Previous Experimental Outcomes
- Very low [S] and very high [P] -> chosen condition was too late / too fast -> move to earlier observation times and/or lower cat0.
- Very little product formation -> chosen condition was too early / too slow -> increase t and/or increase cat0.

4. Balance Exploration and Efficiency
- Do not repeat or slightly perturb conditions that are likely to provide redundant information.

5. Output Requirements
- t and cat0 MUST lie within the hard limits in System Constraints. Double-check before answering.
Return JSON ONLY in the following format:
{{"t": float, "cat0": float, "reasoning": "brief explanation of why this experiment should be most informative"}}"""

        if is_initial:
            all_mechs_info = []
            for i in range(N_MECHANISMS):
                mech_name = f"M{i + 1}"
                desc = self.config.get("mech_descriptions", {}).get(mech_name, "Unknown mechanism")
                all_mechs_info.append(f"- {mech_name}: {desc}")
            all_mechs_str = "\n".join(all_mechs_info)
            user_content = f"""Current Status:
- Phase: 1. Initialization
- Steps: 0
- Prior Knowledge: You are given the full landscape of 20 possible mechanisms.

Candidate Mechanisms Mapping:
{all_mechs_str}

Task:
No previous experimental history is available. Propose an initial widely-informative condition (t, cat0) to kick off the mechanistic discrimination.
Return JSON ONLY.
"""
        else:
            probs = self.current_probs if self.current_probs is not None else self.predict(self.observed_samples)
            unc = self.current_uncertainty or {}
            current_entropy = unc.get("total_entropy", float(entropy(probs)))
            top_indices = np.argsort(probs)[-5:][::-1]
            candidates_info = []
            for idx in top_indices:
                mech_name = f"M{idx + 1}"
                desc = self.config.get("mech_descriptions", {}).get(mech_name, "Unknown mechanism")
                candidates_info.append(f"- {mech_name} ({probs[idx]:.1%}): {desc}")
            history_lines = [
                f"- Exp {i + 1}: t = {s['t']:.2f}, cat0 = {s['cat0']:.3f} | Observed: [S] = {s['St']:.3f}, [P] = {s.get('Pt', 0.0):.3f}"
                for i, s in enumerate(self.observed_samples)
            ]
            user_content = f"""Current Status:
- Phase: 2. Iterative Active Learning
- Steps: {len(self.observed_samples)}
- Ensemble Entropy: {current_entropy:.4f} (Goal: < 0.5)

Top Candidate Mechanisms:
{chr(10).join(candidates_info)}

Experimental History:
{chr(10).join(history_lines)}

Task:
Reason about the experimental history and the top candidate mechanisms, then propose the next experiment.
Return JSON ONLY."""

        max_retries = 4
        retry_delays = [2, 5, 10, 10]
        base_user_content = user_content
        last_sugg = None

        def _in_range(t_val, cat_val):
            return (self.t_min <= t_val <= self.t_max) and (self.cat_min <= cat_val <= self.cat_max)

        for attempt in range(max_retries):
            try:
                self._log(f"[LLM] Prompt length {len(user_content)} chars (attempt {attempt + 1}/{max_retries})")
                res = self.client.chat.completions.create(
                    model=self.config["model_name"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                )
                content = res.choices[0].message.content
                self._log("[LLM] Raw response received.")
                if content is None:
                    raise ValueError("Empty LLM content")
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                sugg_dict = json.loads(content)
                last_sugg = sugg_dict
                t_val = float(sugg_dict.get("t", self.t_min))
                cat_val = float(sugg_dict.get("cat0", self.cat_min))
                if not _in_range(t_val, cat_val):
                    self._log(
                        f"[LLM] Out-of-range (t={t_val:.2f}, cat0={cat_val:.4f}); "
                        f"valid t in [{self.t_min:.1f},{self.t_max:.1f}], "
                        f"cat0 in [{self.cat_min},{self.cat_max}]. Re-asking."
                    )
                    user_content = base_user_content + (
                        f"\n\nIMPORTANT: Your previous proposal (t={t_val:.2f}, cat0={cat_val:.4f}) "
                        f"was OUT OF RANGE and rejected. You MUST return t within "
                        f"[{self.t_min:.1f}, {self.t_max:.1f}] and cat0 within "
                        f"[{self.cat_min}, {self.cat_max}] (inclusive)."
                    )
                    raise ValueError("Proposal out of range")

                hist_entry = {
                    "step": len(self.observed_samples),
                    "system": system_prompt,
                    "user": user_content,
                    "response": content,
                }
                self.llm_history.append(hist_entry)
                return sugg_dict
            except Exception as e:
                self._log(f"[LLM] Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                    self._log(f"[LLM] Waiting {delay}s before retry...")
                    time.sleep(delay)

        if last_sugg is not None:
            t_c = float(np.clip(float(last_sugg.get("t", self.t_min)), self.t_min, self.t_max))
            c_c = float(np.clip(float(last_sugg.get("cat0", self.cat_min)), self.cat_min, self.cat_max))
            self._log(f"[LLM] Fallback clamp -> t={t_c:.2f}, cat0={c_c:.4f}")
            fb = {
                "t": t_c,
                "cat0": c_c,
                "reasoning": last_sugg.get("reasoning", "Clamped to valid range after out-of-range retries."),
            }
        else:
            t_c = float(np.clip(self.t_norm * 0.1, self.t_min, self.t_max))
            self._log(f"[LLM] No valid response. Safe default t={t_c:.2f}, cat0={self.cat_max:.4f}")
            fb = {"t": t_c, "cat0": self.cat_max, "reasoning": "Fallback due to repeated API/range error."}

        hist_entry = {
            "step": len(self.observed_samples),
            "system": system_prompt,
            "user": user_content,
            "response": json.dumps(fb, ensure_ascii=False),
        }
        self.llm_history.append(hist_entry)
        return fb
