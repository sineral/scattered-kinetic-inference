"""Run ensemble active sampling: uniform random, adaptive random, or LLM.

Usage (from Scattered_Set_Transformer/):
    python src/active_sampling/run.py --method uniform_random --gpu 0
    python src/active_sampling/run.py --method adaptive_random --gpu 0
    python src/active_sampling/run.py --method llm --gpu 0
    # OpenAI instead of DashScope:
    #   python src/active_sampling/run.py --method llm --base-url https://api.openai.com/v1 --llm-model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.chdir(PROJECT_ROOT)

from active_sampling.common import (
    CAT0_MAX,
    CAT0_MIN,
    EnsembleEvaluator,
    KineticOracle,
    T_MIN,
    append_log,
    collect_ensemble_ckpts,
    load_ensemble,
    load_samples,
    load_yaml,
    make_run_dir,
    save_campaign,
    save_summary_plot,
    uncertainty,
)

CAT0_LAYER_COUNTS = [18, 22, 18, 12, 10]
CAT0_LAYERS = [
    (0.010, 0.025),
    (0.025, 0.045),
    (0.045, 0.065),
    (0.065, 0.085),
    (0.085, 0.100),
]


def parse_args():
    p = argparse.ArgumentParser(description="Active sampling with a Deep Ensemble evaluator")
    p.add_argument(
        "--method",
        required=True,
        choices=["uniform_random", "adaptive_random", "llm"],
        help="Acquisition strategy (all three use the same ensemble evaluator)",
    )
    p.add_argument("--ensemble", default="default", choices=["default", "combined"])
    p.add_argument("--ckpt-dir", default=os.path.join(PROJECT_ROOT, "checkpoints"))
    p.add_argument(
        "--samples",
        default=os.path.join(PROJECT_ROOT, "data", "active_sampling", "samples.yaml"),
    )
    p.add_argument(
        "--mech-config",
        default=os.path.join(PROJECT_ROOT, "config", "mechanisms.yaml"),
    )
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--n-max-input", type=int, default=20)
    p.add_argument("--repeats", type=int, default=None, help="Default: 3 for random methods, 1 for LLM")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results", "active_sampling"))
    p.add_argument("--llm-model", default="qwen3.5-plus")
    p.add_argument("--api-key", default=None, help="Default: DASHSCOPE_API_KEY or OPENAI_API_KEY")
    p.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    return p.parse_args()


def scale_quotas(counts, n_target, original_total=80):
    scaled = [max(1, round(c * n_target / original_total)) for c in counts]
    diff = n_target - sum(scaled)
    order = np.argsort(scaled)[::-1]
    i = 0
    while diff != 0 and len(order) > 0:
        j = order[i % len(order)]
        if diff > 0:
            scaled[j] += 1
            diff -= 1
        elif scaled[j] > 1:
            scaled[j] -= 1
            diff += 1
        i += 1
    return scaled


class AdaptiveRandomDesigner:
    """Random draws inside cat0 layers whose time windows adapt from [S],[P]."""

    def __init__(self, t_norm, n_steps, cat0_alpha=0.7, late_st=0.2, late_pt=0.8, early_st=0.8, early_pt=0.2, t_shrink=0.55, t_expand=1.35):
        self.t_norm = t_norm
        self.late_st, self.late_pt = late_st, late_pt
        self.early_st, self.early_pt = early_st, early_pt
        self.t_shrink, self.t_expand = t_shrink, t_expand
        self.layer_quotas = scale_quotas(CAT0_LAYER_COUNTS, n_steps, 80)
        self.layer_t_lo = [T_MIN] * len(CAT0_LAYERS)
        self.layer_t_hi = []
        for lo, hi in CAT0_LAYERS:
            c_mid = 0.5 * (lo + hi)
            self.layer_t_hi.append(t_norm * (0.01 / c_mid) ** cat0_alpha)
        self.cat0_lo, self.cat0_hi = CAT0_MIN, CAT0_MAX

    def _layer_index(self, cat0):
        for i, (lo, hi) in enumerate(CAT0_LAYERS):
            if lo <= cat0 <= hi:
                return i
        mids = [0.5 * (lo + hi) for lo, hi in CAT0_LAYERS]
        return int(np.argmin([abs(m - cat0) for m in mids]))

    def classify(self, St, Pt):
        if St < self.late_st and Pt > self.late_pt:
            return "late"
        if St > self.early_st and Pt < self.early_pt:
            return "early"
        return "ok"

    def update(self, St, Pt, cat0):
        tag = self.classify(St, Pt)
        i = self._layer_index(cat0)
        t_lo, t_hi = self.layer_t_lo[i], self.layer_t_hi[i]
        if tag == "late":
            self.layer_t_hi[i] = max(t_lo + 0.05, t_hi * self.t_shrink)
            self.cat0_hi = max(self.cat0_lo + 0.005, self.cat0_hi * 0.92)
        elif tag == "early":
            self.layer_t_hi[i] = min(self.t_norm, t_hi * self.t_expand)
            self.cat0_hi = min(CAT0_MAX, self.cat0_hi * 1.03)
        return tag, i

    def suggest(self):
        avail = [i for i, q in enumerate(self.layer_quotas) if q > 0]
        if not avail:
            t = np.exp(np.random.uniform(np.log(T_MIN), np.log(self.t_norm)))
            cat0 = np.random.uniform(self.cat0_lo, self.cat0_hi)
            return float(t), float(cat0), -1
        i = int(np.random.choice(avail))
        lo, hi = CAT0_LAYERS[i]
        cat0 = np.random.uniform(lo, hi)
        t_lo = max(T_MIN, self.layer_t_lo[i])
        t_hi = max(t_lo + 1e-3, min(self.layer_t_hi[i], self.t_norm))
        t = float(np.exp(np.random.uniform(np.log(t_lo), np.log(t_hi))))
        self.layer_quotas[i] -= 1
        return t, cat0, i


class LLMDesigner:
    def __init__(self, oracle, evaluator, mech_descriptions, model_name, api_key, base_url, log_path, mech_idx):
        from openai import OpenAI

        self.oracle = oracle
        self.evaluator = evaluator
        self.mech_descriptions = mech_descriptions
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.log_path = log_path
        self.mech_idx = mech_idx
        self.observed = []
        self.current_probs = None

    def _log(self, msg):
        append_log(self.log_path, msg)

    def suggest(self):
        is_initial = len(self.observed) == 0
        system_prompt = f"""You are an expert Active Learning Scientist in Chemical Kinetics.

Your Goal:
Identify the true mechanism (M1-M20) by sequentially designing the next experiment to minimize prediction entropy as efficiently as possible.

System Constraints:
- Time (t): {T_MIN} min to {self.oracle.t_norm:.1f} min
- Catalyst loading (cat0): {CAT0_MIN} to {CAT0_MAX}
- Strict minimum for cat0: {CAT0_MIN}
- Standard initial condition: S0 = 1.0, P0 = 0.0 (unitless concentrations)

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
- The uncertainty signal is provided by a Deep Ensemble of {len(self.evaluator.models)} independently trained Set Transformer models.
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
Return JSON ONLY in the following format:
{{"t": float, "cat0": float, "reasoning": "brief explanation of why this experiment should be most informative"}}"""

        if is_initial:
            all_mechs = "\n".join(
                f"- {name}: {self.mech_descriptions.get(name, 'Unknown Mechanism')}"
                for name in [f"M{i+1}" for i in range(20)]
            )
            user_content = f"""Current Status:
- Phase: 1. Initialization
- Steps: 0
- Prior Knowledge: You are given the full landscape of 20 possible mechanisms.

Candidate Mechanisms Mapping:
{all_mechs}

Task:
No previous experimental history is available. Propose an initial widely-informative condition (t, cat0) to kick off the mechanistic discrimination.
Return JSON ONLY.
"""
            log_entropy, log_gt = "Initial", "Initial"
        else:
            probs = self.current_probs
            current_entropy, mutual_info = uncertainty(probs, self.evaluator.member_probs)
            top_indices = np.argsort(probs)[-5:][::-1]
            candidates = "\n".join(
                f"- M{idx+1} ({probs[idx]:.1%}): {self.mech_descriptions.get(f'M{idx+1}', 'Unknown Mechanism')}"
                for idx in top_indices
            )
            history = "\n".join(
                f"- Exp {i+1}: t = {s['t']:.2f}, cat0 = {s['cat0']:.3f} | Observed: [S] = {s['St']:.3f}, [P] = {s.get('Pt', 0.0):.3f}"
                for i, s in enumerate(self.observed)
            )
            user_content = f"""Current Status:
- Phase: 2. Iterative Active Learning
- Steps: {len(self.observed)}
- Ensemble Prediction Entropy (total uncertainty): {current_entropy:.4f} (Goal: < 0.5)
- Ensemble Disagreement / Mutual Information (epistemic uncertainty): {mutual_info:.4f}

Top Candidate Mechanisms:
{candidates}

Experimental History:
{history}

Task:
Reason about the experimental history and the top candidate mechanisms, then propose the next experiment.
Return JSON ONLY."""
            log_entropy = f"{current_entropy:.4f}"
            log_gt = f"{probs[self.mech_idx]:.2%}"

        step_num = len(self.observed) + 1
        self._log("\n" + "=" * 80)
        self._log(f"=== PHASE: {'1. INITIALIZATION' if is_initial else '2. ITERATIVE LOOP'} [Step {step_num}] ===")
        self._log(f"Pre-Decision State -> Entropy: {log_entropy} | GT_Prob: {log_gt}")
        self._log("-" * 40 + "\n[SYSTEM PROMPT]:\n" + system_prompt)
        self._log("-" * 40 + "\n[USER CONTENT / HISTORY]:\n" + user_content)
        self._log("=" * 80 + "\n")

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
            )
            content = res.choices[0].message.content
            self._log(f"[LLM RESPONSE]:\n{content}\n")
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            return json.loads(content)
        except Exception as e:
            err = f"LLM Error: {e}"
            print(err)
            self._log(f"[ERROR]: {err}")
            return {
                "t": self.oracle.t_norm * np.random.uniform(0.1, 0.5),
                "cat0": 0.05,
                "reasoning": "Fallback due to API error.",
            }

    def conclude(self):
        if self.current_probs is None:
            return
        top_idx = int(np.argmax(self.current_probs))
        top_mech = f"M{top_idx+1}"
        top_prob = self.current_probs[top_idx]
        history = "\n".join(
            f"- Exp {i+1}: t={s['t']:.2f}, cat0={s['cat0']:.3f} | [S]={s['St']:.3f}, [P]={s.get('Pt', 0.0):.3f}"
            for i, s in enumerate(self.observed)
        )
        prompt = f"""Current Status:
- Phase: 3. Final Conclusion
- The active learning session has concluded after {len(self.observed)} steps.
- Final Neural Model Prediction: {top_mech} (Confidence: {top_prob:.1%})
- Mechanism Description: {self.mech_descriptions.get(top_mech, "")}

Full Experimental Trajectory:
{history}

Please act as a Chemical Kinetics Expert and provide a formal conclusion:
1. Does the collected experimental history logically support the prediction of {top_mech}?
2. Which specific experiment step(s) contributed the most critical insight for this mechanism over others?
Provide a concise, scientifically grounded response.
"""
        self._log("\n" + "=" * 80)
        self._log("=== PHASE: 3. FINAL CONCLUSION ===")
        self._log("=" * 80 + "\n")
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a highly capable Chemical Kinetics Scientist. Provide expert final evaluation.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            conclusion = res.choices[0].message.content
            print(f"[FINAL LLM ASSESSMENT]:\n{conclusion}")
            self._log(f"\n[FINAL LLM ASSESSMENT]:\n{conclusion}\n")
        except Exception as e:
            print(f"Conclusion Error: {e}")


def record_step(step, sample, probs, member_probs, mech_idx, extra=None):
    ent, mi = uncertainty(probs, member_probs)
    p_gt = float(probs[mech_idx])
    row = {**sample, "step": step, "entropy": ent, "mutual_info": mi, "game_prob_gt": p_gt}
    if extra:
        row.update(extra)
    member_recs = []
    for m_idx, mp in enumerate(member_probs):
        rec = {"step": step, "member_id": f"model_{m_idx+1}", "pred_label": f"M{int(np.argmax(mp))+1}"}
        rec.update({f"P_M{c+1}": float(mp[c]) for c in range(20)})
        member_recs.append(rec)
    top_k = np.argsort(probs)[-5:][::-1]
    top_str = ", ".join([f"M{idx+1}: {probs[idx]:.1%}" for idx in top_k])
    return row, member_recs, ent, mi, p_gt, top_str


def write_header(log_path, method, sample, oracle, ckpt_paths, device, args, extra_lines=None):
    append_log(log_path, "=" * 80)
    append_log(log_path, f"EXPERIMENT LOG")
    append_log(log_path, "=" * 80)
    append_log(log_path, "CORE CONFIGURATION:")
    append_log(log_path, f"  - Strategy: {method}")
    append_log(log_path, f"  - Target Mechanism: {sample['mech_id']}")
    append_log(log_path, f"  - Instance ID: {sample['instance_id']}")
    if "inhibitor" in sample["k"]:
        append_log(log_path, f"  - Inhibitor Initial Value: {sample['k']['inhibitor']}")
    append_log(log_path, f"  - Simulation Noise (std): {args.noise_std}")
    append_log(log_path, f"  - GPU Device: {device}")
    append_log(log_path, f"  - Total Steps: {args.n_steps}")
    append_log(log_path, f"  - Max Input Points: {args.n_max_input}")
    append_log(log_path, f"  - T_norm: {oracle.t_norm:.2f} min")
    append_log(log_path, f"  - Search Space: t=[{T_MIN}, {oracle.t_norm:.1f}] min, cat0=[{CAT0_MIN}, {CAT0_MAX}]")
    if extra_lines:
        for line in extra_lines:
            append_log(log_path, line)
    append_log(log_path, "PATHS:")
    append_log(log_path, f"  - Deep Ensemble Checkpoints ({len(ckpt_paths)} members):")
    for p in ckpt_paths:
        append_log(log_path, f"      * {os.path.basename(p)}")
    append_log(log_path, "=" * 80 + "\n")


def run_one(method, sample, mech_bank, models, feat_idx, ckpt_paths, device, args):
    mech_def = mech_bank[sample["mech_id"]]
    oracle = KineticOracle(mech_def, sample["k"], sample["mech_id"], sample["instance_id"])
    evaluator = EnsembleEvaluator(
        models, feat_idx, device, n_max_input=args.n_max_input, noise_std=args.noise_std
    )
    extra_folder = args.llm_model.replace("/", "_") if method == "llm" else None
    save_dir = make_run_dir(
        args.out_dir, method, args.noise_std, sample["mech_id"], sample["instance_id"], extra=extra_folder
    )
    log_path = os.path.join(save_dir, "experiment.log")
    extra_lines = None
    designer = None
    llm = None
    if method == "adaptive_random":
        designer = AdaptiveRandomDesigner(oracle.t_norm, args.n_steps)
        extra_lines = [f"  - Initial layer t_hi: {[round(x, 2) for x in designer.layer_t_hi]}"]
    if method == "llm":
        extra_lines = [f"  - LLM Model: {args.llm_model}"]
        descs = {name: m.get("description", "")[:150] for name, m in mech_bank.items()}
        llm = LLMDesigner(
            oracle,
            evaluator,
            descs,
            args.llm_model,
            args.api_key,
            args.base_url,
            log_path,
            sample["mech_idx"],
        )
    write_header(log_path, method, sample, oracle, ckpt_paths, device, args, extra_lines=extra_lines)

    history, member_records = [], []
    observed = []
    print(f"Oracle Ready: {sample['mech_id']} | instance={sample['instance_id']} | T_norm={oracle.t_norm:.1f} min")
    for step in range(1, args.n_steps + 1):
        extra = {}
        if method == "uniform_random":
            t = float(np.random.uniform(T_MIN, oracle.t_norm))
            cat0 = float(np.random.uniform(CAT0_MIN, CAT0_MAX))
        elif method == "adaptive_random":
            t, cat0, layer_i = designer.suggest()
            extra["layer"] = layer_i
        else:
            sugg = llm.suggest()
            t = float(sugg.get("t", oracle.t_norm * 0.3))
            cat0 = float(sugg.get("cat0", 0.05))
            extra["reasoning"] = sugg.get("reasoning", "")

        sample_pt = oracle.observe(t, cat0)
        observed.append(sample_pt)
        if method == "adaptive_random":
            tag, _ = designer.update(sample_pt["St"], sample_pt["Pt"], sample_pt["cat0"])
            extra["design_tag"] = tag
        if method == "llm":
            llm.observed = observed

        probs = evaluator.predict(observed)
        if method == "llm":
            llm.current_probs = probs
        row, recs, ent, mi, p_gt, top_str = record_step(
            step, sample_pt, probs, evaluator.member_probs, sample["mech_idx"], extra=extra
        )
        history.append(row)
        member_records.extend(recs)
        print(
            f"[{step:<2}] t={sample_pt['t']:<6.2f} | c={sample_pt['cat0']:<5.3f} | "
            f"RP={sample_pt['RP']:<4.2f} | Ent={ent:<6.4f} | MI={mi:<5.3f} | P_GT={p_gt:<6.2%}"
        )
        append_log(
            log_path,
            f"Step {step} Executed: t={sample_pt['t']:.2f}, c={sample_pt['cat0']:.3f} | "
            f"[S]={sample_pt['St']:.3f}, [P]={sample_pt['Pt']:.3f}, RP={sample_pt['RP']:.2f}",
        )
        append_log(log_path, f"   >> Post-Exp State: Ent={ent:.3f}, MI={mi:.3f}, P_GT={p_gt:.1%} | Top5: {top_str}")

    if method == "llm":
        llm.conclude()
    df = save_campaign(save_dir, history, member_records)
    save_summary_plot(save_dir, oracle, df)
    append_log(log_path, f"\nFinal results saved to: {save_dir}")
    print(f"Saved {save_dir}")
    return save_dir


def main():
    args = parse_args()
    if args.repeats is None:
        args.repeats = 1 if args.method == "llm" else 3
    if args.gpu is not None:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.method == "llm":
        args.api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not args.api_key:
            raise SystemExit("LLM method needs --api-key or DASHSCOPE_API_KEY / OPENAI_API_KEY")

    samples_path = args.samples if os.path.isabs(args.samples) else os.path.join(PROJECT_ROOT, args.samples)
    mech_path = args.mech_config if os.path.isabs(args.mech_config) else os.path.join(PROJECT_ROOT, args.mech_config)
    ckpt_dir = args.ckpt_dir if os.path.isabs(args.ckpt_dir) else os.path.join(PROJECT_ROOT, args.ckpt_dir)
    samples = load_samples(samples_path)
    mech_bank = load_yaml(mech_path)["mechanisms"]
    ckpt_paths = collect_ensemble_ckpts(ckpt_dir, args.ensemble)
    models, feat_idx = load_ensemble(ckpt_paths, device)
    print(f"Deep Ensemble Ready: {len(models)} members on {device}")
    print(
        f"Starting {args.method}: {len(samples)} samples x {args.repeats} repeats, "
        f"noise={args.noise_std}, steps={args.n_steps}"
    )

    for sample in samples:
        for r in range(args.repeats):
            print(
                f"\n---> [{args.method} | {sample['mech_id']} | {sample['instance_id']} | "
                f"repeat {r+1}/{args.repeats}]"
            )
            try:
                run_one(args.method, sample, mech_bank, models, feat_idx, ckpt_paths, device, args)
            except Exception as e:
                print(f"[ERROR] {sample['mech_id']} repeat {r+1}: {e}")
                traceback.print_exc()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if args.method == "llm":
                time.sleep(1)
    print("Done.")


if __name__ == "__main__":
    main()
