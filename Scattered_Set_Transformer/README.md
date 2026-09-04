# Scattered Set Transformer

Public code for **20-class reaction mechanism classification** from **scattered kinetic points** (M1-M20), using a Set Transformer.

Two Set Transformer ensembles are shipped. They share the same architecture; they differ in the **initial catalyst loading (`cat0`) range** seen during training.

| Name | Data | Checkpoints | `cat0` coverage |
| --- | --- | --- | --- |
| **Default** | `data/default/` | `checkpoints/default_member_{0-4}.pth` | In-distribution only: **1–10 mol%** (`cat0 ∈ [0.01, 0.10]`) |
| **Combined** | `data/combined/` | `checkpoints/combined_member_{0-4}.pth` | Default **plus** low-`cat0` **0.1–1 mol%** (`[0.001, 0.01]`) **plus** high-`cat0` **10–20 mol%** (`[0.10, 0.20]`) |

The combined model is trained on a wider `cat0` window than the default 1–10 mol% split.

Omitted on purpose: feature-addition studies, strategy ablations, attention notebooks, and extra OOD-only splits.

## Data

Parquet files (~3 GB) are on Zenodo: **https://doi.org/10.5281/zenodo.21991341**. Schema: `data/README`.

Download `scattered-kinetic-data.zip` and unpack **in this folder** (`Scattered_Set_Transformer/`):

```bash
unzip scattered-kinetic-data.zip
```

That creates `data/default/{train,val,test}.parquet` and `data/combined/{train,val,test}.parquet`. To rebuild the zip for a new Zenodo upload: `bash data/pack_zenodo.sh`.

To regenerate the **default** split from the M1–M20 ODEs (screen `k`, then sample 80 points per instance), see `data_generation/README.md`. The published Zenodo files are enough for training and evaluation; generation is optional and CPU-heavy.

## Install

```bash
conda create -n scattered_bench python=3.10
conda activate scattered_bench
pip install -r requirements.txt
```

`requirements.txt` pins a **CUDA 12.1** PyTorch wheel. For CPU or another CUDA version, change the `--extra-index-url` on the first line, or install PyTorch from https://pytorch.org first.

Run all commands below from **this folder** (`Scattered_Set_Transformer/`), not the git monorepo root. Notebooks (`experiments/*.ipynb`) need Jupyter (`pip install notebook` or VS Code / Cursor).

## Generate data (optional)

The published parquet files are enough to train and evaluate. To rebuild the **default** (1–10 mol% `cat0`) split from the ODEs:

```bash
python data_generation/screen_k.py M8          # repeat for M1–M20
python data_generation/generate_scattered_dataset.py
```

Details, smoke-test flags, and the 80-point sampling design: `data_generation/README.md`. The **combined** split is not produced by these scripts.

## Evaluate shipped ensembles

Default: 10 points per group, no extra noise. `--ensemble` selects both the weights (`{name}_member_*.pth`) and, unless `--data-dir` is set, the matching dataset.

```bash
python src/eval.py --ensemble default --n-points 10 --noise-std 0.0
python src/eval.py --ensemble combined --n-points 10 --noise-std 0.0
```

Single member:

```bash
python src/eval.py --data-dir data/default --ckpt checkpoints/default_member_0.pth
```

Reference (combined test, n=10, noise=0, 87308 groups): member mean Top-1 **0.8663**, Top-3 **0.9889**.

## Train

Each run overwrites `--save-path`. Train noise is drawn from `{0, 0.5, 1, 2}%`; validation uses 1%. Architecture: hidden 256, 8 heads, 6 layers, 16 ISAB inducing points, dropout 0.1.

```bash
python src/train.py --data-dir data/default --save-path checkpoints/default_member_0.pth --gpu 0
python src/train.py --data-dir data/combined --save-path checkpoints/combined_member_0.pth --gpu 0
```

Repeat with different `--save-path` (and independent runs) to build an ensemble.

## Active sampling

Three ensemble strategies share the same Deep Ensemble evaluator (`checkpoints/default_member_*.pth`) and the same 10-step budget on `(t, cat0)`. They differ only in how the next condition is chosen:

| Method | CLI | Who chooses `(t, cat0)` |
| --- | --- | --- |
| Uniform random | `--method uniform_random` | i.i.d. uniform in the design box |
| Adaptive random | `--method adaptive_random` | random inside `cat0` layers whose time windows adapt from `[S],[P]` |
| LLM-guided | `--method llm` | language model, given history + ensemble entropy |

The 20 ground-truth instances (one per mechanism) are listed with explicit rate constants in `data/active_sampling/samples.yaml`. ODE definitions are in `config/mechanisms.yaml`.

```bash
python src/active_sampling/run.py --method uniform_random --gpu 0
python src/active_sampling/run.py --method adaptive_random --gpu 0
# LLM: DashScope by default (qwen3.5-plus). For OpenAI, set OPENAI_API_KEY and
#   --base-url https://api.openai.com/v1 --llm-model gpt-4o-mini   (example)
python src/active_sampling/run.py --method llm --gpu 0
```

Defaults: 10 steps, observation noise 1%, `cat0 ∈ [0.01, 0.10]`, `S0=1`, `P0=0`. Uniform/adaptive default to 3 repeats per instance; LLM defaults to 1. Outputs go to `results/active_sampling/` (`results.csv`, `member_probs.csv`, `experiment.log`, `summary_plot.png`).

## Predict from experiment

Notebook: `experiments/predict_mechanism.ipynb` — point `DATA_PATH` at your own CSV. Template: `experiments/example.csv` (`Step`, `Time`, `Cat0`, `[S]`, `[P]`).

- Photoinduced iron-catalyzed cycloisomerization (Lehnherr *et al.*, *JACS* 2018): `experiments/photoinduced_iron_cycloisomerization/` (default 5-member ensemble).
- Ru-catalyzed RCM (closed-loop, 0.1–1 mol%): `experiments/ru_catalyzed_rcm/` (combined 5-member ensemble; three runs, notebook visualizes one).
- Proline-catalyzed intramolecular aldol (closed-loop, 10–20 mol%): `experiments/proline_catalyzed_intramolecular_aldol/` (combined 5-member ensemble; three runs, notebook visualizes one).

## Self-Driving Lab web app

Gradio closed loop: an LLM proposes `(t, cat0)`, a 5-member ensemble updates P(M1–M20). Default hardware is a **mock** chromatogram so the app runs without instruments. Lab HTTP hooks are documented in `web_app/README.md`.

```bash
export DASHSCOPE_API_KEY=...   # or OPENAI_API_KEY; OpenAI also needs OPENAI_BASE_URL
python web_app/app.py          # http://127.0.0.1:7863  (mock hardware still needs an LLM key to run the loop)
```

## Protocol

- Input: unordered 20-D set of kinetic points (6 basic + 14 engineered features).
- Set size at train time is sampled in `[6, 15]`.
- Relative Gaussian noise is applied to `St` and `Pt` only, then features are recomputed; concentrations are clipped at 0.

## Layout

```
src/train.py
src/eval.py
src/active_sampling/run.py
src/models/set_transformer.py
src/utils/scattered_dataset.py
config/mechanisms.yaml
config/k_generation.yaml
data_generation/
data/README
data/default/
data/combined/
data/active_sampling/samples.yaml
experiments/predict_mechanism.ipynb
experiments/photoinduced_iron_cycloisomerization/
experiments/ru_catalyzed_rcm/
experiments/proline_catalyzed_intramolecular_aldol/
web_app/app.py
checkpoints/default_member_0.pth ... default_member_4.pth
checkpoints/combined_member_0.pth ... combined_member_4.pth
```

## License

MIT License. Copyright (c) 2026 Zheng Xian, Mo-Group, Zhejiang University. See `LICENSE`.
