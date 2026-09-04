# Dataset generation

Rebuild the scattered kinetic tables from ODEs. Two steps: **screen** rate-constant vectors, then **sample** 80 points per passing instance with RP × `cat0` quotas.

The published parquet files on Zenodo were produced this way for the **default** split (`cat0 ∈ [0.01, 0.10]`, 10 000 instances per mechanism before the 70/15/15 split). You do not need to rerun this to train or evaluate; it is for reproducing or extending the data.

## Requirements

Same environment as the rest of the repo (`pip install -r requirements.txt`). Screening and generation are CPU-heavy (ODE solves in many processes).

Configs:

| File | Role |
| --- | --- |
| `config/mechanisms.yaml` | Species and mass-action reactions (M1–M20) |
| `config/k_generation.yaml` | Log-uniform `k` prior, `initial_ranges`, and conversion-window screening rules |

## 1. Screen kinetic constants

For each mechanism, draw `k` from a log-uniform prior, require ≥50% conversion at 1 mol% catalyst, then apply the mechanism’s screening expressions (if any).

```bash
# From Scattered_Set_Transformer/
python data_generation/screen_k.py M8
python data_generation/screen_k.py M1 --n-accept 50 --max-attempts 20000 --jobs 8
```

Paper-scale defaults: `--n-accept 20000`, `--max-attempts 50000000`. A smoke test should use much smaller values.

Outputs go to `data/k_screening/<mech>/`:

- `{mech}_N{max}-{naccept}_{timestamp}.csv` — accepted rows (`passed=True`) plus a capped set of rejects
- matching `.png` — `log10(k)` traces for a subset of accepted vectors

Repeat for M1–M20 (or only the mechanisms you need).

## 2. Sample scattered observations

Reads the latest screening CSV per mechanism and writes parquet + `k_lookup.csv`.

```bash
python data_generation/generate_scattered_dataset.py
python data_generation/generate_scattered_dataset.py --n-pools 20 --mechanisms M1,M2 --workers 8
```

`--n-pools` is the number of **successful** `k` instances per mechanism (paper default 10000). Each instance contributes 80 rows.

Default `--out-dir` is `data/generated/`:

```
data/generated/
  generation.log
  k_lookup.csv
  train.parquet          # concatenated (omit with --no-concat)
  val.parquet
  test.parquet
  train/M1.parquet …     # per-mechanism files
  val/
  test/
```

Column schema: `data/README`. To train on a freshly generated split:

```bash
python src/train.py --data-dir data/generated --save-path checkpoints/custom_member_0.pth --gpu 0
```

## Sampling design (80 points)

**Reaction progress (RP)** bins (counts sum to 80):

| RP interval | Points |
| --- | ---: |
| [0.00, 0.05] | 10 |
| [0.05, 0.20] | 10 |
| [0.20, 0.50] | 20 |
| [0.50, 0.80] | 20 |
| [0.80, 0.98] | 10 |
| [0.98, 1.00] | 10 |

**Initial catalyst** layers (1–10 mol%), with a minority of same-excess ICs (`S0 + P0 = 1`):

| `cat0` | Points | Same-excess fraction |
| --- | ---: | ---: |
| [0.010, 0.025] | 18 | 0.25 |
| [0.025, 0.045] | 22 | 0.25 |
| [0.045, 0.065] | 18 | 0.25 |
| [0.065, 0.085] | 12 | 0.30 |
| [0.085, 0.100] | 10 | 0.30 |

A trajectory is kept only if it can still fill a bin that has remaining quota. Unreachable draws are pushed back onto the IC queue (up to 500 attempts per `k`).

The **combined** Zenodo split additionally includes low-`cat0` (0.1–1 mol%) and high-`cat0` (10–20 mol%) bands; that generator is not in this folder. Change `CAT0_LAYERS` in `generate_scattered_dataset.py` if you need a different `cat0` window.

## Notes

- Worker processes pin BLAS to one thread (`OMP_NUM_THREADS=1`) so the process pool owns parallelism.
- Integrator: SciPy `solve_ivp` method **BDF**, with a wall-clock timeout event (no `SIGALRM`).
- `k_hash` is `sha1` of the sorted `k*` map, truncated to 8 hex characters.
- Train/val/test are split on **instance** (`k_hash`), not on individual rows.
