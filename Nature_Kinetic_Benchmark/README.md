# Nature Kinetic Benchmark

Public code for **20-class organic reaction mechanism classification** on the Nature kinetic-profile benchmark (M1-M20).

This repository contains the **final** training and evaluation scripts used in the paper comparison:

| Model | Input | Script |
| --- | --- | --- |
| Set Transformer | unordered 20-D set | `src/train_set_transformer.py` / `src/eval_set_transformer.py` |
| DeepSets | unordered 4-D points `[cat0, t, S, P]` | `src/train_deepsets.py` / `src/eval_deepsets.py` |
| Time-Series Transformer (TST) | ordered traces, early-fuse x1 into every timestep | `src/train_tst.py` / `src/eval_tst.py` |

Shipped weights for each method are under `checkpoints/`. Intermediate ablations (linspace tests, shuffled-input tests, DeepSets with 20-D features) are omitted on purpose.

## Data

Download the Nature kinetic-profile dataset from:

**https://doi.org/10.48420/16965292**

Unpack the archive and copy the nine `.pkl` files into `data/nature_data/` (if the archive has a nested folder, copy only the pickle files, not the extra directory). Disk space: about **10 GB** (`x2_train` alone is ~9.3 GB). The files are not included in this repository.

```
x1_train_M1_M20_train_val_test_set.pkl
x2_train_M1_M20_train_val_test_set.pkl
y_train_M1_M20_train_val_test_set.pkl
x1_val_M1_M20_train_val_test_set.pkl
x2_val_M1_M20_train_val_test_set.pkl
y_val_M1_M20_train_val_test_set.pkl
x1_test_M1_M20_train_val_test_set.pkl
x2_test_M1_M20_train_val_test_set.pkl
y_test_M1_M20_train_val_test_set.pkl
```

See `data/nature_data/README` for file formats. Source paper: *Organic Reaction Mechanism Classification with Machine Learning*.

- `x1`: `(N, 4)` initial catalyst loadings (four traces per sample)
- `x2` train/val: `(N, 21, 12)` = time axis x four traces of `[t, S, P]`
- `x2` test: nested dict keyed by number of timepoints and noise level
- `y`: labels in `{0,...,19}` for M1-M20

## Install

```bash
conda create -n nature_bench python=3.10
conda activate nature_bench
pip install -r requirements.txt
```

PyTorch CUDA wheels: `requirements.txt` uses the cu121 extra index. Change that URL for a different CUDA version, or install a CPU wheel from https://pytorch.org.

Run all commands below from **this folder** (`Nature_Kinetic_Benchmark/`), not the git monorepo root.

## Evaluate shipped weights

Defaults load `checkpoints/<method>/best_model.pth`. GPU is used when available.

```bash
python src/eval_set_transformer.py --n-points 10 --manual-noise 0.0
python src/eval_deepsets.py --eval-tps 2 --manual-noise 0.0
python src/eval_tst.py --eval-tps 6 --manual-noise 0.0
```

Reference numbers on the Nature test set (100k samples, dataset tps=6, dataset noise=0, extra Gaussian noise=0):

| Model | Setting | Top-1 | Top-3 |
| --- | --- | --- | --- |
| Set Transformer | `--n-points 10` | 0.8968 | 0.9970 |
| DeepSets | `--eval-tps 2` | 0.6352 | 0.9105 |
| TST | `--eval-tps 6` | 0.9128 | 0.9999 |

The shipped Set Transformer was trained with **8 attention layers and 64 ISAB inducing points**. `config/train_set_transformer.yaml` is the default for *new* training (6 layers, 16 inducing points) and will not reproduce that checkpoint.

`--eval-tps` for DeepSets/TST is the number of **extra non-zero** timesteps besides `t=0`.

Sweep the paper grid (points x noise 0-5%):

```bash
python src/run_eval_grid.py --model st --gpu 0
python src/run_eval_grid.py --model deepsets --gpu 0
python src/run_eval_grid.py --model tst --gpu 0
```

## Train

Each script overwrites `checkpoints/<method>/best_model.pth`. Training is GPU-oriented (Set Transformer: 500 epochs, 1M samples/epoch).

```bash
# Set Transformer (batch 512, dynamic train noise {0, 0.5, 1, 2}%)
python src/train_set_transformer.py --gpu 0

# DeepSets (500 epochs, batch 512)
python src/train_deepsets.py --gpu 0

# TST paper default: attn readout, d=128, 3 layers, force t=0
python src/train_tst.py --readout attn --dim-hidden 128 --num-layers 3 --gpu 0
```

Logs are written to `logs/` (not shipped).

## Protocol notes

- **Set Transformer** samples set elements from non-zero times; `t=0` enters only as `cat_0 / sub_0 / prod_0` features. Set size `n` is the number of set elements.
- **DeepSets / TST** evaluation **force-includes t=0**, then samples `eval-tps` additional times. Reported "Pts" in the comparison tables is `4 * eval-tps` non-zero observations (four traces).
- Noise is relative Gaussian noise on concentrations (`S`, `P`); `cat0` and time are not noised.

## Layout

```
config/train_set_transformer.yaml
checkpoints/set_transformer/best_model.pth
checkpoints/deepsets/best_model.pth
checkpoints/tst/best_model.pth
src/models/          # set_transformer, deep_sets, ts_transformer
src/utils/           # Nature -> set conversion and 20-D features
src/train_*.py
src/eval_*.py
src/run_eval_grid.py
data/nature_data/    # pickle files (download from the DOI above)
```

## License

MIT License. Copyright (c) 2026 Zheng Xian, Mo-Group, Zhejiang University. See `LICENSE`.
