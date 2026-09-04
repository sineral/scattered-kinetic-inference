# Scattered Kinetic Inference

Permutation-invariant neural inference of reaction mechanisms with scattered kinetic data.

Code: https://github.com/sineral/scattered-kinetic-inference

This is a **two-folder** repository. Install and run commands **inside one folder**, not from this top-level directory.

| Directory | Role |
| --- | --- |
| [`Nature_Kinetic_Benchmark/`](Nature_Kinetic_Benchmark/README.md) | Set Transformer / DeepSets / TST on the Nature kinetic-profile benchmark (M1–M20) |
| [`Scattered_Set_Transformer/`](Scattered_Set_Transformer/README.md) | Set Transformer for unordered scattered kinetic points, ensembles, active sampling, experimental cases, and a Gradio closed-loop web app |

```bash
# Nature kinetic-profile benchmark (~10 GB pickles, separate download)
cd Nature_Kinetic_Benchmark
conda create -n nature_bench python=3.10
conda activate nature_bench
pip install -r requirements.txt
# then follow Nature_Kinetic_Benchmark/README.md (data DOI + eval)

# Scattered points, ensembles, notebooks, web app
cd Scattered_Set_Transformer
conda create -n scattered_bench python=3.10
conda activate scattered_bench
pip install -r requirements.txt
# unpack Zenodo zip here, then follow Scattered_Set_Transformer/README.md
```

Scattered parquet files (~3 GB) are on Zenodo: https://doi.org/10.5281/zenodo.21991341. Unpack `scattered-kinetic-data.zip` **inside** `Scattered_Set_Transformer/`. Nature `.pkl` files are downloaded separately (see `Nature_Kinetic_Benchmark/README.md`).
