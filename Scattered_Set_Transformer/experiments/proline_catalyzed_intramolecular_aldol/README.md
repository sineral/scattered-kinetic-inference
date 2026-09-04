# Proline-catalyzed intramolecular aldol reaction

A proline-catalyzed intramolecular aldol cyclization, used here as a closed-loop experimental case at **high catalyst loading**. The reaction is a representative organocatalytic C–C bond-forming cyclization; the experimental loading window is **10–20 mol%**.

## Data

Three independent 10-point runs: `run1.csv`, `run2.csv`, `run3.csv`.

Columns: `Step`, `Time (min)`, `Cat0`, `[S]`, `[P]`. `Cat0` is a mole fraction (`0.10`–`0.20`); `[S]` and `[P]` are normalized concentrations. Initial conditions are `S0 = 1`, `P0 = 0`.

## Predict

`predict.ipynb` uses one run (`RUN = 1` by default; switch to `2` or `3`). Points are added in `Step` order; after each step the **combined** 5-member ensemble predicts, and a 2×2 figure shows entropy, mechanism probabilities, top-1 confidence, and the `(t, cat0)` design.
