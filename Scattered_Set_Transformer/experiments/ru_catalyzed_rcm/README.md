# Ru-catalyzed ring-closing metathesis

A ruthenium-catalyzed ring-closing metathesis (RCM) of a diene to a cyclic alkene, used here as a closed-loop experimental case at **low catalyst loading**. The reaction is a standard C–C bond-forming cyclization; under the originally simulated 1–10 mol% window it is too fast to collect informative partial-conversion points, and the Ru stock is solubility-limited, so the practical loading is **0.1–1 mol%**.

## Data

Three independent 10-point runs: `run1.csv`, `run2.csv`, `run3.csv`.

Columns: `Step`, `Time (min)`, `Cat0`, `[S]`, `[P]`. `Cat0` is a mole fraction (`0.001`–`0.010`); `[S]` and `[P]` are normalized concentrations. Initial conditions are `S0 = 1`, `P0 = 0`.

## Predict

`predict.ipynb` uses one run (`RUN = 1` by default; switch to `2` or `3`). Points are added in `Step` order; after each step the **combined** 5-member ensemble predicts, and a 2×2 figure shows entropy, mechanism probabilities, top-1 confidence, and the `(t, cat0)` design.
