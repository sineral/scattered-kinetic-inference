# Photoinduced iron-catalyzed cycloisomerization

A photoinduced iron-catalyzed cycloisomerization of alkynols to cyclic enol ethers, used here as a representative photochemical-activation case. The reaction is synthetically relevant because it uses an earth-abundant iron catalyst to access cyclic enol ethers, and it poses a mechanistic challenge involving light-driven catalyst activation and catalyst speciation.

**Reference.** Lehnherr, D.; Ji, Y.; Neel, A. J.; Cohen, R. D.; Brunskill, A. P. J.; Yang, J.; Reibarkh, M. Discovery of a Photoinduced Dark Catalytic Cycle Using in Situ LED-NMR Spectroscopy. *J. Am. Chem. Soc.* **2018**, *140* (42), 13843–13853. https://doi.org/10.1021/jacs.8b08596

## Data (`kinetics.txt`)

Whitespace-separated table: four traces, each `(Time, A, catT)` in groups of three columns. The `t = 0` row supplies `S0` and `cat0` only and is not sampled.

| Curve | `[S]_0` (raw) | `catT` (raw) |
| --- | ---: | ---: |
| 1–3 | 0.6 | 0.00225, 0.009, 0.018 |
| 4 | 0.3 | 0.00225 |

**Skip curve 4.** Curves 1–3 share the same initial substrate; only catalyst loading changes. Curve 4 uses a different `[S]_0`, so dropping it keeps a matched initial-composition set.

**Keep curve 4.** Uses every published trace, including the lower-`[S]_0` experiment.

## Predict

`predict.ipynb` runs both settings with the default 5-member ensemble and 20 random 12-point subsets (`t > 0`).
