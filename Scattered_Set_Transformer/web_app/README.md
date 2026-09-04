# Self-Driving Lab web app

Gradio UI that closes the loop between an **LLM experiment designer** and a **5-member Set Transformer ensemble** (M1–M20). Each iteration:

1. The LLM proposes the next `(t, cat0)` inside the bounds you set.
2. Hardware (real or mock) returns a chromatogram.
3. `[S]` and `[P]` are obtained from PDA peaks vs an internal standard.
4. The ensemble updates `P(mechanism)` and predictive entropy.
5. Results are autosaved under `results/web_app/autosave_*`.

This is a portable rewrite of the original Windows lab app. Paths, checkpoints, and mechanism text now follow this repository. **Do not commit API keys or lab IPs.**

## Requirements

Same conda env as the rest of this folder (`pip install -r requirements.txt` from `Scattered_Set_Transformer/`). You need:

- Shipped weights in `checkpoints/{combined,default}_member_{0-4}.pth`
- An OpenAI-compatible LLM key for the designer (DashScope / OpenAI / any `base_url`)

The kinetic parquet files on Zenodo are **not** required to run the app.

## Quick start (no lab hardware)

From this directory’s parent (`Scattered_Set_Transformer/`):

```bash
conda activate scattered_bench   # or your env
export DASHSCOPE_API_KEY=...     # or OPENAI_API_KEY
python web_app/app.py
```

Open `http://127.0.0.1:7863`. Leave **Hardware backend** on `mock`. The mock device waits a few seconds, then writes a synthetic PDA JSON with three Gaussians at the `[S]` / `[P]` / `[IS]` retention times you set.

The mock conversion is a **toy first-order curve**, not an M1–M20 ODE. Use it to check the UI, LLM, ensemble, autosave, and export. For scientific campaigns, connect real instruments (`lab`) or replace `MockHardware` with your own backend.

Optional flags:

```bash
python web_app/app.py --port 7863 --server-name 0.0.0.0
python web_app/app.py --share          # Gradio public URL (use with care)
```

Environment overrides:

| Variable | Role |
| --- | --- |
| `DASHSCOPE_API_KEY` / `OPENAI_API_KEY` | LLM key if the UI field is empty |
| `OPENAI_BASE_URL` | Compatible-mode base URL |
| `WEBAPP_LLM_MODEL` | Default model name in the UI |
| `WEBAPP_HARDWARE` | `mock` or `lab` |
| `WEBAPP_MOCK_DELAY` | Seconds the mock device waits before `result_file` (default `2`) |

## Ensembles and design box

| UI ensemble | Weights | Suggested `cat0` box |
| --- | --- | --- |
| **combined** (default) | `checkpoints/combined_member_*.pth` | 0.001–0.20 (0.1–20 mol%) |
| **default** | `checkpoints/default_member_*.pth` | 0.01–0.10 (1–10 mol%) |

Changing the ensemble dropdown resets `cat0` min/max; you can still edit them. The original high-loading closed-loop runs used `cat0 ∈ [0.10, 0.20]` and `t ∈ [15, 600]`.

## Lab instruments

Select **Hardware backend = lab**. Implement or confirm the HTTP contract in `hardware.py` (`LabHardware`):

```text
GET  {LAB_HUB_URL}/GetStatus
POST {LAB_HUB_URL}/next_conditions
     JSON: {"exp_ID": int, "conditions": {"cat0": float, "time": float}}
POST {LAB_LCMS_URL}
     JSON: {"DataFile": "<path>.lcd", "PDA3DData": 0, "MSSpecturm": 0}
```

Set endpoints in the environment (examples only — use your site’s hosts):

```bash
export LAB_HUB_URL=http://127.0.0.1:8001/WebService1
export LAB_LCMS_URL=http://LCMS_HOST:8080/api/GetResultData
export WEBAPP_HARDWARE=lab
python web_app/app.py
```

`GetStatus` should eventually include `result_file` (path of the `.lcd` on the LC-MS PC). The app polls until that field is non-empty, then requests JSON. Peak assignment uses the retention times and response factors in the UI.

If your robot/LC-MS API differs, edit **only** `LabHardware` in `web_app/hardware.py`. The Gradio loop talks to the `HardwareBackend` interface (`check_status`, `send_conditions`, `fetch_lcms`).

## LC-MS calibration

`St = (area_S / area_IS) * factor_s`, same for `P`. Each peak is matched to at most one of `[S]`, `[P]`, `[IS]` (nearest RT within the tolerance). Tune these on a known standard mixture before a real campaign.

## Resume and export

- Each finished loop writes `results/web_app/autosave_<stamp>/loop_XXX/` (history CSV, member probabilities, HTML plots, LLM trace, logs).
- Check **Resume from autosave** and optionally paste an `autosave_*` path.
- **Pack and download** builds a zip of the in-memory run.

`results/` is gitignored.

## Layout

```
web_app/app.py        # Gradio UI and closed loop
web_app/learner.py    # ensemble load + LLM prompt
web_app/hardware.py   # mock device + lab HTTP stubs
web_app/lcms.py       # PDA JSON → chromatogram and [S]/[P]
web_app/io_utils.py   # autosave / resume / zip
web_app/plots.py
web_app/config.py
```
