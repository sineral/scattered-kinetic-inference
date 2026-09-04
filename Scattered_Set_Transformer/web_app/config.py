"""Paths and default experiment bounds for the Self-Driving Lab web app.

Run all commands from the Scattered_Set_Transformer repository root.
Secrets (LLM API keys, lab URLs) come from environment variables or the UI —
never commit them.
"""

from __future__ import annotations

import glob
import os

# Repository root (parent of web_app/).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MECH_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "mechanisms.yaml")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "web_app")
LCMS_SAVE_DIR = os.path.join(RESULTS_DIR, "lcms_data")

# Named ensembles shipped with this repo. Combined covers the high-cat0
# (10–20 mol%) window used in the original closed-loop lab runs.
ENSEMBLE_CHOICES = ("default", "combined")
DEFAULT_ENSEMBLE = "combined"

DEFAULT_T_MIN = 15.0
DEFAULT_T_MAX = 600.0
DEFAULT_T_NORM = 600.0

# cat0 is mole fraction (0.10 = 10 mol%). Combined weights were trained on
# 0.1–20 mol%; default weights only on 1–10 mol%. The UI can override these.
ENSEMBLE_CAT0_BOUNDS = {
    "default": (0.01, 0.10),
    "combined": (0.001, 0.20),
}

N_MAX_INPUT = 20
N_MECHANISMS = 20

DEFAULT_LLM_MODEL = os.environ.get("WEBAPP_LLM_MODEL", "qwen3.5-plus")
DEFAULT_LLM_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Hardware backend: "mock" runs without instruments; "lab" calls HTTP APIs.
DEFAULT_HARDWARE_BACKEND = os.environ.get("WEBAPP_HARDWARE", "mock")

# Lab endpoints — set these in your environment. Do not hard-code site IPs.
LAB_HUB_URL = os.environ.get("LAB_HUB_URL", "http://127.0.0.1:8001/WebService1")
LAB_LCMS_URL = os.environ.get("LAB_LCMS_URL", "http://127.0.0.1:8080/api/GetResultData")

# How long the mock device waits before publishing a result (seconds).
MOCK_DEVICE_DELAY_SEC = float(os.environ.get("WEBAPP_MOCK_DELAY", "2.0"))


def collect_ensemble_ckpts(ckpt_dir: str, ensemble: str) -> list[str]:
    members = sorted(glob.glob(os.path.join(ckpt_dir, f"{ensemble}_member_*.pth")))
    if not members:
        raise FileNotFoundError(
            f"No {ensemble}_member_*.pth under {ckpt_dir}. "
            "Place the shipped checkpoints in checkpoints/ first."
        )
    return members


def resolve_api_key(ui_value: str | None = None) -> str | None:
    """Prefer the UI field, then DASHSCOPE_API_KEY / OPENAI_API_KEY."""
    if ui_value and ui_value.strip() and ui_value.strip() not in {"sk-", "YOUR_KEY"}:
        return ui_value.strip()
    return os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
