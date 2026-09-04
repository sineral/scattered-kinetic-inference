"""Instrument I/O for the closed-loop web app.

Two backends:

* ``mock`` (default) — no lab hardware. The app invents a chromatogram so you
  can exercise the LLM + ensemble loop on any machine.
* ``lab`` — HTTP calls to a local experiment hub and an LC-MS result service.
  Endpoints and payload shapes are site-specific; fill in ``LabHardware``
  for your own robot / LC-MS PC.

Set ``WEBAPP_HARDWARE=lab`` or choose "Lab instruments" in the UI.

Lab environment variables (never commit real IPs or tokens)::

    LAB_HUB_URL=http://127.0.0.1:8001/WebService1
    LAB_LCMS_URL=http://LCMS_HOST:8080/api/GetResultData
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np

from web_app.config import (
    LAB_HUB_URL,
    LAB_LCMS_URL,
    LCMS_SAVE_DIR,
    MOCK_DEVICE_DELAY_SEC,
)

LogFn = Optional[Callable[[str], None]]


def _log(log_cb: LogFn, msg: str) -> None:
    if log_cb:
        log_cb(msg)


class HardwareBackend(ABC):
    """Minimal contract used by the automated loop."""

    @abstractmethod
    def check_status(self, log_cb: LogFn = None) -> Optional[dict]:
        """Return a status dict, or None if the hub is unreachable."""

    @abstractmethod
    def send_conditions(self, exp_id: int, cat0: float, t: float, log_cb: LogFn = None) -> bool:
        """Dispatch the next (cat0, time) pair. Return True on HTTP / mock success."""

    @abstractmethod
    def fetch_lcms(self, data_file_path: str, log_cb: LogFn = None) -> Optional[str]:
        """Download LC-MS JSON for ``data_file_path`` and return the local save path."""


# ---------------------------------------------------------------------------
# Mock backend — fully runnable without instruments
# ---------------------------------------------------------------------------

def _gaussian_peak(x: np.ndarray, center: float, height: float, width: float = 0.12) -> np.ndarray:
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)


def _mock_concentrations(cat0: float, t_min: float) -> tuple[float, float]:
    """Cheap first-order stand-in so mock loops produce changing [S]/[P].

    This is **not** a kinetic model of M1–M20. It only gives the chromatogram
    parser something non-trivial to integrate.
    """
    k_eff = 0.35  # 1 / min at cat0 = 1 (mole fraction)
    conversion = 1.0 - np.exp(-k_eff * max(cat0, 1e-6) * t_min)
    conversion = float(np.clip(conversion, 0.02, 0.98))
    st = float(np.clip(1.0 - conversion, 0.0, 1.0))
    pt = float(np.clip(conversion * 0.95, 0.0, 1.0))
    return st, pt


def build_mock_lcms_payload(
    st: float,
    pt: float,
    rt_s: float,
    rt_p: float,
    rt_is: float,
    factor_s: float,
    factor_p: float,
    t_end_min: float = 15.0,
) -> dict:
    """Synthesize a PDA-like JSON blob compatible with ``lcms.process_lcms_json``."""
    n = 2000
    start_ms, end_ms = 0.0, t_end_min * 60_000.0
    time_min = np.linspace(0.0, t_end_min, n)
    area_is = 1.0e6
    area_s = (st / max(factor_s, 1e-9)) * area_is
    area_p = (pt / max(factor_p, 1e-9)) * area_is
    # Peak height proxy (area ~ height * width * sqrt(2 pi))
    width = 0.12
    scale = width * np.sqrt(2.0 * np.pi)
    intensity = (
        _gaussian_peak(time_min, rt_s, area_s / scale)
        + _gaussian_peak(time_min, rt_p, area_p / scale)
        + _gaussian_peak(time_min, rt_is, area_is / scale)
        + np.random.default_rng(0).normal(0.0, 2.0, size=n)
    )
    intensity = np.clip(intensity, 0.0, None)
    trapz = getattr(np, "trapezoid", np.trapz)
    total = float(trapz(intensity, time_min)) or 1.0

    def _peak(rt: float, area: float) -> dict:
        return {"Rt": float(rt), "Area": float(area), "AreaPct": float(100.0 * area / total)}

    return {
        "Data": {
            "LSSPDAData": {
                "ListPDAData": [
                    {
                        "PDAChroma": {
                            "StartRT": start_ms,
                            "EndRT": end_ms,
                            "ChromList": intensity.tolist(),
                        },
                        "PDAPeakData": [
                            _peak(rt_s, area_s),
                            _peak(rt_p, area_p),
                            _peak(rt_is, area_is),
                        ],
                    }
                ]
            }
        }
    }


class MockHardware(HardwareBackend):
    """In-process stand-in for the experiment hub + LC-MS PC."""

    def __init__(
        self,
        save_dir: str = LCMS_SAVE_DIR,
        delay_sec: float = MOCK_DEVICE_DELAY_SEC,
        lcms_params: Optional[dict] = None,
    ):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.delay_sec = delay_sec
        self.lcms_params = lcms_params or {
            "rt_s": 8.3,
            "rt_p": 9.1,
            "rt_is": 11.0,
            "factor_s": 1.43,
            "factor_p": 1.16,
        }
        self._pending: Optional[dict] = None
        self._ready_at: float = 0.0

    def check_status(self, log_cb: LogFn = None) -> Optional[dict]:
        if self._pending is None:
            return {
                "Status": {
                    "done": 0,
                    "result_file": "",
                    "raw_data": [["t", "cat0", "note"], []],
                }
            }
        done = time.time() >= self._ready_at
        result_file = self._pending["result_file"] if done else ""
        return {
            "Status": {
                "done": 1 if done else 0,
                "result_file": result_file,
                "raw_data": [
                    ["t", "cat0", "done", "note"],
                    [
                        self._pending["t"],
                        self._pending["cat0"],
                        1 if done else 0,
                        "mock device",
                    ],
                ],
            }
        }

    def send_conditions(self, exp_id: int, cat0: float, t: float, log_cb: LogFn = None) -> bool:
        result_file = f"mock_exp_{int(exp_id)}.lcd"
        self._pending = {"exp_id": int(exp_id), "cat0": float(cat0), "t": float(t), "result_file": result_file}
        self._ready_at = time.time() + self.delay_sec
        _log(log_cb, f"[Hardware/mock] Accepted exp {exp_id}: t={t:.2f} min, cat0={cat0:.4f}")
        return True

    def fetch_lcms(self, data_file_path: str, log_cb: LogFn = None) -> Optional[str]:
        if self._pending is None:
            _log(log_cb, "[Hardware/mock] No pending experiment.")
            return None
        st, pt = _mock_concentrations(self._pending["cat0"], self._pending["t"])
        payload = build_mock_lcms_payload(
            st,
            pt,
            rt_s=float(self.lcms_params["rt_s"]),
            rt_p=float(self.lcms_params["rt_p"]),
            rt_is=float(self.lcms_params["rt_is"]),
            factor_s=float(self.lcms_params["factor_s"]),
            factor_p=float(self.lcms_params["factor_p"]),
        )
        base_name = os.path.splitext(os.path.basename(data_file_path))[0]
        save_path = os.path.join(self.save_dir, f"{base_name}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(log_cb, f"[Hardware/mock] Wrote synthetic LC-MS JSON: {os.path.basename(save_path)}")
        return save_path


# ---------------------------------------------------------------------------
# Lab backend — HTTP hooks (adapt to your site)
# ---------------------------------------------------------------------------

class LabHardware(HardwareBackend):
    """Talk to a local experiment hub and LC-MS result API.

    The original Windows deployment used approximately::

        GET  {LAB_HUB_URL}/GetStatus
        POST {LAB_HUB_URL}/next_conditions
             body: {"exp_ID": int, "conditions": {"cat0": float, "time": float}}
        POST {LAB_LCMS_URL}
             body: {"DataFile": "<path>.lcd", "PDA3DData": 0, "MSSpecturm": 0}

    Replace URLs via ``LAB_HUB_URL`` / ``LAB_LCMS_URL``. If your hub uses a
    different schema, edit the methods below — they are the only hardware
    coupling points in this app.
    """

    def __init__(self, hub_url: str = LAB_HUB_URL, lcms_url: str = LAB_LCMS_URL, save_dir: str = LCMS_SAVE_DIR):
        self.hub_url = hub_url.rstrip("/")
        self.lcms_url = lcms_url
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def check_status(self, log_cb: LogFn = None) -> Optional[dict]:
        import requests

        url = f"{self.hub_url}/GetStatus"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except Exception:
                    _log(log_cb, "[Hardware/lab] GetStatus returned non-JSON.")
                    return {"raw": resp.text}
            _log(log_cb, f"[Hardware/lab] GetStatus HTTP {resp.status_code}")
        except Exception as e:
            _log(log_cb, f"[Hardware/lab] GetStatus failed: {e}")
        return None

    def send_conditions(self, exp_id: int, cat0: float, t: float, log_cb: LogFn = None) -> bool:
        import requests

        url = f"{self.hub_url}/next_conditions"
        payload: dict[str, Any] = {"exp_ID": exp_id, "conditions": {"cat0": cat0, "time": t}}
        _log(log_cb, f"[Hardware/lab] POST {url} | {payload}")
        try:
            resp = requests.post(url, json=payload, timeout=5)
            _log(log_cb, f"[Hardware/lab] next_conditions HTTP {resp.status_code}")
            return resp.status_code == 200
        except Exception as e:
            _log(log_cb, f"[Hardware/lab] next_conditions failed: {e}")
            return False

    def fetch_lcms(self, data_file_path: str, log_cb: LogFn = None) -> Optional[str]:
        import requests

        payload = {"DataFile": data_file_path, "PDA3DData": 0, "MSSpecturm": 0}
        _log(log_cb, f"[Hardware/lab] FETCH LCMS {self.lcms_url} | {data_file_path}")
        try:
            resp = requests.post(self.lcms_url, json=payload, timeout=15)
            if resp.status_code != 200:
                _log(log_cb, f"[Hardware/lab] LCMS HTTP {resp.status_code}")
                return None
            result = resp.json()
            base_name = os.path.basename(data_file_path).replace(".lcd", "")
            save_path = os.path.join(self.save_dir, f"{base_name}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            _log(log_cb, f"[Hardware/lab] Saved LC-MS JSON: {os.path.basename(save_path)}")
            return save_path
        except Exception as e:
            _log(log_cb, f"[Hardware/lab] LCMS fetch failed: {e}")
            return None


def make_hardware(backend: str, lcms_params: Optional[dict] = None) -> HardwareBackend:
    name = (backend or "mock").strip().lower()
    if name in {"lab", "real", "hardware"}:
        return LabHardware()
    return MockHardware(lcms_params=lcms_params)
