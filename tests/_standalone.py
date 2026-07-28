"""Lets `tests/test_calibration.py` run against a bare clone of this repo.

`calibration.py` is shipped byte-identical to the copy running on air, and it imports
three things from the surrounding engine (`config.CONFIG`, `ledger.CAL_BINS`,
`models.now_ms`) plus, lazily, `runtime.ledger`. Those modules are PYTHIA's, not ours,
and this repo deliberately ships no copy of them — an `engine/config.py` here would sit
in the same directory the README tells you to copy over a real checkout's `engine/`,
where it could overwrite the real one. Same reasoning that keeps `engine/__init__.py`
out. So the stand-ins live here in the test tree, where nothing installs them.

Imported for its side effect, BEFORE anything imports `engine.calibration`.

Deliberately a FALLBACK, not an override: each stub is installed only if the real module
is not importable. Inside a PYTHIA checkout with these patches applied, every one of them
resolves and this file does nothing, so the tests exercise the real config there.

The values below are copied verbatim from PYTHIA's own defaults and must be updated if
those change. They are fixed rather than read from the environment on purpose — a stranger
who clones this repo should get the same result as CI regardless of what CAL_* happens to
be exported in their shell.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import time
import types
from pathlib import Path

# `python3 tests/test_calibration.py` puts tests/ on sys.path but not the repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _real(name: str) -> bool:
    """Is engine.<name> genuinely importable (i.e. are we inside a PYTHIA checkout)?"""
    try:
        return importlib.util.find_spec(f"engine.{name}") is not None
    except (ImportError, ValueError):
        return False


def _install(name: str, **attrs) -> None:
    if _real(name):
        return
    mod = types.ModuleType(f"engine.{name}")
    mod.__doc__ = f"Stand-in for PYTHIA's engine.{name} (see tests/_standalone.py)."
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[f"engine.{name}"] = mod
    setattr(importlib.import_module("engine"), name, mod)


class _Config:
    """Only the CAL_* tunables `calibration.py` reads — mirrors engine/config.py."""

    def __init__(self) -> None:
        self.cal_enabled = True
        self.cal_feedback = True
        self.cal_method = "platt"
        self.cal_min_n = 60             # resolved forecasts before a map is fit
        self.cal_min_class_n = 10       # minority-class floor (base rate ~11%)
        self.cal_refit_min_new = 20     # walk-forward refit cadence
        self.cal_ridge = 0.01           # logistic conditioning, not shrinkage
        self.cal_clip_eps = 1e-6
        self.cal_clip = 0.01            # output clamp
        self.cal_slope_min = 0.05       # → 1-param fallback below/above
        self.cal_slope_max = 3.0
        self.cal_boot_resamples = 2000


class _NullLedger:
    """An empty ledger: standalone, there is no forecast history to fit on.

    `_resolved()` reads .forecasts/.resolutions and finds nothing; `_walk_forward()`
    reads .path and gets FileNotFoundError, which it already handles as `except OSError`.
    Tests that need real rows substitute their own ledger over this.
    """

    def __init__(self) -> None:
        self.forecasts: dict = {}
        self.resolutions: dict = {}
        self.path = Path(__file__).with_name("_no_such_ledger.jsonl")


_install("config", CONFIG=_Config())
_install("ledger", CAL_BINS=[(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)])
_install("models", now_ms=lambda: int(time.time() * 1000))
_install("runtime", ledger=_NullLedger())
