"""Tests for engine/calibration.py (the self-calibration layer).

Run: `uv run python tests/test_calibration.py`  (stdlib unittest; no pytest needed).
"""
import json
import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Side-effect import, must precede `engine`: stands in for the engine modules
# calibration.py imports, so this repo's tests run on their own. Inert inside a real
# PYTHIA checkout, where the actual modules win. See tests/_standalone.py.
import _standalone  # noqa: E402,F401

from engine import calibration as C  # noqa: E402
from engine.config import CONFIG  # noqa: E402


def _apply(A, B, p):
    return C._sigmoid(A * C._logit(p) + B)


class FitMath(unittest.TestCase):
    def test_sigmoid_logit_roundtrip(self):
        for p in (0.01, 0.1, 0.5, 0.9, 0.99):
            self.assertAlmostEqual(C._sigmoid(C._logit(p)), p, places=6)

    def test_recovers_known_map(self):
        rng = random.Random(7)
        A0, B0 = 0.4, -1.8
        pairs = []
        for _ in range(4000):
            p = rng.uniform(0.02, 0.98)
            y = 1.0 if rng.random() < _apply(A0, B0, p) else 0.0
            pairs.append((p, y))
        A, B, fb = C.fit_platt(pairs)
        self.assertEqual(fb, "none")
        self.assertAlmostEqual(A, A0, delta=0.12)
        self.assertAlmostEqual(B, B0, delta=0.25)

    def test_overconfident_pulls_down(self):
        # wide-range p with a real (but compressed) slope: truth = sigmoid(0.45*logit p - 1.4).
        # 2-param fit should recover a slope < 1 (compression) and pull 0.8 well below 0.8.
        rng = random.Random(3)
        pairs = []
        for _ in range(2000):
            p = rng.uniform(0.05, 0.95)
            y = 1.0 if rng.random() < _apply(0.45, -1.4, p) else 0.0
            pairs.append((p, y))
        A, B, fb = C.fit_platt(pairs)
        self.assertEqual(fb, "none")          # genuine slope signal -> 2-param map
        self.assertLess(A, 0.95)              # compression, not identity
        self.assertLess(_apply(A, B, 0.8), 0.4)

    def test_monotone(self):
        A, B, _ = C.fit_platt([(0.2, 0.0), (0.4, 0.0), (0.6, 1.0), (0.8, 1.0)] * 30)
        self.assertTrue(C._monotone(lambda p: _apply(A, B, p)))

    def test_one_param_fallback_on_flat_signal(self):
        # outcome independent of p -> MLE slope ~0 < cal_slope_min -> 1-param fallback (A==1)
        rng = random.Random(11)
        pairs = [(rng.uniform(0.05, 0.95), 1.0 if rng.random() < 0.1 else 0.0) for _ in range(500)]
        A, B, fb = C.fit_platt(pairs)
        self.assertEqual(fb, "one_param")
        self.assertEqual(A, 1.0)
        self.assertTrue(math.isfinite(B))

    def test_all_negative_window_is_finite(self):
        A, B, fb = C.fit_platt([(0.3, 0.0)] * 80)   # zero positives: smoothing must keep it finite
        for p in (0.01, 0.5, 0.99):
            v = _apply(A, B, p)
            self.assertTrue(0.0 < v < 1.0 and math.isfinite(v))

    def test_auc(self):
        self.assertAlmostEqual(C._auc([(0.9, 1.0), (0.8, 1.0), (0.2, 0.0), (0.1, 0.0)]), 1.0)
        self.assertIsNone(C._auc([(0.5, 1.0), (0.6, 1.0)]))          # one class only
        rng = random.Random(5)
        noise = [(rng.random(), float(rng.random() < 0.5)) for _ in range(2000)]
        self.assertAlmostEqual(C._auc(noise), 0.5, delta=0.05)       # no discrimination


class _Cfg:
    """Context manager: temporarily override CONFIG fields, restore on exit."""
    def __init__(self, **kw):
        self.kw = kw
        self.old = {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = getattr(CONFIG, k)
            setattr(CONFIG, k, v)
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            setattr(CONFIG, k, v)


def _rows(specs):
    """specs: list of (p, y, count); expands to synthetic resolved rows in ts order."""
    rows, t = [], 0
    for p, y, n in specs:
        for _ in range(n):
            t += 1
            rows.append({"fid": f"f{t}", "p": p, "y": float(y), "cal": None,
                         "ts": t, "brief_id": f"b{t // 5}", "horizon": "24h", "resolved_ms": t + 1000})
    return rows


class CalibratorBehavior(unittest.TestCase):
    def _cal(self, rows):
        cal = C.Calibrator()
        cal._resolved = lambda: rows   # instance attr shadows the class staticmethod;
        # a plain lambda (not staticmethod(...)) because staticmethod objects only
        # became directly callable in CPython 3.10 (bpo-43682) and this repo claims 3.8+
        cal.refit(force=True)
        return cal

    def test_disabled_returns_none(self):
        with _Cfg(cal_enabled=False):
            cal = C.Calibrator()
            self.assertIsNone(cal.calibrate(0.8))   # None, not raw: no companion is stamped

    def test_below_min_n_returns_none(self):
        with _Cfg(cal_enabled=True, cal_min_n=60, cal_min_class_n=10):
            cal = self._cal(_rows([(0.7, 1, 5), (0.3, 0, 20)]))   # n=25 < 60
            self.assertFalse(cal._fitted)
            self.assertIsNone(cal.calibrate(0.8))   # pre-fit companion stays empty (not counted in OOS)

    def test_minority_class_gate(self):
        # n=100 >= min_n but only 5 positives < min_class_n -> not fitted (no garbage map)
        with _Cfg(cal_enabled=True, cal_min_n=60, cal_min_class_n=10):
            cal = self._cal(_rows([(0.7, 1, 5), (0.2, 0, 95)]))
            self.assertFalse(cal._fitted)
            self.assertIsNone(cal.calibrate(0.8))

    def test_beta_fallback_stays_fitted(self):
        # regression (verify finding #1): in beta mode, a non-monotone beta fit must fall back
        # to the Platt map it computed — NOT silently disable calibration.
        with _Cfg(cal_enabled=True, cal_method="beta", cal_min_n=60, cal_min_class_n=10):
            cal = C.Calibrator()
            cal._resolved = lambda: _rows([(0.8, 1, 12), (0.6, 0, 50), (0.3, 0, 50), (0.2, 1, 8)])

            def fake_fit_beta(pairs):          # simulate a rejected (non-monotone) beta fit
                cal._A, cal._B, _ = C.fit_platt(pairs)
                return None, "one_param"
            cal._fit_beta = fake_fit_beta
            cal.refit(force=True)
            self.assertTrue(cal._fitted)                  # fitted via the Platt fallback, not disabled
            self.assertEqual(cal._method_live, "platt")
            self.assertIsNotNone(cal.calibrate(0.8))      # a real companion, not None

    def test_overconfident_companion_pulls_down(self):
        with _Cfg(cal_enabled=True, cal_min_n=60, cal_min_class_n=10, cal_method="platt"):
            cal = self._cal(_rows([(0.8, 1, 12), (0.7, 0, 40), (0.6, 0, 40), (0.3, 0, 40), (0.2, 1, 8)]))
            self.assertTrue(cal._fitted)
            self.assertLess(cal.calibrate(0.8), 0.8)     # headline stays raw; companion is honest-lower

    def test_feedback_direction_overconfident(self):
        with _Cfg(cal_enabled=True, cal_feedback=True, cal_min_n=60, cal_min_class_n=10):
            cal = self._cal(_rows([(0.7, 0, 50), (0.5, 0, 40), (0.3, 1, 12), (0.2, 0, 20)]))
            note = cal.feedback_note()
            self.assertIn("OVERCONFIDENT", note)
            self.assertIn("Calibrate DOWN", note)

    def test_feedback_direction_underconfident(self):
        with _Cfg(cal_enabled=True, cal_feedback=True, cal_min_n=60, cal_min_class_n=10):
            cal = self._cal(_rows([(0.2, 1, 40), (0.3, 1, 40), (0.5, 1, 20), (0.1, 0, 20)]))
            note = cal.feedback_note()
            self.assertIn("UNDER-confident", note)
            self.assertIn("Calibrate UP", note)

    def test_feedback_off_when_disabled(self):
        with _Cfg(cal_enabled=True, cal_feedback=False, cal_min_n=60, cal_min_class_n=10):
            cal = self._cal(_rows([(0.7, 0, 50), (0.3, 1, 12), (0.2, 0, 20)]))
            self.assertEqual(cal.feedback_note(), "")

    def test_summary_brier_raw_matches_manual(self):
        rows = _rows([(0.7, 0, 50), (0.5, 1, 12), (0.2, 0, 20)])
        with _Cfg(cal_enabled=True, cal_min_n=60, cal_min_class_n=10):
            cal = self._cal(rows)
            s = cal.summary()
            manual = sum((r["p"] - r["y"]) ** 2 for r in rows) / len(rows)
            self.assertAlmostEqual(s["brier_raw"], round(manual, 4), places=4)
            self.assertEqual(s["base_rate"], round(sum(r["y"] for r in rows) / len(rows), 4))


class WalkForwardNoLookahead(unittest.TestCase):
    """Synthetic append-only ledger: assert the walk-forward never trains on an outcome
    resolved at/after the forecast it scores, and produces a coherent claim block."""

    def _fake_ledger(self, path):
        class L:
            def __init__(s):
                s.path = path
                s.forecasts, s.resolutions = {}, {}
                for line in path.read_text().splitlines():
                    r = json.loads(line)
                    if r["kind"] == "forecast":
                        s.forecasts[r["id"]] = r
                    else:
                        s.resolutions[r["id"]] = r
        return L()

    def test_walkforward_time_gate_and_claim(self):
        rng = random.Random(42)
        recs = []
        # overconfident generator: raw p high, truth ~ sigmoid(0.4*logit(p) - 1.8)
        for i in range(400):
            ts = 1000 + i * 10
            p = round(rng.uniform(0.3, 0.9), 2)
            y = 1.0 if rng.random() < C._sigmoid(0.4 * C._logit(p) - 1.8) else 0.0
            recs.append({"kind": "forecast", "id": f"f{i}", "probability": p, "ts": ts,
                         "brief_id": f"b{i // 4}", "horizon": "24h"})
            recs.append({"kind": "resolution", "id": f"f{i}", "outcome": y,
                         "resolved_ms": ts + 500})   # resolves AFTER its own ts
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in recs))
            fake = self._fake_ledger(path)
            import engine.runtime as rt
            old = rt.ledger
            rt.ledger = fake
            try:
                with _Cfg(cal_enabled=True, cal_method="platt", cal_min_n=60,
                          cal_min_class_n=10, cal_refit_min_new=20, cal_boot_resamples=300):
                    cal = C.Calibrator()
                    cal.refit(force=True)
                    ins = cal.insights()
            finally:
                rt.ledger = old
            wf = ins["walk_forward"]
            self.assertEqual(wf["mode"], "deployed")
            self.assertGreater(wf["n_eval"], 0)
            # calibration should help on cleanly-overconfident synthetic data
            self.assertLess(wf["brier_cal_eval"], wf["brier_raw_eval"])
            self.assertIsNotNone(wf["delta_ci95"])
            self.assertEqual(len(wf["delta_ci95"]), 2)
            self.assertLessEqual(wf["delta_ci95"][0], wf["delta_ci95"][1])
            self.assertIn("discrimination_auc", ins)
            self.assertIn("brier_base_rate", ins)

    def _run(self, recs, **cfg):
        """Fit the walk-forward over a synthetic ledger and return its claim block."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in recs))
            import engine.runtime as rt
            old = rt.ledger
            rt.ledger = self._fake_ledger(path)
            try:
                with _Cfg(cal_enabled=True, cal_method="platt", cal_min_n=60,
                          cal_min_class_n=10, cal_refit_min_new=20,
                          cal_boot_resamples=200, **cfg):
                    cal = C.Calibrator()
                    cal.refit(force=True)
                    return cal.insights()["walk_forward"]
            finally:
                rt.ledger = old

    def test_walkforward_does_not_peek_at_later_outcomes(self):
        """The look-ahead property, actually pinned.

        `test_walkforward_time_gate_and_claim` above asserts the claim block is
        COHERENT — deployed mode, n_eval > 0, calibration beats raw, well-ordered CI.
        None of that can distinguish a prequential fit from one that trains on the
        whole final-outcome set: look-ahead makes the calibrated Brier *better*, so a
        leak satisfies every one of those assertions. Verified — replacing the refit
        with `fit_platt` over all final outcomes leaves that test green.

        This one separates them with a REGIME FLIP. The first half is honest
        (y ~ Bernoulli(p)); the second half is wildly overconfident (logit shifted
        -3.0). A prequential calibrator scores the honest half with a map fit only on
        honest data, so it leaves those forecasts roughly alone and only corrects once
        the flip is in its training window. A peeking calibrator scores every forecast
        with one pooled compromise map, which is wrong for BOTH halves — and it lands a
        visibly better Brier because it already knows how the story ends.

        Measured on this fixture (seed 7, 200 + 200): prequential 0.2473, look-ahead
        0.1959, raw 0.2993. The bound below sits between the first two. If a future
        change moves these numbers, re-measure both arms before touching the constant —
        loosening it to make a red test green would retire the only assertion here that
        has any teeth.
        """
        rng = random.Random(7)
        recs = []
        for i in range(400):
            ts = 1000 + i * 10
            p = round(rng.uniform(0.30, 0.90), 2)
            if i < 200:
                y = 1.0 if rng.random() < p else 0.0                       # honest
            else:
                y = 1.0 if rng.random() < C._sigmoid(C._logit(p) - 3.0) else 0.0
            recs.append({"kind": "forecast", "id": f"f{i}", "probability": p, "ts": ts,
                         "brief_id": f"b{i // 4}", "horizon": "24h"})
            recs.append({"kind": "resolution", "id": f"f{i}", "outcome": y,
                         "resolved_ms": ts + 500})
        wf = self._run(recs)
        self.assertEqual(wf["mode"], "deployed")
        self.assertGreater(wf["n_eval"], 0)
        # calibration must still earn its keep on data this badly miscalibrated
        self.assertLess(wf["brier_cal_eval"], wf["brier_raw_eval"])
        # ...but it must NOT reach the score only a peeking fit can reach
        self.assertGreater(
            wf["brier_cal_eval"], 0.22,
            f"brier_cal_eval={wf['brier_cal_eval']} is at or below the look-ahead arm "
            "(0.1959) — the walk-forward is training on outcomes it should not yet know")


if __name__ == "__main__":
    unittest.main(verbosity=2)
