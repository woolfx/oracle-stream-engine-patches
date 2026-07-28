"""Self-calibration — PYTHIA learns a model of its OWN overconfidence.

The learning brain (brain.py) learns how markets move after events. This module learns
something more inward: how the oracle's stated probabilities relate to what actually
happens. On 261 resolved forecasts the reliability curve was badly overconfident (said
80% for things that happen 16% of the time). A walk-forward backtest showed a fitted map
cuts Brier ~0.27 -> ~0.09 — but almost entirely by UNDOING the overconfidence (predicting
near the ~11% base rate), NOT by adding forecasting skill: discrimination is near zero
(the reliability curve is nearly flat; AUC ≈ 0.5). Honest framing matters — at a base
rate of 0.107 the no-skill "predict the base rate for everything" Brier is only ≈0.096,
so the raw 0.264 is actually WORSE than base-rate-parroting, and the calibrated ~0.09 is
merely REACHING it. This layer makes the numbers honest, it does not make them sharp.

DESIGN (Harvey's call, 2026-07-22 — "companion + self-teach"):
  • The PUBLISHED headline probability stays RAW (the council's own conviction). The
    brain's FORECAST_TRIGGER_P gate, the ORACLE CALL cards, momentum and the public
    Brier all read the raw number, so nothing on-air goes quiet and the scoreboard stays
    reproducible from the public mirror. We do NOT let calibration drive the headline —
    honest calibration collapses almost everything to ~0.11, which would have wrecked the
    show and starved the brain.
  • Each forecast carries a COMPANION `calibrated_probability` (an honest "what your
    track record says this really is"), and /calibration reports Brier BOTH ways with the
    discrimination stat beside it so a base-rate flatten can never be sold as skill.
  • The actual intelligence upgrade is the FEEDBACK: a direction-aware note built from the
    live reliability curve is injected into the oracle's own forecast prompt, teaching it
    to stop clustering at high confidence. Self-correcting — rebuilt from the current
    curve every pass, so if the oracle ever OVER-corrects into under-confidence the note
    flips to "raise your numbers". This is the one part that changes on-air forecasting
    behaviour (indirectly, by sharpening raw conviction) — gated by CAL_FEEDBACK.

No look-ahead: the map is fit from ALREADY-resolved forecasts and applied only to NEW
(future-resolving) ones, whose raw published number is frozen at creation. A forecast is
never calibrated by a map that saw its own outcome. The gold honest number is the
forward-accumulating out-of-sample Brier over forecasts that carry a stored
`calibrated_probability` (each was calibrated by a strictly-earlier map). The retrospective
walk-forward on /calibration is revision-safe (replays the append-only ledger for the
outcome known AS OF each forecast's time, not the later audit-revised label), cadence-
faithful (refits at the production rate, not per forecast), and its "calibration helps"
claim is gated on a block-bootstrap CI (resampling whole briefs) AND a discrimination
floor — heavy work that runs only on the /calibration route, never in the hot path.

Pure stdlib (no numpy/sklearn — none installed), keyless, no LLM. The ledger IS the
state: the map re-fits from the ledger on a cadence; nothing new is persisted.
"""
from __future__ import annotations

import json
import logging
import math
import random
from typing import Optional

from .config import CONFIG
from .ledger import CAL_BINS
from .models import now_ms

log = logging.getLogger("pythia.calibration")

_MIN_BIN_N = 8         # a reliability bin needs this many resolved to speak in the feedback note
_OVERCONF_GAP = 0.05   # mean(predicted) must beat mean(observed) by this to call it miscalibration
_DISC_FLOOR = 0.55     # raw AUC must clear this for a "calibration improved forecasts" claim


# ── numerics ──
def _clip(p: float) -> float:
    e = CONFIG.cal_clip_eps
    return min(1 - e, max(e, p))


def _logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    if z < -35:
        return 1e-15
    if z > 35:
        return 1 - 1e-15
    return 1.0 / (1.0 + math.exp(-z))


def fit_platt(pairs: list[tuple[float, float]]) -> tuple[float, float, str]:
    """Platt scaling q = σ(A·logit(r) + B), Newton with Platt target-smoothing + a light
    conditioning ridge. Fallback ladder: 2-param → 1-param intercept shift → identity.
    Returns (A, B, fallback) where fallback ∈ {"none","one_param","identity"}."""
    n = len(pairs)
    if n == 0:
        return 1.0, 0.0, "identity"
    npos = sum(1 for _, y in pairs if y >= 0.5)
    nneg = n - npos
    lam = CONFIG.cal_ridge
    # Platt target smoothing (Platt 1999): keeps B finite even with a zero-positive fold
    xs = [_logit(r) for r, _ in pairs]
    ts = [((npos + 1) / (npos + 2)) if y >= 0.5 else (1 / (nneg + 2)) for _, y in pairs]

    def newton2() -> Optional[tuple[float, float]]:
        A, B = 1.0, 0.0
        for _ in range(60):
            gA = gB = h00 = h01 = h11 = 0.0
            for x, t in zip(xs, ts):
                q = _sigmoid(A * x + B)
                w = max(q * (1 - q), 1e-9)
                r = q - t
                gA += r * x
                gB += r
                h00 += w * x * x
                h01 += w * x
                h11 += w
            gA += lam * (A - 1.0)
            gB += lam * B
            h00 += lam
            h11 += lam
            det = h00 * h11 - h01 * h01
            if abs(det) < 1e-12:
                break
            dA = (h11 * gA - h01 * gB) / det
            dB = (h00 * gB - h01 * gA) / det
            A -= dA
            B -= dB
            if max(abs(dA), abs(dB)) < 1e-9:
                break
        if not (math.isfinite(A) and math.isfinite(B)):
            return None
        return A, B

    def newton1() -> Optional[float]:
        """A ≡ 1, fit B only (1-D Newton — always convergent)."""
        B = 0.0
        for _ in range(60):
            g = h = 0.0
            for x, t in zip(xs, ts):
                q = _sigmoid(x + B)
                g += (q - t)
                h += max(q * (1 - q), 1e-9)
            g += lam * B
            h += lam
            if h < 1e-12:
                break
            d = g / h
            B -= d
            if abs(d) < 1e-9:
                break
        return B if math.isfinite(B) else None

    r2 = newton2()
    if r2 is not None and CONFIG.cal_slope_min <= r2[0] <= CONFIG.cal_slope_max:
        return r2[0], r2[1], "none"
    b1 = newton1()
    if b1 is not None:
        return 1.0, b1, "one_param"
    return 1.0, 0.0, "identity"


class Calibrator:
    def __init__(self) -> None:
        self.method = "beta" if CONFIG.cal_method == "beta" else "platt"
        self._A = 1.0
        self._B = 0.0
        self._beta3: Optional[list[float]] = None   # only used when method == "beta"
        self._fallback = "identity"
        self._method_live = None                    # the map actually in use (beta may fall back to platt)
        self._fitted = False
        self._fitted_sig: Optional[tuple] = None    # (n, Σoutcome) — changes on a NEW resolution OR a revision
        self._fitted_ms = 0
        self._insights_cache: Optional[tuple[tuple, dict]] = None

    # ── data ──
    @staticmethod
    def _resolved() -> list[dict]:
        """Every resolved forecast (final labels, matching scorecard's predicate), in
        resolution-time order. `p` is the RAW published probability (the headline is never
        calibrated in this design, so f['probability'] is always raw)."""
        try:
            from .runtime import ledger
        except Exception:  # noqa: BLE001 — standalone use
            return []
        rows = []
        for fid, f in ledger.forecasts.items():
            r = ledger.resolutions.get(fid)
            if not r or r.get("outcome") is None:
                continue
            rows.append({
                "fid": fid, "p": float(f["probability"]), "y": float(r["outcome"]),
                "cal": f.get("calibrated_probability"),
                "ts": int(f.get("ts", 0)), "brief_id": f.get("brief_id"),
                "horizon": f.get("horizon", "?"), "resolved_ms": int(r.get("resolved_ms", 0)),
            })
        rows.sort(key=lambda d: d["resolved_ms"])
        return rows

    # ── fit (cheap, live path) ──
    def refit(self, force: bool = False) -> bool:
        if not CONFIG.cal_enabled:
            self._fitted = False
            return False
        rows = self._resolved()
        n = len(rows)
        # signature = (count, Σoutcome): a pure outcome REVISION (audit flip that leaves the
        # count unchanged) still moves Σoutcome, so the live map re-fits on it too.
        sig = (n, round(sum(r["y"] for r in rows), 2))
        if not force and sig == self._fitted_sig:
            return self._fitted
        npos = sum(1 for r in rows if r["y"] >= 0.5)
        eligible = n >= CONFIG.cal_min_n and min(npos, n - npos) >= CONFIG.cal_min_class_n
        self._fitted_sig = sig
        self._insights_cache = None
        if not eligible:
            self._fitted = False
            self._fallback = "identity"
            self._method_live = None
            return False
        pairs = [(r["p"], r["y"]) for r in rows]
        if self.method == "beta":
            # _fit_beta always leaves a USABLE map: beta3 when monotone, else Platt in _A/_B.
            # Either way we're fitted — _apply falls through to _A/_B when beta3 is None.
            self._beta3, self._fallback = self._fit_beta(pairs)
            self._method_live = "beta" if self._beta3 is not None else "platt"
        else:
            self._A, self._B, self._fallback = fit_platt(pairs)
            self._method_live = "platt"
        self._fitted = True
        self._fitted_ms = now_ms()
        log.info("calibrator: %s map fit on %d resolved (live=%s fallback=%s) A=%.3f B=%.3f",
                 self.method, n, self._method_live, self._fallback, self._A, self._B)
        return self._fitted

    def _fit_beta(self, pairs) -> tuple[Optional[list[float]], str]:
        """Opt-in Kull-2017 beta calibration on features [1, ln p, -ln(1-p)]. Falls back
        to Platt (recorded as one_param) if the fit is non-monotone or degenerate."""
        n = len(pairs)
        X = [[1.0, math.log(_clip(p)), -math.log(1 - _clip(p))] for p, _ in pairs]
        y = [yy for _, yy in pairs]
        beta = _newton_nd(X, y, CONFIG.cal_ridge)
        if beta is not None and _monotone(lambda p: _sigmoid(beta[0] + beta[1] * math.log(_clip(p))
                                                             - beta[2] * math.log(1 - _clip(p)))):
            return beta, "none"
        self._A, self._B, fb = fit_platt(pairs)   # fall back to the simpler map
        return None, fb

    # ── apply (live path) ──
    def _apply(self, p: float) -> float:
        if self.method == "beta" and self._beta3 is not None:
            b = self._beta3
            q = _sigmoid(b[0] + b[1] * math.log(_clip(p)) - b[2] * math.log(1 - _clip(p)))
        else:
            q = _sigmoid(self._A * _logit(p) + self._B)
        c = CONFIG.cal_clip
        return min(1 - c, max(c, q))

    def calibrate(self, p: float) -> Optional[float]:
        """The honest companion probability for a raw forecast probability. Lazily (re)fits.
        Returns None — NOT the raw p — when disabled or not yet fittable, so a fresh install
        (< cal_min_n resolved, or too few of the minority class) leaves calibrated_probability
        genuinely empty rather than stamping an identity value that would then be miscounted as
        'calibrated' in the forward OOS Brier."""
        if not CONFIG.cal_enabled:
            return None
        self.refit()   # cheap: early-returns unless the resolved signature changed
        return round(self._apply(p), 4) if self._fitted else None

    # ── the self-teach feedback (the actual intelligence loop) ──
    def feedback_note(self) -> str:
        """A direction-aware calibration note for the oracle's forecast prompt, built from
        the live reliability curve so it stays honest and self-correcting. Empty until
        there's enough resolved history AND a real miscalibration to report."""
        if not (CONFIG.cal_enabled and CONFIG.cal_feedback):
            return ""
        rows = self._resolved()
        if len(rows) < CONFIG.cal_min_n:
            return ""
        pred = [r["p"] for r in rows]
        obs = [r["y"] for r in rows]
        mean_pred, mean_obs = sum(pred) / len(pred), sum(obs) / len(obs)
        off = []
        for lo, hi in CAL_BINS:
            b = [(p, o) for p, o in zip(pred, obs) if lo <= p < hi]
            if len(b) >= _MIN_BIN_N:
                mp, mo = sum(p for p, _ in b) / len(b), sum(o for _, o in b) / len(b)
                if abs(mp - mo) >= _OVERCONF_GAP:
                    off.append((lo, hi, mo))
        if not off or abs(mean_pred - mean_obs) < _OVERCONF_GAP:
            return ""
        lines = [f"- when you said {int(lo*100)}–{int(min(hi,1.0)*100)}%, the event actually "
                 f"happened {round(mo*100)}% of the time" for lo, hi, mo in off[:4]]
        if mean_pred > mean_obs:
            head = ("Your forecasts have been systematically OVERCONFIDENT. From your own "
                    f"{len(rows)} resolved forecasts (only {round(mean_obs*100)}% of which came true):")
            tail = ("Calibrate DOWN: reserve probabilities above ~30% for claims a base rate, a "
                    "scheduled fixture or a hard live signal genuinely supports; prefer lower, humbler "
                    "numbers for speculative events. A well-placed 15% beats a confident, wrong 70% — "
                    "most specific world events you can name do NOT occur inside a short window.")
        else:
            head = (f"Your forecasts have been UNDER-confident. From your own {len(rows)} resolved "
                    "forecasts:")
            tail = ("Calibrate UP where the evidence is strong: when a scheduled fixture, a hard live "
                    "signal or a firm base rate supports an event, state the higher probability it "
                    "deserves rather than hedging toward the middle.")
        return ("=== CALIBRATION CHECK (your own resolved track record — context, not a target) ===\n"
                + head + "\n" + "\n".join(lines) + "\n" + tail)

    # ── honesty surfaces ──
    @staticmethod
    def _brier(pairs) -> Optional[float]:
        return round(sum((p - y) ** 2 for p, y in pairs) / len(pairs), 4) if pairs else None

    def summary(self) -> dict:
        """Cheap calibration summary for /scorecard (no O(n²) work — /scorecard is called
        every oracle pass via _persona_weights, so this must stay light)."""
        self.refit()
        rows = self._resolved()
        raw = [(r["p"], r["y"]) for r in rows]
        base = sum(y for _, y in raw) / len(raw) if raw else None
        oos = [(float(r["cal"]), r["y"]) for r in rows if r["cal"] is not None]
        return {
            "enabled": CONFIG.cal_enabled,
            "feedback": CONFIG.cal_feedback and bool(self.feedback_note()),
            "method": self._method_live if self._fitted else None,   # the map actually in use
            "fallback": self._fallback if self._fitted else None,
            "fitted": self._fitted,
            "resolved_n": len(rows),
            "base_rate": round(base, 4) if base is not None else None,
            "brier_raw": self._brier(raw),                          # == scorecard.brier (self-check)
            "brier_base_rate": round(base * (1 - base), 4) if base is not None else None,  # no-skill yardstick
            "brier_calibrated_oos": self._brier(oos),              # honest forward number (accumulates)
            "brier_calibrated_oos_n": len(oos),
            "note": ("Published headline probability is RAW conviction (unchanged); this is the honest "
                     "companion. Calibration fixes overconfidence, not discrimination — see /calibration."),
        }

    def insights(self) -> dict:
        """Full honesty panel for GET /calibration. Runs the revision-safe, cadence-faithful
        walk-forward + block-bootstrap + AUC here (route-only; cached per resolved count)."""
        self.refit()
        rows = self._resolved()
        n = len(rows)
        if self._insights_cache and self._insights_cache[0] == self._fitted_sig:
            return self._insights_cache[1]

        raw = [(r["p"], r["y"]) for r in rows]
        base = sum(y for _, y in raw) / len(raw) if raw else 0.0
        cal_insample = [(self._apply(r["p"]), r["y"]) for r in rows] if self._fitted else []

        def rel(pairs):
            out = []
            for lo, hi in CAL_BINS:
                b = [(p, y) for p, y in pairs if lo <= p < hi or (hi >= 1.0 and p >= 1.0)]
                if b:
                    out.append({"bin": f"{int(lo*100)}–{int(min(hi,1.0)*100)}%", "n": len(b),
                                "avg_predicted": round(sum(p for p, _ in b) / len(b), 3),
                                "observed": round(sum(y for _, y in b) / len(b), 3)})
            return out

        wf = self._walk_forward()
        auc = _auc(raw)
        out = {
            "note": ("PYTHIA's model of its OWN calibration. brier_raw is the live public score; "
                     "brier_base_rate is the no-skill 'predict the base rate for everything' yardstick "
                     "(raw > it means overconfidence actively HURTS). brier_calibrated_insample is the "
                     "CURRENT map on today's clean labels (optimistic). walk_forward is the conservative "
                     "historical estimate: refit at production cadence, trained on outcomes AS OF each "
                     "forecast's time (revision-safe) — so it is depressed by the pre-audit INC-010/014/017 "
                     "judge labels since voided, i.e. pessimistic vs future clean-label performance. "
                     "discrimination (AUC) modestly above 0.5 means most of the win is undoing "
                     "overconfidence, not new skill — a base-rate flatten alone cannot clear the floor."),
            "enabled": CONFIG.cal_enabled,
            "feedback_active": bool(self.feedback_note()),
            "method": self._method_live if self._fitted else None,
            "fallback": self._fallback,
            "params": (([round(x, 4) for x in self._beta3] if self._method_live == "beta" and self._beta3
                        else [round(self._A, 4), round(self._B, 4)]) if self._fitted else None),
            "degenerate": bool(self._fitted and self._method_live == "platt" and self._A < 0.10),
            "resolved_n": n,
            "base_rate": round(base, 4),
            "brier_raw": self._brier(raw),
            "brier_base_rate": round(base * (1 - base), 4),        # no-skill "predict base rate" Brier
            "brier_calibrated_insample": self._brier(cal_insample),
            "discrimination_auc": auc,
            "disc_floor_passed": auc is not None and auc >= _DISC_FLOOR,
            "walk_forward": wf,
            "claim_supported": bool(wf.get("delta_ci95") and wf["delta_ci95"][0] > 0
                                    and auc is not None and auc >= _DISC_FLOOR),
            "reliability_raw": rel(raw),
            "reliability_calibrated_insample": rel(cal_insample),
            "curve": ([{"p": round(x / 10, 1), "calibrated": round(self._apply(x / 10), 3)}
                       for x in range(1, 10)] if self._fitted else []),
        }
        self._insights_cache = (self._fitted_sig, out)
        return out

    def _walk_forward(self) -> dict:
        """Revision-safe, cadence-faithful prequential walk-forward over the append-only
        ledger. Trains each forecast's map on outcomes KNOWN AS OF its time (replaying
        resolution revisions in order), refits only every CAL_REFIT_MIN_NEW resolutions
        (production cadence), and block-bootstraps the paired Brier difference by brief_id."""
        try:
            from .runtime import ledger
            path = ledger.path
        except Exception:  # noqa: BLE001
            return {"n_eval": 0}
        # rebuild the append-only event stream: forecast (ts) + each resolution (resolved_ms)
        fmeta: dict[str, dict] = {}
        events: list[tuple[int, int, str, Optional[float]]] = []   # (time, kind 0=forecast 1=resolve, fid, outcome)
        final_outcome: dict[str, float] = {}
        try:
            for line in path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                k, fid = rec.get("kind"), rec.get("id")
                if k == "forecast" and fid:
                    fmeta[fid] = {"p": float(rec.get("probability", 0.5)), "ts": int(rec.get("ts", 0)),
                                  "brief_id": rec.get("brief_id"), "horizon": rec.get("horizon", "?")}
                    events.append((int(rec.get("ts", 0)), 0, fid, None))
                elif k == "resolution" and fid and rec.get("outcome") is not None:
                    o = float(rec["outcome"])
                    events.append((int(rec.get("resolved_ms", 0)), 1, fid, o))
                    final_outcome[fid] = o
        except OSError:
            return {"n_eval": 0}
        events.sort(key=lambda e: (e[0], e[1]))   # time order; at a tie, resolutions before forecasts
        known: dict[str, float] = {}              # fid -> outcome AS OF now (revisions overwrite)
        train: list[tuple[float, float]] = []
        A, B = 1.0, 0.0
        fitted = False
        since_fit = 0
        evald: list[dict] = []
        min_class = CONFIG.cal_min_class_n
        for _t, kind, fid, outcome in events:
            if kind == 1:
                if fid in known:                  # a revision: replace its training pair
                    train = [pr for pr in train if pr[0] != fid]
                known[fid] = outcome
                m = fmeta.get(fid)
                if m:
                    train.append((fid, m["p"], outcome))
                since_fit += 1
                if since_fit >= CONFIG.cal_refit_min_new:
                    pairs = [(p, y) for _, p, y in train]
                    npos = sum(1 for _, y in pairs if y >= 0.5)
                    if len(pairs) >= CONFIG.cal_min_n and min(npos, len(pairs) - npos) >= min_class:
                        A, B, _fb = fit_platt(pairs)
                        fitted = True
                    since_fit = 0
            else:  # forecast: score it out-of-sample if we ever learn its final outcome
                if fid not in final_outcome or not fitted:
                    continue
                m = fmeta.get(fid)
                if not m:
                    continue
                y = final_outcome[fid]
                cal = min(1 - CONFIG.cal_clip, max(CONFIG.cal_clip, _sigmoid(A * _logit(m["p"]) + B)))
                evald.append({"raw": m["p"], "cal": cal, "y": y, "brief_id": m["brief_id"],
                              "horizon": m["horizon"]})
        if not evald:
            return {"n_eval": 0, "mode": "deployed"}
        braw = sum((e["raw"] - e["y"]) ** 2 for e in evald) / len(evald)
        bcal = sum((e["cal"] - e["y"]) ** 2 for e in evald) / len(evald)
        # block bootstrap by brief_id — a brief spawns up to 16 correlated forecasts, so
        # i.i.d. resampling would understate the CI and over-claim.
        blocks: dict[str, list[float]] = {}
        for e in evald:
            d = (e["raw"] - e["y"]) ** 2 - (e["cal"] - e["y"]) ** 2
            blocks.setdefault(e["brief_id"] or e.get("horizon") or "?", []).append(d)
        keys = list(blocks.keys())
        rng = random.Random(1729)   # fixed seed: reproducible CI, no Date/rand-at-import issues
        means = []
        # Need enough INDEPENDENT blocks (distinct briefs) for the interval to mean anything:
        # with <10 the resample is near-deterministic and the CI collapses to ~zero width, which
        # would let claim_supported fire on almost no independent evidence. Below the floor → no CI.
        if len(keys) >= 10:
            for _ in range(CONFIG.cal_boot_resamples):
                pick = [blocks[keys[rng.randrange(len(keys))]] for _ in keys]
                flat = [d for blk in pick for d in blk]
                if flat:
                    means.append(sum(flat) / len(flat))
        means.sort()
        ci = ([round(means[int(0.025 * len(means))], 4),
               round(means[int(0.975 * len(means))], 4)] if means else None)
        # per-horizon tripwire (censoring: early on, a fresh year forecast is scored by a 24h-heavy map)
        per_h: dict[str, list[dict]] = {}
        for e in evald:
            per_h.setdefault(e["horizon"], []).append(e)
        per_horizon = [{"horizon": h, "n": len(v),
                        "brier_raw": round(sum((e["raw"] - e["y"]) ** 2 for e in v) / len(v), 4),
                        "brier_cal": round(sum((e["cal"] - e["y"]) ** 2 for e in v) / len(v), 4)}
                       for h, v in sorted(per_h.items())]
        return {"mode": "deployed", "n_eval": len(evald), "brier_raw_eval": round(braw, 4),
                "brier_cal_eval": round(bcal, 4), "delta": round(braw - bcal, 4),
                "delta_ci95": ci, "per_horizon": per_horizon}


# ── shared small helpers (beta fit + monotonicity + AUC) ──
def _solve(A: list[list[float]], b: list[float]) -> Optional[list[float]]:
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(n):
            if r != col:
                f = M[r][col] / pv
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[r][n] / M[r][r] for r in range(n)]


def _newton_nd(X, y, lam, iters=50) -> Optional[list[float]]:
    if not X:
        return None
    d = len(X[0])
    beta = [0.0] * d
    for _ in range(iters):
        g = [0.0] * d
        H = [[0.0] * d for _ in range(d)]
        for xi, yi in zip(X, y):
            mu = _sigmoid(sum(b * x for b, x in zip(beta, xi)))
            w = max(mu * (1 - mu), 1e-6)
            r = mu - yi
            for a in range(d):
                g[a] += r * xi[a]
                for c in range(d):
                    H[a][c] += w * xi[a] * xi[c]
        for a in range(d):
            g[a] += lam * beta[a]
            H[a][a] += lam
        delta = _solve(H, g)
        if delta is None:
            return None
        beta = [b - dta for b, dta in zip(beta, delta)]
        if max((abs(x) for x in delta), default=0.0) < 1e-8:
            break
    return beta if all(math.isfinite(b) for b in beta) else None


def _monotone(fn) -> bool:
    grid = [i / 40 for i in range(1, 40)]
    vals = [fn(p) for p in grid]
    return all(b >= a - 1e-6 for a, b in zip(vals, vals[1:]))


def _auc(pairs: list[tuple[float, float]]) -> Optional[float]:
    """Mann-Whitney AUC of the raw score vs the binary outcome — the discrimination stat.
    ~0.5 means the probabilities don't separate yes from no (calibration can't fix that)."""
    pos = [p for p, y in pairs if y >= 0.5]
    neg = [p for p, y in pairs if y < 0.5]
    if not pos or not neg:
        return None
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return round((wins + 0.5 * ties) / (len(pos) * len(neg)), 3)
