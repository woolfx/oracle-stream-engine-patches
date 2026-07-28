# Engine patches — grading an AI forecaster honestly

Two modules from [The Oracle Stream](https://www.youtube.com/@TheOracleStream1/live), a
24/7 livestream where an AI forecasts world events and grades itself in public. Both
exist because the stream's scoreboard was wrong in ways that flattered it, and both are
the deterministic parts an LLM can't talk its way past.

They are patches for [PYTHIA](https://github.com/jangles-byte/Pythia) (MIT), which is
not our project — we run and extend it. This repo is only the extensions.

| Module | What it does |
|---|---|
| [`engine/relevance.py`](engine/relevance.py) | Decides whether a piece of world data may be used as evidence for a given forecast, and whether it may confirm one. Pure stdlib, no LLM. |
| [`engine/calibration.py`](engine/calibration.py) | Learns the forecaster's own overconfidence from its resolved track record — Platt/beta map, revision-safe walk-forward, block-bootstrap CI, AUC. Pure stdlib, no numpy. |

## Why relevance.py exists

An LLM judge was grading the forecasts, and it turned out to cite whatever large event
happened to be in its evidence dump. A Fed-rate forecast was resolved YES on the strength
of a Mexican earthquake. Hurricane forecasts in three different countries were all
confirmed by one placeless cyclone notification. A Phoenix flood claim was resolved by a
warning issued in Tucson.

**43 resolutions were voided.** The write-up is
[here](https://github.com/woolfx/oracle-stream-ledger/blob/main/audits/2026-07-18-ledger-audit.md).

The module is deliberately dumb: place sets, hazard-concept sets, token overlap, and
numeric floors (a claimed magnitude 6.0 is not confirmed by an M5.2; a Category 2 claim is
not confirmed by a Category 1). Two bars, not one — `related()` decides what the judge is
allowed to *see*, `citable()` decides what may support a **yes**. It has to be cheap enough
to run on every verdict and simple enough that a human can read this file and check it.

It also carries the two premature-resolution guards. A forecast that dates its own event
past its horizon ("…at the FOMC meeting on July 28th", written at a 24h horizon) was being
auto-graded NO the moment its window closed — eight days before the event existed. Then the
same defect for fixtures named *without* a date ("the EIA weekly petroleum status report").
**21 more resolutions voided**
([one](https://github.com/woolfx/oracle-stream-ledger/blob/main/audits/2026-07-20-premature-dated-resolutions.md),
[two](https://github.com/woolfx/oracle-stream-ledger/blob/main/audits/2026-07-22-premature-scheduled-resolutions.md)).

## Why calibration.py exists

The forecaster is badly overconfident, and we wanted to measure that rather than argue
about it. Over 673 resolved forecasts it scores Brier 0.226 while only 7.0% of its
forecasts come true — a rock that always says "no" scores 0.070. When it says 50%, the
observed rate is about 6%.

The module fits a calibration map on already-resolved forecasts and applies it only to
forecasts that resolve later, so nothing is ever calibrated by a map that saw its own
outcome. Three deliberate design choices worth arguing with:

- **The published headline stays raw.** The calibrated number ships as a companion, never
  as the score. An honest calibration collapses nearly everything toward the base rate, and
  publishing that as "the forecast" would be its own kind of dishonesty.
- **Discrimination is reported next to calibration.** AUC is currently 0.639 — the ranking
  carries real information, but most of what calibration buys is undoing overconfidence,
  not adding skill. A base-rate flatten can always be sold as a Brier improvement, so the
  AUC floor sits beside it to stop that.
- **The walk-forward is revision-safe.** It replays the append-only ledger for the outcome
  known *as of* each forecast's time, so a label later corrected by audit can't leak
  backwards, and it refits at production cadence rather than per forecast. The paired Brier
  difference is block-bootstrapped by brief, because one brief spawns up to 16 correlated
  forecasts and i.i.d. resampling would over-claim.

## Tests

`tests/test_relevance.py` needs nothing but Python — no dependencies, no network, no
fixtures. Every case in it is a real mis-resolution from the audits above:

```bash
python3 -m unittest discover -s tests -p 'test_relevance.py' -v
```

23 tests. `tests/test_calibration.py` (17 tests) exercises the numerics, the no-look-ahead
property and the fallback ladder, and needs the surrounding engine — run it from inside a
PYTHIA checkout with these files applied.

## Applying them

Copy `engine/*.py` into a PYTHIA checkout's `engine/` directory. `relevance.py` is
standalone (stdlib `re` only). `calibration.py` imports three things from the engine —
`config.CONFIG`, `ledger.CAL_BINS`, `models.now_ms` — and is wired in from `pipeline.py`,
`loop.py`, `oracle.py`, `server.py` and `runtime.py`; those call sites are not included
here. No `__init__.py` ships, so nothing in this repo can overwrite the package's own.

## The rest of it

- **Live stream** — https://www.youtube.com/@TheOracleStream1/live
- **Public forecast ledger** — https://github.com/woolfx/oracle-stream-ledger — every
  forecast committed before its outcome is known, every audit, the running scorecard
- **The engine** — https://github.com/jangles-byte/Pythia (MIT, not ours)

MIT licensed. Numbers current as of 2026-07-27; the ledger repo has the live ones.
