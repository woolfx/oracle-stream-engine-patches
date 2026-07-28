# Engine patches — grading an AI forecaster honestly

Two modules from [The Oracle Stream](https://www.youtube.com/@TheOracleStream1/live), a
24/7 livestream where an AI forecasts world events and grades itself in public. Both
exist because the stream's scoreboard was wrong in ways that flattered it, and both are
the deterministic parts an LLM can't talk its way past.

They are patches for [PYTHIA](https://github.com/jangles-byte/Pythia) (MIT), which is
not our project — we run and extend it. This repo is only the extensions.

| Module | What it does |
|---|---|
| [`engine/relevance.py`](engine/relevance.py) | Decides whether a piece of world data may be used as evidence for a given forecast, whether it may confirm one, and whether it happened early enough for the forecast to have been a forecast. Pure stdlib, no LLM. |
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

Then the ordering bug. `citable()` checks *what* a signal says — subject, place, magnitude —
and never checked *when* it happened. Disaster feed lines carry their own event time and
linger in the world brief for days, so an event could sit in the brief, be drafted into a
forecast by the oracle that was looking at it, then be cited back as proof that the forecast
came true. Every INC-010 guardrail passes: same subject, same place, threshold met. "A major
earthquake will strike Mexico within the next fortnight", published 17:32 UTC, was resolved
YES on the M7.3 that struck at 14:48 UTC that morning — 2.7 hours before it was written, with
a shareable receipt card already cut from it. The events are real, the citations accurate, the
arithmetic right. The forecasts simply were not forecasts. **2 more voided**
([audit](https://github.com/woolfx/oracle-stream-ledger/blob/main/audits/2026-07-28-retrodiction-resolutions.md)).

The guard is `signal_event_ms()` plus a `made_ms` argument to `filter_signals()`: a signal
whose own text dates its event before the forecast was published never enters the candidate
pool, so the judge is never offered it. Conservative by construction — only unambiguous,
fully-specified timestamps are read, an ambiguous `05/06/2026` returns `None` rather than
guess a locale, and **an unparseable signal stays admissible**. Most evidence states no
absolute time at all, and treating silence as guilt would starve the judge and collapse every
verdict to "no". That is also the limit of it: of 100 resolved-correct records scanned, 7
state a parseable absolute timestamp and 2 of those predate their forecast — **a floor, not a
rate**. And the guard is prospective. It keeps retrodictions out of future evidence pools; it
does not re-grade what is already in the ledger, so the published hit rate still carries an
unquantified retrodiction component bounded below by 2 and above by nothing established here.

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

All three suites run on a bare clone — nothing but Python 3.8+, no dependencies, no network,
no fixtures:

```bash
python3 -m unittest discover -s tests -v
```

57 tests. `tests/test_relevance.py` (23) is every real mis-resolution from the audits
above. `tests/test_retrodiction.py` (17) is the ordering guard — the live record above pinned
as a regression, plus the two properties that matter more than the catch: an unparseable
signal stays admissible, and an ambiguous date refuses to guess. `tests/test_calibration.py`
(17) exercises the numerics, the no-look-ahead property and the fallback ladder.

`calibration.py` reads its tunables from the surrounding engine, which this repo does not
ship (see below), so the calibration suite stands those modules in from
[`tests/_standalone.py`](tests/_standalone.py) — in the test tree, never in `engine/`. It
is a fallback, not an override: inside a PYTHIA checkout with these patches applied, the
real modules are importable and the shim does nothing, so the same suite runs against the
real config there.

## Applying them

Copy `engine/*.py` into a PYTHIA checkout's `engine/` directory. `relevance.py` is
standalone (stdlib `re` and `datetime`, nothing else) — but one call site is not optional:
the retrodiction guard is inert unless the judge hands it the forecast's publication time.
`made_ms` defaults to `None`, which preserves the old unguarded behaviour, so in
`oracle.py::judge()` the call must be `filter_signals(..., made_ms=forecast["ts"])`. Copying
the module without that changes nothing about retrodiction.

`calibration.py` imports three things from the engine — `config.CONFIG`, `ledger.CAL_BINS`,
`models.now_ms` — plus `runtime.ledger` lazily, and is wired in from `pipeline.py`,
`loop.py`, `oracle.py`, `server.py` and `runtime.py`; those call sites are not included here.

`engine/` contains those two modules and nothing else — no `__init__.py`, and no stand-ins
for the four above — so nothing in this repo can overwrite a file of the package's own.

## The rest of it

- **Live stream** — https://www.youtube.com/@TheOracleStream1/live
- **Public forecast ledger** — https://github.com/woolfx/oracle-stream-ledger — every
  forecast committed before its outcome is known, every audit, the running scorecard
- **The engine** — https://github.com/jangles-byte/Pythia (MIT, not ours)

MIT licensed. The figures here are snapshots — the calibration numbers as of 2026-07-27, the
retrodiction audit 2026-07-28; the ledger repo has the live ones.
