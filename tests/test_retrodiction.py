"""Tests for the retrodiction guard (2026-07-28, engine/relevance.py).

A forecast is only a forecast if its evidence came AFTER it. `citable()` checks
subject, place and thresholds but never ORDERING, so an event already sitting in
the oracle's own brief could be forecast and then graded "correct" against itself.

Found live, not hypothetically: `pred_33040ed18b` — "A major earthquake will strike
Mexico within the next fortnight", published 2026-07-17 17:32 UTC, resolved YES
citing the GDACS M7.3 of 14:48 UTC **the same morning**, 2.7 h before the forecast
was written. GDACS lines carry their own event time and linger in the brief for
days, so the stale timestamp rode the "[now]" snapshot into the numbered evidence.
A shareable receipt card had already been cut from it.

Three properties are load-bearing:

1. **It blocks the real case.** The exact live signal/timestamp pair must be
   dropped for a forecast written after it. `test_the_live_regression` is that
   record, pinned.
2. **It does not over-block.** Only unambiguous timestamps are read. NWS product
   lines and market snapshots state no absolute time, and they are the bulk of the
   judge's evidence — if silence were treated as guilt the judge would starve and
   every verdict would collapse to "no". Silence must stay ADMISSIBLE.
3. **It refuses to guess.** "05/06/2026" is 5 June or 6 May depending on locale.
   Guessing wrong invents a retrodiction that never happened (or misses a real
   one). Ambiguity must return None, not a coin flip.

Run: `uv run python tests/test_retrodiction.py`  (stdlib unittest; no pytest needed).
"""
import datetime as dt
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.relevance import filter_signals, signal_event_ms  # noqa: E402


def ms(y, mo, d, h=0, mi=0) -> int:
    return int(dt.datetime(y, mo, d, h, mi, tzinfo=dt.timezone.utc).timestamp() * 1000)


# The verbatim signal text from the live failure, and the forecast that cited it.
LIVE_SIGNAL = ("[now] Green earthquake (Magnitude 7.3M, Depth:18.584km) in Mexico "
               "17/07/2026 14:48 UTC, 60 thousand (in MMI>=VII)")
LIVE_CLAIM = "A major earthquake will strike Mexico within the next fortnight"
LIVE_PUBLISHED = ms(2026, 7, 17, 17, 32)      # 2.7 h AFTER the quake it cited


class TestSignalEventMs(unittest.TestCase):
    """The parser reads only what it can read without guessing."""

    def test_gdacs_dmy_utc(self):
        self.assertEqual(signal_event_ms(LIVE_SIGNAL), ms(2026, 7, 17, 14, 48))

    def test_gdacs_prose_mdy_pm(self):
        t = "On 7/17/2026 2:48:39 PM, an earthquake occurred in Mexico"
        self.assertEqual(signal_event_ms(t), ms(2026, 7, 17, 14, 48))

    def test_gdacs_prose_mdy_am(self):
        t = "On 7/17/2026 9:05:00 AM, an earthquake occurred in Mexico"
        self.assertEqual(signal_event_ms(t), ms(2026, 7, 17, 9, 5))

    def test_midnight_and_noon_do_not_collide(self):
        """12 AM is 00:00 and 12 PM is 12:00 — the classic %12 off-by-twelve."""
        self.assertEqual(signal_event_ms("On 7/17/2026 12:00:00 AM, x"), ms(2026, 7, 17, 0, 0))
        self.assertEqual(signal_event_ms("On 7/17/2026 12:00:00 PM, x"), ms(2026, 7, 17, 12, 0))

    def test_iso(self):
        self.assertEqual(signal_event_ms("recorded 2026-07-19 08:30 by AIS"),
                         ms(2026, 7, 19, 8, 30))

    def test_ambiguous_day_month_refuses_to_guess(self):
        """Both components <= 12: could be DD/MM or MM/DD. None, never a guess."""
        self.assertIsNone(signal_event_ms("event at 05/06/2026 10:00 UTC"))

    def test_impossible_date_is_none(self):
        self.assertIsNone(signal_event_ms("event at 32/13/2026 10:00 UTC"))

    def test_nws_product_line_has_no_absolute_time(self):
        t = ("Severe Thunderstorm Warning issued July 16 at 6:38PM MDT until "
             "July 16 at 7:45PM MDT by NWS Billings MT")
        self.assertIsNone(signal_event_ms(t))

    def test_market_snapshot_has_no_absolute_time(self):
        t = "[now] WTI Crude Oil: 86.75 $/bbl (+2.17%) — 6-mo curve in backwardation"
        self.assertIsNone(signal_event_ms(t))

    def test_bare_date_without_time_is_not_enough(self):
        """A date with no clock time cannot be ordered against an intraday
        publication moment without assuming midnight — so it stays unparsed."""
        self.assertIsNone(signal_event_ms("IMF PortWatch transit calls as of 2026-07-19"))


class TestFilterSignals(unittest.TestCase):
    """The gate drops what predates the forecast and nothing else."""

    CLAIM = "A major earthquake will strike Mexico"

    def test_the_live_regression(self):
        """pred_33040ed18b, pinned: the quake predates the forecast, so the judge
        must never be offered it."""
        kept = filter_signals(LIVE_CLAIM, "Mexico", [LIVE_SIGNAL], made_ms=LIVE_PUBLISHED)
        self.assertEqual(kept, [], "retrodiction reached the judge's evidence list")

    def test_same_signal_admissible_for_an_earlier_forecast(self):
        """The identical text is legitimate evidence for a forecast written BEFORE
        it — this is the difference between prediction and retrodiction, and the
        only thing separating the two cases is made_ms."""
        earlier = ms(2026, 7, 17, 9, 0)       # 5.8 h before the quake
        kept = filter_signals(LIVE_CLAIM, "Mexico", [LIVE_SIGNAL], made_ms=earlier)
        self.assertEqual(kept, [LIVE_SIGNAL])

    def test_unparseable_signals_stay_admissible(self):
        """Silence is not guilt. If this inverts, the judge starves."""
        sigs = ["[now] Green earthquake (Magnitude 6.1M) in Mexico, no timestamp given"]
        kept = filter_signals(self.CLAIM, "Mexico", sigs, made_ms=LIVE_PUBLISHED)
        self.assertEqual(kept, sigs)

    def test_omitting_made_ms_preserves_old_behaviour(self):
        """Backwards compatibility: the pre-2026-07-28 call shape is unfiltered."""
        kept = filter_signals(LIVE_CLAIM, "Mexico", [LIVE_SIGNAL])
        self.assertEqual(kept, [LIVE_SIGNAL])

    def test_mixed_pool_keeps_only_the_admissible_half(self):
        stale = LIVE_SIGNAL
        fresh = ("[now] Green earthquake (Magnitude 6.4M, Depth:12km) in Mexico "
                 "18/07/2026 09:15 UTC, 20 thousand (in MMI>=VII)")
        untimed = "[now] Green earthquake (Magnitude 5.9M) in Mexico reported by USGS"
        kept = filter_signals(LIVE_CLAIM, "Mexico", [stale, fresh, untimed],
                              made_ms=LIVE_PUBLISHED)
        self.assertEqual(kept, [fresh, untimed])

    def test_dedupe_and_order_survive_the_new_filter(self):
        a = "[now] Green earthquake (Magnitude 6.4M) in Mexico 18/07/2026 09:15 UTC"
        b = "[now] Green earthquake (Magnitude 6.6M) in Mexico 19/07/2026 09:15 UTC"
        kept = filter_signals(LIVE_CLAIM, "Mexico", [a, b, a], made_ms=LIVE_PUBLISHED)
        self.assertEqual(kept, [a, b])

    def test_cap_still_applies_and_keeps_the_last_entries(self):
        sigs = [f"[now] Green earthquake (Magnitude 6.{i}M) in Mexico USGS report {i}"
                for i in range(9)]
        kept = filter_signals(self.CLAIM, "Mexico", sigs, cap=3, made_ms=LIVE_PUBLISHED)
        self.assertEqual(kept, sigs[-3:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
