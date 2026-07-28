"""Regression tests for engine/relevance.py — the evidence-admissibility guard.

Every case in `TestIncident010` is a real mis-resolution taken from the public
audit at github.com/woolfx/oracle-stream-ledger/blob/main/audits/2026-07-18-ledger-audit.md,
where an LLM judge resolved forecasts YES by citing whatever large event happened
to be in its evidence dump. 43 resolutions were voided. These are the cases that
must never pass again.

`TestIncident014` and `TestIncident017` cover the two premature-resolution
classes: a forecast that dates its own event past its horizon, and one that names
a scheduled fixture without a date. 21 more resolutions were voided for those.

Pure stdlib, no fixtures, no network:

    python3 -m unittest discover -s tests -v
"""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.relevance import (  # noqa: E402
    citable,
    event_end_ms,
    filter_signals,
    related,
    scheduled_end_ms,
)


def _ms(y, m, d) -> int:
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


class TestIncident010(unittest.TestCase):
    """The 2026-07-18 audit: evidence that had nothing to do with the claim."""

    def test_earthquake_cannot_resolve_a_fed_rate_forecast(self):
        # pred_fe77d727d9, graded YES on this evidence. The headline case.
        self.assertFalse(related(
            "The US Federal Reserve will raise interest rates by 50 basis points", None,
            "A magnitude 7.3M earthquake occurred in Mexico on 2026-07-17, "
            "which is within the window."))

    def test_earthquake_cannot_resolve_an_airstrike_forecast(self):
        # pred_40adcc6066 — graded YES, and a receipt card was cut from it.
        self.assertFalse(related(
            "The US will launch additional airstrikes against Iran targets", None,
            "A magnitude 7.3M earthquake occurred in Mexico on July 17, 2026, "
            "potentially affecting 60 million people"))

    def test_a_cyclone_cannot_resolve_a_thunderstorm_claim(self):
        # pred_55fbc0b586: a New York storm claim "confirmed" by cyclone ELIDA-26.
        self.assertFalse(related(
            "A severe thunderstorm warning will be issued for the state of New York", None,
            "Green notification for tropical cyclone ELIDA-26: Population affected by "
            "Category 1 (120 km/h) wind"))

    def test_a_storm_in_the_wrong_state_is_inadmissible(self):
        # pred_2da2b372e2: a Denver claim resolved off an Eastern-time warning.
        self.assertFalse(related(
            "A severe thunderstorm warning will be issued for the Denver metropolitan area",
            "Denver, CO",
            "Severe Thunderstorm Warning issued July 17 at 3:49PM EDT until July 17 at "
            "4:30PM EDT by NWS Mount Holly NJ"))

    def test_the_right_state_is_admissible(self):
        # pred_fe6b35a2a5 — kept by the audit. The guard must not be a blanket "no".
        self.assertTrue(related(
            "A severe thunderstorm warning will be issued for the state of New Jersey", None,
            "A Severe Thunderstorm Warning was issued for New Jersey on July 17"))

    def test_neighbouring_cities_are_not_interchangeable(self):
        # pred_7ee8013c66: a Phoenix claim resolved off a Tucson warning (~180 km).
        self.assertFalse(related(
            "Flash flooding will occur in the Phoenix metropolitan area", "Phoenix, AZ",
            "A Flash Flood Warning issued by NWS Tucson AZ on 2026-07-16 at 2:51PM MST"))

    def test_one_chokepoint_cannot_resolve_another(self):
        # A Hormuz claim must not resolve off Bab el-Mandeb: different waterways.
        self.assertFalse(related(
            "The Strait of Hormuz will experience a brief oil tanker congestion", None,
            "Chokepoint CRITICAL: Bab el-Mandeb — 6.2M bpd oil / LIVE SHIPS: 0."))


class TestCitableIsStricterThanRelated(unittest.TestCase):
    """`citable` is the bar for a signal cited to say YES."""

    def test_a_smaller_quake_cannot_confirm_a_claimed_magnitude(self):
        # pred_ca9fe5b2b7: a "magnitude 6.0" claim resolved YES against a M5.2 in
        # the same region. Right place, right hazard, wrong number.
        claim = "A magnitude 6.0 earthquake will occur in the Mariana Islands region"
        signal = "A magnitude 5.2 earthquake struck 48 km E of the Mariana Islands"
        self.assertTrue(related(claim, None, signal))    # admissible as evidence
        self.assertFalse(citable(claim, None, signal))   # but cannot confirm a YES

    def test_an_equal_or_larger_quake_does_confirm(self):
        self.assertTrue(citable(
            "A magnitude 6.0 earthquake will occur in the Mariana Islands region", None,
            "M6.4 earthquake — 48 km E of the Mariana Islands"))

    def test_a_weaker_hurricane_cannot_confirm_a_category_claim(self):
        self.assertFalse(citable(
            "A Category 2 hurricane will make landfall in Honduras", "Honduras",
            "Green notification for tropical cyclone in Honduras: Category 1 (120 km/h)"))

    def test_a_placeless_signal_cannot_confirm_a_located_claim(self):
        # The ELIDA-26 pattern: one placeless cyclone line was used to confirm
        # landfall in Mexico, the Philippines and Honduras at once.
        signal = "Green notification for tropical cyclone ELIDA-26, population affected: 0"
        for place in ("Mexico", "the Philippines", "Honduras"):
            with self.subTest(place=place):
                self.assertFalse(citable(
                    f"A tropical cyclone will make landfall in {place}", place, signal))


class TestFilterSignals(unittest.TestCase):

    def test_drops_inadmissible_keeps_order_and_dedupes(self):
        claim = "A severe thunderstorm warning will be issued for the Chicago area"
        signals = [
            "A magnitude 7.3M earthquake occurred in Mexico",                  # wrong topic
            "Severe Thunderstorm Warning issued for the Chicago area, IL",     # admissible
            "Severe Thunderstorm Warning issued for the Chicago area, IL",     # duplicate
            "Chokepoint CRITICAL: Strait of Hormuz",                           # wrong topic
        ]
        out = filter_signals(claim, "Chicago, IL", signals)
        self.assertEqual(out, ["Severe Thunderstorm Warning issued for the Chicago area, IL"])

    def test_respects_the_cap(self):
        claim = "Flash flooding will occur in Tucson, Arizona"
        signals = [f"Flash Flood Warning issued for Tucson AZ, event {i}" for i in range(30)]
        self.assertEqual(len(filter_signals(claim, "Tucson, AZ", signals, cap=5)), 5)


class TestIncident014(unittest.TestCase):
    """Statements that date their own event beyond the forecast window."""

    def test_reads_a_wordy_date(self):
        ref = _ms(2026, 7, 20)
        end = event_end_ms(
            "The Fed will raise rates at the next FOMC meeting on July 28th", ref)
        self.assertEqual(end, _ms(2026, 7, 29))          # END of July 28
        self.assertGreater(end, ref + 24 * 3600 * 1000)  # ...past a 24h horizon

    def test_reads_an_iso_date(self):
        self.assertEqual(
            event_end_ms("Brent will close above $92 on 2026-08-31", _ms(2026, 7, 20)),
            _ms(2026, 9, 1))

    def test_takes_the_latest_date_named(self):
        self.assertEqual(
            event_end_ms("Between July 20 and August 3, OPEC will cut output",
                         _ms(2026, 7, 19)),
            _ms(2026, 8, 4))

    def test_a_yearless_date_far_behind_rolls_forward(self):
        # A late-December forecast naming "January 5" means next January.
        self.assertEqual(
            event_end_ms("The report lands on January 5", _ms(2026, 12, 20)),
            _ms(2027, 1, 6))

    def test_undated_statements_return_none(self):
        self.assertIsNone(event_end_ms(
            "A major cyberattack will occur on a US government agency", _ms(2026, 7, 20)))


class TestIncident017(unittest.TestCase):
    """Statements naming a scheduled fixture with no date — the INC-014 sequel."""

    FIXTURES = [
        ("EIA Weekly Petroleum Status Report", _ms(2026, 7, 30)),
        ("EIA Weekly Natural Gas Storage Report", _ms(2026, 7, 24)),
        ("FOMC Meeting", _ms(2026, 7, 29)),
        ("FOMC Meeting", _ms(2026, 9, 16)),
    ]

    def test_separates_the_petroleum_report_from_the_gas_one(self):
        # The 13 voided resolutions were all this statement, graded NO a week early.
        self.assertEqual(
            scheduled_end_ms("The EIA weekly petroleum status report will move CL=F by +2.5%",
                             self.FIXTURES, _ms(2026, 7, 22)),
            _ms(2026, 7, 30))

    def test_matches_the_gas_report_on_its_own_terms(self):
        self.assertEqual(
            scheduled_end_ms("Natural gas storage will surprise to the upside",
                             self.FIXTURES, _ms(2026, 7, 22)),
            _ms(2026, 7, 24))

    def test_the_next_fomc_wins_over_a_later_one(self):
        self.assertEqual(
            scheduled_end_ms("The Fed will hold at the next FOMC meeting",
                             self.FIXTURES, _ms(2026, 7, 22)),
            _ms(2026, 7, 29))

    def test_past_fixtures_are_ignored(self):
        self.assertIsNone(
            scheduled_end_ms("The Fed will hold at the next FOMC meeting",
                             self.FIXTURES, _ms(2026, 10, 1)))

    def test_an_unrelated_statement_matches_nothing(self):
        self.assertIsNone(
            scheduled_end_ms("A tropical cyclone will make landfall in the Philippines",
                             self.FIXTURES, _ms(2026, 7, 22)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
