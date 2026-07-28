"""Evidence relevance — the guard between a forecast and the signal that resolves it.

The 2026-07-18 receipts audit (INC-010) found the judge citing whatever big event
was in its evidence dump: a Fed-rates forecast resolved YES off a Mexico earthquake,
a Denver storm warning "confirmed" by a Georgia one. The judge's LLM is small and
eager to please; this module is the deterministic part it can't talk its way past.

Two public calls, shared by the judge (engine/loop.py + oracle.judge), the
re-resolution audit tooling, and the receipt catcher (infra/receipts.py):

    related(statement, location, signal) -> bool
        Could `signal` possibly bear on this forecast? Hard vetoes:
        both sides name places and none overlap; both sides carry a
        recognized hazard/topic concept and they're disjoint.

    filter_signals(statement, location, signals, cap=SIGNAL_CAP, made_ms=None) -> list[str]
        The admissible-evidence list shown to the judge, order preserved.
        Given `made_ms` (the forecast's publication time) it also drops any
        signal whose own text dates its event earlier — a forecast may not be
        confirmed by something that had already happened when it was written.

Pure stdlib, no LLM: it must stay cheap enough to run on every verdict and
receipt, and dumb enough to be auditable by a human reading this file.
"""
from __future__ import annotations

import re

# ── tokens ──
_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "by", "for", "and", "or",
    "will", "be", "is", "are", "was", "were", "its", "it", "this", "that",
    "with", "within", "next", "hours", "hour", "days", "day", "week", "weeks",
    "area", "region", "due", "occur", "occurred", "happen", "issued", "make",
    "made", "strike", "hit", "major", "significant", "severe", "large", "new",
    "green", "orange", "red", "notification", "warning", "alert", "reported",
    "during", "after", "before", "near", "into", "from", "over", "under",
}

def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9']+", text.lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


# ── hazard / topic concepts ──
CONCEPTS: dict[str, set[str]] = {
    "quake":    {"earthquake", "quake", "seismic", "aftershock", "tremor", "magnitude"},
    "cyclone":  {"hurricane", "cyclone", "typhoon", "tropical"},
    "storm":    {"thunderstorm", "thunderstorms", "tornado", "hail"},
    "flood":    {"flood", "floods", "flooding", "rainfall", "deluge", "inundation"},
    "wildfire": {"wildfire", "wildfires", "forest", "blaze", "bushfire"},
    "volcano":  {"volcano", "volcanic", "eruption", "ash", "lava"},
    "cyber":    {"cyberattack", "cyber", "hack", "hacked", "hacker", "breach",
                 "ransomware", "malware", "ddos", "phishing"},
    "military": {"airstrike", "airstrikes", "missile", "missiles", "drone",
                 "military", "offensive", "troops", "forces", "invasion",
                 "shelling", "bombing", "ceasefire"},
    "sanctions": {"sanction", "sanctions", "embargo", "tariff", "tariffs"},
    "market":   {"oil", "price", "prices", "stock", "stocks", "dollar", "euro",
                 "currency", "rate", "rates", "fed", "reserve", "inflation",
                 "trading", "crude", "brent", "futures", "markets"},
    "shipping": {"tanker", "tankers", "shipping", "vessel", "vessels", "strait",
                 "straits", "chokepoint", "canal", "blockade", "maritime",
                 "mandeb", "hormuz", "bosphorus", "suez"},
    "power":    {"outage", "blackout", "grid", "electricity", "npp", "nuclear",
                 "reactor"},
    "health":   {"outbreak", "epidemic", "pandemic", "virus", "disease",
                 "cholera", "ebola", "measles", "influenza"},
    "unrest":   {"protest", "protests", "riot", "riots", "unrest", "coup",
                 "demonstrators"},
}

def _concepts(text: str) -> set[str]:
    toks = set(re.findall(r"[a-z][a-z0-9']+", text.lower()))
    return {name for name, words in CONCEPTS.items() if toks & words}


# ── places ──
# Canonical keys: "us:XX" for states, "cc:<name>" for countries/territories.
_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington state": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}
_STATE_ABBRS = set(_STATES.values()) | {"DC"}

_CITIES = {  # city -> state; the WFO names that show up in NWS product lines, plus claim staples
    "houston": "TX", "galveston": "TX", "dallas": "TX", "san angelo": "TX",
    "midland": "TX", "odessa": "TX", "el paso": "TX", "shreveport": "LA",
    "phoenix": "AZ", "tucson": "AZ", "flagstaff": "AZ", "grand canyon": "AZ",
    "denver": "CO", "pueblo": "CO", "grand junction": "CO", "boulder": "CO",
    "chicago": "IL", "kansas city": "MO", "st louis": "MO", "duluth": "MN",
    "minneapolis": "MN", "billings": "MT", "great falls": "MT", "missoula": "MT",
    "salt lake city": "UT", "las vegas": "NV", "elko": "NV", "reno": "NV",
    "los angeles": "CA", "san francisco": "CA", "san diego": "CA",
    "sacramento": "CA", "seattle": "WA", "portland": "OR", "boise": "ID",
    "new york city": "NY", "nyc": "NY", "manhattan": "NY", "buffalo": "NY",
    "albany": "NY", "boston": "MA", "norton": "MA", "caribou": "ME",
    "mount holly": "NJ", "newark": "NJ", "philadelphia": "PA",
    "pittsburgh": "PA", "baltimore": "MD", "washington dc": "DC",
    "atlanta": "GA", "peachtree city": "GA", "miami": "FL",
    "jacksonville": "FL", "tallahassee": "FL", "tampa": "FL",
    "charlotte": "NC", "nashville": "TN", "memphis": "TN", "morristown": "TN",
    "louisville": "KY", "indianapolis": "IN", "wilmington": "OH",
    "cincinnati": "OH", "cleveland": "OH", "detroit": "MI", "bismarck": "ND",
    "riverton": "WY", "albuquerque": "NM", "santa teresa": "NM",
    "anchorage": "AK", "new orleans": "LA", "oklahoma city": "OK",
    "wichita": "KS", "omaha": "NE", "des moines": "IA", "milwaukee": "WI",
    "columbia": "SC", "charleston": "SC", "richmond": "VA", "norfolk": "VA",
}

_REGIONS = {  # region phrase -> set of state abbrs
    "southeastern united states": {"FL", "GA", "SC", "NC", "AL", "MS", "TN", "LA", "AR", "VA"},
    "southeast us": {"FL", "GA", "SC", "NC", "AL", "MS", "TN", "LA", "AR", "VA"},
    "southern united states": {"TX", "LA", "MS", "AL", "GA", "FL", "SC", "NC",
                               "TN", "AR", "OK", "KY", "VA"},
    "gulf coast": {"TX", "LA", "MS", "AL", "FL"},
    "east coast": {"ME", "NH", "MA", "RI", "CT", "NY", "NJ", "PA", "DE", "MD",
                   "DC", "VA", "NC", "SC", "GA", "FL"},
    "west coast": {"CA", "OR", "WA"},
    "mid-atlantic": {"NY", "NJ", "PA", "DE", "MD", "DC", "VA"},
    "mid atlantic": {"NY", "NJ", "PA", "DE", "MD", "DC", "VA"},
    "new england": {"ME", "NH", "VT", "MA", "RI", "CT"},
    "midwest": {"OH", "IN", "IL", "MI", "WI", "MN", "IA", "MO", "ND", "SD",
                "NE", "KS"},
    "pacific northwest": {"WA", "OR", "ID"},
    "bay area": {"CA"},
    "great plains": {"ND", "SD", "NE", "KS", "OK", "TX"},
}

_COUNTRIES = {
    # name -> canonical; includes territories/areas GDACS+USGS actually emit
    "mexico": "mexico", "mexican": "mexico", "guatemala": "guatemala",
    "honduras": "honduras", "nicaragua": "nicaragua", "cuba": "cuba",
    "haiti": "haiti", "canada": "canada", "canadian": "canada",
    "japan": "japan", "japanese": "japan", "philippines": "philippines",
    "philippine": "philippines", "vietnam": "vietnam", "taiwan": "taiwan",
    "china": "china", "chinese": "china", "india": "india",
    "indonesia": "indonesia", "malaysia": "malaysia", "bangladesh": "bangladesh",
    "pakistan": "pakistan", "iran": "iran", "iranian": "iran", "iraq": "iraq",
    "israel": "israel", "israeli": "israel", "gaza": "gaza",
    "palestinian": "gaza", "lebanon": "lebanon", "syria": "syria",
    "yemen": "yemen", "saudi": "saudi arabia", "turkey": "turkey",
    "turkish": "turkey", "ukraine": "ukraine", "ukrainian": "ukraine",
    "russia": "russia", "russian": "russia", "poland": "poland",
    "germany": "germany", "german": "germany", "france": "france",
    "french": "france", "austria": "austria", "austrian": "austria",
    "czech": "czech republic", "italy": "italy", "spain": "spain",
    "portugal": "portugal", "greece": "greece", "greek": "greece",
    "netherlands": "netherlands", "belgium": "belgium",
    "united kingdom": "uk", "britain": "uk", "british": "uk", "england": "uk",
    "norway": "norway", "sweden": "sweden", "finland": "finland",
    "peru": "peru", "peruvian": "peru", "chile": "chile", "chilean": "chile",
    "colombia": "colombia", "colombian": "colombia", "ecuador": "ecuador",
    "argentina": "argentina", "brazil": "brazil", "bolivia": "bolivia",
    "venezuela": "venezuela", "fiji": "fiji", "tonga": "tonga",
    "vanuatu": "vanuatu", "new zealand": "new zealand",
    "papua new guinea": "papua new guinea", "solomon islands": "solomon islands",
    "australia": "australia", "kermadec": "kermadec",
    "mariana": "mariana", "guam": "mariana",
    "south sandwich": "south sandwich", "aleutian": "us-alaska",
    "kamchatka": "kamchatka", "sumatra": "indonesia", "java": "indonesia",
    "timor": "timor leste", "egypt": "egypt", "libya": "libya",
    "sudan": "sudan", "ethiopia": "ethiopia", "somalia": "somalia",
    "kenya": "kenya", "nigeria": "nigeria", "south africa": "south africa",
    "morocco": "morocco", "algeria": "algeria", "tajikistan": "tajikistan",
    "afghanistan": "afghanistan", "nepal": "nepal", "myanmar": "myanmar",
    "thailand": "thailand", "south korea": "south korea",
    "north korea": "north korea", "puerto rico": "us:PR",
    "virgin islands": "us:VI", "georgia (country)": "georgia-country",
}

# Waterways/chokepoints are places too — a Hormuz claim must not resolve off Bab el-Mandeb.
_WATERWAYS = {
    "hormuz": "hormuz", "bab el-mandeb": "mandeb", "bab el mandeb": "mandeb",
    "suez": "suez", "panama canal": "panama canal",
    "turkish straits": "turkish straits", "bosphorus": "turkish straits",
    "danish straits": "danish straits", "cape of good hope": "good hope",
    "malacca": "malacca", "gibraltar": "gibraltar",
}

def _places(text: str) -> tuple[set[str], set[str]]:
    """(coarse, cities): coarse = country/state/waterway keys, cities = city keys."""
    t = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "
    # the oceanic ridge is not the US Mid-Atlantic region
    t = t.replace(" mid atlantic ridge ", " midatlanticridge ")
    coarse: set[str] = set()
    cities: set[str] = set()
    for name, abbr in _STATES.items():
        if f" {name} " in t:
            coarse.add("us:" + abbr)
    for name, states in _REGIONS.items():
        if f" {name.replace('-', ' ')} " in t:
            coarse |= {"us:" + s for s in states}
    for name, st in _CITIES.items():
        if f" {name} " in t:
            cities.add(name)
            coarse.add("us:" + st)
    for name, cc in _COUNTRIES.items():
        if f" {name} " in t:
            coarse.add("cc:" + cc)
    for name, key in _WATERWAYS.items():
        if f" {name.replace('-', ' ')} " in t:
            coarse.add("ww:" + key)
    # bare state abbreviations, as NWS writes them ("NWS Duluth MN", "Columbia SC")
    for abbr in re.findall(r"\b([A-Z]{2})\b", text):
        if abbr in _STATE_ABBRS:
            coarse.add("us:" + abbr)
    return coarse, cities


def related(statement: str, location: str | None, signal: str) -> bool:
    """Could this signal bear on this forecast? False = inadmissible as evidence."""
    claim = f"{statement} {location or ''}"
    c_coarse, c_cities = _places(claim)
    s_coarse, s_cities = _places(signal)

    # Hard veto 1: both sides name recognizable places that share nothing.
    if c_coarse and s_coarse and not (c_coarse & s_coarse):
        return False
    # Hard veto 2: both name specific known cities, none shared (Tucson ≠ Phoenix).
    if c_cities and s_cities and not (c_cities & s_cities):
        return False
    # Hard veto 3: both carry recognized topic concepts that are fully disjoint
    # (an earthquake can't resolve a cyclone claim, even in the same country).
    c_con, s_con = _concepts(claim), _concepts(signal)
    if c_con and s_con and not (c_con & s_con):
        return False
    if c_con & s_con:
        return True
    # No shared concept vocabulary: fall back on informative token overlap.
    return len(_tokens(claim) & _tokens(signal)) >= 3


# "magnitude 6.0" / "Magnitude 5.7M" / "M4.5" — but not "6.2M bpd" (no \b inside
# "2M"), "km", "million", or depth figures. Plus the reversed "4.5 magnitude" form.
_MAG_PRE = re.compile(r"(?:magnitude\s+|\bm)(\d(?:\.\d+)?)", re.IGNORECASE)
_MAG_POST = re.compile(r"\b(\d(?:\.\d+)?)\s+magnitude", re.IGNORECASE)
_CAT_RE = re.compile(r"category\s+(\d)", re.IGNORECASE)


def _magnitude(text: str) -> float | None:
    vals = [float(m.group(1)) for m in _MAG_PRE.finditer(text)]
    vals += [float(m.group(1)) for m in _MAG_POST.finditer(text)]
    return max(vals) if vals else None


def _thresholds_met(claim: str, signal: str) -> bool:
    """Quantitative floors the LLM judge reliably fumbles: a claimed magnitude or
    hurricane category must be met by the cited signal's own number (llama3.1
    happily called a M5.7 signal proof of a 'magnitude 6.0' forecast). A signal
    with no number at all cannot confirm a numbered claim."""
    cm = _magnitude(claim)
    if cm is not None and cm >= 3:          # ignore stray small numbers
        sm = _magnitude(signal)
        if sm is None or sm + 0.05 < cm:
            return False
    cc = _CAT_RE.search(claim)
    if cc:
        sc = _CAT_RE.search(signal)
        if sc is None or int(sc.group(1)) < int(cc.group(1)):
            return False
    return True


def citable(statement: str, location: str | None, signal: str) -> bool:
    """Stricter bar for a signal cited to CONFIRM a forecast ("yes"):
    related(), and if the claim names a place the signal must name a compatible
    one — a location-less signal (ELIDA-26's "population affected: 0") must never
    confirm a landfall in Mexico, the Philippines, and Honduras at once — and any
    claimed magnitude/category floor must be met by the signal's own numbers."""
    if not related(statement, location, signal):
        return False
    if not _thresholds_met(statement, signal):
        return False
    claim = f"{statement} {location or ''}"
    c_coarse, _ = _places(claim)
    if not c_coarse:
        return True
    s_coarse, _ = _places(signal)
    return bool(c_coarse & s_coarse)


# How many admissible signals reach the judge's numbered list. Exported because
# `filter_signals` keeps the LAST `cap` entries: a caller appending a new class of
# evidence to a full pool would silently evict the archived window signals unless
# it widens the window by what it added. Bind to this rather than re-typing 14.
SIGNAL_CAP = 14


# ── retrodiction guard (2026-07-28) ──
# A forecast is only a forecast if its evidence came AFTER it. citable() checks
# subject, place and thresholds but never ORDERING, so an event already sitting in
# the oracle's own brief could be forecast, then graded "correct" against itself.
# Found live: pred_33040ed18b, "A major earthquake will strike Mexico within the
# next fortnight", published 2026-07-17 17:32 UTC and resolved YES citing the M7.3
# of 14:48 UTC the SAME MORNING — 2.7 h before it was written. GDACS lines carry
# their own event time and linger in the brief for days, so the stale timestamp
# rode the "[now]" snapshot straight into the numbered evidence.
#
# Conservative by construction: only unambiguous, fully-specified timestamps are
# read, and an ambiguous day/month pair yields None. None means "cannot prove
# anything", NEVER "fine" — an unparseable signal is left admissible, because the
# common case (NWS product lines, market snapshots) states no absolute time and
# dropping those would gut the judge's evidence.
_EV_ISO = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)[T ](\d{1,2}):(\d{2})")
_EV_DMY = re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d\d)\s+(\d{1,2}):(\d{2})\s*UTC")
_EV_MDY = re.compile(r"\bOn (\d{1,2})/(\d{1,2})/(20\d\d) (\d{1,2}):(\d{2}):\d{2}\s*(AM|PM)")


def signal_event_ms(text: str) -> int | None:
    """Absolute event time stated inside a signal, epoch ms — or None if the text
    states none, or states one that cannot be read without guessing."""
    import datetime as _dt

    m = _EV_ISO.search(text)
    if m:
        y, mo, d, h, mi = (int(x) for x in m.groups())
    elif (m := _EV_DMY.search(text)):
        d, mo, y, h, mi = (int(x) for x in m.groups())
        if d <= 12:
            return None                     # could equally be MM/DD — refuse to guess
    elif (m := _EV_MDY.search(text)):
        mo, d, y, h, mi = (int(m.group(i)) for i in range(1, 6))
        h = h % 12 + (12 if m.group(6) == "PM" else 0)
    else:
        return None
    try:
        return int(_dt.datetime(y, mo, d, h, mi,
                                tzinfo=_dt.timezone.utc).timestamp() * 1000)
    except ValueError:
        return None


def filter_signals(statement: str, location: str | None,
                   signals: list[str], cap: int = SIGNAL_CAP,
                   made_ms: int | None = None) -> list[str]:
    """Admissible evidence for the judge, order preserved, deduped, capped.

    `made_ms` is the forecast's publication time. When given, any signal whose own
    text dates its event BEFORE that moment is dropped: the judge cannot cite what
    it never sees, so retrodiction is blocked at the source rather than patched at
    the ledger. Omitted (the default) preserves the pre-2026-07-28 behaviour.
    """
    out: list[str] = []
    seen: set[str] = set()
    for s in signals:
        s = s.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        if made_ms is not None:
            ev = signal_event_ms(s)
            if ev is not None and ev < made_ms:
                continue
        if related(statement, location, s):
            out.append(s)
    return out[-cap:]


# ── explicit event dates (INC-014) ──
# The 2026-07-20 audit found eight public MISSes on "the FOMC will raise rates
# at the meeting on July 28th" — graded days before July 28 existed, because
# the statements carried 24h horizons and the judge's rule "window closed with
# no confirming signal → no" is blind to the statement dating its own event
# beyond the window. A dated claim is unresolvable before its day ends; these
# helpers let the loop defer it (and the pipeline size horizons correctly).

_MONTH_NUM = {m: i + 1 for i, m in enumerate((
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"))}
_DATE_WORDY = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october"
    r"|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?\b", re.I)
_DATE_ISO = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)\b")


def event_end_ms(statement: str, ref_ms: int) -> int | None:
    """Latest explicit calendar date in the statement, as END of that day in
    epoch ms (UTC) — or None when no explicit date is named. Wordy dates
    without a year take the ref year, rolling one year forward when that
    lands more than 60 days behind ref (a late-December forecast naming
    "January 5"). Deterministic and dumb on purpose, like the rest of this
    module: month-name + day, optional year, or ISO YYYY-MM-DD."""
    import datetime as _dt

    ref = _dt.datetime.fromtimestamp(ref_ms / 1000, _dt.timezone.utc).replace(tzinfo=None)
    ends: list[_dt.datetime] = []
    for m in _DATE_WORDY.finditer(statement):
        mon, day = _MONTH_NUM[m.group(1).lower()], int(m.group(2))
        year = int(m.group(3)) if m.group(3) else ref.year
        try:
            d = _dt.datetime(year, mon, day)
        except ValueError:
            continue
        if not m.group(3) and (ref - d).days > 60:
            d = d.replace(year=year + 1)
        ends.append(d)
    for m in _DATE_ISO.finditer(statement):
        try:
            ends.append(_dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            continue
    if not ends:
        return None
    end = max(ends) + _dt.timedelta(days=1)   # end of the named day
    return int(end.replace(tzinfo=_dt.timezone.utc).timestamp() * 1000)


# Generic scheduling boilerplate — shared by every fixture title, so it must NOT
# count toward a match (else the petroleum report and the gas report look identical).
_SCHED_GENERIC = frozenset({
    "the", "and", "for", "report", "reports", "weekly", "monthly", "quarterly",
    "meeting", "meetings", "status", "data", "update", "index", "day", "date",
    "release", "scheduled", "confirmed", "first", "second", "round", "decision",
    "rate", "rates", "will", "announcement", "results", "next", "upcoming",
})


def _sched_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower())
            if len(t) >= 3 and not t.isdigit() and t not in _SCHED_GENERIC}


def scheduled_end_ms(statement: str, fixtures: "list[tuple[str, int]]", ref_ms: int) -> int | None:
    """End-of-day ms of the NEXT scheduled fixture a statement refers to by name
    (e.g. "the EIA weekly petroleum status report" -> that report's date), or None.
    Companion to event_end_ms for INC-014: a forecast that names a scheduled fixture
    carries no explicit date, so without this it gets a 24h horizon and is graded
    "no" before the fixture ever occurs (13 EIA-report + 5 FOMC premature NOs found
    2026-07-22). `fixtures` = (title, end_of_day_ms) from the [SCHEDULED] feed.

    A SPECIFIC match (>=2 shared content tokens — generic scheduling words excluded)
    wins and cleanly separates the petroleum report from the natural-gas one; failing
    that, a shared acronym anchor (EIA / FOMC / CPI ...) matches the fixture type.
    Among matches, the soonest future occurrence wins ("the next FOMC")."""
    s_tokens = _sched_tokens(statement)
    if not s_tokens:
        return None
    specific: list[int] = []
    anchored: list[int] = []
    for title, end_ms in fixtures:
        if end_ms <= ref_ms:
            continue
        overlap = s_tokens & _sched_tokens(title)
        if len(overlap) >= 2:
            specific.append(end_ms)
        elif overlap & {a.lower() for a in re.findall(r"\b[A-Z]{2,6}\b", title)}:
            anchored.append(end_ms)
    pool = specific or anchored          # prefer the specific fixture over a bare acronym
    return min(pool) if pool else None


def scheduled_fixtures(events) -> "list[tuple[str, int]]":
    """(title, end_of_day_ms) for each [SCHEDULED] world event — what
    scheduled_end_ms matches against. Duck-typed on .category/.raw/.title so this
    module stays import-light; shared by the pipeline (commit) and resolve loop."""
    import datetime as _dt
    out: list = []
    for e in events or []:
        if getattr(e, "category", None) != "scheduled":
            continue
        raw = getattr(e, "raw", None) or {}
        ds = raw.get("date")
        if not ds:
            continue
        try:
            d = _dt.datetime.strptime(str(ds)[:10], "%Y-%m-%d") + _dt.timedelta(days=1)
        except ValueError:
            continue
        out.append((raw.get("title") or getattr(e, "title", ""),
                    int(d.replace(tzinfo=_dt.timezone.utc).timestamp() * 1000)))
    return out
