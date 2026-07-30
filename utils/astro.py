"""Local astrological chart computation, via kerykeion (Swiss Ephemeris) —
used as a data-supplying tool for the chat model. See utils/tools.py and
utils/tool_router.py for the general mechanism: the router decides a tool is
needed, extracts a single string argument from the user's message, the
tool's run() function turns that into a plain-text result, and a short
follow-up LLM call turns that result into a natural-language answer in the
user's own language (routes/chat.py:_handle_tool_request). None of that
plumbing changes here — this file only adds two new entries to
utils.tools.TOOL_REGISTRY.

Why this lives in ycplt and not ycplt_img: the project's hard rule is that
*models* (chat, vision, image generation/editing) belong exclusively in the
graphics service; this chat app hosts only the chat LLM and intent/tool
classification. kerykeion isn't a model at all — it's a deterministic
ephemeris calculation library (Swiss Ephemeris under the hood, via
pyswisseph), the same category of dependency as the sentence-transformers
embedding step already run in-process here for RAG. There's no inference
cost or hardware concern that would justify moving it to the other machine.

Offline by design: every call here passes online=False plus explicit
lat/lng/tz_str, so no network call or API key is ever needed (kerykeion's
online=True mode exists only for optional city-name geocoding via GeoNames,
deliberately not used, to keep this fully local like the rest of the app).

Licensing note: kerykeion is AGPL-3.0. Importing it directly into this
project (as done here) means this project's own distribution should be
under an AGPL-compatible license if that matters for how you distribute
ycplt — kerykeion's own docs flag this explicitly (their hosted "Astrologer
API" exists specifically so closed-source users can avoid it).

Extensibility — this is meant to grow into a general "astro engine", not
stay a one-off: ASTRO_OPERATIONS is a small registry, the same pattern as
utils/tools.py's TOOL_REGISTRY and ycplt_img's model-factory registry
(conf/models.py). Adding a new capability (synastry between two people, a
composite chart, birth-time rectification, electional/event-time search for
"when does X next happen") means writing one run_xxx(spec) -> str function
below plus one registry entry — nothing else needs to change. Only natal
and transit charts are implemented so far, since that's what's needed to
unblock the RAG-based interpretation experiment; the rest can be added
incrementally as those use cases become concrete.

Argument format (the single string a tool receives, per utils/tools.py):
free text is the expected/primary form — copy the birth date, time, and
place straight from the user's own message, e.g.
  "5 июля 1976 года в 4:30 в Одессе, 46°28'00\"N;30°44'00\"E"
_extract_fields() below parses this with a handful of regexes (ISO date,
DD.MM.YYYY, or "D <russian month> YYYY"; H:MM time; coordinates as either
decimal or degree-minute-second with N/S/E/W); the timezone is then
resolved automatically from the coordinates via timezonefinder (also fully
offline — no API call). This is deliberately tolerant rather than requiring
strict key=value input: earlier attempts required the *classifying* model
(utils/tool_router.py's small, cheap, zero-shot call) to reformat dates,
convert coordinates, and look up an IANA timezone name in a single short
completion, which turned out to fail in practice — asking that same small
model to just quote the user's own wording back verbatim is a much easier,
more reliable task. Strict "key=value;key=value" pairs (date=1990-03-12;
time=14:30;lat=55.7558;lon=37.6173;tz=Europe/Moscow) are still accepted and
take priority over free-text parsing when present, since they're
unambiguous and cost nothing extra to support.

Required, one way or another: date, time, and coordinates (lat+lon).
Optional: name (defaults to "Subject"); tz (auto-resolved from coordinates
if omitted). The transit operation additionally accepts moment (ISO-ish
"YYYY-MM-DDTHH:MM", default "now", key=value only — not covered by the
free-text parser) and moment_lat/moment_lon/moment_tz (default: same as
birth place — override these if the person is asking about a moment
somewhere other than where they were born, since houses depend on
location). If date/time/coordinates can't be found at all, run_natal/
run_transit return a Russian-language explanation of what's missing instead
of raising — that string flows straight into the same natural-language
follow-up step as a successful result, so the user sees a normal
clarifying question rather than an error. The tool descriptions in
utils/tools.py explicitly tell the classifying model never to invent
placeholder birth data and to skip the tool entirely if it isn't present in
the conversation.

A bare city name with no coordinates at all is handled too, as a last
resort after the coordinate parsing above finds nothing: _lookup_city()
looks it up via the geonamescache package (~34k cities worldwide, bundled
with the package itself — no download or file to place anywhere, still
fully offline). Matching is approximate rather than a real geocoder: exact
name match first, then a same-first-5-characters "stem" fallback (since a
Russian city name in a sentence is usually declined — "в Одессе", not the
gazetteer's nominative "Одесса" — and this project doesn't have a
morphological analyzer), and a name shared by multiple places worldwide
(e.g. "Odessa" in both Ukraine and Texas) resolves to whichever has the
larger population. Good enough for the common case, not a substitute for
a real geocoding service.
"""
import re
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

# Chart text (below) is rendered directly in Russian, not left for the
# follow-up LLM call to translate — an earlier version kept this in
# English and asked the interpreting model to translate sign/aspect names
# as part of the same generation, which turned out to be unreliable in
# practice: signs and aspect types came out wrong or invented (confirmed
# by testing — Mars in Leo was reported back as "в знаке Весов", "square"
# became "сферический аспект", neither of which mean anything). Moving
# this fixed vocabulary into deterministic Python removes that failure
# mode entirely; the interpreting model only has to reason over already-
# correct facts, not also transcribe unfamiliar astrological terminology.
_SIGN_NAMES_RU = {
    "Ari": "Овен", "Tau": "Телец", "Gem": "Близнецы", "Can": "Рак",
    "Leo": "Лев", "Vir": "Дева", "Lib": "Весы", "Sco": "Скорпион",
    "Sag": "Стрелец", "Cap": "Козерог", "Aqu": "Водолей", "Pis": "Рыбы",
}

_POINT_NAMES_RU = {
    "Sun": "Солнце", "Moon": "Луна", "Mercury": "Меркурий", "Venus": "Венера",
    "Mars": "Марс", "Jupiter": "Юпитер", "Saturn": "Сатурн", "Uranus": "Уран",
    "Neptune": "Нептун", "Pluto": "Плутон",
    "True_North_Lunar_Node": "Северный узел",
    "Ascendant": "Асцендент", "Medium_Coeli": "Середина неба (MC)",
}

_ASPECT_NAMES_RU = {
    "conjunction": "соединение", "opposition": "оппозиция", "trine": "трин",
    "square": "квадрат", "sextile": "секстиль",
}

_ASPECT_MOVEMENT_RU = {
    "Applying": "сходящийся, усиливается",
    "Separating": "расходящийся, ослабевает",
    "Static": "статичный",
}


def _point_ru(name: str) -> str:
    return _POINT_NAMES_RU.get(name, name)


def _aspect_ru(name: str) -> str:
    return _ASPECT_NAMES_RU.get(name, name)


def _movement_ru(name: str) -> str:
    return _ASPECT_MOVEMENT_RU.get(name, name)


_HOUSE_ORDER = [
    "First_House", "Second_House", "Third_House", "Fourth_House",
    "Fifth_House", "Sixth_House", "Seventh_House", "Eighth_House",
    "Ninth_House", "Tenth_House", "Eleventh_House", "Twelfth_House",
]

# (Russian display label, AstrologicalSubject attribute name)
_PLANET_ATTRS = [
    ("Солнце", "sun"), ("Луна", "moon"), ("Меркурий", "mercury"), ("Венера", "venus"),
    ("Марс", "mars"), ("Юпитер", "jupiter"), ("Сатурн", "saturn"),
    ("Уран", "uranus"), ("Нептун", "neptune"), ("Плутон", "pluto"),
    ("Северный узел", "true_north_lunar_node"),
]
_ANGLE_ATTRS = [("Асцендент", "ascendant"), ("Середина неба (MC)", "medium_coeli")]

# kerykeion's own point-name literals, used for AspectsFactory filtering —
# restricting to the classical set keeps the aspect list short and relevant
# (unfiltered, kerykeion also considers dozens of asteroids/fixed stars).
_ASPECT_ACTIVE_POINTS = [
    "Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn",
    "Uranus", "Neptune", "Pluto", "True_North_Lunar_Node",
    "Ascendant", "Medium_Coeli",
]
_MAJOR_ASPECTS = [
    {"name": "conjunction", "orb": 8},
    {"name": "opposition", "orb": 8},
    {"name": "trine", "orb": 7},
    {"name": "square", "orb": 7},
    {"name": "sextile", "orb": 5},
]

_REQUIRED_FIELDS = ("date", "time", "lat", "lon", "tz")

# --- free-text extraction -------------------------------------------------
# See the module docstring for why this exists: the tool_router classifier
# is a small, cheap, zero-shot call, and asking it to reformat dates,
# convert coordinates, and pick an IANA timezone name in one short
# completion proved unreliable in practice. Parsing the user's own wording
# with plain regexes is more robust than relying on that call to do it.

_RU_MONTH_STEMS = [
    # Ordered longest/most-specific stem first so e.g. "марта" matches
    # "март" rather than the generic "ма" stem meant for май/мая.
    ("сентябр", 9), ("октябр", 10), ("ноябр", 11), ("декабр", 12),
    ("феврал", 2), ("апрел", 4), ("август", 8), ("январ", 1),
    ("март", 3), ("июнь", 6), ("июн", 6), ("июль", 7), ("июл", 7),
    ("май", 5), ("мая", 5), ("ма", 5),
]

_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DATE_DMY_NUM_RE = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")
_DATE_RU_RE = re.compile(r"\b(\d{1,2})\s+([А-Яа-яЁё]+)\s+(\d{4})\b")
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_DMS_RE = re.compile(
    r"(\d{1,3})\s*°\s*(\d{1,2})\s*[′']\s*(\d{1,2}(?:\.\d+)?)?\s*[″\"]?\s*([NSEWnsew])"
)
_DECIMAL_PAIR_RE = re.compile(r"(-?\d{1,2}(?:\.\d+))\s*,\s*(-?\d{1,3}(?:\.\d+))")

_tz_finder = None  # lazily-constructed timezonefinder.TimezoneFinder singleton


def warmup() -> None:
    """Eagerly initializes the optional heavy singletons (TimezoneFinder,
    geonamescache's city index, kerykeion/pyswisseph's own internal
    ephemeris loading) instead of leaving them to build lazily on the
    first real astro request. Worth calling once at app startup, measured
    costs: TimezoneFinder's first construction ~18s (bundled timezone-
    boundary data), a first AstrologicalSubject computation ~5s (Swiss
    Ephemeris data), both dropping to near-zero on every call after —
    harmless at startup where the user already expects some load time,
    but surprising if it silently happens during someone's first chat
    request instead, making it look like the app hung. Safe to call even
    when kerykeion/timezonefinder/geonamescache aren't installed — every
    step here already degrades gracefully on its own."""
    try:
        _resolve_timezone(0.0, 0.0)  # coordinates are arbitrary; only the construction cost matters
    except Exception:
        pass
    try:
        _build_city_index()
    except Exception:
        pass
    try:
        from kerykeion import AstrologicalSubject

        AstrologicalSubject(
            name="warmup", year=2000, month=1, day=1, hour=12, minute=0,
            lat=0.0, lng=0.0, tz_str="UTC", online=False,
        )
    except Exception:
        pass


def status() -> dict:
    try:
        import kerykeion  # noqa: F401
    except Exception as e:
        return {"available": False, "error": str(e)}
    try:
        import timezonefinder  # noqa: F401

        tz_auto = True
    except Exception:
        tz_auto = False
    try:
        import geonamescache  # noqa: F401

        city_lookup = True
    except Exception:
        city_lookup = False
    return {"available": True, "auto_timezone": tz_auto, "city_lookup": city_lookup}


def _ru_month_to_num(word: str) -> Optional[int]:
    lowered = word.lower()
    for stem, num in _RU_MONTH_STEMS:
        if lowered.startswith(stem):
            return num
    return None


def _find_date(text: str) -> Optional[str]:
    m = _DATE_ISO_RE.search(text)
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    m = _DATE_DMY_NUM_RE.search(text)
    if m:
        d, mo, y = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    m = _DATE_RU_RE.search(text)
    if m:
        d, month_word, y = m.groups()
        mo = _ru_month_to_num(month_word)
        if mo:
            return f"{int(y):04d}-{mo:02d}-{int(d):02d}"
    return None


def _find_time(text: str) -> Optional[str]:
    m = _TIME_RE.search(text)
    if m:
        hour, minute = m.groups()
        return f"{int(hour):02d}:{int(minute):02d}"
    return None


def _dms_to_decimal(deg: str, minute: str, sec: str, hemisphere: str) -> float:
    value = int(deg) + int(minute) / 60 + (float(sec) if sec else 0.0) / 3600
    if hemisphere.upper() in ("S", "W"):
        value = -value
    return value


def _find_coordinates(text: str) -> Tuple[Optional[float], Optional[float]]:
    lat = lon = None
    for deg, minute, sec, hemisphere in _DMS_RE.findall(text):
        value = _dms_to_decimal(deg, minute, sec, hemisphere)
        if hemisphere.upper() in ("N", "S"):
            lat = value
        else:
            lon = value
    if lat is None and lon is None:
        m = _DECIMAL_PAIR_RE.search(text)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
    return lat, lon


_timezonefinder_import_error: Optional[str] = None  # set on first failed import attempt


def _resolve_timezone(lat: float, lon: float) -> Optional[str]:
    """None means "couldn't resolve" — check _timezonefinder_import_error
    afterward to tell "package not installed" apart from "coordinates
    genuinely didn't resolve to a timezone" (e.g. open ocean), which
    _missing_fields_message uses to give a more actionable explanation."""
    global _tz_finder, _timezonefinder_import_error
    if _tz_finder is None and _timezonefinder_import_error is None:
        try:
            from timezonefinder import TimezoneFinder

            _tz_finder = TimezoneFinder()
        except Exception as e:
            _timezonefinder_import_error = str(e)
    if _tz_finder is None:
        return None
    try:
        return _tz_finder.timezone_at(lat=lat, lng=lon)
    except Exception:
        return None


def _parse_spec(spec: str) -> Dict[str, str]:
    """Strict "key=value;key=value" parsing — the fast path, still accepted
    and preferred whenever present (see _extract_fields)."""
    fields: Dict[str, str] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        fields[key.strip().lower()] = value.strip()
    return fields


_CITY_STEM_LEN = 5
_city_index: Optional[Dict[str, dict]] = None       # lowercased name/alt -> city record
_city_stem_index: Optional[Dict[str, List[dict]]] = None  # first-5-chars -> records, by population desc


def _build_city_index() -> None:
    """Lazy, built once per process (~2s the first time, then cached).
    geonamescache bundles its own data (no download, no file to place
    anywhere) — ~34k cities worldwide with population above a threshold,
    each with lat/lon, country, timezone, and a list of alternate-language
    names (Cyrillic included for most sizeable cities)."""
    global _city_index, _city_stem_index
    if _city_index is not None:
        return
    try:
        import geonamescache
    except Exception:
        _city_index, _city_stem_index = {}, {}
        return

    index: Dict[str, dict] = {}
    stems: Dict[str, List[dict]] = {}

    def _consider(name: str, record: dict) -> None:
        key = name.strip().lower()
        if not key:
            return
        existing = index.get(key)
        if existing is None or record["population"] > existing["population"]:
            index[key] = record
        stems.setdefault(key[:_CITY_STEM_LEN], []).append(record)

    for record in geonamescache.GeonamesCache().get_cities().values():
        _consider(record["name"], record)
        for alt in record.get("alternatenames", []):
            _consider(alt, record)
    for bucket in stems.values():
        bucket.sort(key=lambda r: r["population"], reverse=True)

    _city_index, _city_stem_index = index, stems


def _lookup_city(text: str) -> Optional[dict]:
    """Best-effort offline city lookup, used as a fallback only when no
    explicit coordinates were found anywhere in the spec text. Tries an
    exact (casefolded) match on each word or adjacent word-pair first (for
    multi-word names like "Санкт-Петербург"); if nothing matches exactly,
    falls back to comparing the first few characters ("stem"), since a
    Russian city name in a birth-info sentence is usually in some
    grammatical case ("в Одессе") rather than the gazetteer's nominative
    form ("Одесса") — real morphological analysis would need a dedicated
    library (e.g. pymorphy2), this is a cheap approximation that works for
    most city names since the stem — everything but the last 1-3 letters —
    doesn't change across cases. Name collisions (e.g. "Odessa" exists in
    both Ukraine and Texas) are resolved by picking the more populous
    match, which is usually but not always the intended one — a known,
    accepted limitation rather than something worth a disambiguation
    prompt for this use case. This also matters within one sentence: an
    ordinary word occasionally coincides with an obscure place's alternate
    name somewhere in the world (Russian "года", "of the year", turned out
    to also be a transliterated alternate name for a small Japanese town —
    found by testing this before shipping it) — so every word/word-pair is
    checked and the most populous match across ALL of them wins, not just
    whichever is found first in text order, since a small town matching a
    filler word essentially never beats the sentence's actual, real,
    usually far more populous, named city.
    """
    _build_city_index()
    if not _city_index:
        return None

    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    candidates = words + [f"{a} {b}" for a, b in zip(words, words[1:])]

    exact_matches = [
        record for record in (_city_index.get(c.lower()) for c in candidates) if record
    ]
    if exact_matches:
        return max(exact_matches, key=lambda r: r["population"])

    stem_matches: List[dict] = []
    for candidate in candidates:
        key = candidate.lower()
        if len(key) >= _CITY_STEM_LEN:
            stem_matches.extend(_city_stem_index.get(key[:_CITY_STEM_LEN], []))
    if stem_matches:
        return max(stem_matches, key=lambda r: r["population"])
    return None


def _extract_fields(spec: str) -> Tuple[Dict[str, str], List[str]]:
    """Combines the strict key=value fast path with free-text extraction
    for anything still missing, then auto-resolves tz from lat/lon if it
    wasn't given explicitly. Returns (fields, missing_field_names).

    Coordinate resolution order: explicit lat/lon (key=value) > DMS/decimal
    coordinates found in the text > a bare city name looked up via
    geonamescache (only reached if the first two found nothing) — each
    step only runs if the previous one came up empty."""
    fields = _parse_spec(spec)

    if not fields.get("date"):
        found = _find_date(spec)
        if found:
            fields["date"] = found
    if not fields.get("time"):
        found = _find_time(spec)
        if found:
            fields["time"] = found
    if not fields.get("lat") or not fields.get("lon"):
        lat, lon = _find_coordinates(spec)
        if lat is not None:
            fields.setdefault("lat", str(lat))
        if lon is not None:
            fields.setdefault("lon", str(lon))
    if not fields.get("lat") or not fields.get("lon"):
        city = _lookup_city(spec)
        if city:
            fields.setdefault("lat", str(city["latitude"]))
            fields.setdefault("lon", str(city["longitude"]))
            fields.setdefault("tz", city["timezone"])
    if not fields.get("tz") and fields.get("lat") and fields.get("lon"):
        try:
            tz = _resolve_timezone(float(fields["lat"]), float(fields["lon"]))
        except Exception:
            tz = None
        if tz:
            fields["tz"] = tz

    missing = [k for k in _REQUIRED_FIELDS if not fields.get(k)]
    return fields, missing


def _missing_fields_message(missing: List[str], fields: Dict[str, str]) -> str:
    # Coordinates present but only tz missing means auto-resolution itself
    # failed — a different, more actionable problem than "nothing was
    # given at all", worth telling apart so it's diagnosable without
    # reading server logs.
    if missing == ["tz"] and fields.get("lat") and fields.get("lon"):
        if _timezonefinder_import_error is not None:
            return (
                "Дата, время и координаты рождения есть, но автоматическое "
                "определение часового пояса недоступно на сервере (пакет "
                "timezonefinder не установлен — pip install timezonefinder). "
                "Можно указать часовой пояс вручную (например Europe/Kyiv)."
            )
        return (
            "Дата, время и координаты рождения есть, но по этим координатам "
            "не удалось определить часовой пояс автоматически. Уточните, "
            "пожалуйста, часовой пояс места рождения (например Europe/Kyiv)."
        )
    return (
        "Не хватает данных для расчёта: " + ", ".join(missing) + ". "
        "Нужны точная дата и время рождения и место — координаты (широта и "
        "долгота, в любом виде — например 46.4667, 30.7333 или "
        "46°28'00\"N 30°44'00\"E). Часовой пояс определяется автоматически "
        "по координатам, указывать его отдельно не обязательно."
    )


def _house_number(house_name: Optional[str]) -> Optional[int]:
    if house_name in _HOUSE_ORDER:
        return _HOUSE_ORDER.index(house_name) + 1
    return None


def _format_point_line(label: str, point) -> str:
    sign = _SIGN_NAMES_RU.get(point.sign, point.sign)
    house_num = _house_number(getattr(point, "house", None))
    house_part = f", дом {house_num}" if house_num else ""
    retro = " (ретроградный)" if getattr(point, "retrograde", False) else ""
    return f"{label}: {sign} {point.position:.1f}°{house_part}{retro}"


def _build_subject(fields: Dict[str, str], name: str):
    from kerykeion import AstrologicalSubject

    date_str = fields["date"]
    time_str = fields.get("time", "12:00")
    year, month, day = (int(x) for x in date_str.split("-"))
    hour, minute = (int(x) for x in time_str.split(":"))
    return AstrologicalSubject(
        name=name,
        year=year, month=month, day=day, hour=hour, minute=minute,
        lat=float(fields["lat"]), lng=float(fields["lon"]), tz_str=fields["tz"],
        city=fields.get("city") or None, nation=fields.get("nation") or None,
        online=False,
    )


def _format_natal_text(subject) -> str:
    from kerykeion import AspectsFactory

    m = subject.model()
    lines = [
        f"Натальная карта для {subject.name} "
        f"({m.year:04d}-{m.month:02d}-{m.day:02d} {m.hour:02d}:{m.minute:02d}, "
        f"{m.tz_str}).",
        "Планеты:",
    ]
    for label, attr in _PLANET_ATTRS:
        lines.append("  " + _format_point_line(label, getattr(subject, attr)))
    lines.append("Углы:")
    for label, attr in _ANGLE_ATTRS:
        lines.append("  " + _format_point_line(label, getattr(subject, attr)))
    lines.append("Куспиды домов:")
    for i, attr in enumerate(
        ["first_house", "second_house", "third_house", "fourth_house",
         "fifth_house", "sixth_house", "seventh_house", "eighth_house",
         "ninth_house", "tenth_house", "eleventh_house", "twelfth_house"],
        start=1,
    ):
        cusp = getattr(subject, attr)
        sign = _SIGN_NAMES_RU.get(cusp.sign, cusp.sign)
        lines.append(f"  Дом {i}: {sign} {cusp.position:.1f}°")

    aspects = AspectsFactory.natal_aspects(
        m, active_points=_ASPECT_ACTIVE_POINTS, active_aspects=_MAJOR_ASPECTS,
    ).aspects
    lines.append("Аспекты:")
    if aspects:
        for a in aspects:
            lines.append(
                f"  {_point_ru(a.p1_name)} — {_aspect_ru(a.aspect)} — {_point_ru(a.p2_name)} "
                f"(орбис {a.orbit:.1f}°, {_movement_ru(a.aspect_movement)})"
            )
    else:
        lines.append("  (нет аспектов в пределах орбиса)")
    return "\n".join(lines)


def _format_transit_text(natal, transit) -> str:
    from kerykeion import AspectsFactory

    tm = transit.model()
    lines = [
        f"Положения планет на {tm.year:04d}-{tm.month:02d}-{tm.day:02d} "
        f"{tm.hour:02d}:{tm.minute:02d} ({tm.tz_str}), в сравнении с натальной "
        f"картой {natal.name}.",
        "Текущие положения планет:",
    ]
    for label, attr in _PLANET_ATTRS:
        lines.append("  " + _format_point_line(label, getattr(transit, attr)))

    aspects = AspectsFactory.dual_chart_aspects(
        natal.model(), tm, active_points=_ASPECT_ACTIVE_POINTS, active_aspects=_MAJOR_ASPECTS,
    ).aspects
    lines.append("Транзитные аспекты к натальной карте:")
    shown = 0
    for a in aspects:
        transiting, to_natal = (
            (a.p1_name, a.p2_name) if a.p1_owner == transit.name else (a.p2_name, a.p1_name)
        )
        # Ascendant/Medium_Coeli as the *transiting* point just reflect the
        # moment's time-of-day (they shift ~1°/4min from Earth's rotation),
        # not real planetary movement — not meaningful transits, so they're
        # excluded here. As the *natal* (to_natal) side they're kept: a real
        # transiting planet crossing your natal Ascendant/MC is a standard,
        # meaningful reading.
        if transiting in ("Ascendant", "Medium_Coeli"):
            continue
        lines.append(
            f"  транзитный {_point_ru(transiting)} — {_aspect_ru(a.aspect)} — "
            f"натальный {_point_ru(to_natal)} "
            f"(орбис {a.orbit:.1f}°, {_movement_ru(a.aspect_movement)})"
        )
        shown += 1
    if not shown:
        lines.append("  (нет аспектов в пределах орбиса)")
    return "\n".join(lines)


# Every non-chart string run_natal/run_transit can return starts with one
# of these — used by routes/chat.py to skip the RAG-augmented reasoning
# prompt when there's no actual chart data to interpret yet (just a
# request for missing info, or a computation error). Retrieving and
# injecting the full methodology/context for a placeholder message like
# "не хватает данных" is pure wasted context-window budget on top of
# being pointless — there's nothing to reason about yet.
_ERROR_RESULT_PREFIXES = ("Не хватает данных", "Не удалось", "Ошибка")


def is_error_result(tool_result: str) -> bool:
    return tool_result.startswith(_ERROR_RESULT_PREFIXES)


def run_natal(spec: str) -> str:
    """Tool entry point (utils.tools.TOOL_REGISTRY["astro_natal_chart"]).
    Never raises — any failure becomes a plain-text explanation instead,
    since the caller feeds the return value straight into a follow-up
    generation, not into error-handling code."""
    fields, missing = _extract_fields(spec)
    if missing:
        return _missing_fields_message(missing, fields)
    try:
        subject = _build_subject(fields, name=fields.get("name") or "Subject")
    except Exception as e:
        return f"Не удалось построить натальную карту — некорректные данные ({e})."
    try:
        return _format_natal_text(subject)
    except Exception as e:
        return f"Ошибка при расчёте натальной карты: {e}"


def run_transit(spec: str) -> str:
    """Tool entry point (utils.tools.TOOL_REGISTRY["astro_transit_chart"])."""
    fields, missing = _extract_fields(spec)
    if missing:
        return _missing_fields_message(missing, fields)
    try:
        natal = _build_subject(fields, name=fields.get("name") or "Subject")
    except Exception as e:
        return f"Не удалось построить натальную карту — некорректные данные ({e})."

    moment = (fields.get("moment") or "now").strip()
    try:
        if moment.lower() in ("", "now", "сейчас"):
            dt = datetime.now()
        else:
            dt = datetime.strptime(moment, "%Y-%m-%dT%H:%M")
    except Exception as e:
        return (
            f"Не удалось разобрать момент времени '{moment}' ({e}); "
            "ожидается формат ГГГГ-ММ-ДДTЧЧ:ММ или 'now'."
        )

    moment_lat_str = fields.get("moment_lat")
    moment_lon_str = fields.get("moment_lon")
    moment_lat = float(moment_lat_str) if moment_lat_str else float(fields["lat"])
    moment_lon = float(moment_lon_str) if moment_lon_str else float(fields["lon"])
    moment_tz = fields.get("moment_tz")
    if not moment_tz:
        # Same place as birth -> reuse the already-resolved birth tz. A
        # different place -> re-resolve, since tz depends on location.
        moment_tz = fields["tz"] if not (moment_lat_str or moment_lon_str) else (
            _resolve_timezone(moment_lat, moment_lon) or fields["tz"]
        )

    try:
        from kerykeion import AstrologicalSubject

        transit_subject = AstrologicalSubject(
            name="Transit",
            year=dt.year, month=dt.month, day=dt.day, hour=dt.hour, minute=dt.minute,
            lat=moment_lat, lng=moment_lon, tz_str=moment_tz,
            online=False,
        )
    except Exception as e:
        return f"Не удалось рассчитать текущие положения планет: {e}"

    try:
        return _format_transit_text(natal, transit_subject)
    except Exception as e:
        return f"Ошибка при расчёте транзитов: {e}"


# Registry for future operations (synastry, composite, rectification,
# electional search, ...) — see module docstring. Not yet consumed by
# anything (utils/tools.py wires run_natal/run_transit directly, since
# there are only two so far), but kept as the intended extension point so
# adding a third operation doesn't require inventing a new pattern.
ASTRO_OPERATIONS: Dict[str, Callable[[str], str]] = {
    "natal": run_natal,
    "transit": run_transit,
}
