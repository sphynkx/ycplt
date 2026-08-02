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
from typing import Any, Callable, Dict, List, Optional, Tuple

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

# Prepositional case ("в Раке", not "в Рак") — needed anywhere a sign name
# follows "в" as a real grammatical phrase (get_planet_profiles' RAG
# queries and fact descriptions) rather than standing alone as a label
# (_format_point_line's "Рак 13.2°" doesn't need this). Getting this wrong
# isn't just clumsy Russian — reference material is normally titled/phrased
# in this case ("Солнце в Раке"), so using the nominative form here would
# also weaken the targeted RAG queries' match against it.
_SIGN_NAMES_RU_PREPOSITIONAL = {
    "Овен": "Овне", "Телец": "Тельце", "Близнецы": "Близнецах", "Рак": "Раке",
    "Лев": "Льве", "Дева": "Деве", "Весы": "Весах", "Скорпион": "Скорпионе",
    "Стрелец": "Стрельце", "Козерог": "Козероге", "Водолей": "Водолее", "Рыбы": "Рыбах",
}
# Which preposition ("в" or "во") precedes each sign's prepositional form.
# Plain "в" works before almost every consonant, but Russian swaps to "во"
# before certain consonant clusters that are awkward to pronounce after "в"
# — "Льве" (Leo) is the one case among the 12 signs this actually affects
# ("во Льве", never "в Льве"; confirmed a real bug — a real answer used "в
# Леве", not even the right consonant cluster). Defaults to "в" for every
# sign not listed here.
_SIGN_PREPOSITION = {"Лев": "во"}

_POINT_NAMES_RU = {
    "Sun": "Солнце", "Moon": "Луна", "Mercury": "Меркурий", "Venus": "Венера",
    "Mars": "Марс", "Jupiter": "Юпитер", "Saturn": "Сатурн", "Uranus": "Уран",
    "Neptune": "Нептун", "Pluto": "Плутон",
    "True_North_Lunar_Node": "Северный узел", "True_South_Lunar_Node": "Южный узел",
    "Chiron": "Хирон", "True_Lilith": "Лилит", "Pars_Fortunae": "Парс Фортуны",
    "Vertex": "Вертекс",
    "Ascendant": "Асцендент", "Medium_Coeli": "Середина неба (MC)",
}

_ASPECT_NAMES_RU = {
    "conjunction": "соединение", "opposition": "оппозиция", "trine": "трин",
    "square": "квадрат", "sextile": "секстиль",
    # Minor aspects — see _MINOR_ASPECTS below for why these are included
    # with much tighter orbs than the majors above.
    "semi-sextile": "полусекстиль", "semi-square": "полуквадрат",
    "quintile": "квинтиль", "sesquiquadrate": "полутораквадрат",
    "biquintile": "биквинтиль", "quincunx": "квинконс",
}

_ASPECT_MOVEMENT_RU = {
    "Applying": "сходящийся, усиливается",
    "Separating": "расходящийся, ослабевает",
    "Static": "статичный",
}

# Unicode symbols, embedded directly in the chart text / fact descriptions
# this module produces — NOT left as a "please use these symbols"
# instruction for the interpreting model to follow. That was tried first
# (a table in interpretation_methodology.txt) and the model simply ignored
# it even when the table was confirmed to be in context — a small model
# reliably PRESERVING a symbol that's already right there in the data it's
# copying from is a much easier task than reliably RECALLING and
# generating the correct one from scratch. Points/signs/aspects without a
# genuinely standard, widely-recognized glyph (Parts, Vertex, most
# non-classical points) deliberately have no entry here rather than an
# invented one — same principle the methodology document states for the
# model's own output.
_SIGN_SYMBOLS = {
    "Овен": "♈", "Телец": "♉", "Близнецы": "♊", "Рак": "♋", "Лев": "♌", "Дева": "♍",
    "Весы": "♎", "Скорпион": "♏", "Стрелец": "♐", "Козерог": "♑", "Водолей": "♒", "Рыбы": "♓",
}
_POINT_SYMBOLS = {
    "Солнце": "☉", "Луна": "☽", "Меркурий": "☿", "Венера": "♀", "Марс": "♂",
    "Юпитер": "♃", "Сатурн": "♄", "Уран": "♅", "Нептун": "♆", "Плутон": "♇",
    "Северный узел": "☊", "Южный узел": "☋", "Хирон": "⚷", "Лилит": "⚸",
}
_ASPECT_SYMBOLS = {
    "соединение": "☌", "оппозиция": "☍", "трин": "△", "квадрат": "□", "секстиль": "⚹",
    # Semisextile/quincunx/sesquiquadrate have standard, widely-recognized
    # Unicode glyphs (U+26BA/26BB/26BC, the same "Miscellaneous Symbols"
    # block sextile's ⚹ comes from). Semi-square, quintile, and biquintile
    # have no comparably standard single-glyph symbol — left out rather
    # than invented, same principle as elsewhere in this table.
    "полусекстиль": "⚺", "полутораквадрат": "⚼", "квинконс": "⚻",
}


def _with_symbol(name_ru: str, symbols: Dict[str, str]) -> str:
    """Appends " <symbol>" after a Russian name if one exists for it,
    otherwise returns the name unchanged — never invents a symbol for
    something not in the table above."""
    symbol = symbols.get(name_ru)
    return f"{name_ru} {symbol}" if symbol else name_ru


def _point_ru(name: str) -> str:
    return _with_symbol(_POINT_NAMES_RU.get(name, name), _POINT_SYMBOLS)


def _aspect_ru(name: str) -> str:
    return _with_symbol(_ASPECT_NAMES_RU.get(name, name), _ASPECT_SYMBOLS)


def _movement_ru(name: str) -> str:
    return _ASPECT_MOVEMENT_RU.get(name, name)


def _sign_ru(sign_code: str) -> str:
    """Nominative Russian sign name with its symbol, e.g. "Рак ♋"."""
    name = _SIGN_NAMES_RU.get(sign_code, sign_code)
    return _with_symbol(name, _SIGN_SYMBOLS)


def _sign_ru_prepositional(sign_code: str) -> str:
    """Full prepositional phrase with symbol, e.g. "в Раке ♋" or "во Льве
    ♌" — includes the preposition itself (see _SIGN_PREPOSITION), so
    callers should use this as a complete phrase, not prepend their own
    literal "в " in front of it (that was the bug: a hardcoded "в" at each
    call site could never produce "во Льве")."""
    name = _SIGN_NAMES_RU.get(sign_code, sign_code)
    prep_word = _SIGN_NAMES_RU_PREPOSITIONAL.get(name, name)
    preposition = _SIGN_PREPOSITION.get(name, "в")
    return f"{preposition} {_with_symbol(prep_word, _SIGN_SYMBOLS)}"


_HOUSE_ORDER = [
    "First_House", "Second_House", "Third_House", "Fourth_House",
    "Fifth_House", "Sixth_House", "Seventh_House", "Eighth_House",
    "Ninth_House", "Tenth_House", "Eleventh_House", "Twelfth_House",
]

# (Russian display label, AstrologicalSubjectModel attribute name). Chiron,
# Lilith, Part of Fortune and the Vertex were added after real testing
# showed the reference corpus (fixed stars, "fictitious"/hypothetical
# points, Arabic Parts) existed but nothing in this module ever surfaced
# facts about them — these four are the commonly-used-in-mainstream-
# practice subset (out of kerykeion's much larger catalogue of dwarf
# planets/parts/stars) added as regular "planet-like" points with their
# own sign/house. Fixed stars are handled separately below
# (_find_star_conjunctions) since they're conventionally read only via
# tight conjunctions to a personal point, not as their own sign/house
# placement the way a planet is.
_PLANET_ATTRS = [
    ("Солнце", "sun"), ("Луна", "moon"), ("Меркурий", "mercury"), ("Венера", "venus"),
    ("Марс", "mars"), ("Юпитер", "jupiter"), ("Сатурн", "saturn"),
    ("Уран", "uranus"), ("Нептун", "neptune"), ("Плутон", "pluto"),
    ("Северный узел", "true_north_lunar_node"), ("Южный узел", "true_south_lunar_node"),
    ("Хирон", "chiron"), ("Лилит", "true_lilith"), ("Парс Фортуны", "pars_fortunae"),
    ("Вертекс", "vertex"),
]
_ANGLE_ATTRS = [("Асцендент", "ascendant"), ("Середина неба (MC)", "medium_coeli")]

# The extended set of points AstrologicalSubjectFactory.from_birth_data
# should actually compute (see _build_subject) — kerykeion's own
# AstrologicalSubject wrapper class is deprecated and hardcodes a fixed 18-
# point DEFAULT_ACTIVE_POINTS list with no way to extend it (confirmed by
# reading kerykeion's own source: backword.py's compat class doesn't
# accept an active_points argument at all), which is why True_Lilith,
# Pars_Fortunae, Vertex, and every fixed star used to silently come back
# as None regardless of what this module asked for — not a missing
# dependency, a genuinely different, non-deprecated entry point
# (AstrologicalSubjectFactory.from_birth_data) was required.
_FIXED_STARS = ["Regulus", "Aldebaran", "Antares", "Fomalhaut", "Spica", "Algol"]
_FIXED_STAR_NAMES_RU = {
    "Regulus": "Регул", "Aldebaran": "Альдебаран", "Antares": "Антарес",
    "Fomalhaut": "Фомальгаут", "Spica": "Спика", "Algol": "Алголь",
}
_ACTIVE_POINTS_NATAL = [
    "Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus",
    "Neptune", "Pluto", "True_North_Lunar_Node", "True_South_Lunar_Node",
    "Chiron", "True_Lilith", "Pars_Fortunae", "Vertex",
    "Ascendant", "Medium_Coeli", "Descendant", "Imum_Coeli",
] + _FIXED_STARS

# For the *transiting* moment's own subject in run_transit — deliberately
# narrower than _ACTIVE_POINTS_NATAL: fixed stars don't move, so nothing
# meaningful "transits" them, and the Vertex is itself derived from the
# moment/location rather than being a body in motion, so a "transiting
# Vertex" isn't a standard reading either. Chiron/Lilith/Part of Fortune
# ARE conventionally read as transiting bodies, so those stay.
_ACTIVE_POINTS_TRANSIT = [
    p for p in _ACTIVE_POINTS_NATAL if p not in _FIXED_STARS and p != "Vertex"
]

# kerykeion's own point-name literals, used for AspectsFactory filtering —
# restricting to this set keeps the aspect list relevant. Fixed stars are
# deliberately excluded here (see _find_star_conjunctions): the standard
# aspect/orb table below is right for planets and points but far too
# generous for a star, which is conventionally only significant on a tight
# conjunction (~1°), not a wide trine or square.
_ASPECT_ACTIVE_POINTS = [
    "Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn",
    "Uranus", "Neptune", "Pluto", "True_North_Lunar_Node", "Chiron",
    "True_Lilith", "Pars_Fortunae",
    "Ascendant", "Medium_Coeli",
]
_MAJOR_ASPECTS = [
    {"name": "conjunction", "orb": 8},
    {"name": "opposition", "orb": 8},
    {"name": "trine", "orb": 7},
    {"name": "square", "orb": 7},
    {"name": "sextile", "orb": 5},
]
# Minor aspects: present in the indexed reference corpus (per the user's
# real testing) but never computed at all before, so that material was
# unreachable no matter how good the corpus was — the same class of gap
# houses/stars/Parts had before tasks #101-103. Orbs are kept much tighter
# than the majors' above (2-3° vs 5-8°) per standard astrological
# convention: minor aspects are considered meaningful only close to exact,
# unlike majors which stay significant across a wider separation. Quincunx
# gets a slightly wider orb than the other minors — conventionally treated
# as the most significant of this group. kerykeion itself only enables one
# minor aspect (quintile, 1° orb) by default; the rest were never
# reachable via this app at all before this list existed.
_MINOR_ASPECTS = [
    {"name": "semi-sextile", "orb": 2},
    {"name": "semi-square", "orb": 2},
    {"name": "quintile", "orb": 2},
    {"name": "sesquiquadrate", "orb": 2},
    {"name": "biquintile", "orb": 2},
    {"name": "quincunx", "orb": 3},
]
# Combined list passed to AspectsFactory everywhere aspects are computed
# (natal, transit, significant-fact scoring) — kerykeion takes one
# active_aspects list, not separate major/minor calls. Kept as two named
# lists above (rather than one flat one) purely so the rationale for each
# tier's orbs stays attached to it in the source, not because anything
# downstream treats major/minor differently — scoring in
# get_planet_profiles is purely by orb tightness regardless of aspect
# type, per the methodology's "точность важнее типа" rule, so a very tight
# minor aspect can legitimately outrank a loose major one.
_ALL_ASPECTS = _MAJOR_ASPECTS + _MINOR_ASPECTS
# Fixed stars barely move, so what matters is a personal point sitting
# right on top of one — kept tight and separate from _ALL_ASPECTS' orbs.
_STAR_CONJUNCTION_ORB = 1.5

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
        from kerykeion.astrological_subject_factory import AstrologicalSubjectFactory

        # Warms the richer natal active_points set (fixed stars, Lilith,
        # Part of Fortune, Vertex, ...), not just the classical 10 planets
        # — this is the actual code path run_natal/get_planet_profiles
        # exercise per request, and the whole point of warmup() is to move
        # first-use cost here instead of onto a user's request.
        AstrologicalSubjectFactory.from_birth_data(
            name="warmup", year=2000, month=1, day=1, hour=12, minute=0,
            lat=0.0, lng=0.0, tz_str="UTC", online=False,
            active_points=_ACTIVE_POINTS_NATAL,
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


def _format_point_line(label: str, point, house_num_override: Optional[int] = None) -> str:
    """house_num_override lets a caller supply a house number computed
    against a DIFFERENT chart's cusps than `point`'s own (see
    _house_of_degree) — used by _format_transit_text so a transiting
    planet's house is read against the natal chart, not the transit
    moment's own independently-computed houses."""
    sign = _sign_ru(point.sign)
    house_num = (
        house_num_override if house_num_override is not None
        else _house_number(getattr(point, "house", None))
    )
    house_part = f", дом {house_num}" if house_num else ""
    retro = " (ретроградный)" if getattr(point, "retrograde", False) else ""
    return f"{label}: {sign} {point.position:.1f}°{house_part}{retro}"


def _house_cusp_degrees(subject) -> List[float]:
    """Absolute ecliptic degree of each of a chart's 12 house cusps, in
    house order (index 0 = house 1's cusp) — the input _house_of_degree
    needs to place an arbitrary point (e.g. a transiting planet) into
    THIS chart's houses, regardless of which chart the point itself came
    from."""
    attrs = [
        "first_house", "second_house", "third_house", "fourth_house",
        "fifth_house", "sixth_house", "seventh_house", "eighth_house",
        "ninth_house", "tenth_house", "eleventh_house", "twelfth_house",
    ]
    return [getattr(subject, attr).position for attr in attrs]


def _house_of_degree(cusp_degrees: List[float], degree: float) -> int:
    """Which house (1-12) an arbitrary absolute ecliptic degree falls
    into, given a chart's 12 house cusp degrees (house order, index 0 =
    house 1). This is the standard transit-astrology convention for
    reading a transiting planet's house — compare its own degree against
    the NATAL chart's cusps — confirmed with the user as the intended
    behavior over the alternative (and, until this function existed, the
    actual behavior): computing an entirely separate house system for the
    transit moment's own time/location via a second AstrologicalSubject
    build, which is not how transits are conventionally read (a transit
    reading is about which of YOUR houses a planet is currently moving
    through, not what house it would occupy in a chart cast for that
    moment at your birthplace).

    Houses aren't evenly spaced (real house systems, Placidus included,
    have unequal arcs), so this can't just divide by 30° — each house's
    actual arc (cusp i to the next cusp, wrapping past 360° when it
    crosses 0° Aries) has to be checked directly."""
    n = len(cusp_degrees)
    d = degree % 360.0
    for i in range(n):
        start = cusp_degrees[i] % 360.0
        end = cusp_degrees[(i + 1) % n] % 360.0
        in_arc = (start <= d < end) if start <= end else (d >= start or d < end)
        if in_arc:
            return i + 1
    return 12  # unreachable in practice (the loop above is exhaustive) — safe fallback


def _build_subject(fields: Dict[str, str], name: str, active_points: Optional[List[str]] = None):
    """Builds a chart via AstrologicalSubjectFactory.from_birth_data —
    NOT kerykeion's own `AstrologicalSubject` class, which is a deprecated
    backward-compat wrapper (confirmed by reading kerykeion's source,
    backword.py) that hardcodes an 18-point DEFAULT_ACTIVE_POINTS list
    with no way to ask for anything beyond it. That's why Lilith (the
    "true" variant), the Part of Fortune, the Vertex, and every fixed star
    used to silently come back as None regardless of what this module
    tried to do with them — not a missing dependency or a bug in this
    file's own translation tables, a fundamentally different constructor
    was required to get them computed at all.

    active_points defaults to _ACTIVE_POINTS_NATAL (the rich set used for
    a person's own birth chart); run_transit passes _ACTIVE_POINTS_TRANSIT
    for the moving/transiting side instead."""
    from kerykeion.astrological_subject_factory import AstrologicalSubjectFactory

    date_str = fields["date"]
    time_str = fields.get("time", "12:00")
    year, month, day = (int(x) for x in date_str.split("-"))
    hour, minute = (int(x) for x in time_str.split(":"))
    return AstrologicalSubjectFactory.from_birth_data(
        name=name,
        year=year, month=month, day=day, hour=hour, minute=minute,
        lat=float(fields["lat"]), lng=float(fields["lon"]), tz_str=fields["tz"],
        city=fields.get("city") or None, nation=fields.get("nation") or None,
        online=False,
        active_points=active_points if active_points is not None else _ACTIVE_POINTS_NATAL,
    )


def _find_star_conjunctions(subject) -> List[Dict]:
    """Fixed stars are conventionally read only via a tight conjunction to
    a personal point (typically ~1° orb), not as their own sign/house
    placement the way a planet is read, and not via the standard 5-aspect
    table (a fixed star "trine" is essentially meaningless — the star
    hasn't moved, only the natal point's own degree happens to be some
    angle away). This checks each of _FIXED_STARS against each classical
    point/angle for a conjunction within _STAR_CONJUNCTION_ORB, entirely
    separately from AspectsFactory.natal_aspects."""
    classical = [attr for _, attr in _PLANET_ATTRS if attr not in ("chiron", "true_lilith", "pars_fortunae", "vertex")]
    classical += [attr for _, attr in _ANGLE_ATTRS]

    facts: List[Dict] = []
    for star_en in _FIXED_STARS:
        star_point = getattr(subject, star_en.lower(), None)
        if star_point is None:
            continue
        star_ru = _FIXED_STAR_NAMES_RU.get(star_en, star_en)
        for attr in classical:
            point = getattr(subject, attr, None)
            if point is None:
                continue
            separation = abs(point.position - star_point.position)
            separation = min(separation, 360.0 - separation)
            if separation <= _STAR_CONJUNCTION_ORB:
                label = next(l for l, a in (_PLANET_ATTRS + _ANGLE_ATTRS) if a == attr)
                point_ru = _point_ru_from_label(label)
                facts.append(
                    {
                        "kind": "star",
                        "text": f"{point_ru} — соединение со звездой {star_ru} (орбис {separation:.1f}°)",
                        "queries": [f"{point_ru} соединение звезда {star_ru}", f"звезда {star_ru}"],
                        "score": max(0.0, 3.0 - separation),
                    }
                )
    return facts


def _point_ru_from_label(label: str) -> str:
    """_PLANET_ATTRS/_ANGLE_ATTRS labels are already Russian names without
    a symbol (e.g. "Солнце", not "Sun") — this adds the symbol the same
    way _point_ru does for kerykeion's own English literals, without
    needing a reverse lookup back to English first."""
    return _with_symbol(label, _POINT_SYMBOLS)


# Genitive case ("трин Солнца и Луны", not "трин Солнце и Луна") — needed
# for get_planet_profiles' pre-formatted aspect phrases (see "phrase" below).
# This exists because of a real, reproducible grammar failure: the digest/
# final-answer prompts used to just say "<аспект> к <planet>", where
# "к" grammatically demands the DATIVE case ("к Венере"), but the point
# labels everywhere else in this module are nominative ("Венера") — asking
# the model to paraphrase around that mismatch on the fly produced garbled
# results in a real answer ("Лунный аспект с Венерой", "Марский аспект",
# "Юпитерианский аспект" — invented, sometimes ungrammatical adjectives
# standing in for a case the model couldn't confidently produce). Spelling
# out the full genitive phrase here in code, the same way sign names got a
# prepositional-case table, removes the ambiguity instead of hoping the
# model resolves it correctly under time pressure.
_POINT_NAMES_RU_GENITIVE = {
    "Солнце": "Солнца", "Луна": "Луны", "Меркурий": "Меркурия", "Венера": "Венеры",
    "Марс": "Марса", "Юпитер": "Юпитера", "Сатурн": "Сатурна", "Уран": "Урана",
    "Нептун": "Нептуна", "Плутон": "Плутона",
    "Северный узел": "Северного узла", "Южный узел": "Южного узла",
    "Хирон": "Хирона", "Лилит": "Лилит",  # "Лилит" is indeclinable as a name
    "Парс Фортуны": "Парса Фортуны", "Вертекс": "Вертекса",
    "Асцендент": "Асцендента", "Середина неба (MC)": "Середины неба (MC)",
}


def _point_ru_genitive_from_label(label: str) -> str:
    """Genitive counterpart to _point_ru_from_label — same plain-Russian-
    label input (e.g. "Венера"), symbol appended the same way, but the
    word itself declined ("Венеры ♀")."""
    genitive = _POINT_NAMES_RU_GENITIVE.get(label, label)
    return _with_symbol(genitive, _POINT_SYMBOLS)


def _format_natal_text(subject) -> str:
    from kerykeion import AspectsFactory

    lines = [
        f"Натальная карта для {subject.name} "
        f"({subject.year:04d}-{subject.month:02d}-{subject.day:02d} {subject.hour:02d}:{subject.minute:02d}, "
        f"{subject.tz_str}).",
        "Планеты и точки:",
    ]
    for label, attr in _PLANET_ATTRS:
        point = getattr(subject, attr, None)
        if point is not None:
            lines.append("  " + _format_point_line(label, point))
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
        lines.append(f"  Дом {i}: {_sign_ru(cusp.sign)} {cusp.position:.1f}°")

    aspects = AspectsFactory.natal_aspects(
        subject, active_points=_ASPECT_ACTIVE_POINTS, active_aspects=_ALL_ASPECTS,
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

    star_facts = _find_star_conjunctions(subject)
    if star_facts:
        lines.append("Соединения с неподвижными звёздами:")
        for f in star_facts:
            lines.append("  " + f["text"])

    return "\n".join(lines)


def _format_transit_text(natal, transit) -> str:
    from kerykeion import AspectsFactory

    # Each transiting planet's house is read against the NATAL chart's own
    # cusps (_house_of_degree), not `transit`'s own house attribute — see
    # that function's docstring for why. `transit` (built via
    # _ACTIVE_POINTS_TRANSIT) still supplies each planet's current
    # sign/degree/retrograde state; only its OWN house computation is
    # discarded here.
    natal_cusps = _house_cusp_degrees(natal)

    lines = [
        f"Положения планет на {transit.year:04d}-{transit.month:02d}-{transit.day:02d} "
        f"{transit.hour:02d}:{transit.minute:02d} ({transit.tz_str}), в сравнении с натальной "
        f"картой {natal.name}.",
        "Текущие положения планет (дом — натальный, т.е. дом натальной карты, "
        "через который сейчас проходит планета):",
    ]
    for label, attr in _PLANET_ATTRS:
        point = getattr(transit, attr, None)
        if point is not None:
            natal_house = _house_of_degree(natal_cusps, point.position)
            lines.append("  " + _format_point_line(label, point, house_num_override=natal_house))

    aspects = AspectsFactory.dual_chart_aspects(
        natal, transit, active_points=_ASPECT_ACTIVE_POINTS, active_aspects=_ALL_ASPECTS,
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


_ANGULAR_HOUSES = {1, 4, 7, 10}
_SUCCEDENT_HOUSES = {2, 5, 8, 11}

# Cap on how many of a profile's own aspects get their own targeted RAG
# query + get spelled out in the digest prompt — without this, a
# heavily-aspected planet (Sun/Moon/angles routinely have 5-8 aspects in
# _ALL_ASPECTS' now-wider net including minors) would balloon both the
# number of retrieval calls per answer and the digest prompt's length.
# Kept to the tightest-orb ones, per the methodology's own "точность важнее
# типа" priority rule — a wide, barely-in-orb aspect is the first thing to
# drop when something has to give.
_MAX_ASPECTS_PER_PROFILE = 3


def get_planet_profiles(spec: str, top_n: int = 9) -> List[Dict]:
    """Replaces the older get_significant_facts(): instead of ranking
    planet-placement, aspect, and house-cusp facts as independent,
    unrelated items, this builds one PROFILE per significant point —
    its sign, house, retrograde state, and (this is the actual fix) its
    own strongest aspects to other points, each with enough context about
    the *other* point (its sign/house) to judge how that aspect colors
    this one.

    Why this replaced the flat fact list: real end-to-end testing (a full
    answer reviewed by the user) showed the old design's real failure
    mode — a planet's sign+house meaning and its aspects were digested as
    entirely separate, unconnected facts, so the final answer described
    "Юпитер в Овне, 12 дом" as straightforwardly expansive/fortunate
    without ever registering that a 12th-house placement conventionally
    mutes or hides a planet's outward expression, and never once wove an
    aspect into any planet's characterization at all — despite aspects
    being exactly what turns a generic sign/house description into
    something specific to this one chart. Bundling sign+house+aspects into
    one profile, and prompting the digest step (utils/interpret.py) to
    synthesize them together rather than list them, is meant to fix that.

    A second real complaint this addresses: standalone "what does the Nth
    house mean" facts (the old "house" kind) produced their own free-
    floating paragraphs in the final answer that weren't wanted at all —
    house meaning only matters colored by what's actually placed there, so
    there is no more standalone house-kind fact; a house's cusp sign is
    already visible in the raw computed chart text every answer already
    gets, which is enough context on its own without a dedicated fact/RAG
    query for it.

    Selection: scored by the same angularity/retrogradation/aspect-count
    priority rules as before, but two categories are force-included
    regardless of score — Pars Fortunae (an Arabic Part is inherently a
    minor point that will rarely out-score a heavily-aspected classical
    planet on this scale, so it was silently getting dropped every time)
    and any point sitting in a fixed-star conjunction (same reasoning,
    plus these are genuinely rare/notable when they do occur). Both were
    confirmed missing from a real answer despite being present in the
    chart.

    One combined digest LLM call still processes every returned profile
    together (see utils/interpret.py) rather than one call per profile —
    a per-profile call was considered (and is worth revisiting once
    hardware/latency allow), but multiplying an already multi-minute
    CPU-only generation by one call per profile was judged too costly for
    now; bundling richer per-profile context into the existing single
    digest call is the cheaper way to get the same "aspects considered
    together with placement" result.

    Returns [] (not an error) if the chart can't be built at all.

    Natal charts only for now — transit significance needs different
    scoring, left as a follow-up rather than bolted on here."""
    fields, missing = _extract_fields(spec)
    if missing:
        return []
    try:
        subject = _build_subject(fields, name=fields.get("name") or "Subject")
    except Exception:
        return []

    from kerykeion import AspectsFactory

    aspects = AspectsFactory.natal_aspects(
        subject, active_points=_ASPECT_ACTIVE_POINTS, active_aspects=_ALL_ASPECTS,
    ).aspects

    aspect_counts: Dict[str, int] = {}
    for a in aspects:
        aspect_counts[a.p1_name] = aspect_counts.get(a.p1_name, 0) + 1
        aspect_counts[a.p2_name] = aspect_counts.get(a.p2_name, 0) + 1

    # Fixed-star facts, indexed by which classical point/angle they touch
    # (their own "text" already names it first, e.g. "Сатурн ♄ —
    # соединение..."), so each attaches to that point's profile instead of
    # floating as its own unrelated fact kind.
    stars_by_label: Dict[str, List[Dict]] = {}
    for sf in _find_star_conjunctions(subject):
        point_label = sf["text"].split(" — ", 1)[0]
        stars_by_label.setdefault(point_label, []).append(sf)

    kery_name_to_point = _kery_name_to_point_map()

    profiles: List[Dict] = []
    for label, attr in _PLANET_ATTRS + _ANGLE_ATTRS:
        point = getattr(subject, attr, None)
        if point is None:
            continue
        house_num = _house_number(getattr(point, "house", None)) or 0
        retrograde = bool(getattr(point, "retrograde", False))
        kery_name = attr_to_kerykeion_name(attr)
        label_ru = _point_ru_from_label(label)
        own_stars = stars_by_label.get(label_ru, [])

        score = 0.0
        if house_num in _ANGULAR_HOUSES:
            score += 3.0
        elif house_num in _SUCCEDENT_HOUSES:
            score += 1.5
        if retrograde:
            score += 0.5
        score += 0.5 * aspect_counts.get(kery_name, 0)
        if own_stars:
            score += 2.0  # a star conjunction is itself notable, not just a tiebreaker

        # This profile's own aspects, each carrying the OTHER point's
        # sign/house too — that's the piece the old flat aspect facts
        # never carried, and is exactly what lets the digest step reason
        # about how strong/relevant the aspecting influence itself is
        # (e.g. a square from a planet that's itself angular and tightly
        # aspected matters more than one from a weak, cadent placement).
        own_aspects = []
        for a in aspects:
            if a.p1_name == kery_name:
                other_kery = a.p2_name
            elif a.p2_name == kery_name:
                other_kery = a.p1_name
            else:
                continue
            other_label, other_attr = kery_name_to_point.get(other_kery, (other_kery, None))
            other_point = getattr(subject, other_attr, None) if other_attr else None
            other_sign = _sign_ru(other_point.sign) if other_point is not None else ""
            other_house = _house_number(getattr(other_point, "house", None)) if other_point is not None else None
            # Pre-formatted, grammatically correct phrase ("трин Солнца и
            # Луны") for the digest/final-answer prompts to quote directly
            # instead of paraphrasing around a case mismatch themselves —
            # see _POINT_NAMES_RU_GENITIVE's comment for the real failure
            # this fixes.
            phrase = (
                f"{_aspect_ru(a.aspect)} {_point_ru_genitive_from_label(label)} "
                f"и {_point_ru_genitive_from_label(other_label)}"
            )
            own_aspects.append(
                {
                    "orb": a.orbit,
                    "aspect_ru": _aspect_ru(a.aspect),
                    "movement_ru": _movement_ru(a.aspect_movement),
                    "other_label": _point_ru_from_label(other_label),
                    "other_sign": other_sign,
                    "other_house": other_house,
                    "phrase": phrase,
                }
            )
        own_aspects.sort(key=lambda x: x["orb"])
        own_aspects = own_aspects[:_MAX_ASPECTS_PER_PROFILE]

        sign_prep = _sign_ru_prepositional(point.sign)
        retro_text = " (ретроградный)" if retrograde else ""
        house_text = f", {house_num} дом" if house_num else ""

        queries = [f"{label} {sign_prep}"] + ([f"{label} в {house_num} доме"] if house_num else [])
        for asp in own_aspects:
            queries.append(f"{asp['aspect_ru']} {label_ru} и {asp['other_label']}")
        for sf in own_stars:
            queries.extend(sf["queries"])

        profiles.append(
            {
                "kind": "planet",
                "label": label_ru,
                "text": f"{label_ru} {sign_prep}{house_text}{retro_text}",
                "aspects": own_aspects,
                "stars": own_stars,
                "queries": queries,
                "score": score,
                # The Sun and Moon are force-included on their own merit
                # regardless of score — they're the two placements every
                # mainstream reading treats as fundamental, and this
                # scoring model (angularity/aspect-count/retrogradation)
                # has no notion of "luminary" that would otherwise protect
                # them from being crowded out by a chart with several
                # fixed-star conjunctions elsewhere (a real tested case:
                # ten other points force-included by stars alone, which
                # left zero budget for anything score-ranked at all — the
                # Sun and Moon included).
                "force_include": label in ("Солнце", "Луна", "Парс Фортуны") or bool(own_stars),
            }
        )

    forced = [p for p in profiles if p["force_include"]]
    rest = sorted(
        (p for p in profiles if not p["force_include"]), key=lambda p: p["score"], reverse=True
    )
    # top_n is a floor on forced items, not a hard cap on the whole list —
    # a chart with several fixed-star conjunctions (a real, tested case:
    # this one has six) can legitimately have more force-included profiles
    # than top_n on its own, and truncating forced[top_n:] would silently
    # drop exactly the rare points (Pars Fortunae, star conjunctions) this
    # mechanism exists to guarantee. Only the score-ranked "rest" fill is
    # ever capped.
    return forced + rest[: max(0, top_n - len(forced))]


def attr_to_kerykeion_name(attr: str) -> str:
    """Maps a subject attribute name (e.g. "true_north_lunar_node") back to
    kerykeion's own point-name literal (e.g. "True_North_Lunar_Node") as
    used in aspect.p1_name/p2_name — small helper so get_planet_profiles
    can look up per-planet aspect counts without a second parallel table."""
    return {
        "sun": "Sun", "moon": "Moon", "mercury": "Mercury", "venus": "Venus",
        "mars": "Mars", "jupiter": "Jupiter", "saturn": "Saturn",
        "uranus": "Uranus", "neptune": "Neptune", "pluto": "Pluto",
        "true_north_lunar_node": "True_North_Lunar_Node",
        "true_south_lunar_node": "True_South_Lunar_Node",
        "chiron": "Chiron", "true_lilith": "True_Lilith",
        "pars_fortunae": "Pars_Fortunae", "vertex": "Vertex",
        # Angles — missing here was a real bug: the fallback below (the
        # plain lowercase attr name) doesn't match kerykeion's actual
        # p1_name/p2_name capitalization ("Ascendant", not "ascendant"),
        # so get_planet_profiles' reverse lookup (kerykeion name -> label)
        # silently failed for any aspect to an angle, printing the raw
        # English kerykeion name ("к Ascendant") instead of the Russian
        # label with its sign/house — confirmed via a real digest-prompt
        # inspection.
        "ascendant": "Ascendant", "medium_coeli": "Medium_Coeli",
    }.get(attr, attr)


def _kery_name_to_point_map() -> Dict[str, Tuple[str, str]]:
    """kerykeion point-name literal (e.g. "True_North_Lunar_Node") ->
    (Russian label, subject attribute) — shared by get_planet_profiles
    and get_dual_chart_profiles for looking up the *other* side of an
    aspect's own sign/house. Pulled out as its own function (rather than a
    module-level constant) since it depends on attr_to_kerykeion_name,
    defined just above."""
    all_points = _PLANET_ATTRS + _ANGLE_ATTRS
    return {attr_to_kerykeion_name(attr): (label, attr) for label, attr in all_points}


def get_dual_chart_profiles(
    reference_subject,
    other_subject,
    other_active_points: Optional[List[str]] = None,
    top_n: int = 9,
) -> List[Dict]:
    """Two-chart counterpart to get_planet_profiles — one profile per
    significant point of `other_subject` (currently: a transit moment's
    moving planets; intended to also serve synastry as its second
    consumer, see below), each bundling:
      - its placement, with the house computed against
        `reference_subject`'s OWN cusps via _house_of_degree — the
        standard transit-astrology convention (a transiting planet's
        house is which of YOUR houses it's currently moving through, not
        some house system computed fresh for the transit moment itself);
      - its own cross-chart aspects to `reference_subject`'s points, each
        carrying that reference point's sign/house too — the same "other
        side of the aspect" context get_planet_profiles already gives for
        single-chart aspects, needed for the same reason (so the digest
        step can judge how strong/relevant the aspecting relationship is,
        not just that it exists).

    This is the shared layer run_transit is being refactored onto (see
    _format_transit_text, which now also uses _house_of_degree directly
    for its own raw-data listing, and routes/chat.py, which now runs
    transit answers through the same digest/sectioned-answer pipeline
    natal charts already have instead of the generic reasoning-mode
    prompt). Synastry is intended to reuse this same function as its
    second consumer — most likely by calling it twice, once per
    direction (person A's points aspecting B's chart, then B's points
    aspecting A's), since a synastry reading is inherently bidirectional
    in a way a transit reading isn't (a moment doesn't have "its own"
    identity worth profiling the way a second person's chart does) — not
    yet built, left for that follow-up rather than guessed at here.

    other_active_points defaults to _ACTIVE_POINTS_TRANSIT (fixed stars
    and the Vertex excluded from the moving side — see that list's own
    comment for why neither is meaningfully "transiting"). Scoring
    mirrors get_planet_profiles: angularity (by the reference-chart house
    computed above, not other_subject's own), aspect count, retrograde.
    No force-include tier here: Pars Fortunae/star-conjunction force-
    include is a natal-only concept (a transiting Part of Fortune isn't
    conventionally read as a moving body, and other_active_points already
    excludes fixed stars entirely) — only Sun/Moon are force-included,
    since they're the two bodies every mainstream transit reading treats
    as significant regardless of house/aspect count (the Moon especially:
    it moves fast enough that its house/sign alone is often the main
    short-term theme, even with few exact aspects on a given day)."""
    if other_active_points is None:
        other_active_points = _ACTIVE_POINTS_TRANSIT

    from kerykeion import AspectsFactory

    natal_cusps = _house_cusp_degrees(reference_subject)
    aspects = AspectsFactory.dual_chart_aspects(
        reference_subject, other_subject,
        active_points=_ASPECT_ACTIVE_POINTS, active_aspects=_ALL_ASPECTS,
    ).aspects

    other_name = other_subject.name
    aspect_counts: Dict[str, int] = {}
    for a in aspects:
        other_side = a.p1_name if a.p1_owner == other_name else a.p2_name
        aspect_counts[other_side] = aspect_counts.get(other_side, 0) + 1

    kery_name_to_point = _kery_name_to_point_map()

    profiles: List[Dict] = []
    for label, attr in _PLANET_ATTRS:
        kery_name = attr_to_kerykeion_name(attr)
        if kery_name not in other_active_points:
            continue
        point = getattr(other_subject, attr, None)
        if point is None:
            continue

        natal_house = _house_of_degree(natal_cusps, point.position)
        retrograde = bool(getattr(point, "retrograde", False))
        label_ru = _point_ru_from_label(label)

        score = 0.0
        if natal_house in _ANGULAR_HOUSES:
            score += 3.0
        elif natal_house in _SUCCEDENT_HOUSES:
            score += 1.5
        if retrograde:
            score += 0.5
        score += 0.5 * aspect_counts.get(kery_name, 0)

        # This planet's own cross-chart aspects, each carrying the
        # REFERENCE (natal) point's sign/house too — mirrors
        # get_planet_profiles' own_aspects exactly, just across two
        # subjects instead of one.
        own_aspects = []
        for a in aspects:
            if a.p1_owner == other_name and a.p1_name == kery_name:
                other_kery = a.p2_name
            elif a.p2_owner == other_name and a.p2_name == kery_name:
                other_kery = a.p1_name
            else:
                continue
            ref_label, ref_attr = kery_name_to_point.get(other_kery, (other_kery, None))
            ref_point = getattr(reference_subject, ref_attr, None) if ref_attr else None
            ref_sign = _sign_ru(ref_point.sign) if ref_point is not None else ""
            ref_house = (
                _house_number(getattr(ref_point, "house", None)) if ref_point is not None else None
            )
            # Pre-formatted, grammatically correct phrase, same convention
            # as get_planet_profiles' "phrase" — e.g. "квадрат Марса и
            # Сатурна" for a transiting Mars square natal Saturn.
            phrase = (
                f"{_aspect_ru(a.aspect)} {_point_ru_genitive_from_label(label)} "
                f"и {_point_ru_genitive_from_label(ref_label)}"
            )
            own_aspects.append(
                {
                    "orb": a.orbit,
                    "aspect_ru": _aspect_ru(a.aspect),
                    "movement_ru": _movement_ru(a.aspect_movement),
                    "other_label": _point_ru_from_label(ref_label),
                    "other_sign": ref_sign,
                    "other_house": ref_house,
                    "phrase": phrase,
                }
            )
        own_aspects.sort(key=lambda x: x["orb"])
        own_aspects = own_aspects[:_MAX_ASPECTS_PER_PROFILE]

        sign_prep = _sign_ru_prepositional(point.sign)
        retro_text = " (ретроградный)" if retrograde else ""
        house_text = f", натальный {natal_house} дом" if natal_house else ""

        queries = [f"транзитный {label} {sign_prep}"] + (
            [f"транзитный {label} в {natal_house} доме"] if natal_house else []
        )
        for asp in own_aspects:
            queries.append(f"транзит {asp['aspect_ru']} {label_ru} и {asp['other_label']}")

        profiles.append(
            {
                "kind": "transit_planet",
                "label": label_ru,
                "text": f"транзитный {label_ru} {sign_prep}{house_text}{retro_text}",
                "aspects": own_aspects,
                "stars": [],
                "queries": queries,
                "score": score,
                "force_include": label in ("Солнце", "Луна"),
            }
        )

    forced = [p for p in profiles if p["force_include"]]
    rest = sorted(
        (p for p in profiles if not p["force_include"]), key=lambda p: p["score"], reverse=True
    )
    return forced + rest[: max(0, top_n - len(forced))]


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


def _build_transit_subjects(fields: Dict[str, str], name: str) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    """Shared by run_transit and get_transit_profiles — both need the same
    pair of subjects (natal + the transit-moment subject), and duplicating
    this parsing/building logic in two places risked exactly the kind of
    drift that would silently make the digest step (get_transit_profiles)
    see a different moment/location than the raw chart text
    (run_transit/_format_transit_text) shows. Returns (natal,
    transit_subject, None) on success, or (None, None, error_message) on
    any failure — following this module's established "never raise, return
    an explanatory Russian string instead" convention (see run_natal's own
    docstring) rather than letting callers each reimplement their own
    try/except around three separate failure points."""
    try:
        natal = _build_subject(fields, name=name)
    except Exception as e:
        return None, None, f"Не удалось построить натальную карту — некорректные данные ({e})."

    moment = (fields.get("moment") or "now").strip()
    try:
        if moment.lower() in ("", "now", "сейчас"):
            dt = datetime.now()
        else:
            dt = datetime.strptime(moment, "%Y-%m-%dT%H:%M")
    except Exception as e:
        return None, None, (
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
        from kerykeion.astrological_subject_factory import AstrologicalSubjectFactory

        transit_subject = AstrologicalSubjectFactory.from_birth_data(
            name="Transit",
            year=dt.year, month=dt.month, day=dt.day, hour=dt.hour, minute=dt.minute,
            lat=moment_lat, lng=moment_lon, tz_str=moment_tz,
            online=False,
            active_points=_ACTIVE_POINTS_TRANSIT,
        )
    except Exception as e:
        return None, None, f"Не удалось рассчитать текущие положения планет: {e}"

    return natal, transit_subject, None


def run_transit(spec: str) -> str:
    """Tool entry point (utils.tools.TOOL_REGISTRY["astro_transit_chart"])."""
    fields, missing = _extract_fields(spec)
    if missing:
        return _missing_fields_message(missing, fields)
    natal, transit_subject, error = _build_transit_subjects(fields, name=fields.get("name") or "Subject")
    if error:
        return error
    try:
        return _format_transit_text(natal, transit_subject)
    except Exception as e:
        return f"Ошибка при расчёте транзитов: {e}"


def get_transit_profiles(spec: str, top_n: int = 9) -> List[Dict]:
    """Transit counterpart to get_planet_profiles, for routes/chat.py's
    digest step — rebuilds both subjects from `spec` (same duplication-of-
    computation pattern get_planet_profiles already has relative to
    run_natal: the tool call already computed a subject once for the raw
    chart text, and this recomputes it independently for profiling; kept
    consistent with that existing precedent rather than plumbing the
    already-built subject through routes/chat.py) and hands them to
    get_dual_chart_profiles.

    Returns [] (not an error) if fields are missing or subject-building
    fails — same "no profiles available, caller falls back" contract
    get_planet_profiles already has."""
    fields, missing = _extract_fields(spec)
    if missing:
        return []
    natal, transit_subject, error = _build_transit_subjects(fields, name=fields.get("name") or "Subject")
    if error:
        return []
    try:
        return get_dual_chart_profiles(natal, transit_subject, top_n=top_n)
    except Exception:
        return []


# Registry for future operations (synastry, composite, rectification,
# electional search, ...) — see module docstring. Not yet consumed by
# anything (utils/tools.py wires run_natal/run_transit directly, since
# there are only two so far), but kept as the intended extension point so
# adding a third operation doesn't require inventing a new pattern.
ASTRO_OPERATIONS: Dict[str, Callable[[str], str]] = {
    "natal": run_natal,
    "transit": run_transit,
}
