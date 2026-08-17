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
import math
import re
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils import llm as llm_utils  # LLM-first field extraction, see _extract_fields_llm below

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

# Deterministic harmonious/tense/neutral classification per aspect TYPE —
# added after a real, reproducible failure the user caught in a generated
# synastry answer: a trine (Mars-Venus) and a semisextile (Saturn-Jupiter)
# were both labeled "Точка напряжения" ("point of tension") in the final
# prose, despite both being conventionally harmonizing aspects with no
# standard tense/conflict reading at all. The digest prompt already SPELLS
# OUT this same classification in prose (_build_digest_prompt's aspect_rule
# lists "гармоничные (трин, секстиль, ...)" / "напряжённые (квадрат,
# оппозиция, ...)"), but relying on the model to apply that rule correctly,
# consistently, for every single aspect, under the same time/context
# pressure that already produced invented aspect-naming adjectives
# elsewhere (see _POINT_NAMES_RU_GENITIVE's own comment) turned out not to
# be reliable enough — the user's own read: "создается такое впечатление,
# что термины 'напряжение', 'конфликт' итд раскиданы по тексту случайным
# образом". Same fix as that earlier grammar problem: compute the correct
# answer once in Python and hand it to the model as an already-labeled
# fact to copy, rather than a rule to re-derive from scratch each time.
#
# Classification follows standard mainstream convention: trine/sextile are
# unambiguously harmonious; square/opposition are unambiguously tense;
# conjunction is neither — it blends whatever is conjunct, so its "nature"
# genuinely depends on the two points involved, not the aspect type alone.
# Minor aspects: semi-sextile is a mild harmonious aspect (gentle, easy
# blending); semi-square/sesquiquadrate are the minor-aspect counterparts
# of square (irritating friction, tighter but real); quintile/biquintile
# are conventionally read as harmonious-and-creative (talent, a "knack"
# for something) — the same family as trine/sextile, not tense; quincunx
# is the one genuinely ambiguous minor aspect (an "awkward fit" needing
# adjustment rather than a straightforward blend) but is NOT a conflict/
# friction aspect the way square or opposition are, so it's classified as
# "неоднозначный" (ambiguous/adjustment-needed) rather than "напряжённый",
# to avoid the exact overclaiming this table exists to prevent.
_ASPECT_NATURE = {
    "conjunction": "нейтральный (зависит от планет)",
    "opposition": "напряжённый",
    "trine": "гармоничный",
    "square": "напряжённый",
    "sextile": "гармоничный",
    "semi-sextile": "гармоничный (мягкий)",
    "semi-square": "напряжённый (мягкий)",
    "quintile": "гармоничный (творческий)",
    "sesquiquadrate": "напряжённый",
    "biquintile": "гармоничный (творческий)",
    "quincunx": "неоднозначный (требует приспособления, не конфликт)",
}


def _aspect_nature_ru(name: str) -> str:
    """Aspect-type literal (kerykeion's own English name, e.g. "trine") ->
    its pre-classified Russian harmonious/tense/neutral label — see
    _ASPECT_NATURE's own comment for why this exists and the convention it
    follows. Falls back to the neutral-ish generic label for anything not
    in the table (there shouldn't be any, since _ALL_ASPECTS is the only
    source of aspect names used anywhere in this module)."""
    return _ASPECT_NATURE.get(name, "нейтральный")

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

# ---------- Classical (horary/electional) aspect set + luminary-aware orb ----------
# horary_methodology.txt section 4 (electional_methodology.txt explicitly
# reuses this same rule — "в тех же орбисах... что и в хорарной технике")
# specifies a DIFFERENT orb scheme from _ALL_ASPECTS above, on two counts:
# (1) only six classical aspects count at all — conjunction, sextile,
# square, trine, quincunx, opposition — none of _MINOR_ASPECTS' other five
# (semi-sextile/semi-square/quintile/sesquiquadrate/biquintile) are part of
# classical horary/electional doctrine; (2) the orb itself depends on which
# BODIES are involved, not just the aspect type: 8-10° when either party is
# the Sun or Moon, 6-7° between any other two planets, a flat 5° for
# quincunx regardless of which bodies are involved. kerykeion's own
# active_aspects table can only express one orb per aspect NAME (not
# per-pair), so this can't be done with a single AspectsFactory call the
# way _ALL_ASPECTS is used everywhere else — see _CLASSICAL_ASPECTS_WIDE
# and filter_classical_aspects below for how callers actually apply this.
#
# This was previously specified in the methodology doc but never actually
# implemented anywhere in code: utils/horary.py used to define its own
# (correctly reasoned, but never wired up) _HORARY_ASPECTS restricting the
# aspect set, then called AspectsFactory with astro._ALL_ASPECTS anyway —
# dead code, not a real fix. utils/electional.py's own _ELECTIONAL_ASPECTS
# DID correctly restrict the aspect set, but used _MAJOR_ASPECTS'/
# _MINOR_ASPECTS' flat per-type orbs (8/8/7/7/5/3) rather than this
# luminary-aware rule. utils/chart_draw.py's wheel renderer never
# distinguished technique at all, always using _ALL_ASPECTS regardless —
# so a horary or electional chart could draw non-classical minor-aspect
# lines (e.g. a quintile) that shouldn't exist under this doctrine at all,
# with a generic orb determining which of the classical aspects appeared.
_CLASSICAL_ASPECT_NAMES = {
    "conjunction", "opposition", "trine", "square", "sextile", "quincunx",
}
_LUMINARY_KERYKEION_NAMES = {"Sun", "Moon"}
# Fed to AspectsFactory's active_aspects — deliberately the WIDEST orb
# either branch of the luminary-aware rule could ever allow (10° for the
# five non-quincunx classical aspects, 5° for quincunx, which is already
# its final value with no further widening), so kerykeion's own orb
# filtering can never exclude an aspect before filter_classical_aspects
# below gets a chance to apply the real, narrower, per-pair cutoff.
_CLASSICAL_ASPECTS_WIDE = [
    {"name": name, "orb": 10} for name in ("conjunction", "opposition", "trine", "square", "sextile")
] + [{"name": "quincunx", "orb": 5}]


def _classical_orb_limit(aspect_name: str, p1_name: str, p2_name: str) -> float:
    """The real per-pair cutoff described above — quincunx is a flat 5°
    regardless of participants; every other classical aspect gets 10° if
    either party is the Sun or Moon, else 7°. These are the upper bound of
    each range the methodology gives ("8-10°" / "6-7°") — a single flat
    number per case, consistent with how the rest of this app already
    collapses an orb allowance to one cutoff per aspect rather than
    modeling full per-planet moieties."""
    if aspect_name == "quincunx":
        return 5.0
    if p1_name in _LUMINARY_KERYKEION_NAMES or p2_name in _LUMINARY_KERYKEION_NAMES:
        return 10.0
    return 7.0


def filter_classical_aspects(raw_aspects: List[Any]) -> List[Any]:
    """Post-filters a kerykeion aspects list (computed with
    _CLASSICAL_ASPECTS_WIDE as active_aspects, so nothing valid was
    excluded too early) down to real classical horary/electional doctrine:
    only the six aspect names in _CLASSICAL_ASPECT_NAMES, and only within
    that pair's real orb per _classical_orb_limit above. Every caller that
    wants classical (not general _ALL_ASPECTS) aspects — utils/horary.py,
    utils/electional.py, and utils/chart_draw.py's wheel renderer when
    drawing a horary/electional chart — should call AspectsFactory with
    _CLASSICAL_ASPECTS_WIDE and then pass its .aspects list through this
    function before using it for anything (scoring, verdict text, or
    drawing)."""
    return [
        a for a in raw_aspects
        if a.aspect in _CLASSICAL_ASPECT_NAMES
        and a.orbit <= _classical_orb_limit(a.aspect, a.p1_name, a.p2_name)
    ]


# ---------- Per-technique orb: natal / transit-family / synastry ----------
# A real, reported gap in the wheel-chart renderer (utils/chart_draw.py):
# every chart used the one flat, technique-agnostic _ALL_ASPECTS orb
# regardless of which technique it was for, even though real astrology
# software conventionally uses DIFFERENT orbs for a single natal chart's
# own internal aspects vs. a synastry comparison between two people vs. a
# transit/progression/direction/return "one real chart + one derived
# moment" comparison. Extracted from the user's own reference astrology
# software (screenshots of its per-technique aspect/orb configuration
# pages) — three tables below, deliberately NOT covering horary/
# electional (see _CLASSICAL_ASPECTS_WIDE's own comment for why those stay
# on their own, methodology-text-grounded scheme instead), and
# deliberately NOT extending to aspect types beyond what this app already
# computes (the reference software's pages also showed septile/novile/
# decile-family angles at unusual non-round degree values — 51.43°,
# 40°, 36°, 102.86°, 154.29° — which aren't part of any methodology
# document here and were left out rather than guessed at).
#
# Natal's own page gave a genuinely different orb PER BODY (not just per
# aspect type) — the classical "moiety" convention: each body has its own
# orb allowance, and the real orb between two specific bodies is the
# average of their two individual allowances (see _moiety_orb_limit
# below). Transit's and synastry's own pages, by contrast, showed one
# flat orb per aspect type regardless of which two bodies were involved
# (transit differentiates Sun/Moon from everything else on the tightest
# half-aspects only; synastry doesn't differentiate by body at all) — so
# for those two, "_default" is effectively the only entry most rows need,
# and averaging two identical numbers is a no-op, which is why one shared
# _moiety_orb_limit implementation below covers all three tables
# correctly regardless of whether a given table actually varies by body.
#
# Any aspect/body combination with no explicit entry below falls back to
# _ALL_ASPECTS' existing flat per-aspect-type orb — e.g. synastry's own
# page had no row at all for semi-sextile or quincunx, meaning neither
# aspect is treated specially there; they keep behaving exactly as they
# did before any of this existed.
_NATAL_ORB_BY_BODY: Dict[str, Dict[str, float]] = {
    "conjunction": {"Sun": 12.0, "Moon": 10.0, "Jupiter": 8.0, "True_North_Lunar_Node": 0.1, "_default": 5.0},
    "opposition": {"Sun": 12.0, "Moon": 10.0, "Jupiter": 8.0, "True_North_Lunar_Node": 0.1, "_default": 5.0},
    "trine": {"Sun": 12.0, "Moon": 8.0, "True_North_Lunar_Node": 0.1, "_default": 5.0},
    "square": {"Sun": 10.0, "Moon": 8.0, "Jupiter": 7.0, "True_North_Lunar_Node": 0.1, "_default": 5.0},
    "sextile": {"Sun": 6.5, "Moon": 6.0, "_default": 5.0},
    "semi-sextile": {"Sun": 1.5, "Moon": 1.0, "_default": 1.0},
    "semi-square": {"_default": 1.0},
    "quintile": {"Sun": 1.5, "Moon": 1.5, "_default": 1.0},
    "sesquiquadrate": {"_default": 1.0},
    "biquintile": {"_default": 1.0},
    "quincunx": {"_default": 1.0},
}
_TRANSIT_ORB_BY_BODY: Dict[str, Dict[str, float]] = {
    "conjunction": {"_default": 1.0},
    "opposition": {"_default": 1.0},
    "trine": {"_default": 1.0},
    "square": {"_default": 1.0},
    "sextile": {"_default": 1.0},
    "semi-sextile": {"Sun": 1.0, "Moon": 1.0, "_default": 0.5},
    "semi-square": {"Sun": 1.0, "Moon": 1.0, "_default": 0.5},
    "quintile": {"_default": 1.0},
    "sesquiquadrate": {"Sun": 1.0, "Moon": 1.0, "_default": 0.5},
    "biquintile": {"Sun": 1.0, "Moon": 1.0, "_default": 0.5},
    "quincunx": {"_default": 1.0},
}
_SYNASTRY_ORB_BY_BODY: Dict[str, Dict[str, float]] = {
    "conjunction": {"_default": 7.35},
    "opposition": {"_default": 3.67},
    "trine": {"_default": 2.43},
    "square": {"_default": 1.8},
    "sextile": {"_default": 1.17},
    "semi-square": {"_default": 0.87},
    "quintile": {"_default": 1.37},
    "sesquiquadrate": {"_default": 0.53},
    "biquintile": {"_default": 1.15},
    # semi-sextile, quincunx: no row in the reference software's synastry
    # page at all — falls through to _ALL_ASPECTS' generic orb.
}


def _generic_orb_limit(aspect_name: str) -> float:
    """The flat per-aspect-type orb from _ALL_ASPECTS — the fallback used
    wherever a per-technique table above has no entry for this aspect."""
    return next((a["orb"] for a in _ALL_ASPECTS if a["name"] == aspect_name), 8.0)


def _orb_by_body(table: Dict[str, Dict[str, float]], aspect_name: str, body: str) -> Optional[float]:
    per_aspect = table.get(aspect_name)
    if per_aspect is None:
        return None
    return per_aspect.get(body, per_aspect.get("_default"))


def _moiety_orb_limit(table: Dict[str, Dict[str, float]], aspect_name: str, p1_name: str, p2_name: str) -> float:
    """orb(A,B) = average of each body's own per-aspect orb from `table`
    — the classical moiety convention, each body contributing its own
    half of the final orb. Falls back to _generic_orb_limit if `table`
    has no entry at all for this aspect, or (defensively) if a body
    somehow resolves to no value even via "_default"."""
    o1 = _orb_by_body(table, aspect_name, p1_name)
    o2 = _orb_by_body(table, aspect_name, p2_name)
    if o1 is None or o2 is None:
        return _generic_orb_limit(aspect_name)
    return (o1 + o2) / 2.0


def natal_orb_limit(aspect_name: str, p1_name: str, p2_name: str) -> float:
    """Orb for a chart's own INTERNAL aspects — applies universally to any
    single chart read on its own, whichever technique produced it (a real
    natal chart, a progressed chart considered alone, a return chart
    alone, ...), not just utils/astro.py's own run_natal."""
    return _moiety_orb_limit(_NATAL_ORB_BY_BODY, aspect_name, p1_name, p2_name)


def transit_orb_limit(aspect_name: str, p1_name: str, p2_name: str) -> float:
    """Orb for CROSS-chart aspects between one real chart and a technique-
    derived moment — transit, progression, direction, lunar/solar return.
    NOT for synastry (see synastry_orb_limit below) — two real people
    conventionally get a different, wider orb than a single moment
    overlaid on a natal chart."""
    return _moiety_orb_limit(_TRANSIT_ORB_BY_BODY, aspect_name, p1_name, p2_name)


def synastry_orb_limit(aspect_name: str, p1_name: str, p2_name: str) -> float:
    """Orb for CROSS-chart aspects between two real people's charts."""
    return _moiety_orb_limit(_SYNASTRY_ORB_BY_BODY, aspect_name, p1_name, p2_name)


# Per _ALL_ASPECTS' own entries, widened wherever any per-technique table
# above allows a wider orb than the generic default (only natal's
# conjunction/opposition/trine/square/sextile ever do, since its
# Sun/Moon/Jupiter entries exceed the generic 8/8/7/7/5) — fed to
# AspectsFactory's active_aspects for every non-classical chart so
# kerykeion's own orb filtering can never exclude an aspect before
# natal_orb_limit/transit_orb_limit/synastry_orb_limit above get a chance
# to apply the real, technique-specific cutoff. Computed once at import
# time from the tables above, not hand-maintained separately — stays
# correct automatically if those tables ever change.
_PER_TECHNIQUE_TABLES = (_NATAL_ORB_BY_BODY, _TRANSIT_ORB_BY_BODY, _SYNASTRY_ORB_BY_BODY)
_PER_TECHNIQUE_ASPECTS_WIDE = [
    {
        "name": spec["name"],
        # kerykeion's own AspectsFactory validates active_aspects' "orb"
        # as an int (a pydantic model field) — this table only needs to
        # be wide enough that kerykeion never excludes an aspect before
        # the real per-pair natal_orb_limit/transit_orb_limit/
        # synastry_orb_limit gets a chance to apply the true cutoff
        # afterward, so rounding UP to the next whole degree is exactly
        # as safe as the real fractional value and keeps pydantic happy.
        "orb": math.ceil(
            max(
                [spec["orb"]]
                + [
                    max(table[spec["name"]].values())
                    for table in _PER_TECHNIQUE_TABLES
                    if spec["name"] in table
                ]
            )
        ),
    }
    for spec in _ALL_ASPECTS
]


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
# Dash-separated time ("19-28-30", "19-28") — a real, reported gap: a
# real user writes time this way (their own long-standing habit across
# several test messages), and the colon-only regex above silently treats
# that as "no time given at all", which then falls through to the
# free-text fallback and can pick up a time from somewhere else entirely —
# a real, reported bug where a much OLDER time from earlier in the same
# conversation silently won instead of the one actually typed. Kept as a
# SEPARATE regex rather than broadening _TIME_RE's own character class to
# accept "-" as well as ":" — that was tried first and rejected after
# testing: it also matches the tail of an ISO date ("2026-08-06" contains
# "08-06", misread as 08:06), a real, confirmed false positive. This
# regex requires a plausible clock hour (00-23) and minute/second (00-59),
# and is NOT preceded by a 4-digit year and dash — blocking exactly that
# ISO-date collision — without touching _TIME_RE's own already-safe
# colon-based matching at all.
_TIME_DASH_RE = re.compile(r"(?<!\d{4}-)\b([01]?\d|2[0-3])-([0-5]\d)(?:-[0-5]\d)?\b")
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
    m = _TIME_RE.search(text) or _TIME_DASH_RE.search(text)
    if m:
        hour, minute = m.group(1), m.group(2)
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


_LOCATIVE_PREPOSITIONS = {
    "в", "во", "из", "к", "ко", "у", "под", "около", "близ", "г",
}


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
    name somewhere in the world — two real, reproducible cases found by
    testing: Russian "года" ("of the year") is also a transliterated
    alternate name for Gōdo, a small Japanese town, and — found via a real
    horary test — "французский" ("French", the adjective/language) shares
    its first five letters with "Францистаун", the Cyrillic name for
    Francistown, Botswana, and is close enough in overall length that even
    a strict stem check doesn't tell them apart. Two independent
    mitigations, in order: (1) candidates immediately preceded by a
    locative preposition (в/из/к/у/под/...) are preferred outright over
    plain, preposition-less candidates — a real place mention in a
    birth-info sentence is overwhelmingly phrased "в Одессе", while a
    coincidental match on an ordinary word essentially never has a
    preposition directly in front of it, so this filters both known
    collision cases above without needing real morphological analysis; (2)
    if no candidate has a preposition (still common — a structured
    "Name, DD.MM.YYYY, HH:MM, City" listing has no prepositions at all),
    every word/word-pair is still checked and the most populous match
    across ALL of them wins, same fallback this function always had, since
    a small town matching a filler word essentially never beats the
    sentence's actual, real, usually far more populous, named city on
    population alone either.
    """
    _build_city_index()
    if not _city_index:
        return None

    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    lowered = [w.lower() for w in words]
    has_preposition = [
        i > 0 and lowered[i - 1] in _LOCATIVE_PREPOSITIONS for i in range(len(words))
    ]
    # (candidate_text, preposition_flag) — word-pairs inherit the FIRST
    # word's flag, since that's the word a preposition would sit in front of.
    candidates: List[Tuple[str, bool]] = [
        (w, has_preposition[i]) for i, w in enumerate(words)
    ] + [
        (f"{words[i]} {words[i + 1]}", has_preposition[i]) for i in range(len(words) - 1)
    ]

    def _find_matches(pool: List[Tuple[str, bool]]) -> List[dict]:
        # Exact and stem matches are pooled into ONE list and resolved by a
        # single population comparison across all of them — NOT "exact
        # always wins, stem is only a fallback if there's no exact match at
        # all", an earlier version of this function's actual (if
        # unintended) behavior. That tiering broke the "most populous match
        # across ALL candidates wins" promise this docstring already
        # makes: a real, reproducible case found by testing — "года" being
        # an EXACT match for Gōdo used to short-circuit and win outright
        # even when the same sentence also stem-matched Kyiv (pop. ~2.95M)
        # — the two were never actually compared against each other at
        # all. Pooling them together and comparing every candidate's
        # population in one pass is what makes "most populous wins" hold.
        found: List[dict] = [
            record for record in (_city_index.get(c.lower()) for c, _ in pool) if record
        ]
        for candidate, _ in pool:
            key = candidate.lower()
            max_len = min(len(key), _CITY_STEM_LEN)
            if max_len < 3:
                continue
            # Check a few prefix lengths, not just _CITY_STEM_LEN — a
            # second, independent bug found alongside the one above: a
            # base city name SHORTER than _CITY_STEM_LEN (e.g. "Киев", 4
            # letters) is only ever indexed under its own full-length
            # bucket ("киев"), but a declined form in the text ("Киеве", 5
            # letters, "в Киеве") only ever checked its OWN 5-character
            # bucket ("киеве") — which never matches "киев" — so a short
            # city name's declined form could never even become a
            # candidate at all before this fix, regardless of the tiering
            # issue above. Checking a couple of shorter prefixes too (down
            # to 3 characters) covers the common case of a Russian
            # declension only adding/changing the last 1-2 letters.
            floor = max(3, max_len - 2)
            for length in range(max_len, floor - 1, -1):
                found.extend(_city_stem_index.get(key[:length], []))
        return found

    preferred = [c for c in candidates if c[1]]
    if preferred:
        matches = _find_matches(preferred)
        if matches:
            return max(matches, key=lambda r: r["population"])
        # A preposition was found but matched nothing real (e.g. "в этом
        # году") — fall through to the full pool below rather than
        # reporting no city at all just because the higher-confidence tier
        # came up empty.

    matches = _find_matches(candidates)
    if matches:
        return max(matches, key=lambda r: r["population"])
    return None


def _lookup_city_exact(text: str) -> Optional[dict]:
    """Stricter sibling of _lookup_city — EXACT (non-fuzzy) name/alternate-
    name matches only, never the stem-bucket tier. Exists for callers where
    a wrong location is a much more serious error than for most of this
    app's techniques (currently: utils/horary.py, whose entire radicality/
    validity check hinges on getting the exact Ascendant right, so a
    silent, low-confidence geocode is a real correctness risk rather than a
    minor accuracy nit). Skips the exact match itself if the candidate word
    is too short to be a real place name (matches _lookup_city's own >=3
    character floor for its stem tier, for the same reason: very short
    words collide with real gazetteer entries far too often to trust)."""
    _build_city_index()
    if not _city_index:
        return None
    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    candidates = [w for w in words if len(w) >= 3] + [
        f"{a} {b}" for a, b in zip(words, words[1:])
    ]
    matches = [
        record for record in (_city_index.get(c.lower()) for c in candidates) if record
    ]
    if matches:
        return max(matches, key=lambda r: r["population"])
    return None


_FIELD_EXTRACTION_PROMPT = """Тебе показан текст запроса на построение астрологической карты (натальной,
транзитной, для ректификации, дирекций, прогрессий, возвращений,
профекций и т.п.).

Извлеки из этого текста:
1. Дату — в формате ГГГГ-ММ-ДД.
2. Время — в 24-часовом формате ЧЧ:ММ, независимо от того, как оно записано
   в тексте (через двоеточие, дефис, словами и т.п.).
3. Место — город и страна; если в тексте прямо даны координаты, верни их
   как "широта, долгота" (например "46.48, 30.72").

Текст:
\"\"\"{text}\"\"\"

Если какого-то из этих пунктов в тексте ДЕЙСТВИТЕЛЬНО нет — напиши "нет" в
соответствующей строке, не выдумывай и не угадывай. Если в тексте описаны
ДВА разных человека (например запрос на синастрию) — извлеки данные только
для ПЕРВОГО упомянутого.

Ответь СТРОГО в этом формате, каждый пункт на отдельной строке, без
пояснений до или после:
ДАТА: <ГГГГ-ММ-ДД или нет>
ВРЕМЯ: <ЧЧ:ММ или нет>
МЕСТО: <город, страна ИЛИ широта, долгота ИЛИ нет>"""


def _parse_labeled_field(label: str, answer: str) -> Optional[str]:
    """Pulls one "LABEL: value" line out of a model's own strict-format
    answer. This is parsing the MODEL's controlled output, not scanning
    raw free-form user text — a categorically safer use of regex than the
    free-text pattern-matching this whole extraction path exists to
    replace (see _extract_fields_llm below)."""
    m = re.search(rf"{label}\s*:\s*(.+)", answer, re.IGNORECASE)
    if not m:
        return None
    value = m.group(1).strip().strip('"').strip()
    if not value or value.lower() in ("нет", "нету", "n/a", "-", "—", "unknown"):
        return None
    return value


def _resolve_place_string(place_text: str) -> Optional[Tuple[float, float, str]]:
    """Resolves an already-isolated place string the model itself named
    (not a whole free-text blob) to (lat, lon, tz). Tries an explicit
    "lat, lon" pair first (a plain split+float, not a regex — safe here
    because it's the model's own clean, single-purpose answer, not raw
    user text), then falls back to an EXACT city-name match
    (_lookup_city_exact) — deliberately never the fuzzier stem-matching
    tier _lookup_city itself still uses elsewhere, since by this point
    there's no large blob of unrelated surrounding text left for a fuzzy
    match to go wrong in the way it did in horary's Francistown bug."""
    parts = [p.strip() for p in place_text.split(",")]
    if len(parts) == 2:
        try:
            lat, lon = float(parts[0]), float(parts[1])
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon, (_resolve_timezone(lat, lon) or "")
        except ValueError:
            pass
    city = _lookup_city_exact(place_text)
    if city:
        return city["latitude"], city["longitude"], city["timezone"]
    return None


def _extract_fields_llm(text: str) -> Optional[Dict[str, str]]:
    """LLM-first date/time/place extraction — the PRIMARY path inside
    _extract_fields below, with the existing regex-based
    _fill_fields_from_text only running afterwards as a mop-up for
    whatever this doesn't resolve (or entirely, if no model is loaded).

    Generalizes to every technique built on _extract_fields (natal,
    transit, directions, returns, profections, rectification) the same
    mechanism utils/horary.py pioneered for horary specifically
    (_extract_horary_fields_llm), after two concrete, reported regex
    failures there: a dash-separated time ("19-28-30") wasn't recognized
    by the colon-only time regex at all, and free-text city search
    accidentally stem-matched an ordinary word against an unrelated,
    obscure place name anywhere in the world (a real test found
    "французский" matching "Францистаун"/Francistown, Botswana). Both are
    failures of pattern-matching text rather than reading it — a model
    that actually understands the sentence doesn't have this failure
    mode, and a capable model is now the app's default (see
    install/.env.example).

    Never invents a field it isn't confident about (the prompt allows
    answering "нет"); returns only whichever of date/time/lat/lon/tz it
    could resolve, or None if no model is loaded, the call errored, or
    nothing useful came back at all. Deliberately NOT shared/imported
    from horary.py's own near-identical helpers, to avoid coupling this
    change to that already-stable, separately-tested module — the minor
    duplication is worth the isolation."""
    if llm_utils.get_llm() is None:
        return None
    try:
        answer = llm_utils.classify_sync(
            _FIELD_EXTRACTION_PROMPT.format(text=text), max_tokens=120, temperature=0.0,
        )
    except Exception:
        return None
    date = _parse_labeled_field("ДАТА", answer)
    time_ = _parse_labeled_field("ВРЕМЯ", answer)
    place = _parse_labeled_field("МЕСТО", answer)
    # Validate the model's own "ДАТА"/"ВРЕМЯ" lines actually landed in the
    # strict numeric format the prompt asked for, instead of trusting them
    # blindly — a real, reported failure on a weaker model (3B-class)
    # showed it answering something like "1976-июл-05" for "5 июля 1976
    # года" (converting the day/year but leaving the Russian month name in
    # place instead of finishing the conversion to a number), which then
    # crashed several calls downstream at `int(x) for x in
    # date_str.split("-")` with a raw, user-facing "invalid literal for
    # int()" error. A value that fails this check is treated exactly like
    # "нет" was already handled — dropped here so _fill_fields_from_text's
    # regex fallback (_find_date, which already handles Russian month
    # names like "5 июля 1976" correctly and always returns a clean
    # zero-padded ISO string or None) gets a chance to fill it instead,
    # rather than a half-converted value blocking that fallback from ever
    # running (_fill_fields_from_text only fills keys not already
    # present).
    if date and not re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", date):
        date = None
    if time_ and not re.match(r"^\d{1,2}:\d{2}$", time_):
        time_ = None
    result: Dict[str, str] = {}
    if date:
        result["date"] = date
    if time_:
        result["time"] = time_
    if place:
        resolved = _resolve_place_string(place)
        if resolved:
            lat, lon, tz = resolved
            result["lat"], result["lon"] = str(lat), str(lon)
            if tz:
                result["tz"] = tz
    return result or None


def _fill_fields_from_text(fields: Dict[str, str], text: str) -> None:
    """Shared free-text fallback used by both _extract_fields (single
    person) and _extract_two_person_fields (synastry's two independent
    text halves, see below) — mutates `fields` in place, only filling
    keys not already present, so explicit key=value input always takes
    priority regardless of which caller it came from. Pulled out of
    _extract_fields (unchanged logic, just no longer duplicated) so
    synastry's per-person field resolution can't silently drift from the
    single-person path's own coordinate-resolution order (explicit
    lat/lon > DMS/decimal coordinates found in the text > a bare city name
    via geonamescache, tz last, from whichever of the previous steps
    actually filled lat/lon)."""
    if not fields.get("date"):
        found = _find_date(text)
        if found:
            fields["date"] = found
    if not fields.get("time"):
        found = _find_time(text)
        if found:
            fields["time"] = found
    if not fields.get("lat") or not fields.get("lon"):
        lat, lon = _find_coordinates(text)
        if lat is not None:
            fields.setdefault("lat", str(lat))
        if lon is not None:
            fields.setdefault("lon", str(lon))
    if not fields.get("lat") or not fields.get("lon"):
        city = _lookup_city(text)
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


def _extract_fields(spec: str) -> Tuple[Dict[str, str], List[str]]:
    """Combines the strict key=value fast path with LLM-first free-text
    extraction (_extract_fields_llm), then falls back to the original
    regex-based extraction (_fill_fields_from_text) for anything still
    missing — which is everything, unchanged, if no model is loaded, so
    this is a pure addition with no regression risk when the LLM path is
    unavailable. Explicit key=value input always wins over both, exactly
    as before (both helpers only ever fill keys not already present).
    Finally auto-resolves tz from lat/lon if it still wasn't given
    explicitly. Returns (fields, missing_field_names).

    lat/lon/tz are merged from the LLM result as one atomic group, not
    field-by-field: if lat/lon are already present (from the explicit
    key=value fast path or anywhere else upstream), the LLM's own tz guess
    — which corresponds to ITS place reading, not necessarily the same
    place — must not be allowed to attach itself to those already-settled
    coordinates. Caught by a real test: explicit lat/lon for Moscow paired
    with a stubbed LLM answer naming Paris otherwise left the fields with
    Moscow's coordinates but Paris's timezone."""
    fields = _parse_spec(spec)
    llm_result = _extract_fields_llm(spec)
    if llm_result:
        for key in ("date", "time"):
            if llm_result.get(key):
                fields.setdefault(key, llm_result[key])
        if not fields.get("lat") and not fields.get("lon") and llm_result.get("lat") and llm_result.get("lon"):
            fields["lat"] = llm_result["lat"]
            fields["lon"] = llm_result["lon"]
            if llm_result.get("tz"):
                fields["tz"] = llm_result["tz"]
    _fill_fields_from_text(fields, spec)
    missing = [k for k in _REQUIRED_FIELDS if not fields.get(k)]
    return fields, missing


def _split_two_person_text(spec: str) -> Tuple[str, str]:
    """Synastry needs TWO independent sets of birth fields out of one
    free-text message — this splits the raw text into two halves, one per
    person, as a best-effort heuristic (same accepted-approximation
    spirit as _lookup_city's stem matching elsewhere in this module, not
    a real parser).

    Finds every recognizable date anywhere in the text (the same three
    formats _find_date checks, via finditer instead of search so ALL
    occurrences are found, not just the first), takes the first two by
    position, then splits between them at the clearest available
    separator, checked in priority order:
      1. The LAST newline strictly between the two dates — a real,
         common phrasing this needs to handle correctly: two people
         listed on separate lines ("Мужчина: 5 июля 1976 в 4:30 в
         Одессе\nЖенщина: 16 февраля 1977 в 16:46 в Днепропетровске"),
         found via real testing. A line break is an even stronger
         separator signal than a comma, so it's preferred whenever both
         happen to be present.
      2. The LAST comma strictly between the two dates — the other
         common phrasing this is built for ("Иван, 5 июля 1976 в 4:30 в
         Одессе, и Мария, 12 марта 1980 в 9:15 в Киеве") separates the
         two people with exactly that comma.
      3. If neither is present, the whitespace character closest to the
         plain midpoint between the two dates — NEVER the raw midpoint
         itself. A real, reproducible bug found via testing: falling
         back to a bare midpoint can land INSIDE a word (a multi-line
         message with no comma between the two dates split "Одесса"
         into "Одес"/"са", silently breaking that person's city lookup
         since neither fragment matches anything) — snapping to the
         nearest whitespace guarantees a word is never cut in half,
         while staying close to the same "split roughly in the middle"
         intent.

    Returns (spec, spec) unchanged if fewer than two dates are found —
    the caller's missing-fields check doesn't specifically catch "both
    halves resolved to the same person", but a synastry request with only
    one person's data in it isn't a coherent request in the first place,
    so there's no meaningfully better degenerate behavior to fall back to
    here."""
    matches = []
    for regex in (_DATE_ISO_RE, _DATE_DMY_NUM_RE, _DATE_RU_RE):
        matches.extend(regex.finditer(spec))
    matches.sort(key=lambda m: m.start())
    if len(matches) < 2:
        return spec, spec

    first, second = matches[0], matches[1]
    between = spec[first.end():second.start()]

    newline_pos = between.rfind("\n")
    comma_pos = between.rfind(",")
    if newline_pos != -1:
        split_point = first.end() + newline_pos + 1
    elif comma_pos != -1:
        split_point = first.end() + comma_pos + 1
    else:
        midpoint_offset = (second.start() - first.end()) // 2
        ws_positions = [m.start() for m in re.finditer(r"\s", between)]
        if ws_positions:
            closest = min(ws_positions, key=lambda p: abs(p - midpoint_offset))
            split_point = first.end() + closest + 1
        else:
            # No whitespace at all between the two dates either (the two
            # dates are directly adjacent with no separator whatsoever) —
            # nothing left to snap to; the raw midpoint is the least-bad
            # option remaining, same as before this fix.
            split_point = first.end() + midpoint_offset
    return spec[:split_point], spec[split_point:]


_LABEL_JUNK_WORDS = {"и", "а"}


def _extract_person_label(prefix_text: str) -> Optional[str]:
    """Best-effort extraction of a short label for one synastry person —
    their actual name if given, or a role word like "Мужчина"/"Женщина"
    — from whatever text immediately precedes their birth date in the
    message. Purely cosmetic: only used so the final answer refers to
    each person the way the USER did ("используй обозначения, которыми
    назвал их сам пользователь" — a real, reported preference: a query
    that said "Мужчина: ... / Женщина: ..." got back an answer that
    called them "Человек A"/"Человек B" instead, which read as needlessly
    generic) — never used for anything that affects the actual chart
    computation, so a wrong or missing guess here is a cosmetic
    imperfection, not a correctness risk the way a wrong date/coordinate
    guess would be.

    Looks at the last few words right before the date, strips trailing
    punctuation and a leading connector word ("и Мария" -> "Мария", since
    "и" precedes the SECOND person's own label, not part of it), and
    accepts the result only if it looks like a plausible label (starts
    with an uppercase letter, no digits, reasonable length) — returns
    None rather than a low-confidence guess otherwise, so the generic
    "Человек A"/"Человек B" fallback (_build_synastry_subjects) is always
    available.

    Known limitation, accepted rather than solved: in the flowing
    comma-separated phrasing ("Иван, 5 июля ... в Одессе, и Мария, 12
    марта ..."), the label for the SECOND person ("Мария") ends up in the
    tail of the FIRST person's own split half (see
    _split_two_person_text — the split point falls right after "и
    Мария,"), not at the head of the second half this function actually
    looks at — so this only reliably picks up labels in phrasings where
    each person's own label sits at the START of their own half
    (`"Мужчина: ...", "Иван - ..."`), not the flowing comma style. Good
    enough for the common structured case; the flowing style simply falls
    back to the generic label instead of a wrong one."""
    text = prefix_text.strip()
    if not text:
        return None
    # Only the CURRENT LINE matters, not any earlier preamble in the same
    # message — a real bug found via testing: "Проанализируй синастрию
    # карт:\nМужчина: " has "Мужчина" as the genuine label, but without
    # this line-restriction the last-3-words heuristic below picked up
    # "синастрию карт: Мужчина" instead (spanning across the newline into
    # the unrelated preamble line), which correctly failed the "starts
    # with an uppercase letter" check (starts with lowercase "синастрию")
    # and silently produced no label at all.
    last_line = text.rsplit("\n", 1)[-1]
    last_line = re.sub(r"[:,\-—\s]+$", "", last_line).strip()  # trailing separator right before the date itself
    if not last_line:
        return None
    words = last_line.split()[-3:]
    while words and words[0].lower() in _LABEL_JUNK_WORDS:
        words = words[1:]
    if not words:
        return None
    label = " ".join(words)
    if not label[:1].isupper() or any(ch.isdigit() for ch in label) or len(label) > 30:
        return None
    return label


def _extract_person_label_before_date(text_half: str) -> Optional[str]:
    """Runs _extract_person_label against whatever precedes the FIRST
    recognizable date within one person's own half of a synastry
    message."""
    matches = []
    for regex in (_DATE_ISO_RE, _DATE_DMY_NUM_RE, _DATE_RU_RE):
        m = regex.search(text_half)
        if m:
            matches.append(m)
    if not matches:
        return None
    earliest = min(matches, key=lambda m: m.start())
    return _extract_person_label(text_half[: earliest.start()])


def _extract_two_person_fields(
    spec: str, split_hint: Optional[Tuple[str, str]] = None
) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    """Synastry counterpart to _extract_fields: pulls two independent
    sets of birth fields ("a"/"b") out of one spec string.

    Fast path: explicit key=value pairs with an _a/_b suffix (date_a=...,
    time_a=..., lat_a=..., lon_a=..., tz_a=..., name_a=..., and the same
    with _b) — _parse_spec already parses arbitrary key=value pairs, so
    no change was needed there; this function just reads the suffixed
    keys back out.

    Fallback: free-text birth info for two people in one message, split
    into independent halves either by split_hint if one is given (see
    below), or otherwise by _split_two_person_text (see its own
    docstring for the heuristic and its limits) — each half is then
    resolved via the exact same _fill_fields_from_text single-person logic
    _extract_fields uses, entirely independently per half, so a field
    given explicitly for one person can never leak into the other's. A
    "name" is also filled in this way if not already given via key=value
    — see _extract_person_label for what it accepts and its known
    limitation.

    split_hint, if given, is (text_a, text_b) — pre-split halves supplied
    by an external caller instead of _split_two_person_text's own date-
    position heuristic. routes/chat.py passes one here as a fallback ONLY
    when the plain heuristic split above left required fields missing for
    either person: utils/intent.split_two_person_text_async runs a single,
    narrow LLM call whose only job is deciding which words of the
    original free text belong to which person (never reformatting a
    date/time/coordinate/city itself — see that function's own comment for
    why this narrow scope keeps it safe to use, unlike the project's
    already-rejected "let a small model reformat birth data" approach).
    Each half — heuristic or hinted — is parsed by the exact same
    deterministic regexes either way, so a bad or hallucinated hint just
    produces "still missing", never a silently wrong date/coordinate.

    Returns (fields_a, fields_b, missing), where `missing` lists which
    _REQUIRED_FIELDS ended up absent from EITHER person, prefixed "a:" or
    "b:" so _missing_synastry_fields_message can tell them apart —
    synastry needs both people's data complete before a chart can be
    built at all."""
    raw = _parse_spec(spec)
    text_a, text_b = split_hint if split_hint is not None else _split_two_person_text(spec)

    def _fields_for(suffix: str, text_half: str) -> Dict[str, str]:
        fields: Dict[str, str] = {}
        for key in ("date", "time", "lat", "lon", "tz", "name"):
            value = raw.get(f"{key}_{suffix}")
            if value:
                fields[key] = value
        _fill_fields_from_text(fields, text_half)
        if not fields.get("name"):
            label = _extract_person_label_before_date(text_half)
            if label:
                fields["name"] = label
        return fields

    fields_a = _fields_for("a", text_a)
    fields_b = _fields_for("b", text_b)

    missing = [
        f"{person}:{k}"
        for person, fields in (("a", fields_a), ("b", fields_b))
        for k in _REQUIRED_FIELDS
        if not fields.get(k)
    ]
    return fields_a, fields_b, missing


def _missing_synastry_fields_message(missing: List[str]) -> str:
    a_missing = [m.split(":", 1)[1] for m in missing if m.startswith("a:")]
    b_missing = [m.split(":", 1)[1] for m in missing if m.startswith("b:")]
    parts = []
    if a_missing:
        parts.append(f"для первого человека — {', '.join(a_missing)}")
    if b_missing:
        parts.append(f"для второго человека — {', '.join(b_missing)}")
    # The format example deliberately uses abstract placeholders
    # (<имя1>/<дата1>/...), NOT plausible-looking names like "Иван"/
    # "Мария" — a real, reproducible confusion found via testing: an
    # earlier version of this message used concrete example names, and
    # the follow-up LLM turning this tool result into a natural-language
    # answer mistook that illustrative EXAMPLE for actual partially-
    # recognized data ("Мы знаем что: Мужчина: Иван (или другие
    # данные)..."), instead of understanding both people's data was
    # simply missing. Abstract placeholders can't be confused with real
    # extracted values the same way.
    return (
        "Для синастрии (сравнения двух карт) не хватает данных: "
        + "; ".join(parts) + ". Нужны точная дата и время рождения и место "
        "(координаты) КАЖДОГО из двух человек — формат: "
        "'<имя1>, <дата1> в <время1> в <город1>, и <имя2>, <дата2> в "
        "<время2> в <город2>' (имена не обязательны). Часовой пояс "
        "определяется автоматически по координатам."
    )


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
    """Absolute ecliptic degree (0-360) of each of a chart's 12 house
    cusps, in house order (index 0 = house 1's cusp) — the input
    _house_of_degree needs to place an arbitrary point (e.g. a transiting
    planet) into THIS chart's houses, regardless of which chart the point
    itself came from.

    Uses `.abs_pos`, NOT `.position` — a real, previously-shipped bug
    found via testing (a user-reported synastry house mixup, tracked down
    by recomputing the exact test case in a sandbox): kerykeion's
    `.position` is the degree WITHIN the point's own sign (0-30, e.g.
    17.6 for something at 17.6° Libra), while `.abs_pos` is the true
    ecliptic longitude (0-360, e.g. 197.6 for that same point). Comparing
    `.position` values across cusps as if they were absolute degrees
    produces essentially arbitrary house assignments — cusps that are
    actually far apart around the circle can have coincidentally similar
    within-sign degrees, and the wrap-around arc math in _house_of_degree
    is meaningless over a value range that resets every ~30° instead of
    spanning the full circle once. Confirmed by direct inspection: this
    function used to return values like [14.88, 3.82, 28.11, 0.07, ...]
    (all under 30) for a chart whose cusps are actually spread across the
    full 0-360 range."""
    attrs = [
        "first_house", "second_house", "third_house", "fourth_house",
        "fifth_house", "sixth_house", "seventh_house", "eighth_house",
        "ninth_house", "tenth_house", "eleventh_house", "twelfth_house",
    ]
    return [getattr(subject, attr).abs_pos for attr in attrs]


def _house_of_degree(cusp_degrees: List[float], degree: float) -> int:
    """Which house (1-12) an arbitrary absolute ecliptic degree falls
    into, given a chart's 12 house cusp degrees (house order, index 0 =
    house 1). `degree` and every value in `cusp_degrees` MUST be absolute
    ecliptic longitude (0-360, kerykeion's `.abs_pos`) — NOT the
    within-sign `.position` (0-30) a point also carries; mixing the two up
    was a real, shipped bug (see _house_cusp_degrees' docstring). This is
    the standard transit-astrology convention for
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
            natal_house = _house_of_degree(natal_cusps, point.abs_pos)
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
                    "nature_ru": _aspect_nature_ru(a.aspect),
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
    other_point_label: Optional[Callable[[str], str]] = None,
    reference_house_label: Optional[Callable[[int], str]] = None,
    own_house_label: Optional[Callable[[int], str]] = None,
    query_prefix: str = "транзитный",
    query_aspect_prefix: str = "транзит",
    force_include_labels: Tuple[str, ...] = ("Солнце", "Луна"),
    kind: str = "transit_planet",
    include_angles: bool = False,
) -> List[Dict]:
    """Two-chart counterpart to get_planet_profiles — one profile per
    significant point of `other_subject` (a transit moment's moving
    planets, or — now that synastry is this function's second consumer,
    see get_synastry_profiles — a second person's own natal points), each
    bundling:
      - its placement, with the house computed against
        `reference_subject`'s OWN cusps via _house_of_degree — the
        standard transit-astrology convention (a transiting planet's
        house is which of YOUR houses it's currently moving through, not
        some house system computed fresh for the transit moment itself),
        and the same overlay convention synastry uses (person B's planet
        falls into whichever of person A's houses its degree lands in);
      - its own cross-chart aspects to `reference_subject`'s points, each
        carrying that reference point's sign/house too — the same "other
        side of the aspect" context get_planet_profiles already gives for
        single-chart aspects, needed for the same reason (so the digest
        step can judge how strong/relevant the aspecting relationship is,
        not just that it exists).

    This is the shared layer run_transit was refactored onto first (see
    _format_transit_text, which now also uses _house_of_degree directly
    for its own raw-data listing, and routes/chat.py, which runs transit
    answers through the same digest/sectioned-answer pipeline natal
    charts have). get_synastry_profiles calls this function TWICE — once
    per direction (person A's points aspecting B's chart, then B's points
    aspecting A's) — since a synastry reading is inherently bidirectional
    in a way a transit reading isn't (a moment doesn't have "its own"
    identity worth profiling the way a second person's chart does).

    other_point_label/reference_house_label/query_prefix/
    query_aspect_prefix/force_include_labels exist ONLY so
    get_synastry_profiles can relabel the generic "транзитный .../
    натальный N дом" phrasing this function was originally written with
    into person-specific phrasing ("Венера ♀ человека Б" / "7 дом
    человека А") — every default below reproduces run_transit's exact
    original wording byte-for-byte, so transit behavior is unchanged
    unless a caller explicitly overrides them.

    own_house_label is None by default (transit's behavior, unchanged):
    a transiting planet doesn't meaningfully have "its own" house the way
    a person does (see _ACTIVE_POINTS_TRANSIT's own comment on why a
    transiting Vertex isn't a standard reading either), so nothing extra
    is shown there. get_synastry_profiles passes a label callable here —
    a real, reported point of confusion (a generated synastry answer
    conflated one person's overlay house with the other's, and separately
    the user asked to see a planet's OWN natal house alongside the
    overlay for easier reading) — so each profile line can show BOTH:
    other_subject's own natal house of that point (straight from
    kerykeion's own per-chart computation, not the overlay math at all)
    together with the overlay house computed against reference_subject's
    cusps.

    other_active_points defaults to _ACTIVE_POINTS_TRANSIT (fixed stars
    and the Vertex excluded from the moving side — see that list's own
    comment for why neither is meaningfully "transiting"; get_synastry_
    profiles passes _ACTIVE_POINTS_NATAL instead, since a second person's
    Vertex/Chiron/Lilith/Part of Fortune ARE conventionally read in
    synastry — nothing about them is transiting-only there). Scoring
    mirrors get_planet_profiles: angularity (by the reference-chart house
    computed above, not other_subject's own), aspect count, retrograde.
    No fixed-star/Pars-Fortunae force-include tier here (that's a natal-
    only concept, see get_planet_profiles) — only force_include_labels
    (default Sun/Moon) are force-included, since they're the two bodies
    every mainstream transit OR synastry reading treats as significant
    regardless of house/aspect count.

    include_angles (default False, unchanged behavior for every existing
    caller — transit/progression/synastry) additionally profiles
    other_subject's own Ascendant/MC (_ANGLE_ATTRS) the same way its
    planets are profiled. Added for lunar/solar returns: unlike a transit
    moment's Ascendant (just reflects time-of-day, not a real "transit" —
    see _ACTIVE_POINTS_TRANSIT's own comment), a RETURN chart's own
    Ascendant is a real, independently meaningful point — solar_return_
    methodology.txt flags the return's own Ascendant/its ruler as the
    single most important point in a solar return, which would otherwise
    never surface as a profiled fact at all (aspects TO it were already
    computed via _ASPECT_ACTIVE_POINTS, but no profile entry ever
    summarized its own sign/house the way a planet's does)."""
    if other_active_points is None:
        other_active_points = _ACTIVE_POINTS_TRANSIT
    if other_point_label is None:
        other_point_label = lambda label_ru: f"транзитный {label_ru}"  # noqa: E731
    if reference_house_label is None:
        reference_house_label = lambda house_num: f"натальный {house_num} дом"  # noqa: E731

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

    attrs_to_profile = _PLANET_ATTRS + (_ANGLE_ATTRS if include_angles else [])

    profiles: List[Dict] = []
    for label, attr in attrs_to_profile:
        kery_name = attr_to_kerykeion_name(attr)
        if kery_name not in other_active_points:
            continue
        point = getattr(other_subject, attr, None)
        if point is None:
            continue

        natal_house = _house_of_degree(natal_cusps, point.abs_pos)
        own_house = _house_number(getattr(point, "house", None))
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
                    "nature_ru": _aspect_nature_ru(a.aspect),
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
        own_house_text = (
            f", {own_house_label(own_house)}" if own_house_label is not None and own_house else ""
        )
        house_text = f", {reference_house_label(natal_house)}" if natal_house else ""
        label_text = other_point_label(label_ru)

        queries = [f"{query_prefix} {label} {sign_prep}"] + (
            [f"{query_prefix} {label} в {natal_house} доме"] if natal_house else []
        )
        for asp in own_aspects:
            queries.append(f"{query_aspect_prefix} {asp['aspect_ru']} {label_ru} и {asp['other_label']}")

        profiles.append(
            {
                "kind": kind,
                "label": label_ru,
                "text": f"{label_text} {sign_prep}{own_house_text}{house_text}{retro_text}",
                "aspects": own_aspects,
                "stars": [],
                "queries": queries,
                "score": score,
                "own_house": own_house,
                "overlay_house": natal_house,
                "force_include": label in force_include_labels,
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


def run_natal_and_subject(spec: str) -> Tuple[str, Optional[Any], None, None]:
    """Full logic for the natal-chart tool, returning the built subject
    alongside its text report — added so routes/chat.py's chart-drawing
    step can reuse the SAME subject instead of rebuilding it a second
    time via a separate get_natal_chart_subject() call (that used to mean
    every natal-chart reply parsed the spec and ran the full ephemeris/
    fixed-star computation twice; see this function's own callers for the
    "_and_subject" convention already established by
    electional.run_electional_chart_and_subject and
    rectification._run_rectification_trutine_full — same idea, applied
    here for the same reason: never do twice what one call can hand to
    both a text report and a chart image).

    Returns (text, subject, None, None) — the trailing None, None keep
    this the same 4-tuple shape as every other "cheap" technique's
    _and_subject sibling (subject, second_subject_or_overlay,
    highlight_house), even though natal charts have no second subject or
    highlighted house — see routes/chat.py's _SIMPLE_AND_SUBJECT_FUNCS
    for why a uniform shape across all of them keeps that dispatch code
    a single generic branch instead of one bespoke branch per technique."""
    fields, missing = _extract_fields(spec)
    if missing:
        return _missing_fields_message(missing, fields), None, None, None
    try:
        subject = _build_subject(fields, name=fields.get("name") or "Subject")
    except Exception as e:
        return f"Не удалось построить натальную карту — некорректные данные ({e}).", None, None, None
    try:
        return _format_natal_text(subject), subject, None, None
    except Exception as e:
        return f"Ошибка при расчёте натальной карты: {e}", None, None, None


def run_natal(spec: str) -> str:
    """Tool entry point (utils.tools.TOOL_REGISTRY["astro_natal_chart"]).
    Never raises — any failure becomes a plain-text explanation instead,
    since the caller feeds the return value straight into a follow-up
    generation, not into error-handling code. Thin wrapper — see
    run_natal_and_subject for the full logic."""
    return run_natal_and_subject(spec)[0]


# Recognized as "no specific non-current moment given" — i.e. right now —
# by _parse_moment below. The tool's own TOOL_REGISTRY description tells
# the routing model to add ";moment=YYYY-MM-DDTHH:MM" ONLY for a moment
# other than right now, and to default to "right now" otherwise — but a
# real, reported failure on a weak 3B routing model showed it adding
# 'moment=текущий момент' anyway when the user's own message said "на
# текущий момент" (for the current moment), presumably echoing the salient
# phrase into the field it thought it should fill rather than actually
# omitting it. That produced a raw Python ValueError surfacing straight
# into the tool's reply text. Widening the "this means now" set to include
# the phrasings a model (or a person typing it by hand) is likely to use
# instead of just "now"/"сейчас" fixes that class of failure without
# depending on any particular model's instruction-following strength.
_MOMENT_NOW_SYNONYMS = (
    "", "now", "сейчас", "текущий момент", "текущий", "на текущий момент",
    "current", "current moment", "today", "сегодня", "right now",
)


def _parse_moment(moment: str) -> Tuple[Optional[datetime], Optional[str]]:
    """Shared by run_transit/run_progression/run_lunar_return/
    run_solar_return/run_profection's identical "moment" field handling.
    Returns (datetime, None) on success, or (None, error_message) only for
    a value that isn't a recognized "now" synonym (see
    _MOMENT_NOW_SYNONYMS) AND doesn't parse under any of the tolerated
    formats — the strict 'YYYY-MM-DDTHH:MM' the tool description asks for,
    plus a couple of near-misses (space instead of "T", or date only) a
    model or a person might plausibly produce instead."""
    normalized = moment.strip().lower()
    if normalized in _MOMENT_NOW_SYNONYMS:
        return datetime.now(), None
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(moment.strip(), fmt), None
        except ValueError:
            continue
    return None, (
        f"Не удалось разобрать момент времени '{moment}'; "
        "ожидается формат ГГГГ-ММ-ДДTЧЧ:ММ или 'now'."
    )


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

    dt, error = _parse_moment(fields.get("moment") or "now")
    if error:
        return None, None, error

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


def run_transit_and_subject(spec: str) -> Tuple[str, Optional[Any], Optional[Any], None]:
    """See run_natal_and_subject's docstring for the "_and_subject"
    convention this follows. Returns (text, natal, transit_subject, None)."""
    fields, missing = _extract_fields(spec)
    if missing:
        return _missing_fields_message(missing, fields), None, None, None
    natal, transit_subject, error = _build_transit_subjects(fields, name=fields.get("name") or "Subject")
    if error:
        return error, None, None, None
    try:
        return _format_transit_text(natal, transit_subject), natal, transit_subject, None
    except Exception as e:
        return f"Ошибка при расчёте транзитов: {e}", None, None, None


def run_transit(spec: str) -> str:
    """Tool entry point (utils.tools.TOOL_REGISTRY["astro_transit_chart"]).
    Thin wrapper — see run_transit_and_subject for the full logic."""
    return run_transit_and_subject(spec)[0]


def get_transit_profiles(spec: str, top_n: int = 12) -> List[Dict]:
    """Transit counterpart to get_planet_profiles, for routes/chat.py's
    digest step — rebuilds both subjects from `spec` (same duplication-of-
    computation pattern get_planet_profiles already has relative to
    run_natal: the tool call already computed a subject once for the raw
    chart text, and this recomputes it independently for profiling; kept
    consistent with that existing precedent rather than plumbing the
    already-built subject through routes/chat.py) and hands them to
    get_dual_chart_profiles.

    top_n raised from 9 to 12 after real user feedback that a transit
    answer felt thinner and less specific than it could be — the same
    "more material available to the digest step produces a more thorough
    answer" lever already used for synastry (top_n_each 7 -> 9). A
    real chart tested here has 15 active transiting points, so 12 still
    leaves real room for the score-based ranking to matter, while giving
    the digest/final-answer steps enough profiles to actually cover
    "several simultaneous transits" (see transit_methodology.txt point 5)
    instead of just the top 2-3.

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


# --- secondary progressions ------------------------------------------------
#
# "Day for a year": the standard secondary-progression convention treats
# each day AFTER birth as symbolically standing for one YEAR of life, so a
# 50-year-old's progressed chart for today is cast for a moment only ~50
# DAYS after their actual birth — not for today's real calendar date at
# all. This is why _format_progression_text below explicitly states both
# the real target date/age and the internal symbolic progressed date: the
# raw numbers otherwise look like an obvious bug (a chart "for 2026"
# showing planetary positions from a date in 1976) if not labeled clearly.
#
# Structurally this reuses the exact same natal-vs-"other moment" overlay
# machinery run_transit/get_transit_profiles already established
# (get_dual_chart_profiles, _house_of_degree against the NATAL cusps) —
# a progressed chart is read the same way a transit is (which of the
# natal houses is now activated), just for a differently-computed moment.
# The one thing that's genuinely different is the moment's own timescale:
# transit's moment is real, exact "now"; progression's is symbolic and
# computed from age, never an independently-meaningful calendar date on
# its own.


def _secondary_progressed_datetime(birth_dt: datetime, target_dt: datetime) -> datetime:
    """The core "day for a year" formula: elapsed real time since birth,
    expressed in years (using the standard 365.25-day tropical-year
    approximation), becomes the same number of DAYS added to the birth
    moment. Fractional days are kept (not rounded) for precision — over a
    50-year span a whole-day rounding error would shift the progressed
    Moon (the fastest-moving progressed point, ~1° per progressed day) by
    a measurable fraction of a degree."""
    elapsed_days = (target_dt - birth_dt).total_seconds() / 86400.0
    age_years = elapsed_days / 365.25
    return birth_dt + timedelta(days=age_years)


def _build_progression_subjects(
    fields: Dict[str, str], name: str
) -> Tuple[Optional[Any], Optional[Any], Optional[datetime], Optional[float], Optional[str]]:
    """Shared by run_progression and get_progression_profiles — mirrors
    _build_transit_subjects' role and shape exactly, just computing the
    progressed moment via _secondary_progressed_datetime instead of using
    a real target moment directly. Returns (natal, progressed_subject,
    target_dt, age_years, None) on success, or (None, None, None, None,
    error_message) on failure — same never-raise convention as the rest of
    this module.

    Reuses the "moment" field convention run_transit already established
    (a key=value ISO moment, or "now"/omitted) — it means the same thing
    here: the real date/time to compute progressions FOR, not the
    internal symbolic progressed date itself (which this function derives,
    not accepts as input). Progressions keep the birth location/timezone
    — no moment_lat/moment_lon/moment_tz override here, unlike transit,
    since a progressed chart isn't conventionally relocated the way a
    transit reading sometimes is; relocated progressions are a real, more
    specialized technique that can be added later if wanted."""
    try:
        natal = _build_subject(fields, name=name)
    except Exception as e:
        return None, None, None, None, f"Не удалось построить натальную карту — некорректные данные ({e})."

    target_dt, error = _parse_moment(fields.get("moment") or "now")
    if error:
        return None, None, None, None, error

    try:
        date_str = fields["date"]
        time_str = fields.get("time", "12:00")
        year, month, day = (int(x) for x in date_str.split("-"))
        hour, minute = (int(x) for x in time_str.split(":"))
        birth_dt = datetime(year, month, day, hour, minute)
    except Exception as e:
        return None, None, None, None, f"Не удалось разобрать дату рождения ({e})."

    age_years = (target_dt - birth_dt).total_seconds() / 86400.0 / 365.25
    progressed_dt = _secondary_progressed_datetime(birth_dt, target_dt)

    try:
        from kerykeion.astrological_subject_factory import AstrologicalSubjectFactory

        progressed_subject = AstrologicalSubjectFactory.from_birth_data(
            name="Progressed",
            year=progressed_dt.year, month=progressed_dt.month, day=progressed_dt.day,
            hour=progressed_dt.hour, minute=progressed_dt.minute,
            lat=float(fields["lat"]), lng=float(fields["lon"]), tz_str=fields["tz"],
            online=False,
            # Same active-point set as transit (see _ACTIVE_POINTS_TRANSIT's
            # own comment: no fixed stars, no Vertex) — a reasonable first
            # scope, not a principled exclusion specific to progressions;
            # revisit if a progressed Vertex/fixed-star reading is wanted
            # later.
            active_points=_ACTIVE_POINTS_TRANSIT,
        )
    except Exception as e:
        return None, None, None, None, f"Не удалось рассчитать прогрессивные положения планет: {e}"

    return natal, progressed_subject, target_dt, age_years, None


def _format_progression_text(natal, progressed, target_dt: datetime, age_years: float) -> str:
    from kerykeion import AspectsFactory

    natal_cusps = _house_cusp_degrees(natal)

    lines = [
        f"Вторичные прогрессии (secondary progressions) для {natal.name} на "
        f"{target_dt.year:04d}-{target_dt.month:02d}-{target_dt.day:02d} "
        f"(возраст {age_years:.1f} лет) — по принципу «день за год» это "
        f"символически представлено датой "
        f"{progressed.year:04d}-{progressed.month:02d}-{progressed.day:02d} "
        f"{progressed.hour:02d}:{progressed.minute:02d} — НЕ реальная дата "
        f"события, а техническая точка расчёта; описывай эти планеты как "
        f"текущее прогрессивное состояние человека, а не как что-то, "
        f"происходящее в этот прогрессивный день буквально.",
        "Прогрессивные положения планет (дом — натальный, т.е. дом "
        "натальной карты, в котором сейчас находится прогрессивная планета):",
    ]
    for label, attr in _PLANET_ATTRS:
        point = getattr(progressed, attr, None)
        if point is not None:
            natal_house = _house_of_degree(natal_cusps, point.abs_pos)
            lines.append("  " + _format_point_line(label, point, house_num_override=natal_house))

    aspects = AspectsFactory.dual_chart_aspects(
        natal, progressed, active_points=_ASPECT_ACTIVE_POINTS, active_aspects=_ALL_ASPECTS,
    ).aspects
    lines.append("Прогрессивные аспекты к натальной карте:")
    shown = 0
    for a in aspects:
        progressing, to_natal = (
            (a.p1_name, a.p2_name) if a.p1_owner == progressed.name else (a.p2_name, a.p1_name)
        )
        if progressing in ("Ascendant", "Medium_Coeli"):
            continue
        lines.append(
            f"  прогрессивный {_point_ru(progressing)} — {_aspect_ru(a.aspect)} — "
            f"натальный {_point_ru(to_natal)} "
            f"(орбис {a.orbit:.1f}°, {_movement_ru(a.aspect_movement)})"
        )
        shown += 1
    if not shown:
        lines.append("  (нет аспектов в пределах орбиса)")
    return "\n".join(lines)


def run_progression_and_subject(spec: str) -> Tuple[str, Optional[Any], Optional[Any], None]:
    """See run_natal_and_subject's docstring for the "_and_subject"
    convention. Returns (text, natal, progressed, None)."""
    fields, missing = _extract_fields(spec)
    if missing:
        return _missing_fields_message(missing, fields), None, None, None
    natal, progressed, target_dt, age_years, error = _build_progression_subjects(
        fields, name=fields.get("name") or "Subject"
    )
    if error:
        return error, None, None, None
    try:
        return _format_progression_text(natal, progressed, target_dt, age_years), natal, progressed, None
    except Exception as e:
        return f"Ошибка при расчёте прогрессий: {e}", None, None, None


def run_progression(spec: str) -> str:
    """Tool entry point (utils.tools.TOOL_REGISTRY["astro_progression_chart"]).
    Thin wrapper — see run_progression_and_subject for the full logic."""
    return run_progression_and_subject(spec)[0]


def get_progression_profiles(spec: str, top_n: int = 12) -> List[Dict]:
    """Progression counterpart to get_transit_profiles, for routes/chat.py's
    digest step — same duplication-of-computation pattern relative to
    run_progression the other get_*_profiles functions already have
    relative to their own run_*() counterparts.

    Returns [] (not an error) if fields are missing or subject-building
    fails — same "no profiles available, caller falls back" contract
    every other get_*_profiles function already has."""
    fields, missing = _extract_fields(spec)
    if missing:
        return []
    natal, progressed, _target_dt, _age_years, error = _build_progression_subjects(
        fields, name=fields.get("name") or "Subject"
    )
    if error:
        return []
    try:
        return get_dual_chart_profiles(
            natal, progressed, top_n=top_n,
            other_point_label=lambda lr: f"прогрессивный {lr}",
            reference_house_label=lambda h: f"натальный {h} дом",
            query_prefix="прогрессивный",
            query_aspect_prefix="прогрессия",
            kind="progression_planet",
        )
    except Exception:
        return []


def _format_synastry_text(person_a, person_b) -> str:
    """Raw chart text for synastry: each person's OWN placements (sign,
    house, retrograde — from their own chart, exactly like a natal
    reading — a synastry reading always starts from "who is each person
    on their own", not just the cross-aspects between them), then every
    cross-chart aspect between the two in one pass (kerykeion's
    dual_chart_aspects already returns the full bidirectional set — A's
    planets aspecting B's points AND B's planets aspecting A's, in a
    single symmetric list — so unlike the profile step below, only one
    aspect computation is needed here, not two)."""
    from kerykeion import AspectsFactory

    lines = [
        f"Синастрия (сравнение натальных карт) {person_a.name} и {person_b.name}.",
        f"Собственные положения планет — {person_a.name}:",
    ]
    for label, attr in _PLANET_ATTRS:
        point = getattr(person_a, attr, None)
        if point is not None:
            lines.append("  " + _format_point_line(label, point))
    lines.append(f"Собственные положения планет — {person_b.name}:")
    for label, attr in _PLANET_ATTRS:
        point = getattr(person_b, attr, None)
        if point is not None:
            lines.append("  " + _format_point_line(label, point))

    aspects = AspectsFactory.dual_chart_aspects(
        person_a, person_b, active_points=_ASPECT_ACTIVE_POINTS, active_aspects=_ALL_ASPECTS,
    ).aspects
    lines.append(f"Межкарточные (синастрические) аспекты между {person_a.name} и {person_b.name}:")
    if aspects:
        for a in aspects:
            p1_owner_name = person_a.name if a.p1_owner == person_a.name else person_b.name
            p2_owner_name = person_a.name if a.p2_owner == person_a.name else person_b.name
            lines.append(
                f"  {_point_ru(a.p1_name)} ({p1_owner_name}) — {_aspect_ru(a.aspect)} — "
                f"{_point_ru(a.p2_name)} ({p2_owner_name}) "
                f"(орбис {a.orbit:.1f}°, {_movement_ru(a.aspect_movement)})"
            )
    else:
        lines.append("  (нет аспектов в пределах орбиса)")
    return "\n".join(lines)


def _build_synastry_subjects(
    fields_a: Dict[str, str], fields_b: Dict[str, str]
) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    """Builds both people's own natal subjects for synastry — the full
    _ACTIVE_POINTS_NATAL set for each side (_build_subject's own default),
    since Vertex/Chiron/Lilith/Part of Fortune ARE conventionally read in
    synastry, unlike transit's narrower moving-body set. Fixed stars are
    NOT part of the cross-chart comparison here — _find_star_conjunctions
    is a natal-only, single-chart mechanism, not reused across two charts
    (a star sitting on person A's Sun doesn't become more or less
    meaningful because of anything in person B's chart). Returns
    (person_a, person_b, None) on success, or (None, None, error_message)
    on failure — same never-raise convention _build_transit_subjects
    already uses."""
    name_a = fields_a.get("name") or "Человек A"
    name_b = fields_b.get("name") or "Человек B"
    try:
        person_a = _build_subject(fields_a, name=name_a)
        person_b = _build_subject(fields_b, name=name_b)
    except Exception as e:
        return None, None, f"Не удалось построить карты для синастрии — некорректные данные ({e})."
    return person_a, person_b, None


def run_synastry_and_subject(
    spec: str, split_hint: Optional[Tuple[str, str]] = None
) -> Tuple[str, Optional[Any], Optional[Any]]:
    """See run_natal_and_subject's docstring for the "_and_subject"
    convention — this one predates that name (routes/chat.py used to call
    _extract_two_person_fields + _build_synastry_subjects a second time
    itself, right after this same computation, specifically so the drawn
    chart's person-A/person-B split matched split_hint; that second build
    is now unnecessary since this function hands both subjects back
    directly). Returns (text, person_a, person_b)."""
    fields_a, fields_b, missing = _extract_two_person_fields(spec, split_hint=split_hint)
    if missing:
        return _missing_synastry_fields_message(missing), None, None
    person_a, person_b, error = _build_synastry_subjects(fields_a, fields_b)
    if error:
        return error, None, None
    try:
        return _format_synastry_text(person_a, person_b), person_a, person_b
    except Exception as e:
        return f"Ошибка при расчёте синастрии: {e}", None, None


def run_synastry(spec: str, split_hint: Optional[Tuple[str, str]] = None) -> str:
    """Tool entry point (utils.tools.TOOL_REGISTRY["astro_synastry_chart"]).

    split_hint is optional and only ever passed by routes/chat.py, never by
    the generic tool-dispatch mechanism (utils.tools.TOOL_REGISTRY callers
    invoke run() with a single positional spec string) — see
    _extract_two_person_fields' own docstring for what it is and why it's
    safe (segmentation only, never reformatting a date/time/coordinate).
    Thin wrapper — see run_synastry_and_subject for the full logic."""
    return run_synastry_and_subject(spec, split_hint=split_hint)[0]


def synastry_fields_missing(spec: str) -> List[str]:
    """Cheap, side-effect-free check of whether the deterministic
    heuristic split/extraction already has everything needed for both
    people — used by routes/chat.py to decide whether the LLM segmentation
    fallback (utils.intent.split_two_person_text_async) is even worth
    calling. Returns the same "a:field"/"b:field"-prefixed list
    _extract_two_person_fields itself returns (empty = nothing missing)."""
    _, _, missing = _extract_two_person_fields(spec)
    return missing


def get_synastry_profiles(
    spec: str, top_n_each: int = 9, split_hint: Optional[Tuple[str, str]] = None
) -> Tuple[List[Dict], List[Dict], str, str]:
    """Synastry counterpart to get_planet_profiles/get_transit_profiles,
    for routes/chat.py's digest step. Rebuilds both subjects independently
    from `spec` (same accepted duplication-of-computation pattern the
    other two get_*_profiles functions already have relative to their
    run_*() counterparts), then calls get_dual_chart_profiles TWICE — once
    per direction — since synastry is inherently bidirectional in a way
    transit isn't: person A's planets overlaid onto person B's houses are
    a genuinely different (and equally significant) set of facts from
    person B's planets overlaid onto person A's, not just the same
    information read backwards.

    other_point_label/reference_house_label/query_prefix are overridden
    (relative to get_dual_chart_profiles' transit-oriented defaults) to
    name each actual person instead of the generic "транзитный"/
    "натальный N дом" phrasing, since two real people's names (or the
    "Человек A"/"Человек B" fallback labels) are what the digest/final-
    answer prompts need to attribute each placement and aspect to the
    right person.

    Returns (profiles_a, profiles_b, name_a, name_b) — profiles_a is
    person A's own points (house computed against person B's cusps),
    profiles_b is person B's own points (house computed against person
    A's cusps). Returns ([], [], "", "") (never raises) if fields are
    missing or subject-building fails — same "no profiles available,
    caller falls back" contract the other two get_*_profiles functions
    already have."""
    fields_a, fields_b, missing = _extract_two_person_fields(spec, split_hint=split_hint)
    if missing:
        return [], [], "", ""
    person_a, person_b, error = _build_synastry_subjects(fields_a, fields_b)
    if error:
        return [], [], "", ""

    name_a, name_b = person_a.name, person_b.name
    try:
        profiles_a = get_dual_chart_profiles(
            person_b, person_a,
            other_active_points=_ACTIVE_POINTS_NATAL, top_n=top_n_each,
            other_point_label=lambda lr, n=name_a: f"{lr} ({n})",
            reference_house_label=lambda h, n=name_b: f"{h} дом у {n}",
            # Alongside the overlay house (person A's point read against
            # person B's cusps, above), also show person A's OWN natal
            # house of that same point — a real, user-requested addition
            # ("интереснее анализировать планету не только в своем доме,
            # но и в доме владельца первой карты") that also happens to
            # make the overlay reading much less ambiguous to a reader (or
            # to the interpreting model): "своя 5 дом" vs "2 дом у Женщина"
            # are visibly two different things, rather than a single bare
            # house number that's easy to misattribute to the wrong chart.
            own_house_label=lambda h, n=name_a: f"свой {h} дом ({n})",
            query_prefix=f"синастрия {name_a}",
            query_aspect_prefix="синастрия",
            kind="synastry_planet",
        )
        profiles_b = get_dual_chart_profiles(
            person_a, person_b,
            other_active_points=_ACTIVE_POINTS_NATAL, top_n=top_n_each,
            other_point_label=lambda lr, n=name_b: f"{lr} ({n})",
            reference_house_label=lambda h, n=name_a: f"{h} дом у {n}",
            own_house_label=lambda h, n=name_b: f"свой {h} дом ({n})",
            query_prefix=f"синастрия {name_b}",
            query_aspect_prefix="синастрия",
            kind="synastry_planet",
        )
    except Exception:
        return [], [], name_a, name_b
    return profiles_a, profiles_b, name_a, name_b


# --- solar arc directions ---------------------------------------------------
#
# Solar arc directions move EVERY natal point by the same arc (the angular
# distance the secondary-progressed Sun has moved from its own natal
# position — reusing _build_progression_subjects/_secondary_progressed_
# datetime rather than recomputing "day for a year" separately), unlike
# secondary progressions where different points move at their own real
# speeds. This is the key thing direction_methodology.txt calls out as
# specific to this technique, and the reason the implementation below
# can't just reuse get_dual_chart_profiles/AspectsFactory the way
# progression did: there is no independent kerykeion chart for a
# "directed" moment at all (no real date/time/place a directed Mars
# actually occupies) — it's a synthetic set of points obtained by adding
# one arc to every natal degree, so AspectsFactory.dual_chart_aspects
# (which requires two real, independently-built subjects) simply isn't
# usable here. Aspect matching below is done directly against
# _ASPECT_ANGLES instead.
_ZODIAC_SIGN_CODES = list(_SIGN_NAMES_RU.keys())

# Standard aspect angles (degrees) — needed only for directions' custom
# aspect matching (see above); every other technique in this module gets
# its aspects from kerykeion's own AspectsFactory, which doesn't need this.
_ASPECT_ANGLES = {
    "conjunction": 0.0, "opposition": 180.0, "trine": 120.0, "square": 90.0,
    "sextile": 60.0, "semi-sextile": 30.0, "semi-square": 45.0,
    "quintile": 72.0, "sesquiquadrate": 135.0, "biquintile": 144.0,
    "quincunx": 150.0,
}


def _sign_from_abs_pos(abs_pos: float) -> Tuple[str, float]:
    """Absolute ecliptic degree (0-360, kerykeion's own convention — see
    _house_cusp_degrees' docstring for why this matters and not
    the within-sign 0-30 `.position`) -> (sign code, within-sign degree),
    e.g. 197.6 -> ("Lib", 17.6). _ZODIAC_SIGN_CODES is in the same
    zodiacal order _SIGN_NAMES_RU already uses (Aries first, Pisces
    last), so a plain 30°-wide bucket lookup is enough — no wraparound
    edge case, since `% 360.0` already normalizes the input."""
    d = abs_pos % 360.0
    idx = int(d // 30.0)
    return _ZODIAC_SIGN_CODES[idx], d - idx * 30.0


def _angular_separation(deg_a: float, deg_b: float) -> float:
    """Shortest angular distance (0-180°) between two absolute ecliptic
    degrees — the input every aspect-angle comparison below needs."""
    diff = abs(deg_a - deg_b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def _direction_aspect_movement(base_abs_pos: float, target_abs_pos: float, aspect_angle: float) -> str:
    """Directions have no real "applying/separating" motion the way a
    transit does (nothing is actually moving at the moment being asked
    about) — but the solar arc itself grows monotonically with age
    (~0.9856°/year, matching the progressed Sun's own mean motion), so
    whether a given direction is still tightening or has already passed
    exact IS a well-defined, real fact about it. Nudging the arc forward
    by a small step and checking whether the resulting orb shrinks or
    grows reuses the exact same "Applying"/"Separating" literals
    _ASPECT_MOVEMENT_RU already knows how to render, so this plugs
    straight into the same _movement_ru/_format_aspect_line machinery as
    every other technique."""
    def _diff(pos: float) -> float:
        return abs(_angular_separation(pos, target_abs_pos) - aspect_angle)

    current = _diff(base_abs_pos)
    nudged = _diff((base_abs_pos + 0.01) % 360.0)
    if nudged < current - 1e-9:
        return "Applying"
    if nudged > current + 1e-9:
        return "Separating"
    return "Static"


def _build_direction_subjects(
    fields: Dict[str, str], name: str
) -> Tuple[
    Optional[Any], Optional[Dict[str, Any]], Optional[datetime], Optional[float], Optional[float], Optional[str]
]:
    """Shared by run_direction and get_direction_profiles — mirrors
    _build_transit_subjects/_build_progression_subjects' role and shape,
    but builds a plain dict of synthetic "directed point" objects
    (kerykeion point-name literal -> a SimpleNamespace exposing .sign/
    .position/.abs_pos/.retrograde, everything _format_point_line and the
    aspect-matching helpers above need) instead of a second real kerykeion
    subject — see this section's module comment for why no real subject
    exists to build here.

    Returns (natal, directed_points, target_dt, age_years, arc_degrees,
    None) on success, or (None, None, None, None, None, error_message) on
    failure — same never-raise convention as the rest of this module."""
    natal, progressed, target_dt, age_years, error = _build_progression_subjects(fields, name=name)
    if error:
        return None, None, None, None, None, error

    # The solar arc: how far the progressed Sun has moved from its own
    # natal position — the single number every natal point below gets
    # shifted by identically (see this section's module comment for why
    # that "same arc for every point" behavior is what defines this
    # technique, unlike progression's per-point speeds).
    arc_degrees = (progressed.sun.abs_pos - natal.sun.abs_pos) % 360.0

    directed_points: Dict[str, Any] = {}
    for label, attr in _PLANET_ATTRS + _ANGLE_ATTRS:
        natal_point = getattr(natal, attr, None)
        if natal_point is None:
            continue
        new_abs = (natal_point.abs_pos + arc_degrees) % 360.0
        sign_code, within_sign = _sign_from_abs_pos(new_abs)
        kery_name = attr_to_kerykeion_name(attr)
        directed_points[kery_name] = SimpleNamespace(
            sign=sign_code,
            position=within_sign,
            abs_pos=new_abs,
            retrograde=bool(getattr(natal_point, "retrograde", False)),
        )
    return natal, directed_points, target_dt, age_years, arc_degrees, None


def _format_direction_text(
    natal, directed_points: Dict[str, Any], target_dt: datetime, age_years: float, arc_degrees: float
) -> str:
    natal_cusps = _house_cusp_degrees(natal)
    kery_name_to_point = _kery_name_to_point_map()

    lines = [
        f"Дирекции (солнечная дуга) для {natal.name} на "
        f"{target_dt.year:04d}-{target_dt.month:02d}-{target_dt.day:02d} "
        f"(возраст {age_years:.1f} лет), дуга направления {arc_degrees:.2f}° — "
        "ВСЕ натальные точки смещены на этот ОДИН И ТОТ ЖЕ угол (в отличие "
        "от прогрессий, где разные точки движутся с разной скоростью).",
        "Направленные положения (дом — натальный, т.е. дом натальной "
        "карты, через который сейчас проходит направленная точка):",
    ]
    for label, attr in _PLANET_ATTRS + _ANGLE_ATTRS:
        kery_name = attr_to_kerykeion_name(attr)
        point = directed_points.get(kery_name)
        if point is None:
            continue
        natal_house = _house_of_degree(natal_cusps, point.abs_pos)
        lines.append("  " + _format_point_line(label, point, house_num_override=natal_house))

    lines.append("Аспекты направленных точек к натальной карте:")
    shown = 0
    for label, attr in _PLANET_ATTRS + _ANGLE_ATTRS:
        kery_name = attr_to_kerykeion_name(attr)
        point = directed_points.get(kery_name)
        if point is None:
            continue
        for label2, attr2 in _PLANET_ATTRS + _ANGLE_ATTRS:
            natal_point2 = getattr(natal, attr2, None)
            if natal_point2 is None:
                continue
            sep = _angular_separation(point.abs_pos, natal_point2.abs_pos)
            for aspect_spec in _ALL_ASPECTS:
                angle = _ASPECT_ANGLES.get(aspect_spec["name"])
                if angle is None:
                    continue
                orb = abs(sep - angle)
                if orb <= aspect_spec["orb"]:
                    movement = _direction_aspect_movement(point.abs_pos, natal_point2.abs_pos, angle)
                    lines.append(
                        f"  направленный {_point_ru_from_label(label)} — "
                        f"{_aspect_ru(aspect_spec['name'])} — натальный {_point_ru_from_label(label2)} "
                        f"(орбис {orb:.1f}°, {_movement_ru(movement)})"
                    )
                    shown += 1
    if not shown:
        lines.append("  (нет аспектов в пределах орбиса)")
    return "\n".join(lines)


def run_direction_and_subject(spec: str) -> Tuple[str, Optional[Any], Optional[List[Dict[str, Any]]], None]:
    """See run_natal_and_subject's docstring for the "_and_subject"
    convention. Returns (text, natal, overlay_points, None) — overlay_points
    already shaped for utils.chart_draw.draw_wheel_svg's own
    second=[{"label", "abs_pos", "retrograde"}, ...] list form, since
    directions have no independently-cast second chart (see
    _build_direction_subjects' own docstring: solar arc directions are
    just every natal point shifted by one shared angle)."""
    fields, missing = _extract_fields(spec)
    if missing:
        return _missing_fields_message(missing, fields), None, None, None
    natal, directed_points, target_dt, age_years, arc_degrees, error = _build_direction_subjects(
        fields, name=fields.get("name") or "Subject"
    )
    if error:
        return error, None, None, None
    try:
        text = _format_direction_text(natal, directed_points, target_dt, age_years, arc_degrees)
    except Exception as e:
        return f"Ошибка при расчёте дирекций: {e}", None, None, None
    kery_to_label = {attr_to_kerykeion_name(attr): label for label, attr in _PLANET_ATTRS + _ANGLE_ATTRS}
    overlay = [
        {"label": kery_to_label[kery_name], "abs_pos": point.abs_pos, "retrograde": point.retrograde}
        for kery_name, point in directed_points.items() if kery_name in kery_to_label
    ]
    return text, natal, overlay, None


def run_direction(spec: str) -> str:
    """Tool entry point (utils.tools.TOOL_REGISTRY["astro_direction_chart"]).
    Thin wrapper — see run_direction_and_subject for the full logic."""
    return run_direction_and_subject(spec)[0]


def get_direction_profiles(spec: str, top_n: int = 12) -> List[Dict]:
    """Direction counterpart to get_transit_profiles/get_progression_
    profiles, for routes/chat.py's digest step — bespoke rather than a
    call to get_dual_chart_profiles, since there's no real second subject
    for AspectsFactory.dual_chart_aspects to compare against (see this
    section's module comment); aspect matching is done directly against
    _ASPECT_ANGLES instead. The resulting per-profile shape (text/aspects/
    stars/queries/score, and each aspect's orb/aspect_ru/movement_ru/
    nature_ru/other_label/other_sign/other_house/phrase keys) matches
    get_dual_chart_profiles' output exactly, so the shared digest/answer-
    prompt machinery in utils/interpret.py works completely unchanged.

    Force-includes the Sun, Moon, Ascendant, and MC regardless of score —
    direction_methodology.txt explicitly treats the directed Ascendant/MC
    as being just as significant as directed planets, unlike transit/
    progression where the moving angles aren't a standard reading at all.

    Returns [] (not an error) if fields are missing or subject-building
    fails — same "no profiles available, caller falls back" contract
    every other get_*_profiles function already has."""
    fields, missing = _extract_fields(spec)
    if missing:
        return []
    natal, directed_points, _target_dt, _age_years, _arc_degrees, error = _build_direction_subjects(
        fields, name=fields.get("name") or "Subject"
    )
    if error:
        return []
    try:
        natal_cusps = _house_cusp_degrees(natal)

        # Every directed point's own aspects, computed once up front so
        # per-point scoring (aspect count) and the profile's own "aspects"
        # list can both reuse the same pass instead of recomputing it.
        aspects_by_point: Dict[str, List[Dict]] = {}
        for label, attr in _PLANET_ATTRS + _ANGLE_ATTRS:
            kery_name = attr_to_kerykeion_name(attr)
            point = directed_points.get(kery_name)
            if point is None:
                continue
            own_aspects = []
            for label2, attr2 in _PLANET_ATTRS + _ANGLE_ATTRS:
                natal_point2 = getattr(natal, attr2, None)
                if natal_point2 is None:
                    continue
                sep = _angular_separation(point.abs_pos, natal_point2.abs_pos)
                for aspect_spec in _ALL_ASPECTS:
                    angle = _ASPECT_ANGLES.get(aspect_spec["name"])
                    if angle is None:
                        continue
                    orb = abs(sep - angle)
                    if orb <= aspect_spec["orb"]:
                        movement = _direction_aspect_movement(point.abs_pos, natal_point2.abs_pos, angle)
                        phrase = (
                            f"{_aspect_ru(aspect_spec['name'])} "
                            f"{_point_ru_genitive_from_label(label)} и "
                            f"{_point_ru_genitive_from_label(label2)}"
                        )
                        own_aspects.append(
                            {
                                "orb": orb,
                                "aspect_ru": _aspect_ru(aspect_spec["name"]),
                                "movement_ru": _movement_ru(movement),
                                "nature_ru": _aspect_nature_ru(aspect_spec["name"]),
                                "other_label": _point_ru_from_label(label2),
                                "other_sign": _sign_ru(natal_point2.sign),
                                "other_house": _house_number(getattr(natal_point2, "house", None)),
                                "phrase": phrase,
                            }
                        )
            own_aspects.sort(key=lambda x: x["orb"])
            aspects_by_point[kery_name] = own_aspects

        profiles: List[Dict] = []
        for label, attr in _PLANET_ATTRS + _ANGLE_ATTRS:
            kery_name = attr_to_kerykeion_name(attr)
            point = directed_points.get(kery_name)
            if point is None:
                continue
            natal_house = _house_of_degree(natal_cusps, point.abs_pos)
            retrograde = bool(point.retrograde)
            label_ru = _point_ru_from_label(label)
            own_aspects_full = aspects_by_point.get(kery_name, [])

            score = 0.0
            if natal_house in _ANGULAR_HOUSES:
                score += 3.0
            elif natal_house in _SUCCEDENT_HOUSES:
                score += 1.5
            if retrograde:
                score += 0.5
            score += 0.5 * len(own_aspects_full)

            own_aspects = own_aspects_full[:_MAX_ASPECTS_PER_PROFILE]

            sign_prep = _sign_ru_prepositional(point.sign)
            retro_text = " (ретроградный)" if retrograde else ""
            house_text = f", натальный {natal_house} дом" if natal_house else ""

            queries = [f"направленный {label} {sign_prep}"] + (
                [f"направленный {label} в {natal_house} доме"] if natal_house else []
            )
            for asp in own_aspects:
                queries.append(f"дирекция {asp['aspect_ru']} {label_ru} и {asp['other_label']}")

            profiles.append(
                {
                    "kind": "direction_planet",
                    "label": label_ru,
                    "text": f"направленный {label_ru} {sign_prep}{house_text}{retro_text}",
                    "aspects": own_aspects,
                    "stars": [],
                    "queries": queries,
                    "score": score,
                    "force_include": label in ("Солнце", "Луна", "Асцендент", "Середина неба (MC)"),
                }
            )

        forced = [p for p in profiles if p["force_include"]]
        rest = sorted((p for p in profiles if not p["force_include"]), key=lambda p: p["score"], reverse=True)
        return forced + rest[: max(0, top_n - len(forced))]
    except Exception:
        return []


# --- lunar & solar returns ---------------------------------------------------
#
# Both built on kerykeion's own PlanetaryReturnFactory (a real Swiss-
# Ephemeris search for the exact moment the Sun/Moon returns to its natal
# degree), so — unlike directions — there IS a genuine independent
# kerykeion subject here (PlanetReturnModel), fully compatible with
# get_dual_chart_profiles/AspectsFactory exactly like a transit or
# progression subject is. Location is ALWAYS the natal birth place's own
# lat/lon/tz — per the user's explicit, confirmed choice — relocated
# returns are a real, more specialized technique deliberately left out for
# now (see lunar_return_methodology.txt/solar_return_methodology.txt).
_RETURN_CYCLE_MARGIN_DAYS = {"Solar": 400, "Lunar": 30}
_RETURN_KIND_RU = {"Solar": "солярный", "Lunar": "лунарный"}
_RETURN_LABEL_RU = {"Solar": "Солнечное возвращение (солар)", "Lunar": "Лунное возвращение (лунар)"}


def _parse_kerykeion_iso(value: str) -> datetime:
    """PlanetReturnModel doesn't expose .year/.month/.day/... directly the
    way a subject built via from_birth_data does — only iso_formatted_
    local_datetime/iso_formatted_utc_datetime strings — so every caller
    that needs the return's own calendar moment as a real datetime parses
    it through here. The 'Z' replacement guards against a UTC-suffixed
    string Python's own fromisoformat can't parse on its own on some
    versions; every other offset form it already handles natively.

    Returns a NAIVE datetime (tzinfo stripped) even when the source string
    carried a UTC offset — target_dt (this module's "now"/moment,
    everywhere else too) is always naive, and subtracting an aware from a
    naive datetime raises TypeError; a real, caught-in-testing failure
    here. Comparing/subtracting wall-clock times without carrying tzinfo
    through is the same convention _build_transit_subjects/_build_
    progression_subjects already use for "now", not a new inconsistency
    introduced here."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.replace(tzinfo=None)


def _build_return_subjects(
    fields: Dict[str, str], name: str, return_type: str
) -> Tuple[
    Optional[Any], Optional[Any], Optional[datetime], Optional[datetime], Optional[datetime], Optional[str]
]:
    """Shared by run_lunar_return/run_solar_return and get_lunar_return_
    profiles/get_solar_return_profiles. return_type is kerykeion's own
    literal ("Solar" or "Lunar").

    PlanetaryReturnFactory.next_return_from_iso_formatted_time() finds the
    NEXT return strictly after the given moment, not the most recently
    started one — so the "current" (still-active) return period is found
    by first probing from (target_dt - a safety margin: 400 days for
    Solar, comfortably past the ~365.25-day tropical year; 30 days for
    Lunar, past the ~27.3-day sidereal month), which is guaranteed to land
    at or before the real current return, then WALKING FORWARD one return
    at a time until passing target_dt.

    The walk (not just the single probe) is required, not a nice-to-have:
    a real, caught-in-testing failure showed a margin picked to be "just
    over one cycle" landing a full EXTRA cycle behind (the probe found a
    return from over a year ago instead of last month's), because
    kerykeion's real return period isn't exactly 365.25/27.3 days — it
    varies by hours depending on where the body actually is in its orbit
    that cycle, so no single fixed margin can be proven tight enough to
    land in exactly the right cycle on its own. Walking forward from a
    safely-early starting probe is robust to that variation regardless of
    how far back the probe itself happened to land.

    Returns (natal, current_return_subject, target_dt, current_return_dt,
    next_return_dt, None) on success, or (None, None, None, None, None,
    error_message) on failure — same never-raise convention as the rest of
    this module."""
    try:
        natal = _build_subject(fields, name=name)
    except Exception as e:
        return None, None, None, None, None, f"Не удалось построить натальную карту — некорректные данные ({e})."

    target_dt, error = _parse_moment(fields.get("moment") or "now")
    if error:
        return None, None, None, None, None, error

    try:
        from kerykeion.planetary_return_factory import PlanetaryReturnFactory

        factory = PlanetaryReturnFactory(
            natal, lat=float(fields["lat"]), lng=float(fields["lon"]), tz_str=fields["tz"], online=False,
        )
        margin_days = _RETURN_CYCLE_MARGIN_DAYS[return_type]
        search_start = target_dt - timedelta(days=margin_days)
        probe = factory.next_return_from_iso_formatted_time(search_start.isoformat(), return_type=return_type)
        current_return_dt = _parse_kerykeion_iso(probe.iso_formatted_local_datetime)
        current_return = probe

        next_return, next_return_dt = None, None
        for _ in range(8):
            candidate = factory.next_return_from_iso_formatted_time(
                (current_return_dt + timedelta(hours=1)).isoformat(), return_type=return_type
            )
            candidate_dt = _parse_kerykeion_iso(candidate.iso_formatted_local_datetime)
            if candidate_dt > target_dt:
                next_return, next_return_dt = candidate, candidate_dt
                break
            current_return, current_return_dt = candidate, candidate_dt
        if next_return is None:
            # Shouldn't happen in practice — 8 iterations comfortably
            # covers the margin/actual-cycle-length mismatch this loop
            # exists to correct for. Fail loudly rather than silently
            # hand back a stale current/next pair.
            raise RuntimeError("не удалось сойтись на текущем цикле возвращения за разумное число шагов")
    except Exception as e:
        return None, None, None, None, None, f"Не удалось рассчитать возвращение: {e}"

    return natal, current_return, target_dt, current_return_dt, next_return_dt, None


def _format_return_text(
    natal, return_subject, return_type: str, target_dt: datetime,
    current_return_dt: datetime, next_return_dt: datetime,
) -> str:
    from kerykeion import AspectsFactory

    kind_ru = _RETURN_KIND_RU[return_type]
    label_ru = _RETURN_LABEL_RU[return_type]
    natal_cusps = _house_cusp_degrees(natal)
    cycle_days = (next_return_dt - current_return_dt).days
    day_in_cycle = (target_dt - current_return_dt).days
    days_to_next = (next_return_dt - target_dt).days

    lines = [
        f"{label_ru} для {natal.name}. Текущий цикл начался "
        f"{current_return_dt.year:04d}-{current_return_dt.month:02d}-{current_return_dt.day:02d} "
        f"{current_return_dt.hour:02d}:{current_return_dt.minute:02d} и длится {cycle_days} дней — "
        f"следующее возвращение "
        f"{next_return_dt.year:04d}-{next_return_dt.month:02d}-{next_return_dt.day:02d} "
        f"(сейчас {day_in_cycle}-й день цикла, до следующего возвращения {days_to_next} дней). "
        "Карта возвращения рассчитана ТОЛЬКО для натального места рождения, без релокации.",
        f"Собственные положения планет {kind_ru} карты:",
    ]
    for label, attr in _PLANET_ATTRS:
        point = getattr(return_subject, attr, None)
        if point is not None:
            lines.append("  " + _format_point_line(label, point))
    lines.append(f"Собственные углы {kind_ru} карты:")
    for label, attr in _ANGLE_ATTRS:
        point = getattr(return_subject, attr, None)
        if point is not None:
            lines.append("  " + _format_point_line(label, point))
    lines.append(f"Куспиды домов {kind_ru} карты (собственная система домов возвращения):")
    for i, attr in enumerate(
        ["first_house", "second_house", "third_house", "fourth_house",
         "fifth_house", "sixth_house", "seventh_house", "eighth_house",
         "ninth_house", "tenth_house", "eleventh_house", "twelfth_house"],
        start=1,
    ):
        cusp = getattr(return_subject, attr, None)
        if cusp is not None:
            lines.append(f"  Дом {i}: {_sign_ru(cusp.sign)} {cusp.position:.1f}°")

    lines.append(f"Положения планет {kind_ru} карты в НАТАЛЬНЫХ домах (для сравнения с натальной картой):")
    for label, attr in _PLANET_ATTRS:
        point = getattr(return_subject, attr, None)
        if point is not None:
            natal_house = _house_of_degree(natal_cusps, point.abs_pos)
            lines.append(f"  {label}: натальный {natal_house} дом")

    aspects = AspectsFactory.dual_chart_aspects(
        natal, return_subject, active_points=_ASPECT_ACTIVE_POINTS, active_aspects=_ALL_ASPECTS,
    ).aspects
    lines.append(f"Аспекты {kind_ru} карты к натальной карте:")
    shown = 0
    for a in aspects:
        returning, to_natal = (
            (a.p1_name, a.p2_name) if a.p1_owner == return_subject.name else (a.p2_name, a.p1_name)
        )
        lines.append(
            f"  {kind_ru} {_point_ru(returning)} — {_aspect_ru(a.aspect)} — "
            f"натальный {_point_ru(to_natal)} "
            f"(орбис {a.orbit:.1f}°, {_movement_ru(a.aspect_movement)})"
        )
        shown += 1
    if not shown:
        lines.append("  (нет аспектов в пределах орбиса)")
    return "\n".join(lines)


def run_lunar_return_and_subject(spec: str) -> Tuple[str, Optional[Any], Optional[Any], None]:
    """See run_natal_and_subject's docstring for the "_and_subject"
    convention. Returns (text, natal, return_subject, None)."""
    fields, missing = _extract_fields(spec)
    if missing:
        return _missing_fields_message(missing, fields), None, None, None
    natal, return_subject, target_dt, current_return_dt, next_return_dt, error = _build_return_subjects(
        fields, name=fields.get("name") or "Subject", return_type="Lunar"
    )
    if error:
        return error, None, None, None
    try:
        text = _format_return_text(natal, return_subject, "Lunar", target_dt, current_return_dt, next_return_dt)
        return text, natal, return_subject, None
    except Exception as e:
        return f"Ошибка при расчёте лунара: {e}", None, None, None


def run_lunar_return(spec: str) -> str:
    """Tool entry point (utils.tools.TOOL_REGISTRY["astro_lunar_return_chart"]).
    Thin wrapper — see run_lunar_return_and_subject for the full logic."""
    return run_lunar_return_and_subject(spec)[0]


def run_solar_return_and_subject(spec: str) -> Tuple[str, Optional[Any], Optional[Any], None]:
    """See run_natal_and_subject's docstring for the "_and_subject"
    convention. Returns (text, natal, return_subject, None)."""
    fields, missing = _extract_fields(spec)
    if missing:
        return _missing_fields_message(missing, fields), None, None, None
    natal, return_subject, target_dt, current_return_dt, next_return_dt, error = _build_return_subjects(
        fields, name=fields.get("name") or "Subject", return_type="Solar"
    )
    if error:
        return error, None, None, None
    try:
        text = _format_return_text(natal, return_subject, "Solar", target_dt, current_return_dt, next_return_dt)
        return text, natal, return_subject, None
    except Exception as e:
        return f"Ошибка при расчёте солара: {e}", None, None, None


def run_solar_return(spec: str) -> str:
    """Tool entry point (utils.tools.TOOL_REGISTRY["astro_solar_return_chart"]).
    Thin wrapper — see run_solar_return_and_subject for the full logic."""
    return run_solar_return_and_subject(spec)[0]


def _get_return_profiles(spec: str, return_type: str, top_n: int) -> List[Dict]:
    """Shared body of get_lunar_return_profiles/get_solar_return_profiles
    (both just fix return_type/top_n and delegate here) — reuses
    get_dual_chart_profiles directly (unlike directions), since a
    kerykeion PlanetReturnModel is a real, AspectsFactory-compatible
    subject just like a transit or progressed one.

    include_angles=True (see get_dual_chart_profiles' own docstring for
    why this parameter exists) and force_include_labels including
    "Асцендент" together give the return chart's own Ascendant a real
    profile entry — both lunar_return_methodology.txt and solar_return_
    methodology.txt single out the return's own Ascendant as one of the
    most important points to read, which the plain transit/progression
    defaults would otherwise never surface at all."""
    fields, missing = _extract_fields(spec)
    if missing:
        return []
    natal, return_subject, _target_dt, _current_return_dt, _next_return_dt, error = _build_return_subjects(
        fields, name=fields.get("name") or "Subject", return_type=return_type
    )
    if error:
        return []
    kind_ru = _RETURN_KIND_RU[return_type]
    try:
        return get_dual_chart_profiles(
            natal, return_subject,
            other_active_points=_ACTIVE_POINTS_NATAL, top_n=top_n,
            other_point_label=lambda lr, k=kind_ru: f"{k} {lr}",
            reference_house_label=lambda h: f"натальный {h} дом",
            own_house_label=lambda h, k=kind_ru: f"свой {h} дом ({k})",
            query_prefix=kind_ru,
            query_aspect_prefix=kind_ru,
            force_include_labels=("Солнце", "Луна", "Асцендент"),
            kind=f"{return_type.lower()}_return_planet",
            include_angles=True,
        )
    except Exception:
        return []


def get_lunar_return_profiles(spec: str, top_n: int = 12) -> List[Dict]:
    """Lunar-return counterpart to get_transit_profiles — see
    _get_return_profiles for the shared mechanism. Force-includes the
    return's own Ascendant alongside Sun/Moon (unlike transit/progression's
    Sun/Moon-only default): lunar_return_methodology.txt calls the return's
    own Moon placement the single most important point, and its own
    Ascendant matters far more here than a moving transit moment's does."""
    return _get_return_profiles(spec, "Lunar", top_n)


def get_solar_return_profiles(spec: str, top_n: int = 12) -> List[Dict]:
    """Solar-return counterpart — see solar_return_methodology.txt: the
    return's own Ascendant/its ruler is the single most important point in
    a solar return (more so than the Sun's own house there), which is why
    Ascendant is force-included here exactly as it is for lunar returns."""
    return _get_return_profiles(spec, "Solar", top_n)


# --- profections (whole-sign, classical rulers) -----------------------------
#
# Pure calendar + rulership arithmetic over the ALREADY-COMPUTED natal
# chart — unlike every other technique in this module, this builds no new
# ephemeris chart at all. Deliberately uses WHOLE-SIGN houses counted from
# the natal Ascendant's own sign — a real, explicit fork from the quadrant
# house system every other technique here uses (kerykeion's natal cusps),
# confirmed with the user as the intended, classical convention for this
# one technique specifically ("целые знаки (классика)") rather than an
# oversight or inconsistency. See profection_methodology.txt.
_CLASSICAL_RULERS_RU = {
    "Ari": "Марс", "Tau": "Венера", "Gem": "Меркурий", "Can": "Луна",
    "Leo": "Солнце", "Vir": "Меркурий", "Lib": "Венера", "Sco": "Марс",
    "Sag": "Юпитер", "Cap": "Сатурн", "Aqu": "Сатурн", "Pis": "Юпитер",
}


def _build_profection_context(
    fields: Dict[str, str], name: str
) -> Tuple[Optional[Any], Optional[datetime], Optional[int], Optional[str]]:
    """Shared by run_profection/get_profection_profiles — builds only the
    natal subject (no second chart at all, unlike every other technique
    here) plus the target moment and the person's completed age in years
    at that moment (the whole-sign profection count, see
    _profection_house_and_ruler). Returns (natal, target_dt,
    age_full_years, None) on success, or (None, None, None,
    error_message) on failure — same never-raise convention as the rest of
    this module."""
    try:
        natal = _build_subject(fields, name=name)
    except Exception as e:
        return None, None, None, f"Не удалось построить натальную карту — некорректные данные ({e})."

    target_dt, error = _parse_moment(fields.get("moment") or "now")
    if error:
        return None, None, None, error

    try:
        date_str = fields["date"]
        time_str = fields.get("time", "12:00")
        year, month, day = (int(x) for x in date_str.split("-"))
        hour, minute = (int(x) for x in time_str.split(":"))
        birth_dt = datetime(year, month, day, hour, minute)
    except Exception as e:
        return None, None, None, f"Не удалось разобрать дату рождения ({e})."

    age_years = (target_dt - birth_dt).total_seconds() / 86400.0 / 365.25
    if age_years < 0:
        return None, None, None, "Профекция рассчитывается только для дат ПОСЛЕ рождения."
    return natal, target_dt, int(age_years), None


def _profection_house_and_ruler(natal, age_full_years: int) -> Tuple[int, str, str]:
    """The whole-sign profection formula itself: the natal Ascendant's own
    sign IS house 1 for "year 0" (birth to the first birthday); each
    completed year of life profects the count by exactly one whole sign,
    cycling every 12 years (age 12 lands back on the natal Ascendant's own
    sign, age 13 on the same sign age 1 had, and so on). Returns
    (profected_house_num, profected_sign_code, ruler_label_ru) — the
    ruler is looked up via classical (7-planet, no outer-planet)
    rulerships, per profection_methodology.txt: this is historically a
    classical technique, and Uranus/Neptune/Pluto rulerships would be an
    anachronism here even though the rest of this app doesn't otherwise
    distinguish classical vs. modern rulership anywhere else."""
    asc_sign_index = _ZODIAC_SIGN_CODES.index(natal.ascendant.sign)
    profected_house_num = (age_full_years % 12) + 1
    profected_sign_index = (asc_sign_index + age_full_years) % 12
    profected_sign_code = _ZODIAC_SIGN_CODES[profected_sign_index]
    ruler_label_ru = _CLASSICAL_RULERS_RU[profected_sign_code]
    return profected_house_num, profected_sign_code, ruler_label_ru


def _format_profection_text(
    natal, target_dt: datetime, age_full_years: int,
    profected_house_num: int, profected_sign_code: str, ruler_label_ru: str,
) -> str:
    from kerykeion import AspectsFactory

    ruler_attr = next(attr for lbl, attr in _PLANET_ATTRS if lbl == ruler_label_ru)
    ruler_kery_name = attr_to_kerykeion_name(ruler_attr)
    ruler_point = getattr(natal, ruler_attr)

    lines = [
        f"Профекция для {natal.name} на "
        f"{target_dt.year:04d}-{target_dt.month:02d}-{target_dt.day:02d} "
        f"(возраст {age_full_years} полных лет, профекционный год №{age_full_years + 1}). "
        "Техника ЦЕЛЫХ ЗНАКОВ (классическая) от натального Асцендента — "
        "ОТЛИЧАЕТСЯ от квадрантной системы домов, используемой для "
        "остальных техник в этом приложении.",
        f"Натальный Асцендент: {_sign_ru(natal.ascendant.sign)} {natal.ascendant.position:.1f}°.",
        f"Профецированный дом: {profected_house_num} (знак {_sign_ru(profected_sign_code)}).",
        f"Управитель года (time lord): {_point_ru_from_label(ruler_label_ru)} — "
        + _format_point_line(ruler_label_ru, ruler_point),
    ]

    aspects = AspectsFactory.natal_aspects(
        natal, active_points=_ASPECT_ACTIVE_POINTS, active_aspects=_ALL_ASPECTS,
    ).aspects
    lines.append(f"Натальные аспекты управителя года ({_point_ru_from_label(ruler_label_ru)}):")
    shown = 0
    for a in aspects:
        if a.p1_name == ruler_kery_name:
            other = a.p2_name
        elif a.p2_name == ruler_kery_name:
            other = a.p1_name
        else:
            continue
        lines.append(
            f"  {_point_ru_from_label(ruler_label_ru)} — {_aspect_ru(a.aspect)} — {_point_ru(other)} "
            f"(орбис {a.orbit:.1f}°, {_movement_ru(a.aspect_movement)})"
        )
        shown += 1
    if not shown:
        lines.append("  (нет аспектов в пределах орбиса)")
    return "\n".join(lines)


def run_profection_and_subject(spec: str) -> Tuple[str, Optional[Any], None, Optional[int]]:
    """See run_natal_and_subject's docstring for the "_and_subject"
    convention. Returns (text, natal, None, house_num) — highlight_house
    (not a second subject) is this technique's own third slot, matching
    chart_draw.draw_wheel_svg's highlight_house param for shading the
    "activated" house wedge."""
    fields, missing = _extract_fields(spec)
    if missing:
        return _missing_fields_message(missing, fields), None, None, None
    natal, target_dt, age_full_years, error = _build_profection_context(
        fields, name=fields.get("name") or "Subject"
    )
    if error:
        return error, None, None, None
    try:
        house_num, sign_code, ruler_ru = _profection_house_and_ruler(natal, age_full_years)
        text = _format_profection_text(natal, target_dt, age_full_years, house_num, sign_code, ruler_ru)
        return text, natal, None, house_num
    except Exception as e:
        return f"Ошибка при расчёте профекции: {e}", None, None, None


def run_profection(spec: str) -> str:
    """Tool entry point (utils.tools.TOOL_REGISTRY["astro_profection_chart"]).
    Unlike every other run_* here, this builds NO new ephemeris chart at
    all — profections are pure calendar+rulership arithmetic over the
    already-computed natal chart (see _build_profection_context/
    _profection_house_and_ruler). Thin wrapper — see
    run_profection_and_subject for the full logic."""
    return run_profection_and_subject(spec)[0]


def get_profection_profiles(spec: str, top_n: int = 9) -> List[Dict]:
    """Profection counterpart to get_planet_profiles — NOT a full per-point
    ranking like that function (only one point, the year's ruler, actually
    matters for this technique): returns exactly two profiles — a
    synthetic "which house/sign/ruler" summary fact (kind=
    "profection_house", no aspects of its own, since it's a calendar fact
    about the YEAR, not a chart point) plus the ruler's own natal profile,
    reused verbatim from get_planet_profiles (top_n=20 there to guarantee
    it's never excluded by that function's own score-based cap — _PLANET_
    ATTRS+_ANGLE_ATTRS together total 18 points, so 20 always returns
    every one of them, forced or not).

    top_n is accepted for interface consistency with every other
    get_*_profiles function, but is otherwise unused — this technique's
    output size isn't tunable, since more of a chart's own points simply
    aren't relevant to a single year's profected ruler."""
    fields, missing = _extract_fields(spec)
    if missing:
        return []
    natal, _target_dt, age_full_years, error = _build_profection_context(
        fields, name=fields.get("name") or "Subject"
    )
    if error:
        return []
    try:
        house_num, sign_code, ruler_ru = _profection_house_and_ruler(natal, age_full_years)
        sign_ru = _sign_ru(sign_code)
        summary_profile = {
            "kind": "profection_house",
            "label": "Профекция",
            "text": (
                f"Профецированный {house_num} дом ({sign_ru}), "
                f"управитель года — {_point_ru_from_label(ruler_ru)}"
            ),
            "aspects": [],
            "stars": [],
            "queries": [
                f"профекция {house_num} дом", f"профекция управитель {ruler_ru}", f"профекция {sign_ru}",
            ],
            "score": 0.0,
            "force_include": True,
        }
        # get_planet_profiles' own "label" field already has the point's
        # unicode symbol appended (_point_ru_from_label) — comparing
        # against the bare ruler_ru name here was a real, caught-in-
        # testing bug (silently matched nothing, e.g. "Меркурий" never
        # equals "Меркурий ☿"), which is why this uses the same
        # symbol-appending helper before comparing.
        ruler_label_with_symbol = _point_ru_from_label(ruler_ru)
        all_profiles = get_planet_profiles(spec, top_n=20)
        ruler_profile = next((p for p in all_profiles if p["label"] == ruler_label_with_symbol), None)
        return [summary_profile] + ([ruler_profile] if ruler_profile else [])
    except Exception:
        return []


# Registry for future operations (composite, rectification, electional
# search, ...) — see module docstring. Not yet consumed by anything
# (utils/tools.py wires each run_* function directly), but kept as the
# intended extension point so adding another operation doesn't require
# inventing a new pattern.
ASTRO_OPERATIONS: Dict[str, Callable[[str], str]] = {
    "natal": run_natal,
    "transit": run_transit,
    "synastry": run_synastry,
    "progression": run_progression,
    "direction": run_direction,
    "lunar_return": run_lunar_return,
    "solar_return": run_solar_return,
    "profection": run_profection,
}
