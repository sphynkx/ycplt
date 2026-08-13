"""Electional astrology — deterministic engine, classical Western method
(Andreeva's "Элективная астрология" for planetary-hour doctrine and Moon-
transit-by-sign/house material, Being-on-Time/Scofield and general classical
sources for lunar-phase and diurnal-cycle rules, Vronsky's marriage-date
checklist consulted for scope but NOT implemented in full — see module
docstring below and electional_methodology.txt section 7 for exactly what
v1 does and doesn't compute). Deliberately does NOT use Globa's lecture
material or the Vedic muhurta (panchanga) system at all — excluded by
explicit user instruction, not merely out of scope.

Structurally this is horary's own engine turned around: horary reads an
already-fixed moment against a QUESTION ("will X happen, given this
moment"); this reads a user-PROPOSED moment against a stated PURPOSE ("is
this moment good for doing X"). Both use the same querent(house I)/
quesited(topic house) significator-and-aspect machinery — see
horary_methodology.txt's own note on this and electional_methodology.txt's
opening section, which is the reasoning-layer statement of the same fact
this module's code mirrors structurally. Deliberately NOT imported/shared
from utils/horary.py itself (dignity tables, Via Combusta/combustion/
siege math, _assess_strength are duplicated below, not reused) — same
"helpful duplication over coupling two independently-evolving, closely
related modules" choice already made for utils/astro.py's own
_extract_fields_llm relative to horary's, for the same reason: horary.py is
an already-stable, separately-tested module, and this shouldn't create a
change-in-one-breaks-the-other risk between them.

Known, accepted v1 scope limits (see electional_methodology.txt section 7
for the reasoning-facing version of this same list — this is the engine-
facing "why" for each):
  - Two request modes, classified by _classify_request_mode (LLM-first,
    like every other free-text decision this module makes — never regex;
    a fixed phrase list was tried first and rejected on real feedback for
    being unable to tell a genuinely proposed moment apart from the
    moment a question just happened to be asked, or to handle
    paraphrases): "single" evaluates ONE named candidate moment directly
    (_resolve_single_moment_request + _compute_electional_chart_core);
    "range" ("на какой день лучше..." — "which day is best for X") scans
    forward from the nearest named moment (or right now, if none was
    named — see _resolve_range_request) across _SEARCH_WINDOW_DAYS days
    at full hourly granularity, evaluating every candidate hour via
    _compute_electional_chart_core and keeping the single best one
    (_search_best_electional_moment) — mirroring how
    rectification_events.py scans a window of candidate BIRTH times,
    scoring each one. Found via real testing: left undetected, a range-
    style question used to have its date/time (typically just the moment
    the question was asked, following horary's own convention of "the
    moment you're asking from") silently treated as the user's real
    proposed moment, producing a confident-looking but meaningless
    verdict about a moment nobody actually proposed. The window length
    (30 days), full-hourly-scan granularity (rather than day-level
    scoring + hour-refinement), and best-in-window tie-break policy
    (rather than stopping at the first candidate that clears the
    "благоприятно" threshold) were each explicit choices the user made
    when this was built, trading search time for not missing a good hour
    hidden inside an otherwise ordinary-looking day. An explicit
    user-named END date for the range (rather than always defaulting to
    the fixed window) isn't parsed yet — a reasonable, small v2.1 step,
    not yet requested.
  - Each activity category maps to exactly ONE quesited house (see
    _CATEGORY_TABLE) — real classical sources sometimes weigh two houses
    together for one activity (e.g. starting a business: X for public
    standing AND II for income); v1 picks the single most decisive house
    per category rather than combining two, to keep the significator-
    /aspect logic identical in shape to horary's own (exactly one querent,
    exactly one quesited).
  - No marriage-specific degree-within-sign Moon rules (Vronsky's ~28-point
    checklist), no lunar-day (lunar-sutki) system, no "which body part is
    being operated on" Moon-sign check for medical elections (only the
    Mars-hour avoidance is checked for that category), no Vedic muhurta,
    no religious/cultural calendar layer. None of these are silently
    assumed — electional_methodology.txt tells the model to say so
    explicitly rather than staying quiet about them.

Two later, explicitly user-requested additions on top of the v1 design
above (both purely additive — neither changes a single-moment/range
result when the condition they check for doesn't apply):
  - Day-of-week ruler checklist (compute_planetary_hour/
    _compute_electional_chart_core's own hour-ruler block): the HOUR
    ruler was already scored against each category's sympathetic/avoid
    sets; the DAY ruler (also already computed, previously only
    displayed) now gets the same +/-1 check — Andreeva's own per-planet
    day-and-hour recommendations, not a separate scheme, so this is the
    same table, just applied twice (once per hour, once per day), not
    two different scoring schemes bolted together.
  - Querent's own natal chart, when one was already built earlier in the
    SAME conversation (_build_querent_natal_subject/
    _querent_natal_transit_score): on top of the generic, always-present
    house-I/house-of-the-matter significators every election already
    checks, this ALSO checks real transits from each candidate moment to
    that specific person's own natal Sun/Moon/Ascendant — the same kind
    of check astro.run_transit already does for "what's happening in
    someone's chart right now", applied to a candidate election moment
    instead of "now". Detected via a dedicated LLM lookup over the FULL
    conversation history (not the current election's own round-scoped
    text — see HISTORY_MARKER), since an earlier natal-chart request is
    its own, separate round by design and would otherwise never be
    visible to this tool at all. Silently absent (not an error) whenever
    no such earlier natal request exists, which is the common case.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from utils import astro
from utils import llm as llm_utils

# Delimiter routes/chat.py uses to append the FULL prior-conversation user
# text (unbounded by _collect_current_round_texts' own round isolation)
# after the normal round-scoped tool_arg it already builds — see
# run_electional_chart's own docstring for why this needs to be a separate
# section rather than just handing everything to the same extraction
# prompts _resolve_single_moment_request/_resolve_range_request already
# use: those prompts are deliberately scoped to ONE round's own text (see
# _FIELD_EXTRACTION_PROMPT's own "без более ранних... сообщений" line) so
# an unrelated earlier round's place/date can't leak into the CURRENT
# election's own fields — the querent-natal-chart lookup below is the only
# thing here that's supposed to look further back than that boundary, so
# it gets its own clearly-delimited channel instead of widening the
# round-scoped prompts themselves. Mirrors rectification_events.py's own
# "single string, clearly-delimited sections" convention (birth data, then
# one event per line) rather than changing utils/tools.py's one-argument-
# per-tool contract.
HISTORY_MARKER = "\n\n===ПРЕДЫДУЩИЕ СООБЩЕНИЯ ДИАЛОГА (только для поиска натальных данных кверента)===\n"

# --- essential dignity, Via Combusta, combustion/siege — duplicated from
# utils/horary.py (see module docstring for why this is duplication, not
# a shared import) ------------------------------------------------------

_EXALTATION: Dict[str, str] = {
    "Солнце": "Ari", "Луна": "Tau", "Меркурий": "Vir", "Венера": "Pis",
    "Марс": "Cap", "Юпитер": "Can", "Сатурн": "Lib",
}
_ZODIAC = astro._ZODIAC_SIGN_CODES


def _opposite_sign(sign_code: str) -> str:
    i = _ZODIAC.index(sign_code)
    return _ZODIAC[(i + 6) % 12]


_RULES: Dict[str, List[str]] = {}
for _sign_code, _ruler_label in astro._CLASSICAL_RULERS_RU.items():
    _RULES.setdefault(_ruler_label, []).append(_sign_code)

_DETRIMENT: Dict[str, List[str]] = {
    planet: [_opposite_sign(s) for s in signs] for planet, signs in _RULES.items()
}
_FALL: Dict[str, str] = {planet: _opposite_sign(sign) for planet, sign in _EXALTATION.items()}


def _dignity(planet_label: str, sign_code: str) -> str:
    if sign_code in _RULES.get(planet_label, []):
        return "обитель"
    if _EXALTATION.get(planet_label) == sign_code:
        return "экзальтация"
    if sign_code in _DETRIMENT.get(planet_label, []):
        return "изгнание"
    if _FALL.get(planet_label) == sign_code:
        return "падение"
    return ""


def _angular_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


_VIA_COMBUSTA_START = 195.0
_VIA_COMBUSTA_END = 225.0
_VIA_COMBUSTA_SPICA_EXCEPTION = (202.5, 204.5)


def _on_via_combusta(abs_pos: float) -> bool:
    d = abs_pos % 360.0
    if not (_VIA_COMBUSTA_START <= d < _VIA_COMBUSTA_END):
        return False
    lo, hi = _VIA_COMBUSTA_SPICA_EXCEPTION
    return not (lo <= d <= hi)


_COMBUST_ORB = 8.0


def _is_combust(point, sun) -> bool:
    if point is sun:
        return False
    return _angular_diff(point.abs_pos, sun.abs_pos) <= _COMBUST_ORB


_SIEGE_ORB = 15.0


def _is_besieged(point, mars, saturn) -> bool:
    lo, hi = sorted([mars.abs_pos % 360.0, saturn.abs_pos % 360.0])
    d = point.abs_pos % 360.0
    if (hi - lo) <= 180.0:
        return lo <= d <= hi
    return d >= hi or d <= lo


def _is_captured(point, mars, saturn) -> bool:
    if not _is_besieged(point, mars, saturn):
        return False
    return (
        _angular_diff(point.abs_pos, mars.abs_pos) <= _SIEGE_ORB
        and _angular_diff(point.abs_pos, saturn.abs_pos) <= _SIEGE_ORB
    )


_ANGULAR_HOUSES = {1, 4, 7, 10}
_CADENT_HOUSES = {3, 6, 9, 12}


def _assess_strength(label: str, point, sign_code: str, house_num: int, sun, mars, saturn) -> Tuple[str, List[str]]:
    strong: List[str] = []
    weak: List[str] = []

    dign = _dignity(label, sign_code)
    if dign in ("обитель", "экзальтация"):
        strong.append(dign)
    elif dign in ("изгнание", "падение"):
        weak.append(dign)

    if house_num in _ANGULAR_HOUSES:
        strong.append(f"угловой дом {house_num}")
    elif house_num in _CADENT_HOUSES:
        weak.append(f"кадентный дом {house_num}")

    if _is_captured(point, mars, saturn):
        weak.append("пленён между Марсом и Сатурном")
    elif _is_besieged(point, mars, saturn):
        weak.append("осаждён между Марсом и Сатурном")

    if _is_combust(point, sun):
        weak.append("в соединении с Солнцем (комбустен)")

    if _on_via_combusta(point.abs_pos):
        weak.append("на Via Combusta")

    if len(strong) > len(weak):
        verdict = "сильный"
    elif weak:
        verdict = "слабый"
    else:
        verdict = "нейтральный"
    return verdict, strong + weak


# --- classical aspect set + orb — see astro._CLASSICAL_ASPECT_NAMES/
# _CLASSICAL_ASPECTS_WIDE/filter_classical_aspects for the shared
# implementation (also used by utils/horary.py, and by utils/chart_draw.py
# for chart rendering) — electional_methodology.txt explicitly reuses
# horary's own orb rule ("в тех же орбисах... что и в хорарной технике",
# section 4 there: 8-10° for aspects involving Sun/Moon, 6-7° between
# other planets, flat 5° for quincunx) rather than defining its own. This
# used to be a local _ELECTIONAL_ASPECTS list restricting only the aspect
# TYPE set (correctly, to the six classical types) but still using
# _MAJOR_ASPECTS'/_MINOR_ASPECTS' flat per-type orbs (8/8/7/7/5/3) with no
# luminary-aware differentiation at all — fixed by using the shared
# astro.py helpers at both call sites below instead. ----------------------

_FAVORABLE_ASPECTS = {"trine", "sextile", "conjunction"}
_HARD_ASPECTS = {"square", "opposition", "quincunx"}


# --- activity category -> (house, sympathetic planets, planets to avoid at
# the hour level, whether Mercury retrograde matters for this category) --
#
# Sympathetic/avoid sets are deliberately small and specific, taken
# directly from Andreeva's own per-hour recommend/avoid lists (see
# electional_methodology.txt section 2) rather than encoding her entire
# per-hour text — only the clearest, most activity-specific
# recommendations are kept as scoring signals; the methodology document
# itself carries the fuller descriptive text for the model's own prose.
_CategoryInfo = Tuple[int, frozenset, frozenset, bool]

_CATEGORY_TABLE: Dict[str, _CategoryInfo] = {
    "брак": (7, frozenset({"Венера", "Юпитер"}), frozenset({"Сатурн"}), False),
    "договор": (7, frozenset({"Меркурий", "Юпитер"}), frozenset({"Луна"}), True),
    "бизнес": (10, frozenset({"Юпитер"}), frozenset(), True),
    "поездка_короткая": (3, frozenset({"Меркурий"}), frozenset(), True),
    "поездка_дальняя": (9, frozenset({"Юпитер"}), frozenset(), True),
    "переезд": (4, frozenset({"Сатурн"}), frozenset({"Солнце"}), False),
    "операция": (6, frozenset(), frozenset({"Марс"}), False),
    "суд": (7, frozenset({"Юпитер"}), frozenset(), False),
    "покупка": (2, frozenset({"Венера", "Юпитер"}), frozenset({"Сатурн"}), False),
    "работа": (10, frozenset({"Юпитер", "Солнце"}), frozenset({"Сатурн"}), False),
    "выступление": (1, frozenset({"Меркурий"}), frozenset({"Сатурн"}), True),
    "посадка": (4, frozenset({"Луна"}), frozenset(), False),
    # Household/domestic chores (cleaning, tidying, small home repairs,
    # laundry) -> house IV, NOT house I. Found via a real misclassification
    # ("уборка в комнате" landed in "начинание"/house I): Tsypin's "Основы
    # элективной астрологии" names IV explicitly "Дом результатов" for
    # household-type elections (malefics there are avoided across several
    # worked examples — apartment renovation, appliance purchase), and
    # Robson states the general principle this follows from — "the house
    # governing the subject-matter of the election" — domestic upkeep is
    # squarely a IV-house (home) matter, not a I-house (self/new-undertaking)
    # one. House I remains _DEFAULT_CATEGORY's own house for a genuinely
    # generic "начинание" with no clearer topical home, per Li Liman's
    # separate "always strengthen house I" convention — that's a universal
    # baseline significator alongside a topical house, not a substitute for
    # one, so it's kept only as the true fallback, not as where domestic
    # chores land. Sympathetic to Moon per Robson's own dedicated
    # housework chapter (timing chores by Moon sign/aspect rather than by
    # house at all in the classical text) — Mars avoided at the hour level
    # (associated with breakage/damage during physical chores).
    "быт": (4, frozenset({"Луна"}), frozenset({"Марс"}), False),
    "начинание": (1, frozenset(), frozenset(), False),
}
_DEFAULT_CATEGORY = "начинание"

_CATEGORY_LABELS_RU = {
    "брак": "брак/свадьба", "договор": "подписание договора",
    "бизнес": "начало бизнеса/регистрация предприятия", "поездка_короткая": "короткая/местная поездка",
    "поездка_дальняя": "дальняя/заграничная поездка", "переезд": "переезд, новое жильё",
    "операция": "операция/медицинская процедура", "суд": "судебное дело/иск",
    "покупка": "крупная покупка", "работа": "собеседование/начало новой работы",
    "выступление": "публичное выступление/экзамен", "посадка": "посадка растений/сельское хозяйство",
    "быт": "домашние/бытовые дела (уборка, ремонт, стирка)",
    "начинание": "начинание без более точной категории",
}

# Keyword fallback for when no model is loaded — deliberately coarse (this
# only needs to be roughly right; the LLM path above is primary, exactly
# the same "LLM-first, deterministic-fallback" split as everywhere else in
# this app). Checked in order, first match wins.
_CATEGORY_KEYWORDS: List[Tuple[List[str], str]] = [
    (["свадьб", "брак", "женить", "женил", "замуж", "венчани", "помолвк"], "брак"),
    (["договор", "контракт", "подписа", "соглашени"], "договор"),
    (["бизнес", "предприят", "регистрац", "фирм", "компани"], "бизнес"),
    (["дальн", "заграниц", "загранич", "эмигра", "виза"], "поездка_дальняя"),
    (["поездк", "путешеств", "командировк", "рейс", "вылет"], "поездка_короткая"),
    (["переезд", "квартир", "жиль", "новосель"], "переезд"),
    (["операц", "хирург", "процедур", "лечени"], "операция"),
    (["суд", "иск", "тяжб"], "суд"),
    (["покупк", "купить", "приобрет"], "покупка"),
    (["собеседовани", "устройств", "работ", "должност"], "работа"),
    (["выступлени", "презентац", "экзамен", "доклад", "речь"], "выступление"),
    (["посадк", "посев", "огород", "сад", "растени"], "посадка"),
    (["уборк", "убира", "стирк", "мытьё", "мытье", "чистк", "домашн", "хозяйств", "ремонт", "генеральн"], "быт"),
]


def _classify_category_keywords(purpose_text: str) -> str:
    lowered = purpose_text.lower()
    for keywords, category in _CATEGORY_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return category
    return _DEFAULT_CATEGORY


_CATEGORY_PROMPT = """Тебе показано описание дела, для которого пользователь хочет выбрать
благоприятный момент (элективная астрология). Определи, к какой из
следующих категорий оно ближе всего:

брак — вступление в брак, свадьба, помолвка
договор — подписание договора, партнёрского соглашения
бизнес — начало бизнеса, регистрация предприятия
поездка_короткая — короткая или местная поездка
поездка_дальняя — дальняя или заграничная поездка
переезд — переезд, новое жильё, покупка недвижимости
операция — операция, медицинская процедура
суд — судебное дело, иск
покупка — крупная покупка (не недвижимость)
работа — собеседование, начало новой работы
выступление — публичное выступление, презентация, экзамен
посадка — посадка растений, начало сельскохозяйственных работ
быт — домашние/бытовые дела: уборка, генеральная уборка, стирка, мытьё,
  мелкий ремонт по дому — НЕ переезд и не покупка недвижимости, а именно
  повседневный уход за уже имеющимся жильём
начинание — ничего из перечисленного не подходит точно (используется
  только когда дело не привязано ни к дому, ни к одной из категорий выше)

Описание дела: "{text}"

Ответь СТРОГО одним словом из списка выше (только само слово-категорию,
без пояснений): """


def _parse_category_answer(answer: str) -> Optional[str]:
    lowered = answer.strip().lower()
    for category in _CATEGORY_TABLE:
        if category in lowered:
            return category
    return None


def _classify_category_llm(purpose_text: str) -> Optional[str]:
    if llm_utils.get_llm() is None:
        return None
    try:
        answer = llm_utils.generate_sync(
            _CATEGORY_PROMPT.format(text=purpose_text), max_tokens=15, temperature=0.0,
        )
    except Exception:
        return None
    return _parse_category_answer(answer)


def _classify_category(purpose_text: str) -> Tuple[str, str]:
    """Returns (category, source) — source is "llm" or "keyword", for
    display only."""
    llm_result = _classify_category_llm(purpose_text)
    if llm_result:
        return llm_result, "llm"
    return _classify_category_keywords(purpose_text), "keyword"


# --- LLM-first field extraction (date/time/place/purpose) — mirrors
# utils/horary.py's own _extract_horary_fields_llm/_resolve_place exactly,
# duplicated rather than imported for the same reason given in the module
# docstring. -----------------------------------------------------------

_FIELD_EXTRACTION_PROMPT = """Тебе показан текст одного "раунда" элективного запроса — предложенный
пользователем момент времени и дело, для которого он хочет узнать,
подходит ли этот момент, а также, возможно, последующие уточнения
пользователя в рамках ОДНОГО И ТОГО ЖЕ запроса (без более ранних, не
относящихся к делу сообщений).

Извлеки из этого текста:
1. Дату предложенного момента — в формате ГГГГ-ММ-ДД.
2. Время предложенного момента — в 24-часовом формате ЧЧ:ММ, независимо
   от того, как оно записано в тексте (через двоеточие, дефис, словами
   и т.п.).
3. Место — город и страна; если в тексте прямо даны координаты, верни их
   как "широта, долгота" (например "46.48, 30.72").
4. Само дело, для которого выбирается момент — дословно, как его
   сформулировал пользователь, не пересказывай своими словами.

Текст:
\"\"\"{text}\"\"\"

Если какого-то из этих пунктов в тексте ДЕЙСТВИТЕЛЬНО нет — напиши "нет" в
соответствующей строке, не выдумывай и не угадывай.

Ответь СТРОГО в этом формате, каждый пункт на отдельной строке, без
пояснений до или после:
ДАТА: <ГГГГ-ММ-ДД или нет>
ВРЕМЯ: <ЧЧ:ММ или нет>
МЕСТО: <город, страна ИЛИ широта, долгота ИЛИ нет>
ДЕЛО: <дословная формулировка или нет>"""


def _parse_extraction_field(label: str, answer: str) -> Optional[str]:
    m = re.search(rf"{label}\s*:\s*(.+)", answer, re.IGNORECASE)
    if not m:
        return None
    value = m.group(1).strip().strip('"').strip()
    if not value or value.lower() in ("нет", "нету", "n/a", "-", "—"):
        return None
    return value


def _extract_electional_fields_llm(round_text: str, require_date_time: bool = True) -> Optional[Dict[str, str]]:
    """require_date_time=False is used by the range-search path
    (_resolve_range_request): a real "which day is best" question often
    names no date/time at all — there's nothing to require there, only
    place and purpose are ever mandatory for it. The single-moment path
    keeps the strict default (both required, else None — a "moment" with
    no actual moment named isn't useful data to hand back)."""
    if llm_utils.get_llm() is None:
        return None
    try:
        answer = llm_utils.generate_sync(
            _FIELD_EXTRACTION_PROMPT.format(text=round_text), max_tokens=200, temperature=0.0,
        )
    except Exception:
        return None
    date = _parse_extraction_field("ДАТА", answer)
    time_ = _parse_extraction_field("ВРЕМЯ", answer)
    if require_date_time and not (date and time_):
        return None
    result: Dict[str, str] = {}
    if date:
        result["date"] = date
    if time_:
        result["time"] = time_
    place = _parse_extraction_field("МЕСТО", answer)
    if place:
        result["place"] = place
    purpose = _parse_extraction_field("ДЕЛО", answer)
    if purpose:
        result["purpose"] = purpose
    return result


def _resolve_place(place_text: str) -> Optional[Tuple[float, float, str]]:
    parts = [p.strip() for p in place_text.split(",")]
    if len(parts) == 2:
        try:
            lat, lon = float(parts[0]), float(parts[1])
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon, (astro._resolve_timezone(lat, lon) or "")
        except ValueError:
            pass
    city = astro._lookup_city_exact(place_text)
    if city:
        return city["latitude"], city["longitude"], city["timezone"]
    return None


# --- querent's own natal chart, if already established earlier in this
# conversation -----------------------------------------------------------
#
# User-requested improvement: if the querent (the person asking) already
# had their OWN natal chart built earlier in this same conversation (e.g.
# via astro_natal_chart), the election shouldn't only check the generic,
# impersonal house-I/house-of-the-matter significators every election
# uses — it can ALSO check real transits from each candidate moment to
# that specific person's own natal Sun/Moon/Ascendant, the same way
# astro.run_transit already reads "what's currently happening" for someone
# by comparing a moment's chart against their birth chart. This is
# genuinely optional and additive: most electional requests carry no
# earlier natal-chart context at all, and the classical significator
# scoring above is already a complete, self-sufficient judgment on its
# own — when no querent birth data is found, nothing here changes.
_QUERENT_NATAL_PROMPT = """Тебе показаны более РАННИЕ сообщения пользователя в этом диалоге — до его
текущего запроса на элективную карту (выбор благоприятного момента).

Определи: называл ли пользователь где-то в них СВОИ СОБСТВЕННЫЕ точные
дату, время и место РОЖДЕНИЯ — то есть данные для построения именно ЕГО
ЛИЧНОЙ натальной карты? Это НЕ дата/время/место самого предполагаемого
дела (свадьбы, поездки и т.п.) и НЕ данные другого человека (партнёра,
ребёнка, компании) — только данные рождения самого спрашивающего.

Более ранние сообщения:
\"\"\"{text}\"\"\"

Если такие данные ЕСТЬ и явно относятся к самому спрашивающему — извлеки
их. Если их нет, или неясно, чьи это данные, или это данные другого
человека — напиши "нет" по всем пунктам; не выдумывай и не угадывай.

Ответь СТРОГО в этом формате, каждый пункт на отдельной строке, без
пояснений:
ДАТА: <ГГГГ-ММ-ДД или нет>
ВРЕМЯ: <ЧЧ:ММ или нет>
МЕСТО: <город, страна ИЛИ широта, долгота ИЛИ нет>"""


def _extract_querent_natal_fields_llm(history_text: str) -> Optional[Dict[str, str]]:
    """LLM-first, same as every other free-text judgment in this module —
    returns {"date", "time"?, "place"} if the querent's OWN birth data was
    found somewhere in history_text, else None. Time is optional (defaults
    to noon downstream, same as astro._build_subject already does
    elsewhere) since even an approximate Sun/Moon check is better than
    none — Ascendant is simply not checked when there is no time (see
    _querent_natal_transit_score). Date and place are both required: no
    natal subject can be built at all without a place (kerykeion needs
    lat/lon/tz), and a person's Sun/Moon sign both hinge on the date."""
    if llm_utils.get_llm() is None or not history_text.strip():
        return None
    try:
        answer = llm_utils.generate_sync(
            _QUERENT_NATAL_PROMPT.format(text=history_text), max_tokens=100, temperature=0.0,
        )
    except Exception:
        return None
    date = _parse_extraction_field("ДАТА", answer)
    place = _parse_extraction_field("МЕСТО", answer)
    if not (date and place):
        return None
    result = {"date": date, "place": place}
    time_ = _parse_extraction_field("ВРЕМЯ", answer)
    if time_:
        result["time"] = time_
    return result


def _build_querent_natal_subject(history_text: str) -> Tuple[Optional[Any], Optional[str]]:
    """Returns (natal_subject, description) if the querent's own birth
    data could be found AND resolved to a real place, else (None, None).
    Deliberately fails silently (no error surfaced to the user) on any
    problem — this is a bonus check layered onto an already-complete
    judgment, never something an election should be blocked on. Uses
    astro._ACTIVE_POINTS_TRANSIT (not the full natal point set) since only
    Sun/Moon/Ascendant are ever read from it (see
    _querent_natal_transit_score) — fixed stars and the Vertex would be
    wasted computation here."""
    querent_birth = _extract_querent_natal_fields_llm(history_text)
    if not querent_birth:
        return None, None
    place_resolved = _resolve_place(querent_birth["place"])
    if place_resolved is None:
        return None, None
    lat, lon, tz = place_resolved
    fields = {
        "date": querent_birth["date"], "time": querent_birth.get("time", "12:00"),
        "lat": str(lat), "lon": str(lon), "tz": tz,
    }
    try:
        subject = astro._build_subject(fields, name="querent", active_points=astro._ACTIVE_POINTS_TRANSIT)
    except Exception:
        return None, None
    desc = (
        f"{querent_birth['date']} {querent_birth.get('time', '12:00 (время не названо, взято по умолчанию)')}, "
        f"{querent_birth['place']}"
    )
    return subject, desc


# Which natal points the transit-to-querent's-own-chart check reads, and
# which classical benefic/malefic planets of the ELECTION moment itself
# count for it — deliberately narrow (Sun/Moon/Ascendant are the three
# most personally-read points in any transit reading; the full 10-planet
# grid astro.get_dual_chart_profiles renders for a prose digest would be
# far more than a single scoring signal needs, and would cost real time
# repeated over up to _MAX_SEARCH_WINDOW_DAYS*24 candidate hours).
_QUERENT_NATAL_TARGETS = {"Sun": "Солнце", "Moon": "Луна", "Ascendant": "Асцендент"}
_TRANSIT_BENEFICS = {"Jupiter", "Venus"}
_TRANSIT_MALEFICS = {"Mars", "Saturn"}


def _querent_natal_transit_score(querent_natal_subject, moment_subject) -> Tuple[int, List[str]]:
    """Lightweight transit-style check: does THIS candidate moment's own
    chart send a favorable or hard aspect from a classical benefic/malefic
    to the querent's own natal Sun/Moon/Ascendant? Uses
    AspectsFactory.dual_chart_aspects directly — the same kerykeion call
    astro.get_dual_chart_profiles wraps for run_transit's own prose digest
    — but pulls out only this module's own small scoring signal instead of
    that function's full per-planet profile text, which isn't needed here
    and isn't cheap to build hundreds of times over a range search.
    Ascendant is skipped entirely when querent_natal_subject has no real
    birth time (kerykeion still returns SOME Ascendant value for a
    default noon time, but it would be meaningless, so it's excluded via
    the caller never having asked for angle-dependent scoring here in the
    first place — Sun/Moon are what actually matter when the time is only
    approximate)."""
    from kerykeion import AspectsFactory

    aspects = AspectsFactory.dual_chart_aspects(
        querent_natal_subject, moment_subject,
        active_points=astro._ASPECT_ACTIVE_POINTS, active_aspects=astro._CLASSICAL_ASPECTS_WIDE,
    ).aspects
    aspects = astro.filter_classical_aspects(aspects)

    score = 0
    notes: List[str] = []
    seen = set()
    for a in aspects:
        moment_pt, natal_pt = (
            (a.p1_name, a.p2_name) if a.p1_owner == moment_subject.name else (a.p2_name, a.p1_name)
        )
        if natal_pt not in _QUERENT_NATAL_TARGETS:
            continue
        if moment_pt not in _TRANSIT_BENEFICS and moment_pt not in _TRANSIT_MALEFICS:
            continue
        key = (moment_pt, natal_pt, a.aspect)
        if key in seen:
            continue
        seen.add(key)
        natal_label = _QUERENT_NATAL_TARGETS[natal_pt]
        moment_label = astro._point_ru(moment_pt)
        if moment_pt in _TRANSIT_BENEFICS and a.aspect in _FAVORABLE_ASPECTS:
            score += 1
            notes.append(f"+1: {moment_label} момента в {astro._aspect_ru(a.aspect)} к натальн. {natal_label} кверента")
        elif moment_pt in _TRANSIT_MALEFICS and a.aspect in _HARD_ASPECTS:
            score -= 1
            notes.append(f"-1: {moment_label} момента в {astro._aspect_ru(a.aspect)} к натальн. {natal_label} кверента")
    return score, notes


# --- new-round detection — mirrors utils/horary.py's own mechanism
# exactly, for the same reason: two different elections asked back to back
# in one conversation must not silently share a place/purpose. -----------

_NEW_ROUND_PROMPT = """Тебе показано ПОСЛЕДНЕЕ сообщение пользователя в диалоге о выборе
благоприятного момента (элективная астрология) — БЕЗ предыдущей истории
переписки. Определи: это НОВЫЙ, самостоятельный запрос на оценку момента
(даже если в нём не хватает каких-то деталей — даты, времени, места или
самого дела, это всё равно НОВЫЙ запрос, если он поднимает новое дело или
новый момент) — или это ПРОДОЛЖЕНИЕ/уточнение уже заданного ранее запроса,
БЕЗ нового дела и без нового кандидата-момента?

ПРОДОЛЖЕНИЕМ считается, в частности: просьба объяснить оценку ("почему
так", "что это значит"), а также просьба расширить/сузить/сдвинуть уже
идущий ПОИСК лучшего момента для ТОГО ЖЕ дела ("расширь диапазон до
года", "поищи подольше", "а если взять полгода", "сузь до двух недель") —
это тоже ПРОДОЛЖЕНИЕ, а не новый запрос, даже если в нём явно названо
число дней/дата: сам диапазон поиска — не новое дело и не новый
кандидат-момент, а уточнение уже заданного.

Сообщение: "{message}"

Ответь СТРОГО одним словом, без пояснений: НОВЫЙ или ПРОДОЛЖЕНИЕ"""


def _classify_new_electional_round(message: str) -> Optional[bool]:
    if llm_utils.get_llm() is None:
        return None
    try:
        answer = llm_utils.generate_sync(
            _NEW_ROUND_PROMPT.format(message=message), max_tokens=10, temperature=0.0,
        )
    except Exception:
        return None
    upper = answer.strip().upper()
    if "НОВ" in upper:
        return True
    if "ПРОДОЛЖ" in upper:
        return False
    return None


_ROUND_LOOKBACK_CAP = 8


def _collect_current_round_texts(prior_user_texts: List[str]) -> List[str]:
    collected: List[str] = []
    for text in reversed(prior_user_texts[-_ROUND_LOOKBACK_CAP:]):
        collected.append(text)
        verdict = _classify_new_electional_round(text)
        if verdict is not False:
            break
    return list(reversed(collected))


def _missing_fields_message(missing: List[str]) -> str:
    return (
        "Не хватает данных для элективной карты: " + ", ".join(missing) + ". "
        "Нужны точные дата и время предлагаемого момента, место (координаты в любом "
        "виде, например 46.4667, 30.7333) и само дело, для которого выбирается момент."
    )


# --- range-search-style request classification ------------------------------
#
# v1 only ever evaluates ONE named candidate moment (see module docstring's
# "Known, accepted v1 scope limits") — there is no date-range search here.
# A request phrased as "на какой день лучше..." ("which day is best for...")
# is asking for exactly that unimplemented search, not for a judgment on a
# single moment someone has already picked. Classified via the LLM, same as
# every other decision this module makes about free-form text (category,
# new-round-or-continuation, field extraction) — a fixed regex phrase list
# was tried first and rejected on real feedback: it can't tell a genuinely
# proposed moment apart from the moment a question just happened to be
# asked, can't handle paraphrases, and is exactly the kind of judgment
# this app's local model is meant to make instead of pattern-matching
# strings (per the project's own established "LLM-first, deterministic
# fallback only as a last resort" rule, applied everywhere else already).
_MODE_PROMPT = """Тебе показан текст запроса пользователя об элективной астрологии (выборе
благоприятного момента для дела).

ВАЖНОЕ ПРАВИЛО: наличие в тексте КАКОЙ-ЛИБО даты и времени НЕ означает,
что это МОМЕНТ. Люди часто указывают дату и время просто потому, что это
момент, когда задаётся сам вопрос (как в хорарной астрологии) — а не
потому, что именно этот момент предлагается оценить. Решающий признак —
это САМА ФОРМУЛИРОВКА вопроса о деле, а не присутствие даты рядом с ним:

- МОМЕНТ: вопрос спрашивает про КОНКРЕТНЫЙ, уже выбранный момент —
  "подходит ли ЭТОТ момент", "хорошо ли начинать СЕЙЧАС", "удачно ли
  6 августа в 15:00 для...".
- ПОИСК: вопрос спрашивает, КАКОЙ момент лучше, БЕЗ указания конкретного
  кандидата для проверки — "когда лучше...", "на какой день лучше...",
  "в какое время лучше...", "какой день выбрать для...". Это ПОИСК, ДАЖЕ
  ЕСЛИ рядом в тексте стоит какая-то дата и время — это просто момент,
  когда задан сам вопрос, а не предложенный кандидат.

Пример: текст 'Сделай элективную карту: 07-08-2026 15:16:00 Одесса "На
какой день лучше планировать уборку в комнате?"' — это ПОИСК, а не
МОМЕНТ: дата 07-08-2026 15:16 здесь — просто время, когда задан вопрос,
а сама формулировка ("на какой день лучше") явно просит подбор, а не
оценку именно этой даты.

Текст запроса:
\"\"\"{text}\"\"\"

Ещё раз: смотри на формулировку вопроса о деле, а НЕ на то, есть ли в
тексте дата. Ответь СТРОГО одним словом, без пояснений: МОМЕНТ или
ПОИСК."""


def _classify_request_mode(text: str) -> str:
    """Returns "range" (the user is asking which day/moment is best — a
    search, not a single judgment) or "single" (one specific moment was
    proposed to evaluate). LLM-first and, deliberately, with no string-
    matching fallback of its own: when no model is loaded at all this
    defaults to "single", the same behavior this tool had before
    range-search detection existed, rather than guessing via keywords.
    Real testing surfaced the need for this: given "07-08-2026 15:16:00
    Одесса" plus "На какой день лучше планировать уборку в комнате?", the
    tool used to quietly treat 07-08-2026 15:16 as the proposed election
    moment (it's just the moment the question happened to be asked,
    mirroring horary's own "cast for when you're asking" convention) and
    hand back a confident verdict about a moment nobody actually
    proposed."""
    if llm_utils.get_llm() is None:
        return "single"
    try:
        answer = llm_utils.generate_sync(
            _MODE_PROMPT.format(text=text), max_tokens=10, temperature=0.0,
        )
    except Exception:
        return "single"
    return "range" if "ПОИСК" in answer.strip().upper() else "single"


# --- planetary hours ------------------------------------------------------
#
# Chaldean-order planetary hours per Andreeva's own formula (see
# electional_methodology.txt section 2): each weekday has its own ruling
# planet (the day's "commissioner"); daylight (sunrise-to-sunset) and
# nighttime (sunset-to-next-sunrise) are each divided into 12 EQUAL, but
# generally unequal-to-60-minutes, "temporal hours"; hour 1 of the day
# (starting at sunrise) is always the day's own ruler, and every
# subsequent hour advances one step through the fixed Chaldean sequence
# (Saturn, Jupiter, Mars, Sun, Venus, Mercury, Moon), wrapping and
# continuing seamlessly across the sunset/sunrise boundary. Computed here
# via pyswisseph's own rise_trans (the same ephemeris kerykeion itself
# wraps) — not a separate approximation.
_CHALDEAN_ORDER = ["Сатурн", "Юпитер", "Марс", "Солнце", "Венера", "Меркурий", "Луна"]
_WEEKDAY_RULER_EN = {
    "Sunday": "Солнце", "Monday": "Луна", "Tuesday": "Марс", "Wednesday": "Меркурий",
    "Thursday": "Юпитер", "Friday": "Венера", "Saturday": "Сатурн",
}


def _find_sun_event(start_jd: float, lat: float, lon: float, rsmi: int) -> Optional[float]:
    import swisseph as swe

    try:
        res, tret = swe.rise_trans(start_jd, swe.SUN, rsmi, (lon, lat, 0.0))
    except Exception:
        return None
    if res != 0:
        return None
    return tret[0]


def _jd_to_utc_datetime(jd: float):
    import swisseph as swe
    from datetime import datetime, timezone

    y, m, d, h = swe.revjul(jd)
    hour = int(h)
    minute_frac = (h - hour) * 60
    minute = int(minute_frac)
    second = int(round((minute_frac - minute) * 60))
    if second == 60:
        second = 0
        minute += 1
    if minute == 60:
        minute = 0
        hour += 1
    return datetime(y, m, d, hour % 24, minute, second, tzinfo=timezone.utc)


def compute_planetary_hour(target_jd: float, lat: float, lon: float, tz_str: str) -> Optional[Dict[str, Any]]:
    """Returns {"day_ruler", "hour_ruler", "hour_index"} for the given
    moment (target_jd, Julian Day UT — e.g. subject.julian_day from
    astro._build_subject), or None if the calculation fails (a genuine
    ephemeris/geometry failure — e.g. a circumpolar location/date where
    the Sun doesn't rise or set at all — not merely "unlikely", so callers
    must treat None as "day/hour ruler unavailable for this moment" and
    skip that part of the report, not as an error to surface to the user)."""
    import swisseph as swe
    from zoneinfo import ZoneInfo

    search_start = target_jd - 1.2
    prev_sunrise: Optional[float] = None
    next_sunrise: Optional[float] = None
    for _ in range(5):
        candidate = _find_sun_event(search_start, lat, lon, swe.CALC_RISE)
        if candidate is None:
            return None
        if candidate <= target_jd:
            prev_sunrise = candidate
            search_start = candidate + 0.0005
        else:
            next_sunrise = candidate
            break
    if prev_sunrise is None or next_sunrise is None:
        return None

    sunset = _find_sun_event(prev_sunrise + 0.0005, lat, lon, swe.CALC_SET)
    if sunset is None:
        return None

    if target_jd < sunset:
        seg_start, seg_end, base_index = prev_sunrise, sunset, 0
    else:
        seg_start, seg_end, base_index = sunset, next_sunrise, 12
    hour_len = (seg_end - seg_start) / 12.0
    if hour_len <= 0:
        return None
    idx = int((target_jd - seg_start) / hour_len)
    idx = max(0, min(11, idx))
    hour_index = base_index + idx

    try:
        utc_dt = _jd_to_utc_datetime(prev_sunrise)
        local_dt = utc_dt.astimezone(ZoneInfo(tz_str)) if tz_str else utc_dt
    except Exception:
        return None
    weekday_en = local_dt.strftime("%A")
    day_ruler = _WEEKDAY_RULER_EN.get(weekday_en)
    if day_ruler is None:
        return None
    hour_ruler = _CHALDEAN_ORDER[(_CHALDEAN_ORDER.index(day_ruler) + hour_index) % 7]
    return {"day_ruler": day_ruler, "hour_ruler": hour_ruler, "hour_index": hour_index}


# --- Moon phase -------------------------------------------------------------

def _moon_phase_info(sun_abs_pos: float, moon_abs_pos: float) -> Dict[str, Any]:
    """Sun-Moon elongation (0-360, Moon minus Sun) classifies the phase
    into the coarse categories electional_methodology.txt section 2
    describes — not a precise 8-phase (new/crescent/first quarter/
    gibbous/full/...) system, just what that section's scoring actually
    needs: waxing vs waning, and whether the moment sits close enough to
    an exact quarter to count as a "crisis point" in its own right."""
    elongation = (moon_abs_pos - sun_abs_pos) % 360.0
    is_new = elongation <= 3.0 or elongation >= 357.0
    is_full = 177.0 <= elongation <= 183.0
    is_exact_quarter = (87.0 <= elongation <= 93.0) or (267.0 <= elongation <= 273.0)
    waxing = elongation < 180.0

    if is_exact_quarter:
        label, score = "точная четверть (кризисная точка)", -1
    elif is_new:
        label, score = "новолуние", 0
    elif is_full:
        label, score = "полнолуние", 0
    elif waxing:
        label, score = "растущая", 1
    else:
        label, score = "убывающая", 0
    return {"elongation": elongation, "label": label, "score": score, "waxing": waxing}


# --- main computation ---------------------------------------------------------

_CHART_ACTIVE_POINTS = astro._ACTIVE_POINTS_TRANSIT


def _point_for_label(subject, label: str):
    attr = next((a for lbl, a in astro._PLANET_ATTRS if lbl == label), None)
    return getattr(subject, attr, None) if attr else None


def _kerykeion_name_for_label(label: str) -> Optional[str]:
    attr = next((a for lbl, a in astro._PLANET_ATTRS if lbl == label), None)
    return astro.attr_to_kerykeion_name(attr) if attr else None


def _third_point_aspects(aspects, target_kery_name: str) -> Dict[str, Tuple[str, float, str]]:
    result: Dict[str, Tuple[str, float, str]] = {}
    for a in aspects:
        if a.p1_name == target_kery_name:
            result[a.p2_name] = (a.aspect_movement, a.orbit, a.aspect)
        elif a.p2_name == target_kery_name:
            result[a.p1_name] = (a.aspect_movement, a.orbit, a.aspect)
    return result


def _resolve_single_moment_request(spec: str) -> Dict[str, Any]:
    """Resolves date/time/place/purpose/category for the single-moment
    path — date and time are REQUIRED here (this is "is this named moment
    good", not a search). Returns {"fields", "purpose_text", "category",
    "category_source"} or {"error": "..."}. Split out of
    _compute_electional_chart so the range-search path
    (_resolve_range_request) can sit next to it without one enormous
    function doing both jobs."""
    llm_fields = _extract_electional_fields_llm(spec)
    raw = astro._parse_spec(spec)

    if llm_fields:
        place_resolved = _resolve_place(llm_fields["place"]) if llm_fields.get("place") else None
        if place_resolved is None:
            return {"error": _missing_fields_message(["место"]) + (
                " Модель распознала дату и время момента, но не смогла уверенно "
                "определить место — уточните его явно (город или координаты)."
            )}
        lat, lon, tz = place_resolved
        fields = {
            "date": llm_fields["date"], "time": llm_fields["time"],
            "lat": str(lat), "lon": str(lon), "tz": tz,
        }
        purpose_text = llm_fields.get("purpose") or "дело без явной формулировки"
        # If the model resolved date/time/place but not a clean purpose
        # phrase, still classify against the raw spec text rather than the
        # display placeholder above — the keyword fallback classifier
        # (_classify_category_keywords) just does substring search, so
        # there's no reason to hand it a sentence that can't possibly
        # contain any of its keywords.
        classification_text = llm_fields.get("purpose") or spec
    else:
        fields, missing = astro._extract_fields(spec)
        if missing:
            return {"error": _missing_fields_message(missing)}
        if not raw.get("lat") or not raw.get("lon"):
            coord_lat, coord_lon = astro._find_coordinates(spec)
            if coord_lat is not None:
                fields["lat"], fields["lon"] = str(coord_lat), str(coord_lon)
            else:
                exact_city = astro._lookup_city_exact(spec)
                if exact_city is None:
                    return {"error": _missing_fields_message(["место"]) + (
                        " В тексте не нашлось ни точных координат, ни однозначного "
                        "названия города — уточните место явно."
                    )}
                fields["lat"] = str(exact_city["latitude"])
                fields["lon"] = str(exact_city["longitude"])
                fields["tz"] = exact_city["timezone"]
            if not fields.get("tz") and fields.get("lat") and fields.get("lon"):
                tz = astro._resolve_timezone(float(fields["lat"]), float(fields["lon"]))
                if tz:
                    fields["tz"] = tz
        purpose_text = "дело без явной формулировки"
        # No model at all: classify straight from the raw spec text — the
        # only source of any category keywords in this fallback path (see
        # the LLM branch's own comment on classification_text above).
        classification_text = spec

    category, category_source = _classify_category(classification_text)
    return {
        "fields": fields, "purpose_text": purpose_text,
        "category": category, "category_source": category_source,
    }


def _resolve_range_request(spec: str) -> Dict[str, Any]:
    """Resolves the anchor moment, place, purpose, and category for a
    range-search request. Unlike _resolve_single_moment_request, date/time
    are OPTIONAL here — require_date_time=False, since a real "which day
    is best" question often names no date at all (e.g. "когда лучше
    подписывать договор в Одессе?"). Place is still required (houses
    can't be computed without it). If no date/time was named, the anchor
    defaults to right now, in the resolved place's own local time — it's
    only ever a starting point for _search_best_electional_moment to scan
    forward from, not something that needs to be named precisely.
    Returns {"anchor_date", "anchor_time", "lat", "lon", "tz",
    "purpose_text", "category", "category_source"} or {"error": "..."}."""
    llm_fields = _extract_electional_fields_llm(spec, require_date_time=False)
    place_text = llm_fields.get("place") if llm_fields else None
    purpose_text = (llm_fields.get("purpose") if llm_fields else None) or spec
    date_str = llm_fields.get("date") if llm_fields else None
    time_str = llm_fields.get("time") if llm_fields else None

    place_resolved = _resolve_place(place_text) if place_text else None
    if place_resolved is None:
        # Deterministic fallback for place only — same lookup path used
        # elsewhere in this module when the model is unavailable.
        coord_lat, coord_lon = astro._find_coordinates(spec)
        if coord_lat is not None:
            place_resolved = (coord_lat, coord_lon, astro._resolve_timezone(coord_lat, coord_lon) or "")
        else:
            exact_city = astro._lookup_city_exact(spec)
            if exact_city:
                place_resolved = (exact_city["latitude"], exact_city["longitude"], exact_city["timezone"])
    if place_resolved is None:
        return {"error": _missing_fields_message(["место"]) + (
            " Для поиска благоприятного момента в любом случае нужно знать "
            "место — уточните его явно (город или координаты)."
        )}
    lat, lon, tz = place_resolved

    if date_str and time_str:
        anchor_date, anchor_time = date_str, time_str
    else:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        try:
            now_local = datetime.now(ZoneInfo(tz)) if tz else datetime.utcnow()
        except Exception:
            now_local = datetime.utcnow()
        anchor_date, anchor_time = now_local.strftime("%Y-%m-%d"), now_local.strftime("%H:%M")

    category, category_source = _classify_category(purpose_text)
    return {
        "anchor_date": anchor_date, "anchor_time": anchor_time,
        "lat": lat, "lon": lon, "tz": tz,
        "purpose_text": purpose_text, "category": category, "category_source": category_source,
    }


# Default search window when the text names no explicit end date for the
# range — the user's own chosen trade-off (see the AskUserQuestion this
# was decided through): 30 days, scanned at full HOURLY granularity (not
# day-level scoring + hour-refinement), because planetary-hour rulership
# changes every hour and contributes its own +/-1 to the score — a
# day-level pass would risk missing a good hour inside an otherwise
# mediocre day. This means up to 30*24 = 720 full chart builds per search;
# accepted explicitly as a cost worth paying for not missing anything, the
# same way rectification_events.py's own window scan accepts scanning a
# whole range of candidate birth times instead of a light heuristic.
_SEARCH_WINDOW_DAYS = 30

# Sane upper bound regardless of what a user explicitly requests (see
# _extract_window_days_llm below) — even an explicit ask for something
# very large (or a misparsed extraction) shouldn't turn into an unbounded
# hourly scan. Raised from an earlier, tighter 180 to 370 specifically to
# accommodate a real, explicitly requested use case ("в диапазоне года")
# without silently clamping a year down to six months — 370*24 = 8880
# candidate hours, measured in practice (see run_electional_chart's own
# runtime-estimate note below) at well under a second per candidate on
# ordinary hardware once ephemeris files are warm, so a year-long search
# taking a few minutes is a real but acceptable cost for what's being
# asked, not a runaway computation.
_MAX_SEARCH_WINDOW_DAYS = 370

# Ordinal rank for the verdict-tier comparison _search_best_electional_moment
# uses ahead of raw score — see that function's own docstring for why
# ranking must go by verdict category first, not flat score.
_VERDICT_RANK = {"неблагоприятно": 0, "смешанно": 1, "благоприятно": 2}

_WINDOW_PROMPT = """Тебе показан текст запроса на поиск благоприятного момента (элективная
астрология). По умолчанию поиск идёт в окне 30 дней вперёд от ближайшей
названной даты. Определи, назвал ли пользователь ЯВНО другую длину или
границу этого периода поиска:

1. Если названо ЧИСЛО дней/недель/месяцев/лет ("в течение 60 дней", "на
   следующей неделе", "в течение полугода", "в диапазоне года", "в
   течение квартала") — переведи в целое число ДНЕЙ (неделя = 7 дней,
   месяц = 30 дней, квартал = 90 дней, полугодие = 180 дней, год = 365
   дней).
2. Если названа КОНКРЕТНАЯ конечная дата или узнаваемый период календаря
   ("до 30 сентября", "до конца сентября", "в октябре") — назови её как
   ГГГГ-ММ-ДД (последний день месяца/периода, если назван период, а не
   точная дата); год бери из контекста запроса, если он явно не назван —
   ближайший будущий.

Если пользователь явно НЕ указал ничего про длину или границу поиска —
по обоим пунктам напиши "нет". Не выдумывай и не угадывай, если в тексте
действительно ничего об этом нет.

Текст запроса:
\"\"\"{text}\"\"\"

Ответь СТРОГО в этом формате, каждый пункт на отдельной строке, без
пояснений:
ДНЕЙ: <целое число или нет>
ДАТА: <ГГГГ-ММ-ДД или нет>"""


def _extract_window_days_llm(text: str, anchor_date: str) -> Optional[int]:
    """Returns an explicit search-window length in days if the user named
    one (a broader/narrower range than the _SEARCH_WINDOW_DAYS default),
    or None to keep the default. Deliberately has the model extract only
    a plain unit count (week/month conversion is fixed and low-risk) or a
    literal calendar date — the actual date-difference ARITHMETIC (end
    date minus anchor date) is done in Python via date subtraction, never
    trusted to the model itself. This follows the same principle already
    applied everywhere else in this app's astro tools (see
    _CATEGORY_TABLE's own comment on category->house mapping being a
    plain dict lookup, never left for the model to compute): never let
    the LLM do arithmetic or invent a fact it could get wrong when a
    handful of lines of real code can do it exactly instead."""
    if llm_utils.get_llm() is None:
        return None
    try:
        answer = llm_utils.generate_sync(
            _WINDOW_PROMPT.format(text=text), max_tokens=30, temperature=0.0,
        )
    except Exception:
        return None
    days_str = _parse_extraction_field("ДНЕЙ", answer)
    date_str = _parse_extraction_field("ДАТА", answer)

    if date_str:
        try:
            from datetime import date as _date
            end = _date.fromisoformat(date_str)
            start = _date.fromisoformat(anchor_date)
            delta = (end - start).days
            if delta > 0:
                return min(delta, _MAX_SEARCH_WINDOW_DAYS)
        except Exception:
            pass
    if days_str:
        try:
            days = int(days_str)
            if days > 0:
                return min(days, _MAX_SEARCH_WINDOW_DAYS)
        except ValueError:
            pass
    return None


def _search_best_electional_moment(
    anchor_date: str, anchor_time: str, lat: float, lon: float, tz: str,
    category: str, category_source: str, purpose_text: str,
    window_days: int = _SEARCH_WINDOW_DAYS,
    querent_natal_subject: Optional[Any] = None, querent_natal_desc: Optional[str] = None,
) -> Dict[str, Any]:
    """Scans forward hour-by-hour from the anchor moment across
    window_days, computing a full deterministic electional judgment for
    every candidate hour via _compute_electional_chart_core, and returns
    the single best one found.

    Ranking is THREE-tiered, deliberately NOT a flat "highest raw score
    wins": (1) never a Moon-void moment if a non-void one exists anywhere
    in the window — Moon void of course is a hard override to
    "неблагоприятно" regardless of score (see
    _compute_electional_chart_core); (2) among candidates equally free of
    that override, prefer by VERDICT CATEGORY first — благоприятно beats
    смешанно beats неблагоприятно — via _VERDICT_RANK; (3) only within
    the same verdict category does the raw point score act as a
    tie-breaker. This was a real correction: comparing candidates by raw
    score alone let a "смешанно" moment with an unusually high positive
    score outrank a genuinely "благоприятно" one with a lower score,
    which isn't what "search for the best day" should mean — the
    qualitative presence of favorable indicators (the verdict itself,
    which the classical factors already determine) should decide first,
    with the numeric score used only as the reference/tie-break detail it
    actually is, not as the primary ranking criterion (see
    _compute_electional_chart_core's own note on the score being this
    module's own engineering device for comparing candidates, not a
    quoted classical formula). Ties within the same tier keep the
    EARLIEST candidate: the loop only ever replaces the running best on a
    STRICT improvement, so a later, equally-good hour never displaces an
    earlier one.

    Stepping is done in UTC (anchor_utc + timedelta(hours=i), converted to
    local time per candidate only for display/kerykeion's own local-time
    input) rather than adding hours directly to a local aware datetime —
    this sidesteps DST fold/gap ambiguity, since exact 1-hour UTC steps
    are always unambiguous where naive local-wall-clock arithmetic across
    a DST transition would not be."""
    from datetime import datetime, timedelta, timezone as dt_timezone
    from zoneinfo import ZoneInfo

    try:
        tzinfo = ZoneInfo(tz) if tz else dt_timezone.utc
    except Exception:
        tzinfo = dt_timezone.utc
    anchor_naive = datetime.strptime(f"{anchor_date} {anchor_time}", "%Y-%m-%d %H:%M")
    anchor_local = anchor_naive.replace(tzinfo=tzinfo)
    anchor_utc = anchor_local.astimezone(dt_timezone.utc)

    total_hours = window_days * 24
    best: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[int, int, int]] = None
    evaluated = 0
    errors = 0

    for i in range(total_hours):
        candidate_utc = anchor_utc + timedelta(hours=i)
        try:
            candidate_local = candidate_utc.astimezone(tzinfo)
        except Exception:
            candidate_local = candidate_utc
        fields = {
            "date": candidate_local.strftime("%Y-%m-%d"),
            "time": candidate_local.strftime("%H:%M"),
            "lat": str(lat), "lon": str(lon), "tz": tz,
        }
        try:
            candidate_result = _compute_electional_chart_core(
                fields, category, category_source, purpose_text,
                querent_natal_subject=querent_natal_subject, querent_natal_desc=querent_natal_desc,
            )
        except Exception:
            # One bad candidate hour (a transient ephemeris edge case)
            # shouldn't abort the whole scan — skip it and keep going,
            # same principle rectification_events.py's own window scan
            # already applies to its candidate birth times.
            errors += 1
            continue
        evaluated += 1
        key = (
            0 if candidate_result["moon_void"] else 1,
            _VERDICT_RANK[candidate_result["verdict"]],
            candidate_result["score"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best = candidate_result

    return {
        "best": best, "evaluated": evaluated, "errors": errors,
        "window_days": window_days, "anchor_date": anchor_date, "anchor_time": anchor_time, "tz": tz,
    }


def _compute_electional_chart_core(
    fields: Dict[str, str], category: str, category_source: str, purpose_text: str,
    querent_natal_subject: Optional[Any] = None, querent_natal_desc: Optional[str] = None,
) -> Dict[str, Any]:
    """Given already-resolved date/time/place fields and an already-
    classified category, computes the full deterministic electional
    judgment (querent/quesited significators, Moon status, day/hour
    rulers, score, verdict). This is the reusable core shared by both
    single-moment evaluation (_compute_electional_chart) and the
    date-range search (_search_best_electional_moment) — factored out so
    the search loop doesn't have to re-run LLM field/category extraction
    or place lookup for every one of potentially hundreds of scanned
    candidate hours; only this deterministic part (ephemeris + houses +
    aspects + scoring — no LLM calls) repeats per candidate.

    querent_natal_subject (optional, see _build_querent_natal_subject) is
    the querent's own already-built natal chart, if one was found earlier
    in the conversation — when given, this also scores real transits from
    THIS candidate moment to that person's own natal Sun/Moon/Ascendant
    (see _querent_natal_transit_score), on top of the classical, always-
    present significator scoring below. querent_natal_desc is a plain
    display string for the report only, carried alongside the subject
    itself purely so the search loop doesn't have to thread a second,
    separately-computed value through every candidate."""
    subject = astro._build_subject(fields, name="electional", active_points=_CHART_ACTIVE_POINTS)
    cusps = astro._house_cusp_degrees(subject)

    quesited_house, sympathetic, avoid_hour, mercury_sensitive = _CATEGORY_TABLE[category]

    result: Dict[str, Any] = {
        "purpose_text": purpose_text,
        "date": fields["date"], "time": fields.get("time", "12:00"), "tz": fields.get("tz", ""),
        "category": category, "category_source": category_source,
        "quesited_house": quesited_house,
    }

    from kerykeion import AspectsFactory

    aspects = AspectsFactory.natal_aspects(
        subject, active_points=astro._ASPECT_ACTIVE_POINTS, active_aspects=astro._CLASSICAL_ASPECTS_WIDE,
    ).aspects
    aspects = astro.filter_classical_aspects(aspects)

    querent_sign, _ = astro._sign_from_abs_pos(cusps[0])
    quesited_sign, _ = astro._sign_from_abs_pos(cusps[quesited_house - 1])
    querent_label = astro._CLASSICAL_RULERS_RU.get(querent_sign)
    quesited_label = astro._CLASSICAL_RULERS_RU.get(quesited_sign)
    result["querent_label"], result["querent_sign"] = querent_label, querent_sign
    result["quesited_label"], result["quesited_sign"] = quesited_label, quesited_sign

    querent_point = _point_for_label(subject, querent_label)
    quesited_point = _point_for_label(subject, quesited_label)
    sun, mars, saturn, moon, mercury = subject.sun, subject.mars, subject.saturn, subject.moon, subject.mercury

    querent_house = astro._house_of_degree(cusps, querent_point.abs_pos)
    quesited_house_actual = astro._house_of_degree(cusps, quesited_point.abs_pos)
    querent_strength, querent_notes = _assess_strength(querent_label, querent_point, querent_point.sign, querent_house, sun, mars, saturn)
    quesited_strength, quesited_notes = _assess_strength(quesited_label, quesited_point, quesited_point.sign, quesited_house_actual, sun, mars, saturn)
    result["querent_strength"], result["querent_notes"] = querent_strength, querent_notes
    result["quesited_strength"], result["quesited_notes"] = quesited_strength, quesited_notes
    result["querent_house"], result["quesited_house_actual"] = querent_house, quesited_house_actual

    same_ruler = querent_label == quesited_label
    result["same_ruler"] = same_ruler

    score = 0
    scoring_notes: List[str] = []

    # --- Moon: void-of-course, phase, via combusta/combust/besieged -----
    moon_remaining_deg = 30.0 - moon.position
    moon_speed = moon.speed if moon.speed else 13.2
    time_to_exit = moon_remaining_deg / abs(moon_speed) if moon_speed else None

    moon_candidates = []
    for a in aspects:
        if a.aspect_movement != "Applying":
            continue
        if a.p1_name != "Moon" and a.p2_name != "Moon":
            continue
        other_speed = a.p2_speed if a.p1_name == "Moon" else a.p1_speed
        relative_speed = abs(moon_speed - other_speed)
        if relative_speed <= 0:
            continue
        time_to_perfect = a.orbit / relative_speed
        if time_to_exit is not None and time_to_perfect < time_to_exit:
            other_name = a.p2_name if a.p1_name == "Moon" else a.p1_name
            moon_candidates.append((time_to_perfect, a.aspect, other_name))

    moon_void = not moon_candidates
    result["moon_void"] = moon_void
    if moon_candidates:
        moon_candidates.sort(key=lambda c: c[0])
        _, next_aspect_type, next_aspect_other = moon_candidates[0]
        result["moon_next_aspect"] = f"{astro._aspect_ru(next_aspect_type)} — {astro._point_ru(next_aspect_other)}"
        if next_aspect_type in _FAVORABLE_ASPECTS and next_aspect_other not in ("Mars", "Saturn"):
            score += 1
            scoring_notes.append("+1: ближайший аспект Луны благоприятен")
        elif next_aspect_type in _HARD_ASPECTS or next_aspect_other in ("Mars", "Saturn"):
            score -= 1
            scoring_notes.append("-1: ближайший аспект Луны неблагоприятен")
    else:
        result["moon_next_aspect"] = None

    phase_info = _moon_phase_info(sun.abs_pos, moon.abs_pos)
    result["moon_phase"] = phase_info
    score += phase_info["score"]
    if phase_info["score"]:
        scoring_notes.append(f"{phase_info['score']:+d}: фаза Луны — {phase_info['label']}")

    moon_via_combusta = _on_via_combusta(moon.abs_pos)
    moon_combust = _is_combust(moon, sun)
    moon_besieged = _is_besieged(moon, mars, saturn)
    result["moon_via_combusta"], result["moon_combust"], result["moon_besieged"] = (
        moon_via_combusta, moon_combust, moon_besieged,
    )
    for flag, note in ((moon_via_combusta, "Луна на Via Combusta"), (moon_combust, "Луна комбустна"), (moon_besieged, "Луна в осаде")):
        if flag:
            score -= 1
            scoring_notes.append(f"-1: {note}")

    # --- Mercury retrograde (only for mercury-sensitive categories) -----
    mercury_retrograde = bool(getattr(mercury, "retrograde", False))
    result["mercury_retrograde"] = mercury_retrograde
    result["mercury_sensitive"] = mercury_sensitive
    if mercury_sensitive and mercury_retrograde:
        score -= 1
        scoring_notes.append("-1: Меркурий ретрограден (значимо для этой категории дела)")

    # --- day/hour ruler ---------------------------------------------------
    hour_info = compute_planetary_hour(subject.julian_day, float(fields["lat"]), float(fields["lon"]), fields.get("tz", ""))
    result["hour_info"] = hour_info
    if hour_info:
        ruler = hour_info["hour_ruler"]
        if ruler in sympathetic:
            score += 1
            scoring_notes.append(f"+1: час {ruler} благоприятствует этой категории дела")
        elif ruler in avoid_hour:
            score -= 1
            scoring_notes.append(f"-1: час {ruler} классически противопоказан для этой категории дела")
        # Day-of-week ruler check (Andreeva's checklist, per user request):
        # previously only the HOUR ruler was scored here — the day ruler
        # was computed (compute_planetary_hour already returns it) and
        # displayed in the report, but never itself checked for the same
        # sympathetic/avoid compatibility the hour ruler gets. Uses the
        # SAME per-category sets as the hour check (Andreeva's own per-
        # planet day AND hour recommendations largely overlap — she
        # doesn't keep two separate lists per category), scored
        # independently and more lightly (+/-1, same weight as the hour
        # check — day and hour are each one line of Andreeva's checklist,
        # not one weighted above the other).
        day_ruler = hour_info["day_ruler"]
        if day_ruler in sympathetic:
            score += 1
            scoring_notes.append(f"+1: день недели ({day_ruler}) благоприятствует этой категории дела")
        elif day_ruler in avoid_hour:
            score -= 1
            scoring_notes.append(f"-1: день недели ({day_ruler}) классически противопоказан для этой категории дела")

    # --- querent's own natal chart, if one was found earlier in this
    # conversation (see _build_querent_natal_subject) — optional, additive
    # on top of everything above. ------------------------------------
    result["querent_natal_desc"] = querent_natal_desc
    if querent_natal_subject is not None:
        natal_score, natal_notes = _querent_natal_transit_score(querent_natal_subject, subject)
        result["querent_natal_checked"] = True
        result["querent_natal_notes"] = natal_notes
        score += natal_score
        scoring_notes.extend(natal_notes)
    else:
        result["querent_natal_checked"] = False
        result["querent_natal_notes"] = []

    # --- significator strength -------------------------------------------
    if same_ruler:
        if querent_strength == "сильный":
            score += 1
            scoring_notes.append(f"+1: общий сигнификатор ({querent_label}) силён")
        elif querent_strength == "слабый":
            score -= 1
            scoring_notes.append(f"-1: общий сигнификатор ({querent_label}) слаб")
    else:
        for label, strength in ((querent_label, querent_strength), (quesited_label, quesited_strength)):
            if strength == "сильный":
                score += 1
                scoring_notes.append(f"+1: сигнификатор ({label}) силён")
            elif strength == "слабый":
                score -= 1
                scoring_notes.append(f"-1: сигнификатор ({label}) слаб")

    # --- direct aspect / translation / collection of light ---------------
    direct_aspect = None
    translation = collection = None
    if not same_ruler:
        querent_kery = _kerykeion_name_for_label(querent_label)
        quesited_kery = _kerykeion_name_for_label(quesited_label)
        for a in aspects:
            names = {a.p1_name, a.p2_name}
            if names == {querent_kery, quesited_kery}:
                direct_aspect = (a.aspect_movement, a.aspect)
                break
        if direct_aspect:
            movement, aspect_type = direct_aspect
            if movement == "Applying" and aspect_type in _FAVORABLE_ASPECTS:
                score += 1
                scoring_notes.append("+1: сходящийся благоприятный аспект между сигнификаторами")
            elif movement == "Applying":
                score -= 1
                scoring_notes.append("-1: сходящийся неблагоприятный аспект между сигнификаторами")
            # a separating aspect scores 0 — already in the past, neither
            # helps nor actively hurts the moment being evaluated now.
        else:
            to_querent = _third_point_aspects(aspects, querent_kery)
            to_quesited = _third_point_aspects(aspects, quesited_kery)
            common = (set(to_querent) & set(to_quesited)) - {querent_kery, quesited_kery}
            for name in common:
                mv_q, _, _ = to_querent[name]
                mv_k, _, _ = to_quesited[name]
                if {mv_q, mv_k} == {"Applying", "Separating"} and translation is None:
                    translation = name
                if mv_q == "Applying" and mv_k == "Applying" and collection is None:
                    collection = name
            if translation or collection:
                score += 1
                scoring_notes.append("+1: передача/собирание света между сигнификаторами")
    result["direct_aspect"] = direct_aspect
    result["translation"] = astro._point_ru(translation) if translation else None
    result["collection"] = astro._point_ru(collection) if collection else None

    result["score"] = score
    result["scoring_notes"] = scoring_notes

    if moon_void:
        # Hard override, not just another scoring point — see
        # electional_methodology.txt section 2 ("самостоятельный и сильный
        # отрицательный фактор... не рекомендуется, даже если всё
        # остальное выглядит хорошо").
        verdict = "неблагоприятно"
    elif score >= 2:
        verdict = "благоприятно"
    elif score <= -2:
        verdict = "неблагоприятно"
    else:
        verdict = "смешанно"
    result["verdict"] = verdict
    return result


def _compute_electional_chart(
    spec: str, querent_natal_subject: Optional[Any] = None, querent_natal_desc: Optional[str] = None,
) -> Dict[str, Any]:
    """Entry point used by run_electional_chart. Classifies the request
    via _classify_request_mode as either a single named moment to judge,
    or a range-search ("which day/moment is best"), and dispatches to the
    matching resolution + computation path.

    Returns a dict shaped for _build_report_lines (same keys either way —
    querent_label, score, verdict, etc., all coming from
    _compute_electional_chart_core), or {"error": "..."}. A range result
    additionally carries "mode": "range" and "search" (the scan's own
    metadata: window scanned, candidates evaluated/skipped, anchor) — the
    winning candidate's own fields are merged straight into the top level
    of the returned dict so _build_report_lines never has to know or care
    whether it's rendering a direct single-moment judgment or the best
    candidate a search happened to find.

    querent_natal_subject/querent_natal_desc (see _build_querent_natal_subject)
    are pass-through only — this function never extracts or resolves them
    itself, since that lookup needs the FULL conversation history
    (spec here is already just the current round's own text, per
    run_electional_chart's HISTORY_MARKER split)."""
    if _classify_request_mode(spec) == "range":
        resolved = _resolve_range_request(spec)
        if "error" in resolved:
            return resolved
        window_days = _extract_window_days_llm(spec, resolved["anchor_date"]) or _SEARCH_WINDOW_DAYS
        search = _search_best_electional_moment(
            resolved["anchor_date"], resolved["anchor_time"],
            resolved["lat"], resolved["lon"], resolved["tz"],
            resolved["category"], resolved["category_source"], resolved["purpose_text"],
            window_days=window_days,
            querent_natal_subject=querent_natal_subject, querent_natal_desc=querent_natal_desc,
        )
        if search["best"] is None:
            return {"error": (
                f"Не удалось построить ни одной карты-кандидата за {search['window_days']} дней "
                f"вперёд от {search['anchor_date']} {search['anchor_time']} ({search['tz']}) — "
                "вероятно, проблема с расчётом восхода/захода для этого места (например, "
                "приполярная широта). Уточните место или конкретный момент для проверки."
            )}
        # lat/lon carried alongside the winning candidate's own date/time/tz
        # (already in search["best"]) purely so run_electional_chart_and_
        # subject can rebuild that candidate's chart for drawing without
        # re-running this whole window scan a second time — see that
        # function's own docstring.
        return {"mode": "range", "search": search, "lat": resolved["lat"], "lon": resolved["lon"], **search["best"]}

    resolved = _resolve_single_moment_request(spec)
    if "error" in resolved:
        return resolved
    result = _compute_electional_chart_core(
        resolved["fields"], resolved["category"], resolved["category_source"], resolved["purpose_text"],
        querent_natal_subject=querent_natal_subject, querent_natal_desc=querent_natal_desc,
    )
    result["lat"], result["lon"] = resolved["fields"]["lat"], resolved["fields"]["lon"]
    return result


# --- report formatting --------------------------------------------------------

_VERDICT_MARKER_RE = re.compile(r"^ИТОГОВАЯ ОЦЕНКА.*$", re.MULTILINE)
_BEST_MOMENT_MARKER_RE = re.compile(r"^ИТОГОВЫЙ ЛУЧШИЙ МОМЕНТ.*$", re.MULTILINE)


def extract_best_recommendation(report_text: str) -> Optional[str]:
    """Same bookend/extractor pattern as horary.py/rectification.py/
    rectification_events.py, for the same reason: a small local model can
    contradict — or, worse, silently invent a different concrete fact
    over — a tool's own computed result if asked to reason over it
    freely. Two possible marker lines depending on which mode produced
    this report (see run_electional_chart): a range-search result's
    headline fact is a DATE (checked first — per
    rectification_events.py's own finding that a model can swap in a
    different, physically implausible date in free prose, exactly the
    same risk here), a single-moment result's headline fact is a
    favorable/unfavorable judgment about the one already-named moment."""
    m = _BEST_MOMENT_MARKER_RE.search(report_text)
    if m:
        return m.group(0)
    m = _VERDICT_MARKER_RE.search(report_text)
    return m.group(0) if m else None


def _build_report_lines(data: Dict[str, Any]) -> List[str]:
    """Renders the querent/quesited/Moon/hour-ruler/score breakdown for
    ONE computed judgment — used both directly (single-moment path) and
    as the "here's the winning candidate's own details" body of a
    range-search report (run_electional_chart prepends the search header
    in that case). Deliberately does not know or care whether `data` came
    from a direct single-moment computation or is the best-in-window
    candidate a search found — the shape is identical either way (see
    _compute_electional_chart's own docstring)."""
    verdict_line = f"ИТОГОВАЯ ОЦЕНКА: {data['verdict'].capitalize()} (суммарный балл {data['score']:+d})."

    lines = [
        f"Дело: «{data['purpose_text']}» — категория: {_CATEGORY_LABELS_RU[data['category']]} "
        f"(дом {data['quesited_house']}); момент {data['date']} {data['time']}, {data['tz']}.",
        "",
        verdict_line,
        "",
        f"Кверент: управитель I дома — {data['querent_label']} "
        f"{astro._sign_ru_prepositional(data['querent_sign'])}, "
        f"дом {data['querent_house']} — {data['querent_strength']}"
        + (f" ({', '.join(data['querent_notes'])})" if data["querent_notes"] else "") + ".",
    ]
    if not data["same_ruler"]:
        lines.append(
            f"Сигнификатор дела: управитель дома {data['quesited_house']} — {data['quesited_label']} "
            f"{astro._sign_ru_prepositional(data['quesited_sign'])}, "
            f"дом {data['quesited_house_actual']} — {data['quesited_strength']}"
            + (f" ({', '.join(data['quesited_notes'])})" if data["quesited_notes"] else "") + "."
        )
    else:
        lines.append("Кверент и сигнификатор дела управляются ОДНОЙ и той же планетой.")

    lines.append("")
    phase = data["moon_phase"]
    lines.append(f"Луна: фаза — {phase['label']} (элонгация от Солнца {phase['elongation']:.1f}°).")
    if data["moon_void"]:
        lines.append("Луна без курса до выхода из текущего знака (значимых аспектов не образует).")
    else:
        lines.append(f"Луна не без курса; ближайший образующийся аспект: {data['moon_next_aspect']}.")
    for flag, note in (
        (data["moon_via_combusta"], "Луна на Via Combusta."),
        (data["moon_combust"], "Луна комбустна (в соединении с Солнцем)."),
        (data["moon_besieged"], "Луна в осаде (между Марсом и Сатурном)."),
    ):
        if flag:
            lines.append(note)

    if data["direct_aspect"]:
        movement, aspect_type = data["direct_aspect"]
        lines.append(
            f"Прямой аспект между сигнификаторами: {astro._aspect_ru(aspect_type)}, "
            f"{astro._movement_ru(movement).lower()}."
        )
    elif data["translation"]:
        lines.append(f"Прямого аспекта между сигнификаторами нет; передача света через {data['translation']}.")
    elif data["collection"]:
        lines.append(f"Прямого аспекта между сигнификаторами нет; собирание света через {data['collection']}.")
    elif not data["same_ruler"]:
        lines.append("Между сигнификаторами нет ни прямого аспекта, ни передачи/собирания света.")

    if data["mercury_sensitive"]:
        lines.append(
            f"Меркурий {'ретрограден' if data['mercury_retrograde'] else 'не ретрограден'} "
            "(значимо для этой категории дела)."
        )

    hour_info = data["hour_info"]
    if hour_info:
        lines.append(
            f"Управитель дня: {hour_info['day_ruler']}; управитель текущего планетарного часа: "
            f"{hour_info['hour_ruler']} (час №{hour_info['hour_index'] + 1} от восхода)."
        )
    else:
        lines.append("Управитель дня/часа не вычислен (не удалось определить восход/закат для этого места и даты).")

    if data.get("querent_natal_checked"):
        lines.append(
            f"Дополнительно проверены транзиты этого момента к натальной карте кверента "
            f"(данные рождения из более раннего сообщения в этом диалоге: {data['querent_natal_desc']})"
            + (
                ": " + "; ".join(n.split(": ", 1)[1] for n in data["querent_natal_notes"])
                if data["querent_natal_notes"]
                else " — значимых транзитных аспектов к Солнцу/Луне/Асценденту не найдено."
            ) + "."
        )

    lines.append("")
    lines.append(
        (
            "Слагаемые оценки (справочная сумма для сравнения моментов между собой — "
            "не цитата из источников, конкретные значения дают только качественную "
            "картину, не заявляй их как классическую формулу): "
            + "; ".join(data["scoring_notes"])
        )
        if data["scoring_notes"]
        else "Слагаемые оценки: нейтрально, значимых факторов не найдено."
    )
    lines.append("")
    lines.append(verdict_line)
    return lines


def run_electional_chart(spec: str) -> str:
    """Tool entry point (utils.tools.TOOL_REGISTRY["astro_electional_chart"])
    — thin wrapper over run_electional_chart_and_subject, discarding the
    winning moment's chart object."""
    return run_electional_chart_and_subject(spec)[0]


def run_electional_chart_and_subject(spec: str) -> Tuple[str, Optional[Any]]:
    """Does the actual work; returns (report_text, winning_moment_subject_
    or_None). Split out from run_electional_chart so routes/chat.py's
    chart-drawing step can get the winning moment's chart WITHOUT
    re-running a potentially expensive range search a second time (up to
    _MAX_SEARCH_WINDOW_DAYS*24 candidate hours) — same reasoning
    rectification.py's own _run_rectification_trutine_full and
    rectification_events.py's own run_rectification_events_and_subject_
    async have, for the same class of expensive-search tool. The subject
    is rebuilt (a single cheap _build_subject call, not a search) from the
    winning candidate's own date/time/lat/lon/tz, which _compute_
    electional_chart carries alongside the result for exactly this reason
    (see its own "lat"/"lon" comments).

    routes/chat.py appends the FULL prior-conversation user text after
    HISTORY_MARKER, past whatever round-scoped tool_arg it already built —
    split that off FIRST, before any of the normal round-scoped
    extraction runs on it (see HISTORY_MARKER's own comment for why this
    needs its own channel rather than widening the round-scoped prompts).
    The history section, if present, is used ONLY to look for the
    querent's own natal birth data (_build_querent_natal_subject) — a
    silent, best-effort lookup that changes nothing about the election
    itself when it comes up empty, which is the common case."""
    main_spec, _, history_text = spec.partition(HISTORY_MARKER)
    querent_natal_subject, querent_natal_desc = _build_querent_natal_subject(history_text) if history_text else (None, None)

    data = _compute_electional_chart(
        main_spec, querent_natal_subject=querent_natal_subject, querent_natal_desc=querent_natal_desc,
    )
    if "error" in data:
        return data["error"], None

    winning_subject: Optional[Any] = None
    try:
        winning_subject = astro._build_subject(
            {"date": data["date"], "time": data["time"], "lat": str(data["lat"]), "lon": str(data["lon"]), "tz": data["tz"]},
            name="electional",
        )
    except Exception:
        winning_subject = None

    lines = _build_report_lines(data)

    if data.get("mode") == "range":
        search = data["search"]
        # Headline fact for a range search is the DATE, not a
        # favorable/unfavorable label — mirrors rectification_events.py's
        # own "ИТОГОВЫЙ ЛУЧШИЙ ВАРИАНТ ВРЕМЕНИ РОЖДЕНИЯ" bookend exactly
        # (same marker-line pattern, same reasoning: a concrete found
        # fact needs to survive the follow-up model's own free prose
        # verbatim, and bookending both ends of the report is the
        # established mitigation for "lost in the middle" small-model
        # attention behaviour — see that module's own comment on this).
        # The score is explicitly labeled "справочно" here — see
        # _build_report_lines' own scoring-notes line and
        # electional_methodology.txt: it's this module's own ranking aid
        # for comparing candidates against each other, not a value quoted
        # from any classical source, and the verdict tier (see
        # _VERDICT_RANK/_search_best_electional_moment) — not the raw
        # score — is what actually decided this was the best candidate.
        best_moment_line = (
            f"ИТОГОВЫЙ ЛУЧШИЙ МОМЕНТ: {data['date']} {data['time']} ({data['tz']}) — "
            f"категория «{_CATEGORY_LABELS_RU[data['category']]}», качественная оценка "
            f"«{data['verdict']}» (балл {data['score']:+d}, справочно, не основной критерий выбора)."
        )
        header = [
            f"Поиск благоприятного момента: окно {search['window_days']} дней вперёд от "
            f"{search['anchor_date']} {search['anchor_time']} ({search['tz']}); "
            f"проверено кандидатов (по часам): {search['evaluated']}"
            + (f", ошибок расчёта: {search['errors']}" if search["errors"] else "") + "."
            + (
                " (окно шире обычного — расчёт такого поиска мог занять заметно больше "
                "времени, чем обычный запрос)."
                if search["window_days"] > 90 else ""
            )
            + (
                " В каждом кандидате также проверялись транзиты к натальной карте кверента "
                f"({data['querent_natal_desc']})."
                if data.get("querent_natal_checked") else ""
            ),
            "",
            best_moment_line,
            "",
            "Ниже — разбор именно этого найденного момента. Ранжирование кандидатов шло "
            "СНАЧАЛА по наличию благоприятных показателей (качественная оценка момента), "
            "и только при совпадении оценок — по справочному баллу; это не обязательно "
            "первый подходящий момент по времени, а лучший из всего проверенного окна.",
        ]
        if data["verdict"] != "благоприятно":
            header.append(
                "Внимание: даже лучший момент в этом окне не набрал уверенно "
                "благоприятную оценку — среди проверенных кандидатов не нашлось "
                "по-настоящему хорошего варианта. Можно попробовать более широкое "
                "окно поиска или другую формулировку дела."
            )
        header.append("")
        lines = header + lines
        # Bookend repeat at the very end — see best_moment_line's own
        # comment above for why.
        lines.append("")
        lines.append(best_moment_line)

    return "\n".join(lines), winning_subject
