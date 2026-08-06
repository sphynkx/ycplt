"""Horary astrology — deterministic engine, classical method per Masenkov's
"Построение хорарной карты" (plus general-technique chapters of Lavoie's
"Lose This Book...", excluding its lost-object-specific chapters and any
Vedic material — see the corpus review this module's design followed).

Single tool, always RAG-interpreted — astro_horary_question (TOOL_REGISTRY,
routes/chat.py's _INTERPRETED_TOOL_NAMES) computes the full chart
deterministically (radicality, significators, dignities, Moon status,
aspects/translation/collection, verdict), then that report becomes the
"computed facts" context for the same RAG-augmented follow-up call every
other astro_* tool already uses, reasoning against horary_methodology.txt
(rag_data/astro_horar/) — exactly the natal/transit/synastry pattern, no
special-casing.

This REPLACES an earlier two-tier design (a separate always-instant,
never-interpreted "short verdict" tool, with a second "give details" tool
the router had to be talked into picking for a follow-up) — reverted after
real testing (a real "who took my things" horary case) showed the
short-verdict-only reply was a poor default: a genuinely rich, radical
chart went completely uninterpreted unless the user knew to explicitly ask
for more, which isn't how a real horary question naturally gets asked.
Real testing also showed the hard radicality veto was too blunt — per
Lavoie, even a non-radical chart still carries situational detail worth
explaining (WHY the matter can't/won't proceed), not a blank refusal — so
_check_radicality's finding no longer skips computation at all, only adds
a caveat to the facts the model reasons over (see _compute_horary_chart).

The rectification tools' own lesson (a small local model reliably
CONTRADICTS a tool's own computed verdict if asked to reason over it) still
applies here just as much — horary_methodology.txt's own "authority of the
computed verdict" section is the mitigation, the same role the prepend/
disclaimer/bookend safety net plays for rectification; this module's own
ИТОГОВЫЙ ВЕРДИКТ bookend (extract_best_recommendation) is the matching
code-side half of that same mitigation, always active now (not gated
behind a config toggle the way rectification's is), since — unlike
rectification's small-model tests — the user explicitly wants this
follow-up call on for horary regardless of model capability.

Known, accepted v1 scope limits (documented here rather than in the
methodology file, since these are engine limitations, not interpretation
guidance):
  - "derived house" resolution (Masenkov's multi-hop chain method for
    questions like "my cousin's dog") is NOT implemented — only a single-hop
    topic->house classification (_TOPIC_HOUSE_KEYWORDS, deterministic
    keyword match, same accepted-approximation spirit as
    rectification_events.py's _EVENT_HOUSE_KEYWORDS) or an explicit
    'house=N' override. Multi-hop questions get whatever single house the
    keywords land on, which may be wrong for a genuinely nested relationship
    question — acceptable for v1's common case (the querent's own direct
    topics: love, career, health, money, a lost object, etc.), a real gap
    for anything further removed.
  - Essential dignity (_EXALTATION table below) covers only the 7 classical
    planets, per traditional horary practice — Uranus/Neptune/Pluto have no
    classical rulership/exaltation at all (they didn't exist as visible
    bodies when the system was codified) and are treated as dignity-neutral,
    scored only by house angularity/combustion/aspect like any other point.
  - Void-of-course and "last aspect of Moon" are computed by projecting each
    applying aspect's time-to-perfection (orb / relative daily speed)
    against the Moon's own time-to-sign-exit (remaining degrees / Moon's
    daily speed) — a real computation, not a guess, but still an
    approximation: it assumes both planets' speeds stay constant over that
    short window, which is accurate enough for the few-hours-to-few-days
    windows involved here.
  - Translation/collection of light are detected from kerykeion's own
    applying/separating classification per pair, not by independently
    re-deriving exact-perfection timestamps for every third-point aspect —
    consistent with how every other aspect in this module (and the rest of
    this app, e.g. astro.py) is read.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from utils import astro

# --- essential dignity (7 classical planets only — see module docstring) ---

_EXALTATION: Dict[str, str] = {
    "Солнце": "Ari", "Луна": "Tau", "Меркурий": "Vir", "Венера": "Pis",
    "Марс": "Cap", "Юпитер": "Can", "Сатурн": "Lib",
}

_ZODIAC = astro._ZODIAC_SIGN_CODES  # Ari..Pis, zodiacal order, reused as-is


def _opposite_sign(sign_code: str) -> str:
    i = _ZODIAC.index(sign_code)
    return _ZODIAC[(i + 6) % 12]


# planet label -> signs it rules (1 or 2 signs), built from astro's own
# rulership table (astro._CLASSICAL_RULERS_RU is sign->ruler; this is the
# inverse) so the two tables can never silently drift apart.
_RULES: Dict[str, List[str]] = {}
for _sign_code, _ruler_label in astro._CLASSICAL_RULERS_RU.items():
    _RULES.setdefault(_ruler_label, []).append(_sign_code)

_DETRIMENT: Dict[str, List[str]] = {
    planet: [_opposite_sign(s) for s in signs] for planet, signs in _RULES.items()
}
_FALL: Dict[str, str] = {planet: _opposite_sign(sign) for planet, sign in _EXALTATION.items()}


def _dignity(planet_label: str, sign_code: str) -> str:
    """One of "обитель"/"экзальтация"/"изгнание"/"падение"/"" (neutral —
    either an outer planet, per the module docstring, or simply none of the
    four apply)."""
    if sign_code in _RULES.get(planet_label, []):
        return "обитель"
    if _EXALTATION.get(planet_label) == sign_code:
        return "экзальтация"
    if sign_code in _DETRIMENT.get(planet_label, []):
        return "изгнание"
    if _FALL.get(planet_label) == sign_code:
        return "падение"
    return ""


# --- degree-math helpers -----------------------------------------------------

def _angular_diff(a: float, b: float) -> float:
    """Shortest angular distance between two absolute ecliptic degrees, 0-180."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


_VIA_COMBUSTA_START = 195.0  # 15° Libra
_VIA_COMBUSTA_END = 225.0    # 15° Scorpio
_VIA_COMBUSTA_SPICA_EXCEPTION = (202.5, 204.5)  # ~23-24° Libra, per Masenkov


def _on_via_combusta(abs_pos: float) -> bool:
    d = abs_pos % 360.0
    if not (_VIA_COMBUSTA_START <= d < _VIA_COMBUSTA_END):
        return False
    lo, hi = _VIA_COMBUSTA_SPICA_EXCEPTION
    return not (lo <= d <= hi)


_COMBUST_ORB = 8.0  # degrees from the Sun — see module docstring's scope note


def _is_combust(point, sun) -> bool:
    if point is sun:
        return False
    return _angular_diff(point.abs_pos, sun.abs_pos) <= _COMBUST_ORB


_SIEGE_ORB = 15.0  # "captured" threshold — see class methodology doc


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
    """Returns (overall_verdict, notes) — overall_verdict is "сильный"/
    "слабый"/"нейтральный", per the point-tally rule in horary_methodology.txt
    section 3 (count of strengthening vs weakening factors, not a weighted
    score)."""
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


# --- radicality (validity) check --------------------------------------------

_RADICALITY_INVALID_DEGREE = 3.0  # first/last 3 degrees of the Ascendant's sign


def _check_radicality(subject, cusps: List[float]) -> Tuple[bool, List[str]]:
    """Returns (is_radical, notes) — is_radical=False means the hard
    criterion fired (Ascendant too close to a sign boundary) and nothing
    else about the chart should be read at all. Otherwise `notes` lists
    whichever SECONDARY (non-blocking, cumulative-caution-only) factors were
    found — see horary_methodology.txt section 1. The Saturn-applying-
    hard-aspect-to-house-VII-occupant secondary factor is checked
    separately in _compute_horary_chart, once aspects are available."""
    asc = subject.first_house
    if asc.position < _RADICALITY_INVALID_DEGREE or asc.position > 30.0 - _RADICALITY_INVALID_DEGREE:
        return False, [
            f"Асцендент в {asc.position:.1f}° знака — в пределах "
            f"{_RADICALITY_INVALID_DEGREE:.0f}° от начала или конца знака."
        ]

    notes: List[str] = []
    moon = subject.moon
    if _on_via_combusta(moon.abs_pos):
        notes.append("Луна на Via Combusta.")
    if _on_via_combusta(asc.abs_pos):
        notes.append("Асцендент на Via Combusta.")

    saturn_house = astro._house_of_degree(cusps, subject.saturn.abs_pos)
    if saturn_house == 7:
        notes.append("Сатурн в VII доме.")

    seventh_sign, _ = astro._sign_from_abs_pos(cusps[6])
    if seventh_sign in ("Cap", "Aqu"):
        notes.append("Козерог/Водолей на куспиде VII дома.")

    return True, notes


# --- topic -> quesited house (single-hop, see module docstring's scope note) -

_TOPIC_HOUSE_KEYWORDS: List[Tuple[List[str], int]] = [
    (["свадьб", "брак", "женить", "женил", "замуж", "партнер", "партнёр", "супруг", "роман", "отношени"], 7),
    (["любов", "влюб", "встреч", "познаком", "свидан"], 5),
    (["ребен", "ребён", "беремен", "зачат", "роды", "родила", "родилс"], 5),
    (["работ", "должност", "карьер", "повышен", "увол", "трудоустрой", "вакан", "испытательн"], 10),
    (["бизнес", "предприят", "сделк", "контракт", "клиент"], 7),
    (["денег", "доход", "займ", "кредит", "зарплат", "финанс", "выигрыш"], 2),
    (["суд", "иск", "тяжб", "юрист", "адвокат"], 7),
    (["путешеств", "поездк", "командировк", "виза", "эмиграц", "переезд за границ"], 9),
    (["экзамен", "учеб", "институт", "универ", "диплом", "школ"], 9),
    (["операц", "болезн", "здоров", "диагноз", "выздоров", "госпитал"], 6),
    (["потеря", "пропал", "найд", "украл", "кража", "вещ", "предмет"], 2),
    (["дом", "квартир", "недвижим", "жиль"], 4),
    (["друг", "дружб", "надежд"], 11),
    (["брат", "сестр", "сосед"], 3),
    (["мать", "мама", "родител"], 10),
    (["отец", "отц", "пап"], 4),
    (["смерт", "умр", "наследств"], 8),
    (["враг", "конкурент"], 7),
    (["питомц", "собак", "кошк", "кот "], 6),
]
_DEFAULT_QUESITED_HOUSE = 1


def _classify_topic_house(question_text: str) -> int:
    lowered = question_text.lower()
    for keywords, house in _TOPIC_HOUSE_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return house
    return _DEFAULT_QUESITED_HOUSE


_QUESTION_SENTENCE_RE = re.compile(r"[^.!?]*\?")


def _extract_question_text(spec: str) -> str:
    """Best-effort recovery of the question's own wording, for display only
    (never used to affect the actual chart computation). Horary questions
    are almost always phrased with a literal "?", so the primary path looks
    for one (avoids the mangled leftovers of stripping dates/coordinates
    out of a whole free-text blob — an earlier version did that and
    produced garbage like "года в в Киеве," from the unmatched remainder of
    a date/city phrase, confirmed via testing).

    Picks the LONGEST "?"-terminated match, not the first or the last — a
    real, reproducible bug found via testing: `spec` here is
    routes/chat.py's own concatenation of [req.query, decision.tool_arg,
    history_context] for _INTERPRETED_TOOL_NAMES, so more than one "?" can
    legitimately appear (the router's own short paraphrase of the same
    question, or an unrelated "?" from earlier conversation history) —
    picking a fixed position (first or last) picked up a short, contentless
    fragment in one observed case (a bare "?" with nothing useful before
    it), while the real, fully-phrased question was a much longer match
    elsewhere in the same string. The longest match is the best available
    proxy for "the real question" without actually knowing which part of
    `spec` came from which source. Falls back to the strip-everything
    approach only if no "?" is found at all (or every match is empty after
    stripping punctuation). Same accepted-approximation spirit as
    astro._lookup_city/_split_two_person_text — good enough to echo the
    question back to the user and to let a later "explain this" follow-up
    re-find it in conversation history, not a real NLP parse."""
    matches = [m.strip(" ,.;:-") for m in _QUESTION_SENTENCE_RE.findall(spec)]
    matches = [m for m in matches if m]
    if matches:
        return max(matches, key=len)

    text = spec
    for regex in (
        astro._DATE_ISO_RE, astro._DATE_DMY_NUM_RE, astro._DATE_RU_RE,
        astro._TIME_RE, astro._DMS_RE, astro._DECIMAL_PAIR_RE,
    ):
        text = regex.sub(" ", text)
    text = re.sub(r"[A-Za-zА-Яа-яЁё_]+\s*=\s*\S+", " ", text)  # key=value tokens
    text = re.sub(r"\s+", " ", text).strip(" ,.;:-")
    return text or "вопрос без явной формулировки"


# --- fields / missing-data message ------------------------------------------

def _missing_fields_message(missing: List[str]) -> str:
    return (
        "Не хватает данных для хорарной карты: " + ", ".join(missing) + ". "
        "Нужны точные дата и время, когда был ЗАДАН вопрос (не дата рождения), "
        "и место, где он был задан (координаты — в любом виде, например "
        "46.4667, 30.7333 или 46°28'00\"N 30°44'00\"E). Часовой пояс "
        "определяется автоматически по координатам."
    )


# --- main computation ---------------------------------------------------------

_CHART_ACTIVE_POINTS = astro._ACTIVE_POINTS_TRANSIT  # no fixed stars/Vertex — not read in horary


def _point_for_label(subject, label: str):
    attr = next((a for lbl, a in astro._PLANET_ATTRS if lbl == label), None)
    return getattr(subject, attr, None) if attr else None


def _kerykeion_name_for_label(label: str) -> Optional[str]:
    attr = next((a for lbl, a in astro._PLANET_ATTRS if lbl == label), None)
    return astro.attr_to_kerykeion_name(attr) if attr else None


def _third_point_aspects(aspects, target_kery_name: str) -> Dict[str, Tuple[str, float, str]]:
    """{other_point_kerykeion_name: (movement, orbit, aspect_type)} for every
    aspect involving target_kery_name — used to find translation/collection
    of light candidates (a third point aspecting both significators)."""
    result: Dict[str, Tuple[str, float, str]] = {}
    for a in aspects:
        if a.p1_name == target_kery_name:
            result[a.p2_name] = (a.aspect_movement, a.orbit, a.aspect)
        elif a.p2_name == target_kery_name:
            result[a.p1_name] = (a.aspect_movement, a.orbit, a.aspect)
    return result


_FAVORABLE_ASPECTS = {"trine", "sextile", "conjunction"}
_HARD_ASPECTS = {"square", "opposition", "quincunx"}


def _compute_horary_chart(spec: str) -> Dict[str, Any]:
    """Returns a dict of every computed fact plus the final verdict, or
    {"error": "..."} if the moment/place couldn't be resolved. Never skips
    computation on a failed radicality check anymore (see module
    docstring) — is_radical/radicality_notes are carried in the result
    alongside a real verdict either way."""
    fields, missing = astro._extract_fields(spec)
    if missing:
        return {"error": _missing_fields_message(missing)}

    raw = astro._parse_spec(spec)
    question_text = _extract_question_text(spec)

    subject = astro._build_subject(fields, name="horary", active_points=_CHART_ACTIVE_POINTS)
    cusps = astro._house_cusp_degrees(subject)

    is_radical, radicality_notes = _check_radicality(subject, cusps)

    from kerykeion import AspectsFactory

    aspects = AspectsFactory.natal_aspects(
        subject, active_points=astro._ASPECT_ACTIVE_POINTS, active_aspects=astro._ALL_ASPECTS,
    ).aspects

    # Secondary radicality factor that needs aspects to be known: an applying
    # hard aspect from Saturn to a planet actually sitting in house VII.
    # Checked regardless of is_radical (not gated behind it) — this is
    # informational context either way, and the hard radicality failure no
    # longer stops the rest of the computation from running at all (see
    # module docstring: even a non-radical chart still gets a full read,
    # per Lavoie, just with this caveat carried alongside the verdict
    # rather than blocking it).
    house7_occupants = {
        attr for label, attr in astro._PLANET_ATTRS
        if getattr(subject, attr, None) is not None
        and astro._house_of_degree(cusps, getattr(subject, attr).abs_pos) == 7
    }
    saturn_kery = "Saturn"
    for a in aspects:
        if a.aspect not in _HARD_ASPECTS or a.aspect_movement != "Applying":
            continue
        other = a.p2_name if a.p1_name == saturn_kery else (a.p1_name if a.p2_name == saturn_kery else None)
        if other is None:
            continue
        other_attr = next((attr for lbl, attr in astro._PLANET_ATTRS if astro.attr_to_kerykeion_name(attr) == other), None)
        if other_attr in house7_occupants:
            radicality_notes.append(f"Сходящийся {astro._aspect_ru(a.aspect)} Сатурна к планете в VII доме.")

    result: Dict[str, Any] = {
        "question_text": question_text,
        "date": fields["date"], "time": fields.get("time", "12:00"), "tz": fields.get("tz", ""),
        "is_radical": is_radical,
        "radicality_notes": radicality_notes,
    }

    quesited_house = _DEFAULT_QUESITED_HOUSE
    if raw.get("house"):
        try:
            quesited_house = max(1, min(12, int(raw["house"])))
        except ValueError:
            pass
    else:
        quesited_house = _classify_topic_house(question_text)
    result["quesited_house"] = quesited_house

    querent_sign, _ = astro._sign_from_abs_pos(cusps[0])
    quesited_sign, _ = astro._sign_from_abs_pos(cusps[quesited_house - 1])
    querent_label = astro._CLASSICAL_RULERS_RU.get(querent_sign)
    quesited_label = astro._CLASSICAL_RULERS_RU.get(quesited_sign)
    result["querent_label"], result["querent_sign"] = querent_label, querent_sign
    result["quesited_label"], result["quesited_sign"] = quesited_label, quesited_sign

    querent_point = _point_for_label(subject, querent_label)
    quesited_point = _point_for_label(subject, quesited_label)
    sun, mars, saturn, moon = subject.sun, subject.mars, subject.saturn, subject.moon

    querent_house = astro._house_of_degree(cusps, querent_point.abs_pos)
    quesited_house_actual = astro._house_of_degree(cusps, quesited_point.abs_pos)
    querent_strength, querent_notes = _assess_strength(querent_label, querent_point, querent_point.sign, querent_house, sun, mars, saturn)
    quesited_strength, quesited_notes = _assess_strength(quesited_label, quesited_point, quesited_point.sign, quesited_house_actual, sun, mars, saturn)
    result["querent_strength"], result["querent_notes"] = querent_strength, querent_notes
    result["quesited_strength"], result["quesited_notes"] = quesited_strength, quesited_notes
    result["querent_house"], result["quesited_house_actual"] = querent_house, quesited_house_actual

    mutual_reception = (
        querent_sign in _RULES.get(quesited_label, []) and quesited_sign in _RULES.get(querent_label, [])
    )
    result["mutual_reception"] = mutual_reception

    # --- Moon: void-of-course + last aspect before leaving its sign --------
    moon_remaining_deg = 30.0 - moon.position
    moon_speed = moon.speed if moon.speed else 13.2  # kerykeion always supplies this; fallback is the classical average
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
        _, last_aspect_type, last_aspect_other = moon_candidates[-1]
        # Dash format, not "к <name>" — matches astro.py's own aspect-line
        # convention (_format_natal_text) rather than a preposition, since a
        # preposition here would need the planet name in dative case (e.g.
        # "к Юпитеру", not the nominative "Юпитер" _point_ru always returns).
        result["moon_last_aspect"] = f"{astro._aspect_ru(last_aspect_type)} — {astro._point_ru(last_aspect_other)}"
    else:
        result["moon_last_aspect"] = None

    # --- direct significator aspect, translation/collection of light -------
    querent_kery = _kerykeion_name_for_label(querent_label)
    quesited_kery = _kerykeion_name_for_label(quesited_label)
    same_ruler = querent_label == quesited_label
    result["same_ruler"] = same_ruler

    direct_aspect = None
    if not same_ruler:
        for a in aspects:
            names = {a.p1_name, a.p2_name}
            if names == {querent_kery, quesited_kery}:
                direct_aspect = (a.aspect_movement, a.aspect)
                break
    result["direct_aspect"] = direct_aspect

    translation = collection = None
    if not same_ruler and direct_aspect is None:
        to_querent = _third_point_aspects(aspects, querent_kery)
        to_quesited = _third_point_aspects(aspects, quesited_kery)
        common = (set(to_querent) & set(to_quesited)) - {querent_kery, quesited_kery}
        for name in common:
            mv_q, _, _ = to_querent[name]
            mv_k, _, _ = to_quesited[name]
            if {mv_q, mv_k} == {"Applying", "Separating"} and translation is None:
                translation = name
            if mv_q == "Applying" and mv_k == "Applying" and collection is None:
                third_attr = next((attr for lbl, attr in astro._PLANET_ATTRS if astro.attr_to_kerykeion_name(attr) == name), None)
                third_point = getattr(subject, third_attr, None) if third_attr else None
                if third_point and third_point.speed is not None:
                    if abs(third_point.speed) < abs(querent_point.speed or 0) and abs(third_point.speed) < abs(quesited_point.speed or 0):
                        collection = name
    # Store the already-translated (Russian) display name — translation/
    # collection are only ever used for display and truthiness checks from
    # here on, never matched back against a kerykeion literal again.
    translation_ru = astro._point_ru(translation) if translation else None
    collection_ru = astro._point_ru(collection) if collection else None
    result["translation"], result["collection"] = translation_ru, collection_ru

    # --- final verdict -------------------------------------------------------
    if moon_void:
        verdict, reason = "negative", "Луна без курса (значимых аспектов до выхода из знака не образует)"
    elif same_ruler:
        verdict = "negative" if querent_strength == "слабый" else "positive"
        reason = "кверент и предмет вопроса имеют общего сигнификатора" + (
            " (сигнификатор слаб)" if verdict == "negative" else ""
        )
    elif direct_aspect:
        movement, aspect_type = direct_aspect
        aspect_ru = astro._aspect_ru(aspect_type)
        if movement == "Applying" and aspect_type in _FAVORABLE_ASPECTS:
            verdict, reason = "positive", f"сходящийся {aspect_ru} между сигнификаторами"
        elif movement == "Applying":
            verdict, reason = "negative", f"сходящийся неблагоприятный аспект ({aspect_ru}) между сигнификаторами"
        else:
            verdict, reason = "negative", f"расходящийся аспект ({aspect_ru}) между сигнификаторами — уже в прошлом"
    elif translation_ru:
        verdict, reason = "positive", f"передача света через {translation_ru}"
    elif collection_ru:
        verdict, reason = "positive", f"собирание света через {collection_ru}"
    else:
        verdict, reason = "negative", "между сигнификаторами нет ни аспекта, ни передачи/собирания света"

    if not is_radical:
        # No longer a hard stop (see module docstring) — the verdict above
        # is still the actual chart-based read, just carried alongside an
        # explicit caveat that the classical validity check itself failed,
        # so the RAG follow-up can present it as "here's what the chart
        # shows, but treat it with real caution" rather than either hiding
        # the caveat or refusing to interpret the chart at all.
        reason = (
            f"{reason}; ОДНАКО карта нерадикальна ({radicality_notes[0] if radicality_notes else 'Асцендент слишком близко к границе знака'}) "
            "— суждение ненадёжно, читай его как предварительное"
        )

    result["verdict"], result["reason"] = verdict, reason
    return result


# --- report formatting --------------------------------------------------------

_VERDICT_MARKER_RE = re.compile(r"^ИТОГОВЫЙ ВЕРДИКТ.*$", re.MULTILINE)


def extract_best_recommendation(report_text: str) -> Optional[str]:
    """Same bookend/extractor pattern as rectification.py/
    rectification_events.py, for the same reason: a small local model
    reliably contradicts a tool's own computed verdict if asked to reason
    over it freely — this is the code-side half of the mitigation (see
    module docstring), always active for this tool (unlike rectification's
    equivalent, which is gated behind a config toggle: horary always gets
    the RAG follow-up by design, so the bookend is always needed here, not
    just optionally)."""
    m = _VERDICT_MARKER_RE.search(report_text)
    return m.group(0) if m else None


def run_horary_question(spec: str) -> str:
    """Tool entry point (utils.tools.TOOL_REGISTRY["astro_horary_question"]).
    Computes the full chart deterministically and returns the FULL
    underlying report (radicality, significators and their dignities, Moon
    status, the key aspect or translation/collection of light, verdict) —
    this becomes the "computed facts" context for the normal RAG-augmented
    follow-up call every other astro_* tool already uses, reasoning against
    horary_methodology.txt (see routes/chat.py: this tool is in
    _INTERPRETED_TOOL_NAMES, no no-followup bypass — see module docstring
    for why this replaced an earlier short-verdict-only design)."""
    data = _compute_horary_chart(spec)
    if "error" in data:
        return data["error"]

    verdict_line = f"ИТОГОВЫЙ ВЕРДИКТ: {'Да' if data['verdict'] == 'positive' else 'Нет'} ({data['reason']})."

    lines = [
        f"Хорарный вопрос: «{data['question_text']}» (момент {data['date']} {data['time']}, {data['tz']}).",
        "",
        verdict_line,
        "",
    ]

    if not data["is_radical"]:
        lines.append(
            "ВНИМАНИЕ: карта НЕ прошла классическую проверку радикальности "
            "(" + (data["radicality_notes"][0] if data["radicality_notes"] else "Асцендент слишком близко к границе знака") + "). "
            "Вердикт ниже основан на реальном чтении карты, но его надёжность снижена — "
            "изложи это как предварительное, повышенно осторожное суждение, а не как обычный уверенный ответ."
        )
    lines.append(f"Тема вопроса определена как дом {data['quesited_house']}.")
    if data["is_radical"] and data["radicality_notes"]:
        # Only shown here for a RADICAL chart — the hard-fail case already
        # displayed its one note in the "ВНИМАНИЕ" block above and would
        # otherwise be repeated verbatim (radicality_notes holds either the
        # single hard-fail note OR the secondary notes list, never both).
        lines.append("Второстепенные факторы радикальности (не блокируют суждение): " + " ".join(data["radicality_notes"]))
    lines.append("")
    lines.append(
        f"Кверент: управитель I дома — {data['querent_label']} "
        f"{astro._sign_ru_prepositional(data['querent_sign'])}, "
        f"дом {data['querent_house']} — {data['querent_strength']}"
        + (f" ({', '.join(data['querent_notes'])})" if data["querent_notes"] else "") + "."
    )
    lines.append(
        f"Предмет вопроса (квесит): управитель дома {data['quesited_house']} — {data['quesited_label']} "
        f"{astro._sign_ru_prepositional(data['quesited_sign'])}, "
        f"дом {data['quesited_house_actual']} — {data['quesited_strength']}"
        + (f" ({', '.join(data['quesited_notes'])})" if data["quesited_notes"] else "") + "."
    )
    if data["same_ruler"]:
        lines.append("Кверент и квесит управляются ОДНОЙ и той же планетой (общий сигнификатор).")
    if data["mutual_reception"]:
        lines.append("Сигнификаторы находятся во взаимной рецепции.")

    lines.append("")
    if data["moon_void"]:
        lines.append("Луна без курса до выхода из текущего знака (значимых аспектов не образует).")
    else:
        lines.append(f"Луна не без курса; последний аспект перед выходом из знака: {data['moon_last_aspect']}.")

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

    lines.append("")
    lines.append(verdict_line)
    return "\n".join(lines)
