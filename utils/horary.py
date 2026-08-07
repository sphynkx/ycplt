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
  - "Derived house" (chart-turning) resolution: the PRIMARY path is now the
    local LLM itself (_classify_derived_chain_llm), asked only to name the
    chain's individual link-houses (a classification task, same shape as
    rectification_events._classify_event_houses_llm) — _derived_house()
    (pure Python) always does the actual turning arithmetic (sum of the
    chain minus (length-1), reduced into 1-12), never the model. This
    covers arbitrarily deep chains in principle ("wife's ring, gifted by
    her mother" — item/gift/wife/her-mother), not just one extra hop. If
    the model call fails, is unavailable, or returns something unparsable,
    falls back to a deterministic 2-hop-only heuristic
    (_PERSON_HOUSE_KEYWORDS + _TOPIC_HOUSE_KEYWORDS) that only recognizes a
    small fixed list of common relations (child, spouse, sibling, parent,
    friend, cousin, boss) and can't express anything deeper.
    Added after two rounds of real testing: first, a "will my daughter
    choose French" case showed the original single-hop-only version
    silently answering as if the QUERENT's own house 9 (education/
    languages) were the question, a different chart position than the
    daughter's own house 9 — and, worse, the model started inventing its
    own (wrong, undocumented) derived-house arithmetic in prose to
    compensate, a real "never invent facts not in the data block"
    violation, precisely because the real computation wasn't being
    surfaced to it. Second, the user pointed out a genuinely multi-link
    real-world example ("where is my wife's ring, a gift from her mother
    at the wedding") that a fixed 2-hop keyword table structurally cannot
    express no matter how large it grows — hence the LLM-classification
    primary path, rather than continuing to hand-enumerate relation
    combinations.
    Still an approximation, same honest caveat as
    _classify_event_houses_llm's own: a genuinely ambiguous question (which
    link is "the item" vs "the gift" vs "the giver" is a real judgment call
    even for a human astrologer, per the user's own "могу что-то попутать"
    when proposing this exact example) can still get a wrong or incomplete
    chain from the model — this is best-effort semantic classification, not
    a guaranteed-correct parse.
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
from utils import llm as llm_utils

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


# --- who the question is about -> derived-house chain (see module docstring) -

# House-from-querent for a NAMED third party the question concerns — checked
# separately from _TOPIC_HOUSE_KEYWORDS above, which answers "what topic",
# not "whose". "Двоюродн(ый/ая)" (cousin) is listed first and deliberately
# encoded as a single fixed house (5) rather than re-derived from a chain at
# lookup time — it's already the classical 2-hop result (3rd-from-3rd, i.e.
# "sibling of my sibling") folded into one constant, so it must be checked
# before the plainer "брат"/"сестр" entry (which would otherwise match the
# same substring first and misclassify a cousin as a plain sibling).
_PERSON_HOUSE_KEYWORDS: List[Tuple[List[str], int]] = [
    (["двоюродн"], 5),
    (["дочь", "дочер", "сын", "сыновь"], 5),
    (["муж", "жену", "жены", "жена", "супруг"], 7),
    (["брат", "сестр"], 3),
    (["мать", "мама", "мамин"], 10),
    (["отец", "отц", "пап"], 4),
    (["друг", "подруг", "приятел"], 11),
    (["начальник", "начальниц", "работодател", "босс"], 10),
    (["сосед"], 3),
]


def _classify_person_house(question_text: str) -> Optional[Tuple[int, str]]:
    """Returns (house_from_querent, matched_keyword) for the third party the
    question is actually about, or None if the question reads as being about
    the querent themselves (the ordinary, single-hop case, left unchanged)."""
    lowered = question_text.lower()
    for keywords, house in _PERSON_HOUSE_KEYWORDS:
        for kw in keywords:
            if kw in lowered:
                return house, kw
    return None


def _derived_house(chain: List[int]) -> int:
    """Classical chart-turning arithmetic for a chain of house numbers (each
    element = "count this many houses from the previous position, which
    becomes the new house I"): total = sum(chain) - (len(chain) - 1), then
    reduced into 1-12. A chain of length 1 returns its own single element
    unchanged, so this is a safe drop-in replacement for a bare topic house
    when no third party is involved. The sum is order-independent (moving
    the "-1 per extra link" adjustment aside, it's a plain sum), so the
    caller never needs to worry about which order the links were found in —
    only which links belong in the chain at all, which is the actual hard
    part (see _classify_derived_chain_llm below)."""
    total = sum(chain) - (len(chain) - 1)
    return ((total - 1) % 12) + 1


# --- derived-house chain classification via the local LLM (primary path) ----
#
# _PERSON_HOUSE_KEYWORDS/_classify_person_house above only handles a single
# extra hop (one named third party). Real questions can nest arbitrarily
# deep ("где кольцо жены, подаренное ей матерью на свадьбу" — item -> gift
# -> wife -> her mother -> the wedding), and no fixed keyword table can
# enumerate every such combination. Rather than growing that table forever,
# this reuses the exact same architecture rectification_events.py already
# established for its own "event -> house" problem (see that module's
# _classify_event_houses_llm): ask the already-loaded local model to name
# the chain's individual link-houses as a plain classification task, and do
# every bit of arithmetic in code afterward via _derived_house() — the
# model is never asked to compute the final turned house itself. This
# matters here specifically because letting the model do that arithmetic
# freely is exactly what went wrong before this feature existed at all: a
# real test ("who took my things") had the model invent its own made-up
# derived-house numbers in prose once it knew the technique existed, a
# fabrication-rule violation traced directly to the real computation not
# being surfaced to it. Keeping the model's job to bare classification (the
# same kind of task _classify_event_houses_llm already does reliably) and
# the arithmetic in Python avoids repeating that failure.
_DERIVED_HOUSE_PROMPT = """В хорарной астрологии дома гороскопа (I-XII) обозначают разные сферы, вещи и людей:
I — сам кверент (тот, кто задаёт вопрос), его тело, начинания
II — движимое имущество, деньги, личные ценности
III — братья/сёстры, соседи, короткие поездки
IV — дом, семья, отец, недвижимость, конец дела
V — дети, романтика, творчество
VI — здоровье, повседневная работа (найм), мелкие животные, подчинённые
VII — партнёр, супруг(а), другая сторона в любом взаимодействии, открытый враг
VIII — чужие деньги, долги, наследство, подарки от других людей, смерть
IX — дальние поездки, высшее образование (институт/университет/языки), право, иностранное
X — карьера, статус, репутация, мать, начальник/власть
XI — друзья, надежды
XII — тайные враги, изоляция, крупные животные, скрытое

Классическая техника ПРОИЗВОДНОГО ДОМА (поворот карты): если вопрос касается
не самого кверента напрямую, а цепочки связанных лиц или вещей (например
"кольцо жены, подаренное ей матерью на свадьбу" — это цепочка звеньев:
жена, её мать, подарок/наследство, сама вещь), нужно перечислить ВСЕ звенья
этой цепочки — каждое как отдельный номер дома (1-12), тот, каким домом это
звено обозначалось бы, если бы отсчёт шёл прямо от кверента. Порядок
перечисления не важен. Если вопрос касается самого кверента без посредников
— верни всего один дом, тему самого вопроса.

НЕ считай итоговый производный дом и не складывай числа сам — это отдельно
сделает программа. Твоя единственная задача — перечислить исходные номера
домов-звеньев.

Вопрос: "{question}"

Ответь СТРОГО в этом формате, без пояснений до или после:
ДОМА: <числа через запятую, например: 7, 4, 8, 2>"""

_DERIVED_HOUSE_ANSWER_RE = re.compile(r"ДОМА\s*:\s*(.+)")


def _parse_derived_house_answer(answer: str) -> Optional[List[int]]:
    """Parses "ДОМА: 7, 4, 8, 2" into [7, 4, 8, 2]. Deliberately keeps
    duplicates and order as given (unlike rectification_events' own event-
    house parser) — a repeated house number in the chain is meaningful here
    (e.g. a cousin is classically 3-from-3, i.e. the chain [3, 3]), not
    noise to be deduplicated. Capped at 6 links as a sanity bound against a
    degenerate answer, not because a real question can't nest that deep."""
    m = _DERIVED_HOUSE_ANSWER_RE.search(answer)
    if not m:
        return None
    houses = [int(tok) for tok in re.findall(r"\d+", m.group(1))]
    houses = [h for h in houses if 1 <= h <= 12][:6]
    return houses or None


_NEW_ROUND_PROMPT = """Тебе показано ПОСЛЕДНЕЕ сообщение пользователя в диалоге о хорарной
астрологии — БЕЗ предыдущей истории переписки. Определи: это НОВЫЙ,
самостоятельный хорарный вопрос (даже если в нём не хватает каких-то
деталей — даты, времени или места, это всё равно НОВЫЙ вопрос, если он
поднимает новую тему) — или это ПРОДОЛЖЕНИЕ/уточнение уже заданного ранее
вопроса (просьба объяснить, уточнить смысл вердикта, "почему так", "что
это значит", "уверен ли ты" и т.п., без новой темы вопроса)?

Сообщение: "{message}"

Ответь СТРОГО одним словом, без пояснений: НОВЫЙ или ПРОДОЛЖЕНИЕ"""


def _classify_new_horary_round(message: str) -> Optional[bool]:
    """Returns True if `message` ALONE (no history at all in the prompt —
    deliberately, so the model can't just default to "continuation" out of
    inertia) reads as a fresh, self-contained horary question — even if it
    happens to be missing a required field, still a NEW question, just an
    incomplete one — False if it reads as a follow-up/clarification about
    an answer already given, or None if the model is unavailable or its
    answer is unparsable (caller falls back to a deterministic heuristic in
    that case, never treats None as a hard failure).

    Added after a real, reported bug: a genuinely new horary question that
    happened to omit its casting location (relying on the same "just search
    everywhere, including history" union routes/chat.py otherwise builds
    for _INTERPRETED_TOOL_NAMES) got its place silently resolved from an
    unrelated EARLIER question's city in the same conversation — this
    answers "is this actually a new question at all" up front, so
    routes/chat.py can deliberately drop history_context for a genuinely
    new round rather than ever quietly blending it with an older,
    unrelated question's data. A real classification task (same shape as
    _classify_derived_chain_llm/rectification_events' own event classifier
    above), not something to approximate with more keyword rules."""
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
    """Walks BACKWARD through prior_user_texts (oldest-first, NOT including
    the current message — routes/chat.py appends that separately) to find
    where the CURRENT horary round actually began: classifies each
    historical message in isolation with _classify_new_horary_round (same
    call the current message itself already went through), collecting them
    as it goes, and stopping at — and including — the most recent one
    judged to be a fresh, self-contained NEW question.

    This is what makes trusting a "CONTINUATION" verdict safe at all: since
    every round boundary is itself detected this same way, everything at
    or after the last NEW verdict genuinely belongs to the current round,
    and nothing further back ever needs to be considered — no matter how
    long the whole conversation has grown, or how many unrelated horary
    questions (Moscow, Chelyabinsk, ...) happen to sit earlier in it. A
    real, reported bug motivated this: even after narrowing a continuation
    to "the union of current message + full history", a stray time and an
    unrelated city from a MUCH earlier, different horary question still
    won out over what the user had actually just supplied — because
    "history" had no notion of where the current round even started.

    Capped at _ROUND_LOOKBACK_CAP messages back purely as a safety bound —
    a real horary round (question, then a couple of clarifying follow-ups)
    essentially never runs longer than this. If the cap is hit, or the
    classifier itself returns None (unavailable/unparsable) partway
    through the walk, the walk stops there rather than guessing further
    back — a bounded, honest "this far and no further", never an unbounded
    re-scan of the entire conversation one classification call at a time."""
    collected: List[str] = []
    for text in reversed(prior_user_texts[-_ROUND_LOOKBACK_CAP:]):
        collected.append(text)
        verdict = _classify_new_horary_round(text)
        if verdict is not False:
            # True (a genuine round boundary) or None (classifier hiccup on
            # this one message) — stop here either way, conservatively.
            break
    return list(reversed(collected))


# --- LLM-first field extraction (date/time/place/question) ------------------
#
# Replaces regex-based free-text scanning (astro._find_date/_find_time/
# _lookup_city) as horary's PRIMARY path — kept only as a fallback in
# _compute_horary_chart for when no model is loaded at all. Two concrete,
# reported failures motivated this: a user-typed dash-separated time
# ("19-28-30") wasn't recognized by the colon-only regex at all, silently
# falling through to search the rest of the (possibly much older) request
# text instead; and free-text city search occasionally stem-matches an
# ordinary word against an unrelated, obscure place name anywhere in the
# world (a real test found "французский" matching Francistown, Botswana).
# Both are failures of PATTERN-MATCHING text rather than READING it — a
# model that actually understands the sentence doesn't have this failure
# mode, it answers "what date/time/place does this describe" directly.
_FIELD_EXTRACTION_PROMPT = """Тебе показан текст одного "раунда" хорарного запроса — исходный вопрос и,
возможно, последующие уточнения пользователя в рамках ОДНОГО И ТОГО ЖЕ
вопроса (без более ранних, не относящихся к делу сообщений).

Извлеки из этого текста:
1. Дату, когда был задан вопрос (НЕ дату рождения!) — в формате ГГГГ-ММ-ДД.
2. Время, когда был задан вопрос — в 24-часовом формате ЧЧ:ММ, независимо
   от того, как оно записано в тексте (через двоеточие, дефис, словами
   и т.п.) — переведи его именно в этот формат.
3. Место, где был задан вопрос — город и страна; если в тексте прямо даны
   координаты, верни их как "широта, долгота" (например "46.48, 30.72").
4. Саму формулировку вопроса — дословно, как её сформулировал пользователь,
   не пересказывай своими словами.

Текст:
\"\"\"{text}\"\"\"

Если какого-то из этих пунктов в тексте ДЕЙСТВИТЕЛЬНО нет — напиши "нет" в
соответствующей строке, не выдумывай и не угадывай.

Ответь СТРОГО в этом формате, каждый пункт на отдельной строке, без
пояснений до или после:
ДАТА: <ГГГГ-ММ-ДД или нет>
ВРЕМЯ: <ЧЧ:ММ или нет>
МЕСТО: <город, страна ИЛИ широта, долгота ИЛИ нет>
ВОПРОС: <дословная формулировка или нет>"""


def _parse_extraction_field(label: str, answer: str) -> Optional[str]:
    """Pulls one "LABEL: value" line out of the model's own strict-format
    answer above — this is parsing the MODEL's controlled output, not
    scanning raw user text, the same safe, established pattern as
    _parse_derived_house_answer/rectification_events' event-house parser,
    not the kind of free-text pattern-matching this whole mechanism exists
    to replace. Returns None for a missing line, an empty value, or an
    explicit "нет" (the model's own "not present" answer)."""
    m = re.search(rf"{label}\s*:\s*(.+)", answer, re.IGNORECASE)
    if not m:
        return None
    value = m.group(1).strip().strip('"').strip()
    if not value or value.lower() in ("нет", "нету", "n/a", "-", "—"):
        return None
    return value


def _extract_horary_fields_llm(round_text: str) -> Optional[Dict[str, str]]:
    """Returns {"date": ..., "time": ..., "question": ...} (always both
    present if this returns non-None — date/time are the two hard
    requirements, everything else this module needs) plus an optional
    "place" key, or None if the model is unavailable, errored, or didn't
    return a usable date+time. Callers treat None as "fall back to the
    regex-based path", never as a hard failure — see _compute_horary_
    chart's own docstring for why that fallback still exists at all."""
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
    if not (date and time_):
        return None
    result = {"date": date, "time": time_}
    place = _parse_extraction_field("МЕСТО", answer)
    if place:
        result["place"] = place
    question = _parse_extraction_field("ВОПРОС", answer)
    if question:
        result["question"] = question
    return result


def _resolve_place(place_text: str) -> Optional[Tuple[float, float, str]]:
    """Returns (lat, lon, tz) for an LLM-identified place string (already
    isolated to just the place itself, not a whole message), or None if it
    can't be resolved. Tries an explicit "lat, lon" pair first — plain
    split+float parsing, not a regex, since this is the model's own clean,
    single-purpose output, not raw free text — then an EXACT city-name
    lookup (astro._lookup_city_exact), never the fuzzy substring-based
    tier astro._lookup_city itself still uses for other techniques: by
    this point there's no wider blob of unrelated conversation text left
    for a fuzzy match to go wrong in, but an exact match is still the more
    honest guarantee for a technique this location-sensitive."""
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


def _classify_derived_chain_llm(question_text: str) -> Optional[List[int]]:
    """Best-effort semantic decomposition of the question into its full
    derived-house chain, using the already-loaded chat model — mirrors
    rectification_events._classify_event_houses_llm exactly (same
    prompt/parse/fallback shape). Returns None on any error, an unparsable
    answer, or if no model is loaded; every caller treats None as "fall back
    to the keyword-based heuristic", never as a hard failure — this keeps
    the deterministic 2-hop path (_classify_person_house +
    _classify_topic_house) as a real safety net, not just a decoration."""
    if llm_utils.get_llm() is None:
        return None
    try:
        answer = llm_utils.generate_sync(
            _DERIVED_HOUSE_PROMPT.format(question=question_text),
            max_tokens=40,
            temperature=0.0,
        )
    except Exception:
        return None
    return _parse_derived_house_answer(answer)


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

# Classical horary aspect set ONLY — deliberately narrower than astro._ALL_ASPECTS
# (which also includes semi-sextile/semi-square/quintile/sesquiquadrate/
# biquintile, correct for natal/transit/synastry reading elsewhere in this
# app, but not part of classical horary doctrine at all — see
# horary_methodology.txt section 4, which lists exactly these six and no
# others). Found via real testing: a "квинтиль" (72°) between the two
# significators was picked as the chart's *direct_aspect* below, and because
# quintile is in neither _FAVORABLE_ASPECTS nor _HARD_ASPECTS above, the
# verdict cascade could only ever read it as a negative outcome — a minor,
# non-classical modern aspect was silently overriding whatever real Ptolemaic
# aspect (or lack of one) the significators actually had. Restricting the
# AspectsFactory call itself to this list (rather than filtering afterward)
# is both the correct methodological scope and prevents this class of bug
# outright, including for the third-point translation/collection-of-light
# search below, which reuses the same `aspects` list.
_HORARY_ASPECTS = [a for a in astro._MAJOR_ASPECTS] + [
    a for a in astro._MINOR_ASPECTS if a["name"] == "quincunx"
]


def _compute_horary_chart(spec: str) -> Dict[str, Any]:
    """Returns a dict of every computed fact plus the final verdict, or
    {"error": "..."} if the moment/place couldn't be resolved. Never skips
    computation on a failed radicality check anymore (see module
    docstring) — is_radical/radicality_notes are carried in the result
    alongside a real verdict either way.

    Field resolution (date/time/place/question) is now LLM-first (see
    _extract_horary_fields_llm) — the regex-based astro._extract_fields
    path below only ever runs as a fallback when the model is unavailable
    or its answer didn't parse. This replaced a purely regex-based
    pipeline after two concrete, reported failures: a user-typed dash-
    separated time ("19-28-30") wasn't recognized by the colon-only time
    regex at all (silently treated as "no time given"), and the free-text
    city search could accidentally stem-match an ordinary word against an
    unrelated, obscure place name anywhere in the world (a real test found
    "французский" matching Francistown, Botswana). Both are exactly the
    class of failure a model that actually reads and understands the
    sentence doesn't have — it isn't pattern-matching substrings, it's
    answering "what date/time/place does this text actually describe."
    The regex fallback is kept, not deleted, purely as a safety net for
    when no model is loaded at all — never as the normal path once one is."""
    llm_fields = _extract_horary_fields_llm(spec)
    raw = astro._parse_spec(spec)  # cheap, unambiguous "key=value" parse — kept for the optional house=N override

    if llm_fields:
        place_resolved = _resolve_place(llm_fields["place"]) if llm_fields.get("place") else None
        if place_resolved is None:
            return {"error": _missing_fields_message(["place"]) + (
                " Модель распознала дату и время вопроса, но не смогла уверенно "
                "определить место, где он был задан — уточните его явно (город "
                "или координаты)."
            )}
        lat, lon, tz = place_resolved
        fields = {
            "date": llm_fields["date"], "time": llm_fields["time"],
            "lat": str(lat), "lon": str(lon), "tz": tz,
        }
        question_text = llm_fields.get("question") or _extract_question_text(spec)
    else:
        # --- fallback: regex-based extraction, only reached with no model loaded ---
        fields, missing = astro._extract_fields(spec)
        if missing:
            return {"error": _missing_fields_message(missing)}

        # Horary-specific location-confidence gate (only relevant on this
        # fallback path — the LLM path above already resolves place from a
        # single, clean, model-identified string, so this class of
        # collision can't happen there at all). astro._extract_fields is
        # happy to accept whatever astro._lookup_city's LOOSE, fuzzy stem-
        # matching tier finds as "the place" — fine for most techniques,
        # not safe enough for horary specifically (see module docstring
        # above for the Francistown case). Unless explicit coordinates
        # were given, require an EXACT (non-fuzzy) city match — if
        # neither holds, report the place as missing rather than silently
        # computing a chart for the wrong city.
        if not raw.get("lat") or not raw.get("lon"):
            coord_lat, coord_lon = astro._find_coordinates(spec)
            if coord_lat is not None:
                fields["lat"], fields["lon"] = str(coord_lat), str(coord_lon)
            else:
                exact_city = astro._lookup_city_exact(spec)
                if exact_city is None:
                    return {"error": _missing_fields_message(["lat", "lon"]) + (
                        " В тексте не нашлось ни точных координат, ни однозначного "
                        "названия города для МОМЕНТА ЭТОГО ВОПРОСА — уточните "
                        "место явно (город или координаты), чтобы избежать "
                        "случайной геопривязки по совпадению слов."
                    )}
                fields["lat"] = str(exact_city["latitude"])
                fields["lon"] = str(exact_city["longitude"])
                fields["tz"] = exact_city["timezone"]
            if not fields.get("tz") and fields.get("lat") and fields.get("lon"):
                tz = astro._resolve_timezone(float(fields["lat"]), float(fields["lon"]))
                if tz:
                    fields["tz"] = tz

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

    topic_house = _classify_topic_house(question_text)
    person_match = _classify_person_house(question_text)
    derived_chain: Optional[List[int]] = None
    chain_source: Optional[str] = None  # "llm" or "keyword" — for display only
    person_house: Optional[int] = None
    person_keyword: Optional[str] = None

    quesited_house = _DEFAULT_QUESITED_HOUSE
    if raw.get("house"):
        try:
            quesited_house = max(1, min(12, int(raw["house"])))
        except ValueError:
            quesited_house = topic_house
    else:
        # Primary path: ask the local model to decompose the question into
        # its full derived-house chain (arbitrary depth, e.g. "wife's
        # ring, gifted by her mother" — item/gift/wife/her-mother, not just
        # the one extra hop the keyword heuristic below can express). Same
        # architecture as rectification_events._classify_event_houses_llm:
        # the model only ever names raw house numbers, _derived_house (pure
        # Python) does the actual turning arithmetic — see that function's
        # docstring for exactly why the arithmetic itself must never be
        # left to the model. Falls back to the deterministic 2-hop keyword
        # heuristic (person + topic) on any failure, and to the plain
        # single-hop topic house if even that finds no third party.
        llm_chain = _classify_derived_chain_llm(question_text)
        if llm_chain:
            derived_chain = llm_chain
            chain_source = "llm"
            quesited_house = _derived_house(llm_chain) if len(llm_chain) > 1 else llm_chain[0]
        elif person_match:
            person_house, person_keyword = person_match
            derived_chain = [person_house, topic_house]
            chain_source = "keyword"
            quesited_house = _derived_house(derived_chain)
        else:
            quesited_house = topic_house
    result["quesited_house"] = quesited_house
    result["topic_house"] = topic_house
    result["derived_chain"] = derived_chain
    result["chain_source"] = chain_source
    result["person_house"] = person_house
    result["person_keyword"] = person_keyword

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
    if data["derived_chain"] and data["chain_source"] == "llm":
        chain = data["derived_chain"]
        if len(chain) > 1:
            lines.append(
                f"Вопрос касается не самого кверента напрямую, а цепочки связанных лиц/вещей — "
                f"звенья цепочки (дома от кверента, каждое следующее считается от предыдущего как от его "
                f"собственного дома I): {', '.join(str(h) for h in chain)}. Перенос дома (классическая "
                f"техника разворота карты, сумма звеньев минус число лишних переносов, приведено к 1-12) "
                f"→ итоговый производный дом {data['quesited_house']} — его управитель и есть значимая "
                "планета (квесит) для конца этой цепочки, а не для кверента напрямую."
            )
        # A single-element LLM chain means "no third party" — same as the
        # plain topic-house case below, nothing extra to say here.
    elif data["derived_chain"] and data["chain_source"] == "keyword":
        lines.append(
            f"Вопрос касается не самого кверента, а третьего лица (определено по слову «{data['person_keyword']}» "
            f"в тексте вопроса) — это лицо обозначается домом {data['person_house']} от кверента. "
            f"Тема вопроса для этого лица (как если бы дом {data['person_house']} был его собственным домом I) — "
            f"дом {data['topic_house']}. Перенос дома (классическая техника разворота карты): "
            f"{data['person_house']} + {data['topic_house']} - 1, приведено к диапазону 1-12 → "
            f"итоговый производный дом {data['quesited_house']} — его управитель и есть значимая планета (квесит) "
            "для ЭТОГО человека и ЭТОЙ темы, а не для кверента напрямую."
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
