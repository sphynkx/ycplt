"""Multi-technique, multi-event birth-time rectification — the user's
original long-term vision (see utils/rectification.py's module docstring,
which implements only the first, simplest piece of it: the Trutine of
Hermes). Instead of one classical rule, THIS module searches a window of
candidate birth times and, for EACH candidate, builds profections,
secondary progressions, solar-arc directions, and transits for EACH of
several known LIFE EVENT dates, scoring how well each technique's moving
points aspect the "elements" of the houses that event type classically
belongs to (marriage -> 7th/1st house, career -> 10th/6th, etc. — see
_EVENT_HOUSE_KEYWORDS). The candidate whose aggregate score is highest
across all events/techniques is the best-SUPPORTED birth time — never a
single confirmed answer, always a ranked list of hypotheses (same framing
as rectification.py's Trutine output, just "higher is better" here instead
of "lower is better", since this counts confirmations rather than a
mismatch error).

"Elements of a house" follows the classical rule described in the user's
own indexed rectification corpus (Shestopalov's method, via REKTIF.TXT):
the planet(s) occupying the house, its ruler (sign on the cusp), and a
CO-ruler if the house's own arc extends more than ~13 degrees into a
further sign (an intercepted/co-ruled house — REKTIF.TXT's own worked
example: a 9th house from 20 Taurus to 16 Gemini has Venus+Mercury+Sun as
elements; if it ran to 15 Cancer, the Moon would join too).

Event -> house classification: the PRIMARY path is now the local LLM
itself (_classify_event_houses_llm/_async below) — given one event
description at a time (with the classical meaning of all 12 houses
spelled out in the prompt), it decides which house(s) that specific event
belongs to, semantically, rather than by substring match. This replaced
an earlier keyword-only design after real testing kept finding phrasings
the fixed dictionary couldn't anticipate (a fixed list can enumerate the
common cases, never all of them); the user explicitly asked for this
change and accepted the added runtime cost. The static keyword dictionary
(_EVENT_HOUSE_KEYWORDS) is NOT removed — it's still computed first, as an
instant, free fallback used whenever the LLM call fails, is unavailable
(model not loaded), or returns an answer that doesn't parse into valid
house numbers; _propagate_prefix_categories (below) still backs up the
keyword layer the same way it always did. This mirrors the same
hybrid-with-deterministic-fallback pattern already used elsewhere in this
app (e.g. synastry's LLM-assisted person-label segmentation) — the LLM
step can only improve a classification, never make it worse than the
keyword-only baseline that existed before it.

The LLM classification runs once per event (not once per candidate birth
time), so its cost is O(n_events), fully independent of how large the
candidate search window is. It runs sequentially, not concurrently —
there's exactly one loaded Llama instance for the whole process, so
parallel dispatch would only contend for the same model, not add real
throughput.

Known, documented simplifications (see rectification_events_methodology.txt
for the full list intended for the model's own final answer):
  - uses this app's existing quadrant house system (whichever kerykeion
    computes by default), NOT the Koch houses the source material
    (REKTIF.TXT) specifically recommends for this exact method;
  - transiting/progressed/directed positions for an event are computed at
    local NOON on the event's date (no event TIME is collected from the
    user), which under-represents the Moon's ~13 degrees/day motion around
    that day — a real, accepted precision loss, not a bug;
  - profection's contribution is a proxy (does the profected whole-sign
    house for that year match one of the event's houses, plus whether the
    year's ruler aspects the event's house elements) — not the "profected
    MC exact 30-degree-multiple aspect" variant some sources describe;
  - an event dated before the candidate birth time is simply skipped for
    that candidate (flagged as inapplicable), not treated as a scoring
    penalty.

Async: building one candidate's natal chart plus its per-event progressed/
transit charts is CPU-bound Swiss-Ephemeris work, independent across
candidates — so all candidates are evaluated concurrently via asyncio and a
dedicated thread pool (same run_in_executor pattern already used throughout
routes/chat.py and tool_router.classify_async), which matters here because a
realistic run evaluates dozens of candidates times several events times
four techniques, unlike Trutine's cheap two-chart-per-candidate search.
"""
import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from utils import astro
from utils import llm as llm_utils
from utils import rectification as trutine  # reuse _parse_hhmm_on_date only

# --- event -> house classification -----------------------------------------

# Checked top to bottom, first matching keyword group wins — ordered so a
# more specific event type is checked before anything that could plausibly
# collide with a more generic later stem (kept short/simple on purpose; see
# module docstring re: LLM-fallback classification as a possible later
# addition instead of growing this list indefinitely).
_EVENT_HOUSE_KEYWORDS: List[Tuple[List[str], List[int]]] = [
    (["свадьб", "брак", "женить", "женил", "замуж", "венчан"], [7, 1]),
    # Break-ups: real usage phrases this many ways beyond "развод" itself
    # ("ушел от жены", "рассталась", "разошлись") — all still partner-
    # house (7th) events, not the generic fallback.
    (["развод", "расста", "ушел от", "ушла от", "разошли", "бросил"], [7, 1]),
    # Meeting/dating a future or current partner — checked BEFORE the
    # generic fallback would otherwise catch it; a real reported gap
    # ("Знакомство с первой женой", "Первая очная встреча со 2й женой").
    (["знаком", "встрет", "встреч", "познаком"], [7, 5]),
    (["любов", "влюб", "роман"], [5, 7]),
    (["роды", "родила", "родилс", "ребен", "ребён", "дочь появ", "сын появ"], [5, 11, 1]),
    (["смерт", "умер", "похорон", "скончал"], [8, 4]),
    # Theft/robbery — deliberately checked BEFORE the move/housing group
    # below: a real bug showed "Ограбление квартиры" being misclassified
    # as a MOVE simply because it mentions "квартира" (apartment), when
    # the actual event (a robbery) is an 8th/2nd/12th-house loss, not a
    # 4th-house home-change.
    (["ограблен", "кража", "грабеж", "обокра", "украл"], [2, 8, 12]),
    (["трудоустрой", "устроил", "уволь", "карьер", "должност", "повышен", "работ", "вакан", "испытательн"], [10, 6]),
    (["операц", "болезн", "госпитал", "травм", "диагноз"], [6, 8, 12]),
    (["эвакуац", "переезд", "перее", "квартир", "жиль"], [4, 3, 9]),
    (["учеб", "институт", "универ", "диплом", "экзамен", "школ"], [9, 3]),
    (["путешеств", "поездк", "командировк"], [9, 3]),
    (["денег", "доход", "наследств", "выигрыш", "займ", "кредит"], [2, 8]),
]
_FALLBACK_EVENT_HOUSES = [1, 10]

# --- event -> house classification via the local LLM (primary path) --------

# Kept in the prompt itself, not just a code comment, so the model reasons
# about the actual classical meaning of each house instead of pattern-
# matching on a handful of examples — this is the same information
# _EVENT_HOUSE_KEYWORDS encodes as substring rules, given to the model as
# genuine domain knowledge instead.
_EVENT_HOUSE_PROMPT = """В натальной астрологии 12 домов гороскопа отвечают за разные сферы жизни:
1 дом — личность, тело, само начало жизни, самоощущение
2 дом — деньги, доход, личное имущество, ценности
3 дом — учёба (школа), братья/сёстры, близкое окружение, короткие поездки
4 дом — родительский дом, семья происхождения, корни, недвижимость
5 дом — влюблённость, романтика, дети, творчество, удовольствия
6 дом — повседневная работа, служба, здоровье, распорядок
7 дом — брак, партнёрство, значимые "вторые половины", открытые враги
8 дом — смерть, кризисы, чужие деньги/наследство, глубокая трансформация
9 дом — высшее образование, дальние поездки, философия, мировоззрение
10 дом — карьера, статус, призвание, отношения с начальством/властью
11 дом — друзья, сообщества, надежды и планы на будущее
12 дом — потери, изоляция, скрытое, подсознание, тайные враги

Определи, к какому дому (или домам — не более трёх, если событие затрагивает
сразу несколько сфер) из списка выше по смыслу относится следующее СОБЫТИЕ
ЖИЗНИ конкретного человека. Перечисли дома от наиболее к наименее значимому.

Событие: "{description}"

Ответь СТРОГО в этом формате, без пояснений до или после:
ДОМА: <числа через запятую, например: 7, 1>"""

_EVENT_HOUSE_ANSWER_RE = re.compile(r"ДОМА\s*:\s*(.+)")


def _parse_event_house_answer(answer: str) -> Optional[List[int]]:
    """Parses the "ДОМА: 7, 1" line into [7, 1]. Deliberately tolerant of
    noise around the numbers (extra words, stray punctuation) — captures
    the rest of the line after "ДОМА:" (stopping at the newline, since `.`
    doesn't match `\\n` without re.DOTALL) and pulls out every standalone
    integer in it, rather than requiring the whole tail to be clean
    digits/commas/spaces; a hallucinated stray word next to real house
    numbers is a much smaller failure than discarding a valid answer
    entirely. Returns None (caller falls back to the keyword dictionary)
    only if the line itself is missing or contains no valid 1-12 house
    number at all."""
    m = _EVENT_HOUSE_ANSWER_RE.search(answer)
    if not m:
        return None
    houses: List[int] = []
    for tok in re.findall(r"\d+", m.group(1)):
        h = int(tok)
        if 1 <= h <= 12 and h not in houses:
            houses.append(h)
    return houses[:3] or None


def _classify_event_houses_llm(description: str) -> Optional[List[int]]:
    """Best-effort semantic classification of one event's houses using the
    already-loaded chat model itself — see the module docstring's "event ->
    house classification" section for why this is now the primary path.
    Returns None on any error, an unparsable answer, or if no model is
    loaded — every caller treats None as "fall back to the keyword
    dictionary", never as a hard failure."""
    if llm_utils.get_llm() is None:
        return None
    try:
        answer = llm_utils.generate_sync(
            _EVENT_HOUSE_PROMPT.format(description=description),
            max_tokens=40,
            temperature=0.0,
        )
    except Exception:
        return None
    return _parse_event_house_answer(answer)


async def _classify_event_houses_llm_async(description: str) -> Optional[List[int]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _classify_event_houses_llm, description)


async def _apply_llm_event_classification(events: List[Dict[str, Any]]) -> None:
    """Mutates `events` in place: for each event, tries the LLM semantic
    classifier and overrides the keyword-based houses/category_matched/
    category_note already computed by _extract_events_and_birth_text when
    it succeeds. Runs sequentially (see module docstring — one shared
    Llama instance, concurrent calls would only contend, not parallelize)
    and is skipped entirely up front if no model is loaded, so it costs
    nothing when the app is running without a model for some reason.
    Never makes a classification worse than the keyword-only baseline: on
    any per-event failure, that event's existing keyword/propagation
    result (computed earlier, always present) is simply left as-is."""
    if llm_utils.get_llm() is None:
        return
    for ev in events:
        houses = await _classify_event_houses_llm_async(ev["description"])
        if houses:
            ev["houses"] = houses
            ev["category_matched"] = True
            ev["category_note"] = None

# Deliberately NOT a keyword (too ambiguous to be safe): "предложени"
# ("предложение") means both a job offer AND a marriage proposal in
# Russian, and adding it as a keyword would misclassify whichever sense
# is less common in a given corpus. Left for _propagate_prefix_categories
# to resolve via shared context instead of a guess.

# A line matching "description: date" is normally treated as an event —
# EXCEPT when the description itself is clearly a birth-data label (a user
# describing their own birth data as "Дата рождения: 05.07.1976" would
# otherwise get misclassified as an "event" called "Дата рождения",
# silently stealing the actual birth date out of the free text the birth-
# field parser needs). Checked before the event-keyword classifier.
_BIRTH_LABEL_EXCLUSIONS = [
    "дата рождени", "время рождени", "место рождени", "координат",
    "часовой пояс", "широта", "долгота",
]

_EVENT_LINE_RE = re.compile(
    r"^\s*(?P<desc>.+?)\s*:\s*(?P<date>\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{4}-\d{1,2}-\d{1,2})\s*$"
)

# A bare date/time token, used to detect the RICHER semicolon-separated
# event format real usage produced (not just the simple "description: date"
# this module originally documented): "description; date; [time]; [place];
# [lat]; [lon]; [free comment]" — e.g. "Первая любовь; 1.11.1986; 12:00;
# Одесса, Одесская обл., Украина; 46n28; 30e44; Поворотный момент...".
# Only the description, date, and (if present) time are actually used —
# place/coordinates/comment are accepted and silently ignored (see
# rectification_events_methodology.txt point 6: event location isn't
# needed for THIS engine's scoring — transiting/progressed/directed
# ecliptic longitude doesn't depend on location, only on the moment in
# time — so there's no accuracy lost by ignoring it, only the convenience
# of not having to strip it back out here).
_DATE_TOKEN_RE = re.compile(r"^\d{1,2}[./]\d{1,2}[./]\d{2,4}$|^\d{4}-\d{1,2}-\d{1,2}$")
_TIME_TOKEN_RE = re.compile(r"^\d{1,2}:\d{2}$")


def _try_parse_semicolon_event(line: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """Returns (description, date_token, time_token_or_None) if `line` looks
    like the richer "description; date; [time]; ..." format (recognized by
    its SECOND semicolon-separated field being a bare date, not by any
    fixed field count — trailing place/coordinate/comment fields are
    simply ignored, however many there are), else None."""
    parts = [p.strip() for p in line.split(";")]
    if len(parts) < 2 or not parts[0] or not _DATE_TOKEN_RE.match(parts[1]):
        return None
    time_token = parts[2] if len(parts) >= 3 and _TIME_TOKEN_RE.match(parts[2]) else None
    return parts[0], parts[1], time_token


# Safety cap on how many event lines a single request will ever score.
# Raised from an earlier, much stricter 20 after real usage showed a
# genuinely reasonable request (42 real life events, for a thorough
# rectification) hitting it — a real many-events run is exactly the kind
# of input this engine exists for, not something to arbitrarily truncate.
# Total cost is instead kept bounded by _effective_max_candidates (below),
# which shrinks the per-candidate search resolution as the event count
# grows, rather than shrinking the event list itself.
_MAX_EVENTS = 80


def _classify_event_houses(description: str) -> Tuple[List[int], bool]:
    """Returns (house numbers, matched) — matched=False means no keyword
    hit and the generic fallback houses were used, which callers surface
    explicitly so the model can flag low-confidence categorization instead
    of silently treating a guess as a certain classical rule."""
    text = description.lower()
    for keywords, houses in _EVENT_HOUSE_KEYWORDS:
        if any(kw in text for kw in keywords):
            return houses, True
    return _FALLBACK_EVENT_HOUSES, False


def _parse_event_date(date_str: str) -> Optional[datetime]:
    """Never raises — an invalid calendar date (e.g. "30.02.2061", a real
    typo a user can make) returns None so the caller can skip that one
    event line with an explicit warning instead of crashing the whole
    request over a single bad line."""
    date_str = date_str.strip()
    for sep in (".", "/"):
        if sep in date_str:
            parts = date_str.split(sep)
            if len(parts) == 3:
                try:
                    day, month, year = (int(p) for p in parts)
                    if year < 100:
                        year += 2000 if year < 70 else 1900
                    return datetime(year, month, day)
                except (ValueError, TypeError):
                    return None
            return None
    if "-" in date_str:
        try:
            year, month, day = (int(x) for x in date_str.split("-"))
            return datetime(year, month, day)
        except (ValueError, TypeError):
            return None
    return None


def _parse_time_token(time_str: str) -> Tuple[int, int]:
    hour, minute = (int(x) for x in time_str.split(":"))
    return hour, minute


def _extract_events_and_birth_text(spec: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """Splits the raw tool argument into (birth_data_text, events,
    warnings). Every line recognized as an event (either the simple
    "description: date" format, or the richer real-world "description;
    date; [time]; [place]; [lat]; [lon]; [comment]" format — see
    _try_parse_semicolon_event; the semicolon form is tried FIRST since
    it's the more specific/constrained match) and not looking like a
    birth-data label (see _BIRTH_LABEL_EXCLUSIONS) is REMOVED from the
    text handed to astro._parse_spec/_fill_fields_from_text — this is
    what lets birth data and dozens of event lines coexist in one
    free-text argument without the birth-date regex accidentally
    latching onto an event's date instead, regardless of whether events
    are listed before, after, or interleaved with birth data."""
    birth_lines: List[str] = []
    events: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for raw_line in spec.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        desc: Optional[str] = None
        date_token: Optional[str] = None
        time_token: Optional[str] = None

        semicolon_match = _try_parse_semicolon_event(line)
        if semicolon_match is not None:
            desc, date_token, time_token = semicolon_match
        else:
            m = _EVENT_LINE_RE.match(line)
            if m is not None:
                desc, date_token = m.group("desc").strip(), m.group("date")

        if desc is None:
            birth_lines.append(line)
            continue
        if any(kw in desc.lower() for kw in _BIRTH_LABEL_EXCLUSIONS):
            birth_lines.append(line)
            continue

        event_date = _parse_event_date(date_token)
        if event_date is None:
            warnings.append(f"Не удалось разобрать дату события «{line}» — строка пропущена.")
            continue

        hour, minute = (12, 0)  # default to local noon when no time is given (see module docstring)
        if time_token is not None:
            try:
                hour, minute = _parse_time_token(time_token)
            except (ValueError, TypeError):
                pass  # keep the noon default rather than dropping an otherwise-valid event
        event_dt = event_date.replace(hour=hour, minute=minute)

        houses, matched = _classify_event_houses(desc)
        events.append({
            "description": desc, "date": event_date, "when": event_dt,
            "houses": houses, "category_matched": matched, "category_note": None,
        })
    _propagate_prefix_categories(events)
    return "\n".join(birth_lines), events, warnings


# Only a colon-prefixed label this long or longer is treated as a shared
# "topic group" for propagation (below) — short prefixes (a couple of
# words) are too likely to coincide by accident between UNRELATED events
# and would propagate a wrong category between them.
_MIN_GROUP_PREFIX_CHARS = 6


def _propagate_prefix_categories(events: List[Dict[str, Any]]) -> None:
    """Mutates `events` in place. Real event lists often describe one
    ongoing situation as several dated sub-events sharing a common label
    before a colon — e.g. "5я работа (судьбоносно важна): Собеседование",
    "...: предложение трудоустройства", "...: дал свое согласие" are all
    the same job search, but only some of those phrasings contain a
    keyword _EVENT_HOUSE_KEYWORDS recognizes (a real, reported gap: "дал
    свое согласие" and "Request to vacancy" fell back to the generic
    houses even though sibling lines under the identical label matched
    confidently). If ANY event under a given label matched confidently,
    that same category is propagated to its unmatched siblings — a static
    keyword dictionary can never cover every way a person phrases "job
    offer" or "gave my consent", but events sharing an explicit label are
    overwhelmingly likely to belong to the same house(s) regardless of
    phrasing, which is a much safer inference than guessing from words
    alone."""
    houses_by_prefix: Dict[str, List[int]] = {}
    for ev in events:
        prefix = ev["description"].split(":", 1)[0].strip().lower()
        if ev["category_matched"] and len(prefix) >= _MIN_GROUP_PREFIX_CHARS:
            houses_by_prefix.setdefault(prefix, ev["houses"])

    for ev in events:
        if ev["category_matched"]:
            continue
        prefix = ev["description"].split(":", 1)[0].strip().lower()
        if prefix in houses_by_prefix:
            ev["houses"] = houses_by_prefix[prefix]
            ev["category_matched"] = True
            ev["category_note"] = "по аналогии с другим событием той же группы (общая часть описания до «:»)"


# --- birth-time search window (mirrors rectification.py's Trutine window
# logic, but with its own, coarser defaults: this engine's per-candidate
# cost is far higher than Trutine's two-chart search, since it builds a
# progressed AND a transit chart per event on top of the natal chart, so a
# 1-minute step across a wide window would be prohibitively expensive) ----

_DEFAULT_WINDOW_MINUTES = 120
_DEFAULT_STEP_MINUTES = 10
_MAX_CANDIDATES = 60

# Rough cap on TOTAL chart builds for one request (natal + progressed +
# transit per event, roughly `1 + 2*n_events` per candidate) — used to
# shrink the candidate count (search resolution) as the event count grows,
# rather than capping the event list itself (see _MAX_EVENTS' own comment:
# a real request with dozens of events is exactly this engine's intended
# use, not something to arbitrarily truncate). A user who gives 3 events
# gets up to _MAX_CANDIDATES candidates; a user who gives 40 gets
# proportionally fewer, coarser-grained candidates instead of the request
# taking many times longer — trading time-resolution for event-count
# breadth automatically.
_TOTAL_CHART_BUDGET = 1200


def _effective_max_candidates(n_events: int) -> int:
    per_candidate_cost = 1 + 2 * max(n_events, 1)
    return max(5, min(_MAX_CANDIDATES, _TOTAL_CHART_BUDGET // per_candidate_cost))


def _extract_birth_window_fields(
    birth_text: str,
) -> Tuple[Dict[str, str], Optional[datetime], Optional[datetime], int, List[str]]:
    fields = astro._parse_spec(birth_text)
    astro._fill_fields_from_text(fields, birth_text)

    missing: List[str] = []
    for key, label in (("date", "дата"), ("lat", "широта"), ("lon", "долгота"), ("tz", "часовой пояс")):
        if not fields.get(key):
            missing.append(label)

    step_minutes = _DEFAULT_STEP_MINUTES
    if fields.get("step_minutes"):
        try:
            step_minutes = max(1, int(fields["step_minutes"]))
        except Exception:
            pass

    if missing:
        return fields, None, None, step_minutes, missing

    date_str = fields["date"]
    window_start = window_end = None
    time_min_str, time_max_str = fields.get("time_min"), fields.get("time_max")
    if time_min_str and time_max_str:
        window_start = trutine._parse_hhmm_on_date(date_str, time_min_str)
        window_end = trutine._parse_hhmm_on_date(date_str, time_max_str)
        if window_start is None or window_end is None:
            missing.append("time_min/time_max (ожидается формат ЧЧ:ММ)")
        elif window_end <= window_start:
            window_end += timedelta(days=1)  # window crosses midnight
    else:
        base_time_str = fields.get("time")
        if not base_time_str:
            missing.append(
                "примерное время рождения (укажите приблизительное время, либо time_min=/time_max=)"
            )
        else:
            base_dt = trutine._parse_hhmm_on_date(date_str, base_time_str)
            if base_dt is None:
                missing.append("время рождения (ожидается формат ЧЧ:ММ)")
            else:
                window_minutes = _DEFAULT_WINDOW_MINUTES
                if fields.get("window_minutes"):
                    try:
                        window_minutes = max(step_minutes, int(fields["window_minutes"]))
                    except Exception:
                        pass
                half = timedelta(minutes=window_minutes / 2)
                window_start, window_end = base_dt - half, base_dt + half

    if missing:
        return fields, None, None, step_minutes, missing
    return fields, window_start, window_end, step_minutes, missing


def _missing_fields_message(missing: List[str]) -> str:
    return (
        "Не хватает данных для многотехничной ректификации: " + ", ".join(missing) + ". "
        "Нужны: примерная дата рождения, место (координаты), и либо приблизительное время "
        "рождения (окно +/-1 час), либо явные границы time_min=ЧЧ:ММ;time_max=ЧЧ:ММ; "
        "плюс хотя бы одно распознанное событие в формате \"описание: дата\", по одному на строке."
    )


def _build_candidate_datetimes(
    window_start: datetime, window_end: datetime, step_minutes: int, max_candidates: int = _MAX_CANDIDATES,
) -> List[datetime]:
    """Same auto-widen-the-step-not-the-window policy as rectification.py's
    version, just against this module's own, much lower candidate cap —
    appropriate to this engine's per-candidate cost (roughly
    1 + 2*len(events) chart builds, vs. Trutine's fixed 2). `max_candidates`
    defaults to the module constant but is normally passed explicitly by
    the caller as `_effective_max_candidates(len(events))`, so the cap
    actually shrinks as the event count grows instead of staying fixed."""
    total_minutes = (window_end - window_start).total_seconds() / 60.0
    if total_minutes <= 0:
        return [window_start]

    effective_step = step_minutes
    naive_count = int(total_minutes / effective_step) + 1
    if naive_count > max_candidates:
        effective_step = max(effective_step, total_minutes / max_candidates)

    candidates: List[datetime] = []
    t = window_start
    step = timedelta(minutes=effective_step)
    while t < window_end:
        candidates.append(t)
        t += step
    candidates.append(window_end)
    return candidates


# --- house "elements" (occupants + ruler + co-ruler) ------------------------

# Deliberately narrower than astro._ACTIVE_POINTS_NATAL: no fixed stars (not
# used by this technique at all) and no Vertex (not a classically rulable
# point, and every occupant/ruler loop below already tolerates a missing
# point gracefully) — keeps each of the many per-candidate chart builds a
# little cheaper.
_CHART_ACTIVE_POINTS = astro._ACTIVE_POINTS_TRANSIT


def _house_elements(natal, cusps: List[float]) -> Dict[int, Dict[str, List]]:
    """Precomputed once per candidate natal chart: for each house 1-12, the
    "elements" per the classical rule this engine follows (occupant
    planets, the house's own ruler, and a co-ruler if the house's arc
    extends more than ~13 degrees into a further sign — see module
    docstring). Returns {house_num: {"labels": [...], "abs_positions": [...]}}."""
    result: Dict[int, Dict[str, List]] = {h: {"labels": [], "abs_positions": []} for h in range(1, 13)}

    for label, attr in astro._PLANET_ATTRS:
        point = getattr(natal, attr, None)
        if point is None:
            continue
        house_num = astro._house_of_degree(cusps, point.abs_pos)
        result[house_num]["labels"].append(f"{label} (в доме)")
        result[house_num]["abs_positions"].append(point.abs_pos)

    for house_num in range(1, 13):
        start = cusps[house_num - 1] % 360.0
        end = cusps[house_num % 12] % 360.0
        arc = (end - start) % 360.0
        if arc <= 0:
            arc = 360.0

        sign_code, within_start = astro._sign_from_abs_pos(start)
        signs_in_house = [sign_code]
        remaining = arc - (30.0 - within_start)
        cursor = (start + (30.0 - within_start)) % 360.0
        while remaining > 0:
            next_sign, _within = astro._sign_from_abs_pos(cursor)
            portion = min(30.0, remaining)
            if portion > 13.0:  # co-ruler threshold, per REKTIF.TXT's worked example
                signs_in_house.append(next_sign)
            cursor = (cursor + portion) % 360.0
            remaining -= portion

        for sign_code_in_house in dict.fromkeys(signs_in_house):  # de-dup, keep order
            ruler_label = astro._CLASSICAL_RULERS_RU.get(sign_code_in_house)
            if not ruler_label:
                continue
            ruler_attr = next((a for lbl, a in astro._PLANET_ATTRS if lbl == ruler_label), None)
            if ruler_attr is None:
                continue
            ruler_point = getattr(natal, ruler_attr, None)
            if ruler_point is None:
                continue
            tag = "управитель" if sign_code_in_house == signs_in_house[0] else "со-управитель"
            result[house_num]["labels"].append(f"{ruler_label} ({tag} дома {house_num})")
            result[house_num]["abs_positions"].append(ruler_point.abs_pos)

    return result


# --- aspect scoring ----------------------------------------------------------

# Per-aspect-type orb, NOT one flat tolerance — reuses astro.py's own
# established major/minor split (astro._MAJOR_ASPECTS: 5-8 degrees;
# astro._MINOR_ASPECTS: 2-3 degrees) instead of a single number. This
# matters a lot here specifically: astro._ASPECT_ANGLES has 11 angles
# spread across 0-180 degrees (many only 6-15 degrees apart, e.g.
# 135/144/150), so a single flat 6-degree tolerance covers most of that
# range between them — nearly ANY moving point would then "match
# something" regardless of the real chart, which was confirmed during
# testing (transit scores came back nearly identical, ~20, for every
# candidate across a 2-hour window — a real, caught bug, not a hypothetical
# one). Tight minor-aspect orbs keep the scoring actually discriminative.
_ASPECT_ORBS: Dict[str, float] = {spec["name"]: float(spec["orb"]) for spec in astro._ALL_ASPECTS}

# Per the sources this engine is modeled on (REKTIF.TXT explicitly: "транзиты
# — более мощное указание" / transits are the stronger indicator), transits
# get the most weight, profections (a coarse whole-sign/yearly technique)
# the least; progressions and directions sit in between since both are
# slow, multi-year techniques.
_TECHNIQUE_WEIGHTS = {
    "transit": 1.5,
    "progression": 1.2,
    "direction": 1.2,
    "profection": 1.0,
}

_MIN_SEPARATION_MINUTES = 20
# How many/how detailed the reported candidates are is NOT a fixed
# constant here — see _adaptive_report_limits, called with the actual
# event count at render time (top_n=6 for a small event list, same as
# this module's original fixed default, shrinking from there).


def _best_aspect(moving_abs: float, elements_abs: List[float], elements_labels: List[str]) -> Optional[Tuple[float, str]]:
    """The single closest (lowest-orb, relative to THAT aspect's own orb —
    see _ASPECT_ORBS) aspect between one moving point and ANY of an
    event's house elements, scored 0-1 (1.0 = exact, 0.0 = right at that
    aspect's own orb edge) — only the best match counts per moving point,
    not every element it happens to loosely aspect, so one very tight hit
    isn't diluted by several coincidental loose ones."""
    best: Optional[Tuple[float, str]] = None
    for elem_abs, elem_label in zip(elements_abs, elements_labels):
        sep = astro._angular_separation(moving_abs, elem_abs)
        for aspect_name, angle in astro._ASPECT_ANGLES.items():
            orb_limit = _ASPECT_ORBS.get(aspect_name)
            if orb_limit is None:
                continue
            orb = abs(sep - angle)
            if orb <= orb_limit:
                score = 1.0 - orb / orb_limit
                if best is None or score > best[0]:
                    best = (score, f"{astro._aspect_ru(aspect_name)} к {elem_label} (орбис {orb:.1f}°)")
    return best


def _score_event_for_candidate(
    natal, house_elements: Dict[int, Dict[str, List]], candidate_birth_dt: datetime,
    fields: Dict[str, str], event: Dict[str, Any],
) -> Dict[str, Any]:
    """Scores ONE event against ONE already-built candidate natal chart,
    across all four techniques. Never raises — a chart-build failure for
    one technique on one event just leaves that technique's score at 0
    rather than aborting the whole candidate."""
    # event["when"] is the event's date at its own stated time if one was
    # given (the richer "description; date; time; ..." format), or local
    # noon on that date otherwise (see _extract_events_and_birth_text) —
    # NOT always noon anymore, despite the variable name history below.
    event_dt_noon = event["when"]
    age_days = (event_dt_noon - candidate_birth_dt).total_seconds() / 86400.0

    base = {
        "description": event["description"], "date": event["date"], "houses": event["houses"],
        "category_matched": event["category_matched"], "category_note": event.get("category_note"),
    }
    if age_days < 0:
        # The event happened before this candidate's birth time — not a
        # real event in this person's life under this hypothesis, so it
        # simply can't confirm or refute it (not a penalty, just N/A).
        return {**base, "applicable": False, "technique_scores": {}, "matches": []}

    elements_abs: List[float] = []
    elements_labels: List[str] = []
    for h in event["houses"]:
        elements_abs.extend(house_elements[h]["abs_positions"])
        elements_labels.extend(house_elements[h]["labels"])

    technique_scores: Dict[str, float] = {}
    matches: List[str] = []

    # --- profection: no new ephemeris chart, pure calendar+rulership -----
    age_full_years = int(age_days / 365.25)
    prof_score = 0.0
    try:
        profected_house, _sign, ruler_label = astro._profection_house_and_ruler(natal, age_full_years)
        if profected_house in event["houses"]:
            prof_score += 1.0
            matches.append(f"профекция: активирован именно дом {profected_house} — точное попадание")
        ruler_attr = next((a for lbl, a in astro._PLANET_ATTRS if lbl == ruler_label), None)
        ruler_point = getattr(natal, ruler_attr, None) if ruler_attr else None
        if ruler_point is not None:
            best = _best_aspect(ruler_point.abs_pos, elements_abs, elements_labels)
            if best:
                prof_score += best[0]
                matches.append(f"профекция: управитель года {ruler_label} — {best[1]}")
    except Exception:
        pass
    technique_scores["profection"] = prof_score * _TECHNIQUE_WEIGHTS["profection"]

    # --- progression + direction share one progressed-subject build ------
    progressed = None
    try:
        from kerykeion.astrological_subject_factory import AstrologicalSubjectFactory

        progressed_dt = astro._secondary_progressed_datetime(candidate_birth_dt, event_dt_noon)
        progressed = AstrologicalSubjectFactory.from_birth_data(
            name="Progressed", year=progressed_dt.year, month=progressed_dt.month, day=progressed_dt.day,
            hour=progressed_dt.hour, minute=progressed_dt.minute,
            lat=float(fields["lat"]), lng=float(fields["lon"]), tz_str=fields["tz"],
            online=False, active_points=_CHART_ACTIVE_POINTS,
        )
    except Exception:
        progressed = None

    prog_score = 0.0
    if progressed is not None:
        for label, attr in astro._PLANET_ATTRS:
            point = getattr(progressed, attr, None)
            if point is None:
                continue
            best = _best_aspect(point.abs_pos, elements_abs, elements_labels)
            if best:
                prog_score += best[0]
                matches.append(f"прогрессия: прогр. {label} — {best[1]}")
    technique_scores["progression"] = prog_score * _TECHNIQUE_WEIGHTS["progression"]

    dir_score = 0.0
    if progressed is not None:
        try:
            arc_degrees = (progressed.sun.abs_pos - natal.sun.abs_pos) % 360.0
            for label, attr in astro._PLANET_ATTRS + astro._ANGLE_ATTRS:
                natal_point = getattr(natal, attr, None)
                if natal_point is None:
                    continue
                directed_abs = (natal_point.abs_pos + arc_degrees) % 360.0
                best = _best_aspect(directed_abs, elements_abs, elements_labels)
                if best:
                    dir_score += best[0]
                    matches.append(f"дирекция: напр. {label} — {best[1]}")
        except Exception:
            pass
    technique_scores["direction"] = dir_score * _TECHNIQUE_WEIGHTS["direction"]

    # --- transit: independent chart at the event date, local noon --------
    transit_score = 0.0
    try:
        from kerykeion.astrological_subject_factory import AstrologicalSubjectFactory

        transit_subject = AstrologicalSubjectFactory.from_birth_data(
            name="Transit", year=event_dt_noon.year, month=event_dt_noon.month, day=event_dt_noon.day,
            hour=event_dt_noon.hour, minute=event_dt_noon.minute,
            lat=float(fields["lat"]), lng=float(fields["lon"]), tz_str=fields["tz"],
            online=False, active_points=_CHART_ACTIVE_POINTS,
        )
        for label, attr in astro._PLANET_ATTRS:
            point = getattr(transit_subject, attr, None)
            if point is None:
                continue
            best = _best_aspect(point.abs_pos, elements_abs, elements_labels)
            if best:
                transit_score += best[0]
                matches.append(f"транзит: трансл. {label} — {best[1]}")
    except Exception:
        pass
    technique_scores["transit"] = transit_score * _TECHNIQUE_WEIGHTS["transit"]

    # Cap how many individual match lines are kept — only for the final
    # human-readable report, doesn't affect the score itself.
    return {**base, "applicable": True, "technique_scores": technique_scores, "matches": matches[:6]}


def _evaluate_candidate(
    fields: Dict[str, str], candidate_birth_dt: datetime, events: List[Dict[str, Any]], name: str,
) -> Optional[Dict]:
    """Builds ONE candidate's natal chart and scores every event against
    it. Returns None (never raises) if the natal chart itself fails to
    build — same "skip this one candidate" convention as rectification.py's
    _evaluate_candidate."""
    try:
        candidate_fields = dict(fields)
        candidate_fields["date"] = candidate_birth_dt.strftime("%Y-%m-%d")
        candidate_fields["time"] = candidate_birth_dt.strftime("%H:%M")
        natal = astro._build_subject(candidate_fields, name=name, active_points=_CHART_ACTIVE_POINTS)
        cusps = astro._house_cusp_degrees(natal)
    except Exception:
        return None

    house_elements = _house_elements(natal, cusps)
    event_results = [
        _score_event_for_candidate(natal, house_elements, candidate_birth_dt, candidate_fields, ev)
        for ev in events
    ]
    total_score = sum(sum(er["technique_scores"].values()) for er in event_results)
    return {"birth_dt": candidate_birth_dt, "total_score": total_score, "event_results": event_results}


def _diverse_top_candidates(results: List[Dict], top_n: int, min_separation_minutes: int) -> List[Dict]:
    """Same rationale as rectification.py's version — the goal is to show
    genuinely DIFFERENT competing hypotheses, not the same local maximum
    and its immediate step-neighbors — just ranked by DESCENDING score
    (higher = more confirmations) instead of ascending error."""
    ranked = sorted(results, key=lambda r: r["total_score"], reverse=True)
    picked: List[Dict] = []
    for r in ranked:
        if all(
            abs((r["birth_dt"] - p["birth_dt"]).total_seconds()) / 60.0 >= min_separation_minutes for p in picked
        ):
            picked.append(r)
        if len(picked) >= top_n:
            break
    return picked


def _adaptive_report_limits(n_events: int) -> Tuple[int, int]:
    """Returns (top_n_candidates, max_matches_per_event) — both shrink as
    the event count grows, so the FINAL REPORT TEXT stays bounded
    regardless of how many events were given.

    This exists because of a real, reported crash: this tool's raw text
    output is later injected WHOLE into the model's context window as an
    always-include RAG chunk (see routes/chat.py's computed_chunk), with
    no size cap of its own at that injection point — a detailed per-event-
    per-candidate report for 42 events, at the ORIGINAL fixed
    _TOP_N_REPORTED=6/6-matches-per-event settings, produced a raw result
    alone equivalent to roughly 89000 tokens, far beyond the model's 32768-
    token context window (`Requested tokens (89155) exceed context window
    of 32768`) — a hard failure, not a quality issue. The fix has to live
    HERE (this module has no way to know or control the model's context
    size), not in routes/chat.py or utils/rag.py, since neither of those
    imposes a size cap on a tool's raw result today."""
    if n_events <= 8:
        return 6, 6
    if n_events <= 15:
        return 4, 3
    if n_events <= 25:
        return 3, 2
    return 2, 1


# Hard safety net on top of _adaptive_report_limits' estimate-based
# shrinking (belt and braces — the estimate could still be wrong for an
# unusual input, e.g. very long event descriptions): if the assembled
# report text is still larger than this after formatting, degrade further
# by dropping match-line detail and showing fewer candidates, rather than
# ever returning an unbounded string. ~20000 characters is a conservative
# few-thousand-token budget (roughly 8000 tokens even at a generous
# 2.5 chars/token for mixed Cyrillic/Latin text) — comfortable room in a
# 32768-token context alongside the retrieved methodology/reference
# chunks (capped separately at config.RAG_ALWAYS_INCLUDE_MAX_CHARS, 16000
# chars), the question itself, and the model's own generated answer.
_MAX_REPORT_CHARS = 20000


def _format_candidate_block(index: int, r: Dict, max_matches_shown: int) -> str:
    lines = [
        f"{index}. Время рождения: {r['birth_dt'].strftime('%Y-%m-%d %H:%M')} "
        f"(суммарный балл совпадений {r['total_score']:.2f})"
    ]
    for er in r["event_results"]:
        if not er["applicable"]:
            lines.append(
                f"   - {er['description']} ({er['date'].strftime('%Y-%m-%d')}): событие раньше этого "
                "варианта времени рождения — не оценивалось."
            )
            continue
        scores = er["technique_scores"]
        scores_str = ", ".join(f"{k}={v:.2f}" for k, v in scores.items())
        if not er["category_matched"]:
            cat_note = " [неопределённая категория событий]"
        elif er.get("category_note"):
            cat_note = f" [{er['category_note']}]"
        else:
            cat_note = ""
        lines.append(
            f"   - {er['description']} ({er['date'].strftime('%Y-%m-%d')}), дома {er['houses']}"
            f"{cat_note}: {scores_str}"
        )
        for m in er["matches"][:max_matches_shown]:
            lines.append(f"       {m}")
    return "\n".join(lines)


# --- async engine ------------------------------------------------------------

# Sized like a typical default ThreadPoolExecutor, capped low — Swiss-
# Ephemeris chart builds are CPU work, not I/O, so there's no benefit to a
# huge pool, only contention; capped at 8 to leave room for the rest of the
# server process's own concurrent work.
_CANDIDATE_EXECUTOR = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4)))


async def run_rectification_events_async(spec: str) -> str:
    """The actual multi-candidate search, run concurrently: each
    candidate's full evaluation (1 natal + up to 2*len(events) further
    charts) is independent of every other candidate's, so all of them are
    dispatched to _CANDIDATE_EXECUTOR at once via asyncio.gather rather
    than evaluated one at a time in a plain loop — see module docstring."""
    birth_text, events, warnings = _extract_events_and_birth_text(spec)
    if not events:
        return (
            "Не найдено ни одного распознанного события. Формат: по одной строке на событие — "
            "либо \"описание: дата\" (например \"брак: 21.01.1983\"), либо \"описание; дата; "
            "[время]; [место]; [широта]; [долгота]; [комментарий]\" (лишние поля после даты можно "
            "не указывать или оставлять пустыми). " + " ".join(warnings)
        ).strip()

    if len(events) > _MAX_EVENTS:
        warnings.append(f"Событий больше {_MAX_EVENTS} — учтены только первые {_MAX_EVENTS}.")
        events = events[:_MAX_EVENTS]

    # Semantic override of the keyword-based classification above — see
    # _apply_llm_event_classification's own docstring. Deliberately run
    # AFTER the _MAX_EVENTS truncation (no point spending LLM calls on
    # events that got cut) and BEFORE the candidate search starts (houses
    # are shared across every candidate, so this only runs once total, not
    # once per candidate).
    await _apply_llm_event_classification(events)

    fields, window_start, window_end, step_minutes, missing = _extract_birth_window_fields(birth_text)
    if missing:
        return _missing_fields_message(missing)

    name = fields.get("name") or "Subject"
    candidates = _build_candidate_datetimes(
        window_start, window_end, step_minutes, _effective_max_candidates(len(events))
    )

    loop = asyncio.get_running_loop()
    raw = await asyncio.gather(*[
        loop.run_in_executor(_CANDIDATE_EXECUTOR, _evaluate_candidate, fields, dt, events, name)
        for dt in candidates
    ])
    results = [r for r in raw if r is not None]
    if not results:
        return "Не удалось рассчитать ни одного варианта карты в заданном окне — проверьте дату/координаты."

    best = max(results, key=lambda r: r["total_score"])

    def render(top_n: int, max_matches_shown: int, max_events_listed: Optional[int]) -> str:
        """Builds the full report text at a given verbosity level — pulled
        out as a closure over the already-computed results/best (the
        expensive part) so degrading verbosity to fit _MAX_REPORT_CHARS
        (see below) is just cheap re-formatting, never recomputation."""
        top = _diverse_top_candidates(results, top_n, _MIN_SEPARATION_MINUTES)
        listed_events = events if max_events_listed is None else events[:max_events_listed]

        lines = [
            f"Многотехничная ректификация для {name} по {len(events)} событиям "
            "(профекции + вторичные прогрессии + дирекции солнечной дуги + транзиты).",
            f"Заданная дата: {fields['date']}. Окно поиска времени рождения: "
            f"{window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')} "
            f"({len(candidates)} вариантов времени x {len(events)} событий x 4 техники "
            f"— шаг перебора мог быть автоматически увеличен относительно запрошенного, "
            "если запрошенное окно/шаг дали бы слишком много вариантов).",
        ]
        # Deliberately placed as the very first substantive content of the
        # report, immediately after the title/window lines and BEFORE the
        # (potentially very long, up to _MAX_EVENTS lines) event echo list —
        # a real, reported failure showed the follow-up model can otherwise
        # miss this line entirely when it's buried after dozens of event
        # lines (it wrote a whole qualitative discussion of "house 1
        # matters" without ever stating a concrete rectified time at all).
        # It is deliberately repeated again at the very end of the report
        # too (see bottom of this function) — "lost in the middle" small-
        # model attention behaviour means bookending both ends is more
        # reliable than relying on either position alone. See
        # rectification_events_methodology.txt point 7: the model is
        # explicitly told to quote THIS line's time verbatim in its answer.
        summary_line = (
            f"ИТОГОВЫЙ ЛУЧШИЙ ВАРИАНТ ВРЕМЕНИ РОЖДЕНИЯ: {best['birth_dt'].strftime('%Y-%m-%d %H:%M')} "
            f"(суммарный балл совпадений {best['total_score']:.2f} — чем ВЫШЕ, тем лучше; это мера "
            "ПОДТВЕРЖДЕНИЯ, противоположная по смыслу рассогласованию из метода «Трутина Гермеса»)."
        )
        lines.append("")
        lines.append(summary_line)
        lines.append("")
        if warnings:
            lines.append("Предупреждения при разборе событий: " + " ".join(warnings))
        lines.append(
            "События и распознанные дома (по ключевым словам; \"неопределённая категория\" — "
            "сработал общий запасной вариант, дома 1 и 10, а не конкретное классическое правило):"
        )
        for ev in listed_events:
            if not ev["category_matched"]:
                cat_note = " [неопределённая категория]"
            elif ev.get("category_note"):
                cat_note = f" [{ev['category_note']}]"
            else:
                cat_note = ""
            lines.append(
                f"  {ev['description']}: {ev['date'].strftime('%Y-%m-%d')} -> дом(а) {ev['houses']}{cat_note}"
            )
        if max_events_listed is not None and len(events) > max_events_listed:
            lines.append(f"  ...и ещё {len(events) - max_events_listed} событий (не показаны, отчёт сокращён).")
        lines.append("")
        lines.append(
            f"Наиболее непохожие друг на друга варианты (не ближе {_MIN_SEPARATION_MINUTES} минут "
            "друг к другу), от лучшего к худшему:"
        )
        for i, r in enumerate(top, start=1):
            lines.append(_format_candidate_block(i, r, max_matches_shown))

        if len(top) > 1:
            gap = top[0]["total_score"] - top[1]["total_score"]
            lines.append(
                f"\nРазница баллов между лучшим и вторым непохожим вариантом: {gap:.2f}. Небольшая "
                "разница означает реальную неоднозначность (несколько времён рождения примерно "
                "одинаково хорошо подтверждаются событиями) — не выдавай второй и далее варианты за "
                "существенно менее вероятные, если разница мала."
            )

        if best["birth_dt"] in (results[0]["birth_dt"], results[-1]["birth_dt"]):
            edge = "начале" if best["birth_dt"] == results[0]["birth_dt"] else "конце"
            lines.append(
                f"\nВНИМАНИЕ: лучший вариант находится на самом краю ({edge}) заданного окна поиска — "
                "балл совпадений, судя по всему, ПРОДОЛЖАЛ РАСТИ за пределами окна, и реальный лучший "
                "результат может лежать ВНЕ проверенного диапазона. Стоит повторить поиск с более "
                "широким окном (time_min=/time_max=) в эту сторону, прежде чем доверять этому "
                "результату как окончательному."
            )

        # Bookend repeat of the summary line — see the comment where
        # summary_line is first built, near the top of this function.
        lines.append("")
        lines.append(summary_line)
        return "\n".join(lines)

    top_n, max_matches = _adaptive_report_limits(len(events))
    report = render(top_n, max_matches, max_events_listed=None)

    # Hard safety net (see _MAX_REPORT_CHARS's own comment): the adaptive
    # estimate above is heuristic, not an exact character count — if it
    # still undershoots for an unusual input (e.g. unusually long event
    # descriptions), degrade further in two escalating steps rather than
    # ever returning an unbounded string that could crash the follow-up
    # generation call the way the original, non-adaptive version did.
    if len(report) > _MAX_REPORT_CHARS:
        report = render(1, 0, max_events_listed=None)
    if len(report) > _MAX_REPORT_CHARS:
        report = render(1, 0, max_events_listed=max(10, _MAX_REPORT_CHARS // 150))
    if len(report) > _MAX_REPORT_CHARS:
        report = report[:_MAX_REPORT_CHARS] + "\n[отчёт обрезан, слишком много данных для одного ответа]"
    return report


def run_rectification_events(spec: str) -> str:
    """Tool entry point (utils.tools.TOOL_REGISTRY) — a plain sync
    string->string function, per that registry's contract (see
    utils/tools.py's module docstring), wrapping the async engine via
    asyncio.run(). Safe to call from a plain worker thread with no event
    loop of its own (exactly how routes/chat.py invokes every tool — see
    `await loop.run_in_executor(None, tool_spec["run"], tool_arg)`), same
    dual sync/async pattern already established by
    tool_router.classify/classify_async."""
    return asyncio.run(run_rectification_events_async(spec))


_BEST_RECOMMENDATION_RE = re.compile(r"^ИТОГОВЫЙ ЛУЧШИЙ ВАРИАНТ.*$", re.MULTILINE)


def extract_best_recommendation(report_text: str) -> Optional[str]:
    """Pulls the "ИТОГОВЫЙ ЛУЧШИЙ ВАРИАНТ ВРЕМЕНИ РОЖДЕНИЯ: ..." line back
    out of this tool's own raw report text verbatim (report_text is
    whatever run_rectification_events returned). Used by routes/chat.py to
    deterministically PREPEND this exact line to the follow-up LLM's final
    answer — a real, reported failure showed the small local model can not
    just omit this number but actively invent a different, physically
    implausible one in its own prose (e.g. stating a time 8 hours away
    from a well-established medical birth time, nowhere near what the
    tool's own search window even covered). No amount of prompt
    reinforcement fully prevented that in testing, so this makes the
    correct number reach the user unconditionally, regardless of what the
    model's own paraphrase says. Returns None if the line isn't present
    (e.g. an error/"missing fields" message instead of a real report) —
    caller then leaves the reply untouched."""
    m = _BEST_RECOMMENDATION_RE.search(report_text)
    return m.group(0) if m else None
