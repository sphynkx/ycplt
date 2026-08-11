"""Birth-time rectification: given an APPROXIMATE birth time (a window of
uncertainty, not one exact moment) plus a place, searches for the candidate
birth time within that window that best satisfies a classical rectification
rule, instead of computing one fixed chart the way every other technique in
utils/astro.py does.

This is the first of what's meant to grow into several rectification
methods (see the module docstring precedent set by ASTRO_OPERATIONS in
utils/astro.py — same "start narrow, add methods as concrete need arises"
philosophy). The user's own stated longer-term vision is a much bigger
multi-technique search: build progressions/directions/lunar returns/
transits for several known LIFE EVENT dates against many candidate birth
times, score how tightly they cluster, and let the model reason about which
direction to adjust — that is real future work, NOT implemented here. What
IS implemented here is the first, simplest, fastest piece of that vision:
the "Trutine of Hermes" (Трутина Гермеса), a single self-contained
classical method that doesn't need any life-event dates at all, only the
approximate birth data itself plus an assumed gestation length.

Trutine of Hermes, as implemented here (see rectification_trutine_
methodology.txt for the full caveat list — the user's own indexed
rectification corpus has the classical theory/history, this module and its
methodology doc only cover what THIS implementation specifically computes
and what its output means):
  as the Moon at birth, so the Ascendant at conception;
  as the Ascendant at birth, so the Moon at conception.
i.e. birth Ascendant's absolute ecliptic degree should equal conception
Moon's degree, and birth Moon's degree should equal conception Ascendant's
degree. Conception is estimated as a fixed number of days before birth (the
gestation length, default 273 — the classical average; adjustable), which
means conception happens at the SAME local clock time as whatever birth
time is being tested, just `gestation_days` earlier — this keeps the two
moments' diurnal circumstances comparable and avoids introducing a second
free variable (an unknown conception TIME as well as an unknown birth
time). Conception location is assumed identical to the birth location (an
explicit, documented simplification — the real conception location is
essentially never known).

Deliberately NOT implemented here (left for a later pass, once/if the
user's corpus confirms the exact rule to use): the classical "diurnal vs.
nocturnal" refinement some sources add, which swaps the Ascendant for the
Descendant depending on whether the Moon was above or below the horizon at
birth. Several real historical sources describe this differently, and
guessing at the wrong variant would be worse than being explicit about
using the simpler, most commonly cited Ascendant/Moon-only swap for now.
"""
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from utils import astro

# Classical average human gestation length used by the Trutine of Hermes
# tradition — NOT a medical claim about any individual pregnancy, just the
# conventional fixed offset this technique uses to estimate a conception
# moment from a birth moment. Overridable per request (gestation_days=...).
_DEFAULT_GESTATION_DAYS = 273

# Total width (not half-width) of the default search window when the
# caller gives one approximate time but no explicit time_min/time_max —
# +/-1 hour around the stated approximate time. Wide enough to be useful
# for a birth record that's only "somewhere around 4:30", narrow enough
# that even a 1-minute step stays a small (~120-candidate) search.
_DEFAULT_WINDOW_MINUTES = 120

# Safety cap on how many candidate birth times a single request will ever
# evaluate (each candidate builds TWO charts — birth and conception — via
# Swiss Ephemeris, so this is the real cost driver). Without this, a
# careless combination of a huge window and a 1-minute step could turn one
# request into thousands of chart builds; if the requested window/step
# combination would exceed this, the step is silently widened just enough
# to fit (see _build_candidate_datetimes) rather than refusing outright.
_MAX_CANDIDATES = 1500

# Only points needed for the Trutine comparison itself (Moon + Ascendant) —
# requesting the full _ACTIVE_POINTS_NATAL set for every one of potentially
# hundreds of candidate charts would multiply the ephemeris work for
# planets/points this technique never looks at. Sun is included too since
# some kerykeion active-points combinations assume it's present.
#
# "Ascendant" MUST be listed explicitly here — confirmed by testing:
# unlike the house cusps themselves (always computed as part of the house
# system), kerykeion's AstrologicalSubjectFactory.from_birth_data() only
# populates `.ascendant` at all when "Ascendant" is in active_points;
# without it, `.ascendant` is silently None instead of raising, which
# would otherwise surface as a confusing `NoneType has no attribute
# abs_pos` deep inside the search loop.
_MINIMAL_ACTIVE_POINTS = ["Sun", "Moon", "Ascendant"]

# How many of the best (lowest total-error) candidates to report in full.
_TOP_N_REPORTED = 8

# Two reported candidates are only both worth showing if they're genuinely
# different hypotheses, not the same local minimum one search-step apart —
# candidates within this many minutes of an already-reported one are
# skipped when building the "top N" list (see _diverse_top_candidates).
_MIN_SEPARATION_MINUTES = 20


def _angular_separation(deg_a: float, deg_b: float) -> float:
    """Same computation as astro._angular_separation — re-exported/reused
    here directly (not duplicated logic) purely so this module doesn't
    need to reach into astro's private helper by name at every call site."""
    return astro._angular_separation(deg_a, deg_b)


def _parse_hhmm_on_date(date_str: str, hhmm: str) -> Optional[datetime]:
    m = re.match(r"^(\d{1,2}):(\d{2})$", hhmm.strip())
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    try:
        year, month, day = (int(x) for x in date_str.split("-"))
        return datetime(year, month, day, hour, minute)
    except Exception:
        return None


def _extract_rectification_fields(
    spec: str,
) -> Tuple[Dict[str, str], Optional[datetime], Optional[datetime], int, int, List[str]]:
    """Rectification-specific field extraction — deliberately NOT a call to
    astro._extract_fields, since that function treats "time" as required
    (the whole point here is that the exact time is what's UNKNOWN). Reuses
    astro._parse_spec (key=value fast path) and astro._fill_fields_from_text
    (free-text date/time/coordinates/city fallback, plus tz auto-resolution)
    for date/lat/lon/tz/name/time exactly like every other technique in this
    app, then separately handles the window/gestation/step parameters that
    only this technique needs.

    The single free-text "time" (if the classifier/user's own wording
    happened to state one, e.g. "около 4:30") becomes the CENTER of a
    _DEFAULT_WINDOW_MINUTES-wide window unless the caller gave explicit
    time_min=/time_max=/window_minutes= overrides, which always take
    priority — same "explicit key=value beats free-text guess" precedent
    every other field in this app already follows.

    Returns (fields, window_start, window_end, gestation_days,
    step_minutes, missing) — missing is a list of human-readable problem
    descriptions (empty = nothing missing); window_start/window_end are
    None only when missing is non-empty."""
    fields = astro._parse_spec(spec)
    astro._fill_fields_from_text(fields, spec)  # date/time/lat/lon/tz/name + tz auto-resolve

    missing: List[str] = []
    for key, label in (("date", "дата"), ("lat", "широта"), ("lon", "долгота"), ("tz", "часовой пояс")):
        if not fields.get(key):
            missing.append(label)

    gestation_days = _DEFAULT_GESTATION_DAYS
    if fields.get("gestation_days"):
        try:
            gestation_days = int(fields["gestation_days"])
        except Exception:
            pass

    step_minutes = 1
    if fields.get("step_minutes"):
        try:
            step_minutes = max(1, int(fields["step_minutes"]))
        except Exception:
            pass

    window_start = window_end = None
    if missing:
        return fields, None, None, gestation_days, step_minutes, missing

    date_str = fields["date"]
    time_min_str, time_max_str = fields.get("time_min"), fields.get("time_max")
    if time_min_str and time_max_str:
        window_start = _parse_hhmm_on_date(date_str, time_min_str)
        window_end = _parse_hhmm_on_date(date_str, time_max_str)
        if window_start is None or window_end is None:
            missing.append("time_min/time_max (ожидается формат ЧЧ:ММ)")
        elif window_end <= window_start:
            # Window crosses midnight (e.g. time_min=23:00, time_max=00:30)
            # — the common, real case for a birth right around midnight,
            # not a mistake to reject.
            window_end += timedelta(days=1)
    else:
        base_time_str = fields.get("time")
        if not base_time_str:
            missing.append(
                "примерное время рождения (укажите либо приблизительное "
                "время, либо time_min=/time_max=)"
            )
        else:
            base_dt = _parse_hhmm_on_date(date_str, base_time_str)
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

    return fields, window_start, window_end, gestation_days, step_minutes, missing


def _missing_rectification_fields_message(missing: List[str]) -> str:
    return (
        "Не хватает данных для ректификации: " + ", ".join(missing) + ". "
        "Нужны: примерная дата рождения, место (координаты — автоматически "
        "определяется часовой пояс), и либо приблизительное время рождения "
        "(будет использовано окно +/-1 час вокруг него), либо явные границы "
        "окна time_min=ЧЧ:ММ;time_max=ЧЧ:ММ."
    )


def _build_candidate_datetimes(window_start: datetime, window_end: datetime, step_minutes: int) -> List[datetime]:
    """Every candidate birth moment to test, at step_minutes resolution,
    always including window_end itself even if it doesn't fall exactly on
    a step boundary (otherwise the requested upper bound could silently
    never be tested at all). Auto-widens the step (never the window) if
    the naive candidate count would exceed _MAX_CANDIDATES — a wide search
    window with a very fine step is a real cost driver (two chart builds
    per candidate), not something to fail outright over."""
    total_minutes = (window_end - window_start).total_seconds() / 60.0
    if total_minutes <= 0:
        return [window_start]

    effective_step = step_minutes
    naive_count = int(total_minutes / effective_step) + 1
    if naive_count > _MAX_CANDIDATES:
        effective_step = max(effective_step, total_minutes / _MAX_CANDIDATES)

    candidates: List[datetime] = []
    t = window_start
    step = timedelta(minutes=effective_step)
    while t < window_end:
        candidates.append(t)
        t += step
    candidates.append(window_end)
    return candidates


def _build_minimal_subject(fields: Dict[str, str], at: datetime, name: str):
    """A birth-data dict identical to `fields` except date/time overridden
    to `at` — used for both the birth-time candidate itself and its
    corresponding conception moment, so both go through astro._build_subject
    exactly the way every other technique's own subjects do (same
    kerykeion factory, same never-raise-let-caller-decide convention)."""
    candidate_fields = dict(fields)
    candidate_fields["date"] = at.strftime("%Y-%m-%d")
    candidate_fields["time"] = at.strftime("%H:%M")
    return astro._build_subject(candidate_fields, name=name, active_points=_MINIMAL_ACTIVE_POINTS)


def _evaluate_candidate(fields: Dict[str, str], birth_dt: datetime, gestation_days: int, name: str) -> Optional[Dict]:
    """Builds the birth chart at birth_dt and the conception chart at
    (birth_dt - gestation_days), then scores how well the Trutine of Hermes
    symmetry holds between them. Returns None (never raises) if either
    chart fails to build — a candidate near the edges of a wide search
    window can occasionally hit a kerykeion/Swiss-Ephemeris edge case;
    skipping just that one candidate is preferable to aborting the whole
    search over it."""
    try:
        birth = _build_minimal_subject(fields, birth_dt, name)
        conception_dt = birth_dt - timedelta(days=gestation_days)
        conception = _build_minimal_subject(fields, conception_dt, f"{name} (зачатие)")
    except Exception:
        return None

    asc_birth, moon_birth = birth.ascendant.abs_pos, birth.moon.abs_pos
    asc_conception, moon_conception = conception.ascendant.abs_pos, conception.moon.abs_pos

    # As the Moon at birth, so the Ascendant at conception (error_moon_asc);
    # as the Ascendant at birth, so the Moon at conception (error_asc_moon).
    error_asc_moon = _angular_separation(asc_birth, moon_conception)
    error_moon_asc = _angular_separation(moon_birth, asc_conception)

    return {
        "birth_dt": birth_dt,
        "conception_dt": conception_dt,
        "asc_birth": asc_birth,
        "moon_birth": moon_birth,
        "asc_conception": asc_conception,
        "moon_conception": moon_conception,
        "error_asc_moon": error_asc_moon,
        "error_moon_asc": error_moon_asc,
        "total_error": error_asc_moon + error_moon_asc,
        # Carried at zero extra cost (birth was already built above) so the
        # winning candidate's own chart can be handed to utils/chart_draw.py
        # without re-running this whole window scan a second time — see
        # run_rectification_trutine_and_subject's own docstring.
        "birth_subject": birth,
    }


def _diverse_top_candidates(results: List[Dict], top_n: int, min_separation_minutes: int) -> List[Dict]:
    """Picks up to top_n candidates by ascending total_error, skipping any
    candidate within min_separation_minutes of one already picked — without
    this, the top N would usually all just be the single best local minimum
    and its immediate neighbors (adjacent search steps around the same
    answer), which tells the model nothing about whether a genuinely
    DIFFERENT time also fits reasonably well (real ambiguity worth
    surfacing) versus there being one clear, isolated best answer."""
    ranked = sorted(results, key=lambda r: r["total_error"])
    picked: List[Dict] = []
    for r in ranked:
        if all(
            abs((r["birth_dt"] - p["birth_dt"]).total_seconds()) / 60.0 >= min_separation_minutes for p in picked
        ):
            picked.append(r)
        if len(picked) >= top_n:
            break
    return picked


def _format_degree(abs_pos: float) -> str:
    sign_code, within = astro._sign_from_abs_pos(abs_pos)
    return f"{astro._sign_ru(sign_code)} {within:.1f}°"


def _format_candidate_block(index: int, r: Dict) -> str:
    bt = r["birth_dt"]
    ct = r["conception_dt"]
    return (
        f"{index}. Время рождения: {bt.strftime('%Y-%m-%d %H:%M')} "
        f"(суммарное рассогласование {r['total_error']:.2f}°)\n"
        f"   Асцендент рождения {_format_degree(r['asc_birth'])} <-> "
        f"Луна зачатия {_format_degree(r['moon_conception'])} "
        f"(расхождение {r['error_asc_moon']:.2f}°)\n"
        f"   Луна рождения {_format_degree(r['moon_birth'])} <-> "
        f"Асцендент зачатия {_format_degree(r['asc_conception'])} "
        f"(расхождение {r['error_moon_asc']:.2f}°)\n"
        f"   Момент зачатия при этом времени рождения: {ct.strftime('%Y-%m-%d %H:%M')}"
    )


def run_rectification_trutine(spec: str) -> str:
    """Tool entry point (utils.tools.TOOL_REGISTRY["astro_rectification_
    trutine"]) — thin wrapper over _run_rectification_trutine_full,
    discarding the winning candidate's chart object (see that function's
    own docstring for why it exists as a separate, richer-returning
    sibling rather than changing this one's plain str contract)."""
    return _run_rectification_trutine_full(spec)[0]


def _run_rectification_trutine_full(spec: str) -> Tuple[str, Optional[Any]]:
    """Does the actual work; returns (report_text, winning_candidate_
    subject_or_None). Split out from run_rectification_trutine so
    routes/chat.py's chart-drawing step can get the winning candidate's
    already-built chart (see _evaluate_candidate's "birth_subject" key)
    WITHOUT re-running this whole window scan a second time — the scan
    itself (potentially hundreds of candidate charts) is the expensive
    part here, unlike a technique with a single fast chart build, where a
    cheap from-scratch recompute (this app's usual pattern — see
    astro.get_transit_profiles' own docstring on why THAT recompute is
    fine) would be wasteful and pointlessly slow for this one instead.

    Never raises — any failure becomes a plain-text explanation instead,
    same convention as every run_* function in utils/astro.py.

    Deliberately does NOT go through get_planet_profiles-style profiling —
    there's no single chart's "significant points" to rank here, just a
    ranked list of candidate birth times. routes/chat.py's digest step
    (interpret.digest_facts_async) is skipped for this tool for exactly
    that reason (no get_rectification_profiles function exists), so this
    tool's raw text goes straight into the generic RAG reasoning-mode
    prompt (rag_utils.build_prompt) alongside whatever the user's own
    indexed rectification corpus retrieves for the question — a good fit,
    since that prompt already asks the model to reason step by step over
    given facts before answering, which is exactly the "which candidate
    looks best, and which direction would refine it further" job this
    output needs done."""
    fields, window_start, window_end, gestation_days, step_minutes, missing = _extract_rectification_fields(spec)
    if missing:
        return _missing_rectification_fields_message(missing), None

    name = fields.get("name") or "Subject"
    candidates = _build_candidate_datetimes(window_start, window_end, step_minutes)

    results = [r for r in (_evaluate_candidate(fields, dt, gestation_days, name) for dt in candidates) if r]
    if not results:
        return "Не удалось рассчитать ни одного варианта карты в заданном окне времени — проверьте дату/координаты.", None

    best = min(results, key=lambda r: r["total_error"])
    top = _diverse_top_candidates(results, _TOP_N_REPORTED, _MIN_SEPARATION_MINUTES)

    lines = [
        f"Ректификация времени рождения методом «Трутина Гермеса» для {name}.",
        f"Заданная дата: {fields['date']}. Окно поиска времени рождения: "
        f"{window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')} "
        f"({len(candidates)} вариантов, шаг перебора ~{(candidates[1]-candidates[0]).total_seconds()/60:.1f} мин "
        f"— шаг мог быть автоматически увеличен относительно запрошенного, если запрошенное окно/шаг "
        f"дали бы слишком много вариантов).",
        f"Гестационный период (интервал зачатие -> рождение): {gestation_days} дней "
        "(классическое среднее значение, не медицинское утверждение о конкретной беременности).",
        "Место рождения использовано и для зачатия — точное место зачатия неизвестно, это принятое "
        "упрощение метода, а не найденный факт.",
        "Метод (упрощённый вариант — без учёта дневного/ночного положения Луны, см. методику): "
        "Асцендент рождения должен совпадать с Луной зачатия, а Луна рождения — с Асцендентом "
        "зачатия; чем меньше суммарное рассогласование, тем лучше кандидат соответствует методу.",
        "",
        f"Лучший найденный вариант — {best['birth_dt'].strftime('%Y-%m-%d %H:%M')} "
        f"(суммарное рассогласование {best['total_error']:.2f}°).",
        "",
        f"Наиболее непохожие друг на друга варианты (не ближе {_MIN_SEPARATION_MINUTES} минут "
        "друг к другу), от лучшего к худшему:",
    ]
    for i, r in enumerate(top, start=1):
        lines.append(_format_candidate_block(i, r))

    if len(top) > 1:
        gap = top[1]["total_error"] - top[0]["total_error"]
        lines.append(
            f"\nРазница суммарного рассогласования между лучшим и вторым непохожим вариантом: "
            f"{gap:.2f}°. Небольшая разница означает реальную неоднозначность (несколько времён "
            "рождения одинаково хорошо подходят под метод в этом окне) — не выдавай второй и "
            "далее варианты за менее вероятные, если разница мала."
        )

    # If the best candidate sits right at either edge of the search window,
    # the error was still improving in that direction when the window ran
    # out — a real, meaningfully different situation from a true interior
    # minimum, and worth flagging explicitly rather than silently reporting
    # an edge value as if it were as reliable as an interior one. Detected
    # against the actual candidate list (not just window_start/window_end)
    # since the step can be auto-widened and no longer land exactly there.
    if best["birth_dt"] in (results[0]["birth_dt"], results[-1]["birth_dt"]):
        edge = "начале" if best["birth_dt"] == results[0]["birth_dt"] else "конце"
        lines.append(
            f"\nВНИМАНИЕ: лучший вариант находится на самом краю ({edge}) заданного окна поиска — "
            "это значит, что рассогласование, судя по всему, ПРОДОЛЖАЛО УМЕНЬШАТЬСЯ за пределами "
            "окна, и реальный лучший результат может лежать ВНЕ проверенного диапазона. Стоит "
            "повторить поиск с более широким окном (time_min=/time_max=) в эту сторону, прежде чем "
            "доверять этому результату как окончательному."
        )
    return "\n".join(lines), best.get("birth_subject")


_BEST_RECOMMENDATION_RE = re.compile(r"^Лучший найденный вариант.*$", re.MULTILINE)


def extract_best_recommendation(report_text: str) -> Optional[str]:
    """Pulls the "Лучший найденный вариант — ..." line back out of this
    tool's own raw report text verbatim. Same rationale and same
    deterministic-prepend usage in routes/chat.py as
    rectification_events.extract_best_recommendation — see that
    function's docstring for the real failure this guards against (the
    follow-up model inventing a different, physically implausible time
    instead of quoting the one actually computed). Returns None if the
    line isn't present (e.g. an error/"missing fields" message instead of
    a real report)."""
    m = _BEST_RECOMMENDATION_RE.search(report_text)
    return m.group(0) if m else None
