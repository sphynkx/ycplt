"""Astrological wheel-chart rendering — pure SVG, no raster dependency
(no PIL/cairo needed to PRODUCE the image; the wheel is just circles,
radial lines, wedge paths, and text, all of which SVG expresses natively
as short XML strings). Deliberately NOT a separate microservice call like
ycplt_img: this renders in milliseconds, in-process, with no model or GPU
involved, so it's wired as a synchronous file attachment on the same
message the tool's own text answer is already returned on — see
routes/chat.py's own comment at the call site for why that's the better
fit than ycplt_img's async job/polling machinery (which exists to survive
a generation that takes minutes on a separate host, not for this).

Deliberately generic across every astro_* technique rather than one
drawer per technique: every technique already ends up with either ONE
kerykeion subject (natal, horary, electional's winning moment,
rectification's winning candidate, profection) or TWO (transit,
synastry, progression, solar/lunar return) — draw_wheel_svg's own
(subject, second=...) signature mirrors that exact split, and the caller
(routes/chat.py) is the only place that needs to know which of those two
shapes a given technique produces. See CHART_DRAWABLE in routes/chat.py
for the per-tool mapping.

Wheel convention (matches the reference screenshots the user supplied
from a real desktop astrology program — ZET): Ascendant at 9 o'clock
(screen left), houses numbered COUNTERCLOCKWISE from there — house IV/IC
at 6 o'clock (bottom), house VII/Descendant at 3 o'clock (right), house
X/MC at 12 o'clock (top). This is the near-universal convention across
Western astrology software, not an arbitrary choice. Absolute ecliptic
degree increases in the SAME direction as house number (both
counterclockwise here) — see _theta_for_degree's own docstring for the
derivation.
"""
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from utils import astro
from utils import llm as llm_utils

# --- whether to draw a chart at all -----------------------------------------
#
# User's own explicit spec: draw a chart by default for every astro_*
# technique reply, EXCEPT when the user's own message explicitly says not
# to. LLM-first, per this app's own established "never regex a semantic
# judgment" rule (see e.g. utils/electional.py's own module docstring on
# why a fixed phrase list was tried and rejected for a similar decision) —
# but with the OPPOSITE default from every other LLM-first classifier in
# this app: those all default to the more conservative/narrower behavior
# when no model is loaded, whereas this one defaults to True (draw) with
# no model, matching "draw unless told not to" being the stated normal
# behavior, not the exceptional one.
_SHOULD_DRAW_PROMPT = """Тебе показано сообщение пользователя в диалоге об астрологии, для которого
уже подготовлен текстовый разбор карты. Определи: просил ли пользователь
явно НЕ рисовать/не показывать изображение карты (например "без картинки",
"не рисуй карту", "только текст", "график не нужен")?

Сообщение: "{text}"

Если пользователь явно попросил обойтись без изображения — ответь СТРОГО
одним словом: НЕТ
Если явного отказа от изображения нет (в том числе если про изображение
вообще ничего не сказано — по умолчанию карту нужно рисовать) — ответь
СТРОГО одним словом: ДА"""


def should_draw_chart(user_text: str) -> bool:
    """True (draw) unless the user's own message explicitly declined an
    image — see this section's own comment for why True is the default
    here even with no model loaded, unlike this app's usual LLM-first
    convention.

    Only the FIRST recognizable word of the answer is checked, not a
    substring search across the whole thing — a real, reported risk with a
    weaker model (e.g. a 3B instruct model) that doesn't reliably follow
    "answer with exactly one word": it can preface the real answer with a
    line of its own reasoning, and a plain "НЕТ" in answer.upper() check
    would misfire on an aside like "нет явного запрета, значит рисуем"
    (whose actual conclusion is to draw) purely because the word "нет"
    appears in it somewhere. Anchoring to the first word keeps this
    tolerant of that kind of preamble (same tolerance
    utils/tool_router.py's own _parse already has for its TOOL:/NONE
    line) without the false-negative risk a blind substring check has."""
    if llm_utils.get_llm() is None:
        return True
    try:
        answer = llm_utils.classify_sync(
            _SHOULD_DRAW_PROMPT.format(text=user_text), max_tokens=10, temperature=0.0,
        )
    except Exception:
        return True
    for line in answer.strip().splitlines():
        words = line.strip().split()
        if not words:
            continue
        token = words[0].strip("*.,:;!?\"'()").upper()
        if not token:
            continue
        if token.startswith("НЕТ"):
            return False
        if token.startswith("ДА"):
            return True
        # First real word was neither — keep scanning subsequent lines in
        # case this one was blank punctuation/emphasis noise, same as
        # tool_router._parse does for its own TOOL:/NONE line.
    return True

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _theta_for_degree(abs_deg: float, asc_deg: float) -> float:
    """Converts an absolute ecliptic degree (0-360, kerykeion's .abs_pos)
    into a screen angle in the "math" convention this module's _polar()
    expects (0 deg = screen-right/3 o'clock, increasing COUNTERCLOCKWISE
    on screen). Derived from the four angular houses' known screen
    positions (Asc=left=180 deg, IC=bottom=270 deg, Dsc=right=0/360 deg,
    MC=top=90 deg) and the fact that house numbers increase
    counterclockwise from the Ascendant (I at Asc, IV at IC, VII at Dsc,
    X at MC, confirmed against the reference screenshots) while a
    chart's house cusps' own absolute degrees increase in that same
    direction (houses are cut out of the ecliptic starting at the
    Ascendant, advancing in the direction of increasing ecliptic
    longitude) — so screen theta is just Asc's own fixed screen angle
    (180) plus however far past the Ascendant's degree this point sits."""
    return (180.0 + (abs_deg - asc_deg)) % 360.0


def _polar(cx: float, cy: float, r: float, theta_deg: float) -> Tuple[float, float]:
    rad = math.radians(theta_deg)
    return cx + r * math.cos(rad), cy - r * math.sin(rad)


def _arc_path(cx: float, cy: float, r: float, theta_start: float, theta_end: float) -> str:
    """SVG arc path (outer boundary only, not a filled wedge) from
    theta_start to theta_end going counterclockwise (increasing theta) —
    used for wedge boundaries and connecting arcs."""
    span = (theta_end - theta_start) % 360.0
    if span == 0:
        span = 360.0
    large_arc = 1 if span > 180.0 else 0
    x1, y1 = _polar(cx, cy, r, theta_start)
    x2, y2 = _polar(cx, cy, r, theta_end)
    # sweep=0 because increasing theta is counterclockwise on screen, and
    # SVG's sweep-flag=0 draws counterclockwise given our y-flipped _polar.
    return f"M {x1:.2f},{y1:.2f} A {r:.2f},{r:.2f} 0 {large_arc} 0 {x2:.2f},{y2:.2f}"


def _wedge_path(cx: float, cy: float, r_inner: float, r_outer: float, theta_start: float, theta_end: float) -> str:
    """Filled annulus segment (used for the 12 zodiac-sign wedges)."""
    span = (theta_end - theta_start) % 360.0
    if span == 0:
        span = 360.0
    large_arc = 1 if span > 180.0 else 0
    xo1, yo1 = _polar(cx, cy, r_outer, theta_start)
    xo2, yo2 = _polar(cx, cy, r_outer, theta_end)
    xi1, yi1 = _polar(cx, cy, r_inner, theta_start)
    xi2, yi2 = _polar(cx, cy, r_inner, theta_end)
    return (
        f"M {xo1:.2f},{yo1:.2f} "
        f"A {r_outer:.2f},{r_outer:.2f} 0 {large_arc} 0 {xo2:.2f},{yo2:.2f} "
        f"L {xi2:.2f},{yi2:.2f} "
        f"A {r_inner:.2f},{r_inner:.2f} 0 {large_arc} 1 {xi1:.2f},{yi1:.2f} Z"
    )


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Palettes — none of this existed before (the app's output was text-only);
# authored fresh rather than trying to exactly match any one reference
# program's idiosyncratic palette. Pastel, one shade per element family
# (fire/earth/air/water), smoothly distinguishable side by side.
# ---------------------------------------------------------------------------

_SIGN_FILL: Dict[str, str] = {
    "Ari": "#FBD9C4", "Leo": "#FCE3AE", "Sag": "#F8CDA8",  # fire — warm peach/gold/tan
    "Tau": "#DCEFC7", "Vir": "#C7E6B0", "Cap": "#B7D9AE",  # earth — light to deeper green
    "Gem": "#FBF4B8", "Lib": "#D6F0EE", "Aqu": "#D7E7F8",  # air — pale yellow/cyan/blue
    "Can": "#CFE3F7", "Sco": "#CAD6F0", "Pis": "#E3D6F5",  # water — blue through lavender
}

# Aspect colors, per the user's explicit brief: trine/sextile = green,
# square = red, conjunction/opposition = blue; everything else that
# already lives in astro._MINOR_ASPECTS (semisextile, semisquare,
# quintile, sesquiquadrate, biquintile, quincunx) = purple/lilac
# regardless of harmonious/tense nature, using the SHADE (light lilac vs
# darker violet) to still carry that distinction via astro._ASPECT_NATURE
# rather than inventing a fourth color family the user didn't ask for.
_MAJOR_ASPECT_COLOR: Dict[str, str] = {
    "conjunction": "#2255CC", "opposition": "#2255CC",
    "trine": "#1E9E4A", "sextile": "#3AB56A",
    "square": "#D1291F",
}
_MINOR_HARMONIOUS_COLOR = "#C9A0E0"   # light lilac
_MINOR_OTHER_COLOR = "#8E44AD"        # darker violet

# Fallback short label for the two points astro._POINT_SYMBOLS has no
# Unicode glyph for (see that dict's own comment: no invented symbols) —
# the full Russian word is far too wide to place on a crowded ring, so a
# short abbreviation is used on the WHEEL specifically (text reports
# elsewhere in the app still use the full word).
_POINT_ABBREV = {"Парс Фортуны": "ПФ", "Вертекс": "Вх"}

# Aspect LINES on the wheel are restricted to the 10 classical planets
# plus the Ascendant/MC — narrower than astro._ASPECT_ACTIVE_POINTS (which
# also includes the lunar node, Chiron, Lilith, and Pars Fortunae, used
# for this app's own TEXT reports). Real testing of this renderer showed
# the full set produces an unreadable web of lines on a wheel this size
# (15 active points is 105 possible pairs before any orb filtering even
# applies) — mainstream wheel-chart software conventionally limits the
# WHEEL's own aspect grid to the 10 planets (+ angles) for exactly this
# readability reason, even when a fuller point set is used elsewhere
# (text delineation, a separate aspect table, etc.). The extra points
# (node/Chiron/Lilith/Pars Fortunae) are still PLACED on the ring as
# glyphs — only their aspect LINES are left out of the wheel.
_WHEEL_ASPECT_POINTS = [
    "Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn",
    "Uranus", "Neptune", "Pluto", "Ascendant", "Medium_Coeli",
]

_ANGLE_AXIS_COLOR = {"asc_dsc": "#D1291F", "mc_ic": "#2255CC"}
_HOUSE_LINE_COLOR = "#7a7a7a"
_ZODIAC_RING_STROKE = "#333333"
_PLANET_TEXT_COLOR = "#1a1a2e"
_RETROGRADE_COLOR = "#D1291F"
_SECOND_CHART_TEXT_COLOR = "#7a2f8a"
# Light-blue background for the dedicated house-number band, per the
# reference screenshots — sits just inside the zodiac sign band, holding
# the house cusp ticks and Roman-numeral labels so they never have to run
# through (and visually collide with) the aspect web deeper in the wheel.
_HOUSEBAND_FILL = "#EAF4FC"


def _aspect_style(aspect_name: str, orbit: float, movement: str, max_orb: float) -> Tuple[str, float, Optional[str]]:
    """Returns (stroke_color, stroke_width, dasharray) for one aspect.
    Width is linearly interpolated between a thick line at an exact (0
    deg orb) aspect and a thin one at max_orb — "how close to exact" the
    way the user asked for, not a fixed width per aspect type. max_orb is
    now passed in by the caller rather than looked up here, since the
    correct value depends on which orb scheme this chart uses — see
    _orb_limit_fn below (astro.natal_orb_limit/transit_orb_limit/
    synastry_orb_limit's per-body moiety orb for most techniques, or
    astro._classical_orb_limit's luminary-aware orb for a horary/
    electional chart — all of them depend on which two bodies are
    actually involved, not just the aspect name). Applying aspects (still
    forming, per astro._ASPECT_MOVEMENT_RU's own convention) are solid;
    separating ones are dashed."""
    tightness = 1.0 - min(abs(orbit), max_orb) / max_orb if max_orb else 1.0
    width = 1.0 + 3.0 * max(0.0, min(1.0, tightness))
    dasharray = None if movement == "Applying" else "6,5"

    if aspect_name in _MAJOR_ASPECT_COLOR:
        color = _MAJOR_ASPECT_COLOR[aspect_name]
    else:
        nature = astro._ASPECT_NATURE.get(aspect_name, "")
        color = _MINOR_HARMONIOUS_COLOR if "гармонич" in nature else _MINOR_OTHER_COLOR
    return color, width, dasharray


def _orb_limit_fn(classical_aspects: bool, is_cross: bool, dual_orb_profile: str) -> Callable[[str, str, str], float]:
    """Picks which of astro.py's per-technique orb functions applies to one
    aspect-computation block (inner-chart or cross-chart) of this wheel.

    classical_aspects=True always wins (horary/electional's own scheme,
    untouched by this per-technique work — see astro._classical_orb_limit).

    Otherwise: inner-chart aspects (is_cross=False) are a single chart's
    own INTERNAL aspects — always astro.natal_orb_limit, regardless of
    which technique produced that chart (a real natal chart, a progressed
    chart read alone, a return chart read alone, ...): the "natal" profile
    models one chart looked at by itself, not the natal technique
    specifically. Cross-chart aspects (is_cross=True, the outer ring
    against the inner one) use astro.synastry_orb_limit or
    astro.transit_orb_limit depending on dual_orb_profile — see
    draw_wheel_svg's own docstring for which techniques pass which."""
    if classical_aspects:
        return astro._classical_orb_limit
    if is_cross:
        return astro.synastry_orb_limit if dual_orb_profile == "synastry" else astro.transit_orb_limit
    return astro.natal_orb_limit


# ---------------------------------------------------------------------------
# Point extraction — accepts either a real kerykeion subject or an
# already-prepared list of overlay points (utils/astro.py's direction
# technique has no second real subject, only a dict of SimpleNamespace
# points — see routes/chat.py's CHART_DRAWABLE for that case).
# ---------------------------------------------------------------------------

PointTuple = Tuple[str, float, bool]  # (label_ru, abs_pos, retrograde)


def _points_from_subject(subject: Any) -> List[PointTuple]:
    """Only astro._PLANET_ATTRS (Sun through Vertex) — deliberately NOT
    astro._ANGLE_ATTRS (Ascendant/MC): those two are already drawn as the
    labeled axis lines every wheel gets (see draw_wheel_svg's own house-
    cusp-line loop), so placing them AGAIN as ring glyphs would just be a
    redundant, overlapping "Асцендент"/"Середина неба (MC)" label sitting
    right on top of the "Asc"/"MC" axis label already there."""
    points: List[PointTuple] = []
    for label, attr in astro._PLANET_ATTRS:
        point = getattr(subject, attr, None)
        if point is None:
            continue
        points.append((label, point.abs_pos, bool(getattr(point, "retrograde", False))))
    return points


def _wheel_glyph(label: str) -> str:
    """Short glyph/abbreviation for a ring label — never the full Russian
    word (see _POINT_ABBREV's own comment: there's no room for it on a
    crowded ring, unlike this app's text reports)."""
    symbol = astro._POINT_SYMBOLS.get(label)
    if symbol:
        return symbol
    return _POINT_ABBREV.get(label, label[:2])


def _normalize_second(second: Union[None, Any, Sequence[Dict[str, Any]]]) -> Optional[List[PointTuple]]:
    if second is None:
        return None
    if isinstance(second, (list, tuple)):
        return [(d["label"], float(d["abs_pos"]), bool(d.get("retrograde", False))) for d in second]
    return _points_from_subject(second)


# ---------------------------------------------------------------------------
# Collision-avoiding radial placement — planets close together in degree
# get nudged into different concentric "lanes" within the planet-ring
# annulus rather than overlapping; a thin tick line from each planet's
# TRUE degree (at the ring's outer edge) to its nudged glyph position
# keeps the exact position visible regardless of the nudge (a standard
# convention in wheel-chart software).
# ---------------------------------------------------------------------------


def _place_in_lanes(points: List[PointTuple], asc_deg: float, min_angular_gap: float = 6.0, lanes: int = 3) -> List[Tuple[PointTuple, int]]:
    ordered = sorted(points, key=lambda p: _theta_for_degree(p[1], asc_deg))
    lane_last_theta: List[Optional[float]] = [None] * lanes
    placed: List[Tuple[PointTuple, int]] = []
    for point in ordered:
        theta = _theta_for_degree(point[1], asc_deg)
        chosen_lane = 0
        for lane_idx in range(lanes):
            last = lane_last_theta[lane_idx]
            if last is None or min((theta - last) % 360.0, (last - theta) % 360.0) >= min_angular_gap:
                chosen_lane = lane_idx
                break
        else:
            chosen_lane = lanes - 1
        lane_last_theta[chosen_lane] = theta
        placed.append((point, chosen_lane))
    return placed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _glyph_span(glyph: str, retro: bool, font_size: float) -> str:
    """Text content for one point's glyph, with retrograde marked as a
    small "R" tag riding below the baseline (SVG's baseline-shift="sub"
    equivalent of HTML <sub>) — per explicit request, a plain letter
    rather than the traditional "℞" character, which was hard to read at
    this size next to the planet glyph itself."""
    if not retro:
        return _esc(glyph)
    sub_size = max(8, round(font_size * 0.6))
    return f'{_esc(glyph)}<tspan baseline-shift="sub" font-size="{sub_size}">R</tspan>'


def _arrow_marker(cx: float, cy: float, tip_r: float, theta_deg: float, color: str, size: float = 12.0, half_width: float = 6.0) -> str:
    """Small filled triangle pointing outward along theta_deg, tip at
    tip_r — marks the Ascendant end of the Asc-Dsc axis line."""
    tip = _polar(cx, cy, tip_r, theta_deg)
    base_x, base_y = _polar(cx, cy, tip_r - size, theta_deg)
    p1 = _polar(base_x, base_y, half_width, theta_deg + 90.0)
    p2 = _polar(base_x, base_y, half_width, theta_deg - 90.0)
    return (
        f'<polygon points="{tip[0]:.1f},{tip[1]:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" '
        f'fill="{color}"/>'
    )


def _circle_marker(cx: float, cy: float, r: float, theta_deg: float, color: str, radius: float = 5.0) -> str:
    """Small filled circle marking the Midheaven end of the MC-IC axis line."""
    x, y = _polar(cx, cy, r, theta_deg)
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}"/>'


def draw_wheel_svg(
    subject: Any,
    second: Union[None, Any, Sequence[Dict[str, Any]]] = None,
    title_lines: Optional[List[str]] = None,
    highlight_house: Optional[int] = None,
    second_label: Optional[str] = None,
    classical_aspects: bool = False,
    dual_orb_profile: str = "transit",
) -> bytes:
    """Renders one astrological wheel as a standalone SVG document
    (UTF-8 bytes, ready to hand to db.repository.add_file with
    mime_type="image/svg+xml").

    subject: the reference kerykeion subject (AstrologicalSubjectFactory
    result via astro._build_subject) — its OWN houses/signs are what the
    whole wheel (including the outer ring, if any) is read against, per
    this app's own already-established dual-chart convention (see
    astro.get_dual_chart_profiles' docstring).

    second: optional. Either a second real kerykeion subject (transit
    moment, synastry partner, progressed/return chart — its own planets
    are drawn in an outer ring and cross-chart aspects are computed via
    AspectsFactory.dual_chart_aspects), or a plain list of
    {"label": str, "abs_pos": float, "retrograde": bool} dicts for a
    technique with no real second subject (astro.py's solar-arc
    directions only ever produces SimpleNamespace overlay points, not a
    full chart) — in that case only the outer-ring points are drawn, with
    no cross-chart aspect lines (there's no real second chart to compute
    them against).

    title_lines: plain text lines for the top-left header block (name,
    date/time/place, house system, ...) — the caller already has this
    formatted for its own text report in most cases; falls back to a
    generic subject.name/date line if not given.

    highlight_house: optional house number (1-12) to shade distinctly
    (used by run_profection for the "activated" house this year).

    classical_aspects: False (default) draws aspects with this app's
    per-technique orb system (see dual_orb_profile below and
    astro.natal_orb_limit/transit_orb_limit/synastry_orb_limit) — correct
    for every technique except the two below, per
    interpretation_methodology.txt (no separate numeric orb scheme is
    prescribed there; whatever orb the code already computes is the
    intended one). Pass True only for a horary or electional chart
    (routes/chat.py sets this from tool_name): those two techniques' own
    methodology docs specify a genuinely different scheme — only six
    classical aspect types (not the five extra minors), with an orb that
    depends on which bodies are involved (8-10° for Sun/Moon, 6-7° for
    other planets, flat 5° for quincunx) rather than a per-body moiety
    table. Before this flag existed, every chart — including
    horary/electional — was always drawn with the general table, so a
    horary/electional wheel could show aspect lines (e.g. a quintile) that
    don't exist under that technique's own doctrine at all, and even its
    classical aspects could appear with the wrong orb cutoff. See
    astro._CLASSICAL_ASPECTS_WIDE/filter_classical_aspects for the shared
    implementation (also used by utils/horary.py and utils/electional.py
    for their own verdict computation, so the picture now matches the
    reasoning).

    dual_orb_profile: only matters when classical_aspects=False AND there
    is a real second subject (routes/chat.py sets this from tool_name).
    "transit" (default) is for one real chart plus a technique-derived
    moment overlaid on it — transit, progression, lunar/solar return
    (direction never reaches this code path at all: its second is a list
    of synthetic overlay points, not a real subject, so is_real_second_
    subject is already False and no cross-chart aspects are computed).
    "synastry" is for astro_synastry_chart specifically — two real
    people's charts conventionally get a different, wider orb than a
    single derived moment. Either way, the INNER chart's own aspects
    (this subject's own planets against each other) always use the natal
    profile (astro.natal_orb_limit) regardless of dual_orb_profile — a
    chart's own internal aspects don't change meaning just because it's
    being compared to something else. See utils/astro.py's own comment
    above _NATAL_ORB_BY_BODY/_TRANSIT_ORB_BY_BODY/_SYNASTRY_ORB_BY_BODY
    for where these numbers come from (the user's reference astrology
    software) and why they're structured as per-body moiety tables."""
    orb_limit_inner = _orb_limit_fn(classical_aspects, is_cross=False, dual_orb_profile=dual_orb_profile)
    orb_limit_cross = _orb_limit_fn(classical_aspects, is_cross=True, dual_orb_profile=dual_orb_profile)
    dual = second is not None
    second_points = _normalize_second(second)
    is_real_second_subject = dual and not isinstance(second, (list, tuple))

    # Layer budget, outer to inner: (dual only) second-chart ring -> zodiac
    # sign band -> house band (own light-blue background, per the reference
    # screenshots — replaces the old approach of running house-cusp lines
    # all the way from the edge to the center, which visually collided with
    # the aspect web) -> planet band -> open center. `margin` reserves room
    # outside the outermost ring for the two angle-axis lines (Asc-Dsc,
    # MC-IC), which poke out past every ring and end in a marker (arrow at
    # Asc, circle at MC) plus their text labels — see the axis-line section
    # below.
    size = 1080 if dual else 900
    cx = cy = size / 2.0
    margin = 75.0 if dual else 62.0
    r_outer2 = size / 2.0 - margin if dual else None
    r_second_inner = (r_outer2 - 38.0) if dual else None
    r_signband_outer = (r_second_inner - 12.0) if dual else (size / 2.0 - margin)
    r_signband_inner = r_signband_outer - (40.0 if dual else 42.0)
    r_houseband_outer = r_signband_inner
    r_houseband_inner = r_houseband_outer - (30.0 if dual else 32.0)
    r_planet_outer = r_houseband_inner - 8.0
    r_planet_inner = r_planet_outer - (78.0 if dual else 85.0)
    r_center_circle = 38.0
    r_degree_dot = r_signband_inner  # true (un-nudged) position marker, right on the zodiac ring's inner edge

    outer_for_lines = r_outer2 if dual else r_signband_outer
    r_minor_tick_outer = outer_for_lines + 8.0
    r_major_axis_outer = outer_for_lines + 20.0
    r_axis_label = r_major_axis_outer + 12.0

    asc_deg = subject.ascendant.abs_pos
    cusp_degrees = astro._house_cusp_degrees(subject)

    svg: List[str] = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="Verdana, Arial, sans-serif">'
    )
    svg.append(f'<rect x="0" y="0" width="{size}" height="{size}" fill="#ffffff"/>')

    # --- 12 zodiac-sign wedges (always exactly 30 deg each, fixed to the
    # tropical zodiac, independent of the chart's own unequal houses) ---
    for i, sign_code in enumerate(astro._ZODIAC_SIGN_CODES):
        theta_start = _theta_for_degree(i * 30.0, asc_deg)
        theta_end = _theta_for_degree((i + 1) * 30.0, asc_deg)
        svg.append(
            f'<path d="{_wedge_path(cx, cy, r_signband_inner, r_signband_outer, theta_start, theta_end)}" '
            f'fill="{_SIGN_FILL[sign_code]}" stroke="{_ZODIAC_RING_STROKE}" stroke-width="0.75"/>'
        )
        mid_theta = _theta_for_degree(i * 30.0 + 15.0, asc_deg)
        gx, gy = _polar(cx, cy, (r_signband_inner + r_signband_outer) / 2.0, mid_theta)
        svg.append(
            f'<text x="{gx:.1f}" y="{gy:.1f}" font-size="20" text-anchor="middle" '
            f'dominant-baseline="middle" fill="{_ZODIAC_RING_STROKE}">{astro._SIGN_SYMBOLS[astro._SIGN_NAMES_RU[sign_code]]}</text>'
        )

    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r_signband_outer:.1f}" fill="none" stroke="{_ZODIAC_RING_STROKE}" stroke-width="1.5"/>')

    # --- house band: its own light-blue ring just inside the zodiac band,
    # per the reference screenshots (replaces house-cusp lines that used to
    # run all the way to the center and cross the aspect web) ---
    svg.append(
        f'<circle cx="{cx}" cy="{cy}" r="{(r_houseband_outer + r_houseband_inner) / 2.0:.1f}" '
        f'fill="none" stroke="{_HOUSEBAND_FILL}" stroke-width="{r_houseband_outer - r_houseband_inner:.1f}"/>'
    )
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r_houseband_outer:.1f}" fill="none" stroke="{_ZODIAC_RING_STROKE}" stroke-width="1"/>')
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r_houseband_inner:.1f}" fill="none" stroke="{_ZODIAC_RING_STROKE}" stroke-width="1"/>')

    # --- optional highlighted house wedge (profection's "activated" house) ---
    if highlight_house and 1 <= highlight_house <= 12:
        h0 = cusp_degrees[highlight_house - 1]
        h1 = cusp_degrees[highlight_house % 12]
        svg.append(
            f'<path d="{_wedge_path(cx, cy, r_center_circle, r_signband_inner, _theta_for_degree(h0, asc_deg), _theta_for_degree(h1, asc_deg))}" '
            f'fill="#FFF3B0" fill-opacity="0.55" stroke="none"/>'
        )

    # --- house cusp ticks (confined to the zodiac + house bands only —
    # deliberately NOT reaching the planet/aspect area, so they can never
    # visually collide with an aspect line) + Roman-numeral labels inside
    # the house band ---
    _ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
    r_house_label = (r_houseband_outer + r_houseband_inner) / 2.0
    for i in range(12):
        deg = cusp_degrees[i]
        theta = _theta_for_degree(deg, asc_deg)
        is_angle = i in (0, 3, 6, 9)
        if not is_angle:
            x1, y1 = _polar(cx, cy, r_houseband_inner, theta)
            x2, y2 = _polar(cx, cy, r_minor_tick_outer, theta)
            svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{_HOUSE_LINE_COLOR}" stroke-width="1"/>')

        next_deg = cusp_degrees[(i + 1) % 12]
        mid_theta = _theta_for_degree((deg + ((next_deg - deg) % 360.0) / 2.0), asc_deg)
        lx, ly = _polar(cx, cy, r_house_label, mid_theta)
        svg.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="14" text-anchor="middle" dominant-baseline="middle" fill="{_HOUSE_LINE_COLOR}">{_ROMAN[i]}</text>')

    # --- the two angle axes (Asc-Dsc, Mc-IC): drawn as single diameters
    # (the two cusps of each pair are exactly 180 deg apart by definition)
    # crossing the whole wheel and poking out past every ring, ending in an
    # arrow at the Ascendant and a small circle at the Midheaven — per
    # explicit request, so the four cardinal points stay identifiable even
    # where they'd otherwise be lost among aspect lines in the center ---
    theta_asc = _theta_for_degree(cusp_degrees[0], asc_deg)
    theta_mc = _theta_for_degree(cusp_degrees[9], asc_deg)
    for theta_a, theta_b, color, marker_at_a in (
        (theta_asc, theta_asc + 180.0, _ANGLE_AXIS_COLOR["asc_dsc"], "arrow"),
        (theta_mc, theta_mc + 180.0, _ANGLE_AXIS_COLOR["mc_ic"], "circle"),
    ):
        xa, ya = _polar(cx, cy, r_major_axis_outer, theta_a)
        xb, yb = _polar(cx, cy, r_major_axis_outer, theta_b)
        svg.append(f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" stroke="{color}" stroke-width="2"/>')
        if marker_at_a == "arrow":
            svg.append(_arrow_marker(cx, cy, r_major_axis_outer, theta_a, color))
        else:
            svg.append(_circle_marker(cx, cy, r_major_axis_outer, theta_a, color))

    for label, theta_deg, dx, dy in (
        ("Asc", theta_asc, -13, 0),
        ("Dsc", theta_asc + 180.0, 13, 0),
        ("MC", theta_mc, 0, -12),
        ("IC", theta_mc + 180.0, 0, 16),
    ):
        x, y = _polar(cx, cy, r_axis_label, theta_deg)
        svg.append(f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" font-size="14" font-weight="bold" text-anchor="middle" fill="{_HOUSE_LINE_COLOR}">{label}</text>')

    # White-filled center circle drawn LAST among the structural elements so
    # it cleanly caps the point where the two axis lines cross, instead of
    # showing a busy X through an outlined hole.
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r_center_circle}" fill="#ffffff" stroke="{_ZODIAC_RING_STROKE}" stroke-width="1"/>')

    # --- inner-chart planets, collision-avoided across concentric lanes.
    # Each point also gets a small filled dot at its TRUE (un-nudged)
    # degree, right on the zodiac ring's inner edge, with a thin tick line
    # to its (possibly nudged) glyph — so an unaspected planet is still
    # unambiguously locatable even when its glyph had to be shifted to
    # avoid overlapping a neighbor. Aspect lines are anchored to these dots
    # (not to the nudged glyphs, and not to a separate floating inner
    # circle) — a real, reported issue with the earlier version was aspect
    # lines converging on an inner ring with nothing marking it, which
    # looked like they were "hanging" in empty space. ---
    inner_points = _points_from_subject(subject)
    placed_inner = _place_in_lanes(inner_points, asc_deg)
    lane_count = 3
    lane_step = (r_planet_outer - r_planet_inner) / max(1, lane_count - 1)
    true_pos_by_label: Dict[str, Tuple[float, float]] = {}

    for (label, abs_deg, retro), lane in placed_inner:
        theta = _theta_for_degree(abs_deg, asc_deg)
        r_glyph = r_planet_outer - lane * lane_step
        dot_x, dot_y = _polar(cx, cy, r_degree_dot, theta)
        gx, gy = _polar(cx, cy, r_glyph, theta)
        svg.append(f'<circle cx="{dot_x:.1f}" cy="{dot_y:.1f}" r="2.5" fill="{_PLANET_TEXT_COLOR}"/>')
        svg.append(f'<line x1="{dot_x:.1f}" y1="{dot_y:.1f}" x2="{gx:.1f}" y2="{gy:.1f}" stroke="#bbbbbb" stroke-width="0.75"/>')
        glyph = _wheel_glyph(label)
        svg.append(f'<text x="{gx:.1f}" y="{gy:.1f}" font-size="18" text-anchor="middle" dominant-baseline="middle" fill="{_PLANET_TEXT_COLOR}">{_glyph_span(glyph, retro, 18)}</text>')
        true_pos_by_label[label] = (dot_x, dot_y)

    # Angles (Ascendant/MC) also get a true-position entry for aspect-line
    # purposes even though they're not drawn as ring glyphs (see
    # _points_from_subject) — a conjunction to the Ascendant is genuinely
    # significant and shouldn't be silently dropped from the aspect web.
    for label, attr in astro._ANGLE_ATTRS:
        point = getattr(subject, attr, None)
        if point is None:
            continue
        theta = _theta_for_degree(point.abs_pos, asc_deg)
        dot_x, dot_y = _polar(cx, cy, r_degree_dot, theta)
        true_pos_by_label[label] = (dot_x, dot_y)

    # --- aspects within the inner chart (restricted point set — see
    # _WHEEL_ASPECT_POINTS' own comment on why) ---
    from kerykeion import AspectsFactory

    aspect_table = astro._CLASSICAL_ASPECTS_WIDE if classical_aspects else astro._PER_TECHNIQUE_ASPECTS_WIDE
    inner_aspects = AspectsFactory.natal_aspects(
        subject, active_points=_WHEEL_ASPECT_POINTS, active_aspects=aspect_table,
    ).aspects
    if classical_aspects:
        inner_aspects = astro.filter_classical_aspects(inner_aspects)
    kery_to_label = {astro.attr_to_kerykeion_name(attr): label for label, attr in astro._PLANET_ATTRS + astro._ANGLE_ATTRS}
    for a in inner_aspects:
        l1, l2 = kery_to_label.get(a.p1_name), kery_to_label.get(a.p2_name)
        if not l1 or not l2 or l1 not in true_pos_by_label or l2 not in true_pos_by_label:
            continue
        max_orb = orb_limit_inner(a.aspect, a.p1_name, a.p2_name)
        if a.orbit > max_orb:
            continue
        color, width, dasharray = _aspect_style(a.aspect, a.orbit, a.aspect_movement, max_orb)
        x1, y1 = true_pos_by_label[l1]
        x2, y2 = true_pos_by_label[l2]
        dash_attr = f' stroke-dasharray="{dasharray}"' if dasharray else ""
        svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width:.2f}"{dash_attr} opacity="0.85"/>')

    # --- outer ring: second chart's own planets + cross-chart aspects ---
    if dual and second_points:
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r_outer2:.1f}" fill="none" stroke="{_SECOND_CHART_TEXT_COLOR}" stroke-width="1.25"/>')
        second_placed = _place_in_lanes(second_points, asc_deg, min_angular_gap=6.0, lanes=2)
        second_lane_step = (r_outer2 - 8 - r_second_inner) / 1.0
        for (label, abs_deg, retro), lane in second_placed:
            theta = _theta_for_degree(abs_deg, asc_deg)
            dot_x, dot_y = _polar(cx, cy, r_signband_outer, theta)
            r_glyph = r_second_inner + lane * second_lane_step + 8
            gx, gy = _polar(cx, cy, r_glyph, theta)
            svg.append(f'<circle cx="{dot_x:.1f}" cy="{dot_y:.1f}" r="2.5" fill="{_SECOND_CHART_TEXT_COLOR}"/>')
            svg.append(f'<line x1="{dot_x:.1f}" y1="{dot_y:.1f}" x2="{gx:.1f}" y2="{gy:.1f}" stroke="#d8b8e6" stroke-width="0.75"/>')
            glyph = _wheel_glyph(label)
            svg.append(f'<text x="{gx:.1f}" y="{gy:.1f}" font-size="17" text-anchor="middle" dominant-baseline="middle" fill="{_SECOND_CHART_TEXT_COLOR}">{_glyph_span(glyph, retro, 17)}</text>')

        if is_real_second_subject:
            cross_aspects = AspectsFactory.dual_chart_aspects(
                subject, second, active_points=_WHEEL_ASPECT_POINTS, active_aspects=aspect_table,
            ).aspects
            if classical_aspects:
                cross_aspects = astro.filter_classical_aspects(cross_aspects)
            for a in cross_aspects:
                inner_name = a.p1_name if a.p1_owner == subject.name else a.p2_name
                outer_name = a.p2_name if a.p1_owner == subject.name else a.p1_name
                l1 = kery_to_label.get(inner_name)
                l2 = kery_to_label.get(outer_name)
                if not l1 or not l2 or l1 not in true_pos_by_label:
                    continue
                outer_point = next((p for p in second_points if p[0] == l2), None)
                if outer_point is None:
                    continue
                theta2 = _theta_for_degree(outer_point[1], asc_deg)
                x2, y2 = _polar(cx, cy, r_signband_outer, theta2)
                max_orb = orb_limit_cross(a.aspect, a.p1_name, a.p2_name)
                if a.orbit > max_orb:
                    continue
                color, width, dasharray = _aspect_style(a.aspect, a.orbit, a.aspect_movement, max_orb)
                x1, y1 = true_pos_by_label[l1]
                dash_attr = f' stroke-dasharray="{dasharray}"' if dasharray else ""
                svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width:.2f}"{dash_attr} opacity="0.7"/>')

    # --- header text block ---
    lines = title_lines if title_lines else [getattr(subject, "name", "chart")]
    if second_label:
        lines = list(lines) + [second_label]
    y0 = 22
    for i, line in enumerate(lines[:6]):
        svg.append(f'<text x="14" y="{y0 + i * 17}" font-size="13" fill="#222222">{_esc(line)}</text>')

    svg.append("</svg>")
    return "".join(svg).encode("utf-8")


def unique_chart_filename(prefix: str = "chart") -> str:
    """Unix-time-based unique filename, per explicit instruction — using
    milliseconds (not whole seconds) so two charts requested in the same
    request/response cycle (e.g. a technique that itself compares two
    moments) can never collide."""
    return f"{prefix}_{int(time.time() * 1000)}.svg"
