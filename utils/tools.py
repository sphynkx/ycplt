"""Registry of tools the router (utils/tool_router.py) can dispatch a chat
message to when a plain text answer from the model isn't enough — e.g. the
model has no notion of "now" and can't reliably do arithmetic on its own.

Each tool is a plain Python function taking a single string argument (the
empty string if the tool needs none) and returning a plain string result.
That result is fed back to the model as context for a short follow-up
generation, which produces the actual answer shown to the user (see
routes/chat.py:_handle_tool_request) — the tool itself never talks to the
user directly.

Adding a new tool is meant to be a one-function, one-line change: write the
function below, register it in TOOL_REGISTRY with a clear one-line
description (the router builds its classifier prompt straight from these
descriptions, so a vague description leads to vague routing), and nothing
else in the app needs to change.
"""
import ast
import operator
from datetime import datetime
from typing import Callable, Dict, TypedDict

from utils import astro
from utils import electional
from utils import horary
from utils import rectification
from utils import rectification_events


def get_current_datetime(_arg: str = "") -> str:
    """Current date, time, and day of week. Local to wherever the server
    process runs — there's no other notion of "now" available to a model
    whose training data has a fixed cutoff."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S, %A")


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("only numeric literals are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"disallowed expression element: {type(node).__name__}")


def calculate(expr: str) -> str:
    """Safely evaluates a numeric arithmetic expression: + - * / // % ** and
    parentheses, nothing else. Deliberately not a plain eval() — the AST
    walk in _eval_node rejects names, calls, attribute access, subscripts,
    comprehensions, and anything else that isn't a number or an allowed
    operator, so a malicious or malformed expression can't do more than
    raise an error."""
    expr = expr.strip()
    if not expr:
        return "Error: no expression given"
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
    except Exception as e:
        return f"Error: could not evaluate '{expr}' ({e})"
    return str(result)


class ToolSpec(TypedDict):
    description: str
    run: Callable[[str], str]


TOOL_REGISTRY: Dict[str, ToolSpec] = {
    "get_current_datetime": {
        "description": "Current date, time, and day of week. Takes no argument.",
        "run": get_current_datetime,
    },
    "calculate": {
        "description": (
            "Evaluates an arithmetic expression, e.g. '12 * (3 + 4) / 2'. "
            "Argument: the expression as a plain string."
        ),
        "run": calculate,
    },
    "astro_natal_chart": {
        "description": (
            "Computes a natal (birth) astrological chart — planet signs/"
            "houses and aspects between them — from someone's exact birth "
            "data (see utils/astro.py for the full computation). Only use "
            "this if a birth date, time, and place or coordinates actually "
            "appear somewhere in this conversation (the current message or "
            "an earlier one) — never invent placeholder birth data; if it's "
            "missing, don't use this tool at all. Argument: simply copy the "
            "birth date, time, and place/coordinates as they were written — "
            "do NOT reformat or convert them yourself, the tool parses "
            "common formats automatically (dates like '5 июля 1976' or "
            "'1990-03-12'; times like '4:30'; coordinates as decimal or "
            "degree-minute-second, e.g. '46°28'00\"N;30°44'00\"E' — the "
            "timezone is resolved automatically from the coordinates, you "
            "never need to state or know it). If only a well-known city is "
            "named with no coordinates, you may add that city's "
            "approximate coordinates from your own knowledge instead of "
            "skipping the tool."
        ),
        "run": astro.run_natal,
    },
    "astro_transit_chart": {
        "description": (
            "Computes current (or a given moment's) planetary positions and "
            "how they aspect someone's natal chart — for questions about "
            "what's currently happening in someone's chart, or on a "
            "specific date. Requires the same birth data as "
            "astro_natal_chart, in the same free-text form (never invent "
            "it — skip this tool if it isn't in the conversation). Only add "
            "';moment=YYYY-MM-DDTHH:MM' if a specific non-current moment "
            "was asked about (default: right now)."
        ),
        "run": astro.run_transit,
    },
    "astro_synastry_chart": {
        "description": (
            "Compares the natal charts of TWO people — for questions about "
            "compatibility, relationship dynamics, or how two specific "
            "people relate astrologically. Only use this if birth date, "
            "time, and place/coordinates for BOTH people actually appear "
            "somewhere in this conversation — never invent placeholder "
            "birth data for either person; if either person's data is "
            "missing, don't use this tool at all (use astro_natal_chart for "
            "a single person instead). Argument: copy BOTH people's birth "
            "date, time, and place/coordinates as written, in the order "
            "they were mentioned, exactly like astro_natal_chart's argument "
            "but for two people back to back (e.g. 'Иван, 5 июля 1976 в "
            "4:30 в Одессе, и Мария, 12 марта 1980 в 9:15 в Киеве') — do "
            "NOT reformat, and do not merge or drop either person's data."
        ),
        "run": astro.run_synastry,
    },
    "astro_progression_chart": {
        "description": (
            "Computes SECONDARY PROGRESSIONS ('day for a year') for "
            "someone's chart — how their planets have symbolically evolved "
            "over their life so far, for questions about long-term "
            "personal development, life stages/chapters, or 'where they "
            "are now' in a slower, deeper sense than current transits "
            "(use astro_transit_chart instead for short-term current "
            "events/mood). Requires the same birth data as "
            "astro_natal_chart, in the same free-text form (never invent "
            "it — skip this tool if it isn't in the conversation). Only "
            "add ';moment=YYYY-MM-DDTHH:MM' if a specific non-current "
            "target date/age was asked about (default: right now)."
        ),
        "run": astro.run_progression,
    },
    "astro_direction_chart": {
        "description": (
            "Computes SOLAR ARC DIRECTIONS ('дирекция'/'directions') for "
            "someone's chart — EVERY natal point shifted by the SAME "
            "precise angular arc (unlike astro_progression_chart, where "
            "different points move at their own different speeds), a "
            "precise, calculable timing technique. Use ONLY when the user "
            "explicitly asks about 'дирекция'/'дирекции'/'directions' — "
            "NOT for general current-period questions (astro_transit_"
            "chart) or slow decades-scale development questions "
            "(astro_progression_chart); those are different techniques "
            "with different tools. Requires the same birth data as "
            "astro_natal_chart, in the same free-text form (never invent "
            "it — skip this tool if it isn't in the conversation). Only "
            "add ';moment=YYYY-MM-DDTHH:MM' if a specific non-current "
            "target date/age was asked about (default: right now)."
        ),
        "run": astro.run_direction,
    },
    "astro_lunar_return_chart": {
        "description": (
            "Computes the LUNAR RETURN ('лунар'/'lunar return') for "
            "someone's chart — an independent chart cast for the moment "
            "the Moon returns to its exact natal degree (roughly every "
            "27-29 days), plus how it aspects the natal chart; for "
            "questions about the current ~month. Use ONLY when the user "
            "explicitly asks about 'лунар'/'lunar return' — NOT for "
            "current transits (astro_transit_chart) or the annual solar "
            "return (astro_solar_return_chart). Uses ONLY the natal "
            "birth location for the return chart (no relocation support). "
            "Requires the same birth data as astro_natal_chart, in the "
            "same free-text form (never invent it — skip this tool if "
            "it isn't in the conversation). Only add "
            "';moment=YYYY-MM-DDTHH:MM' if a specific non-current moment "
            "was asked about (default: right now)."
        ),
        "run": astro.run_lunar_return,
    },
    "astro_solar_return_chart": {
        "description": (
            "Computes the SOLAR RETURN ('солар'/'solar return') for "
            "someone's chart — an independent chart cast for the moment "
            "the Sun returns to its exact natal degree (roughly annually, "
            "on or near the birthday), plus how it aspects the natal "
            "chart; for questions about the current year/'year ahead'. "
            "Use ONLY when the user explicitly asks about 'солар'/'solar "
            "return' — NOT for current transits (astro_transit_chart) or "
            "the monthly lunar return (astro_lunar_return_chart). Uses "
            "ONLY the natal birth location for the return chart (no "
            "relocation support). Requires the same birth data as "
            "astro_natal_chart, in the same free-text form (never invent "
            "it — skip this tool if it isn't in the conversation). Only "
            "add ';moment=YYYY-MM-DDTHH:MM' if a specific non-current "
            "moment/year was asked about (default: right now)."
        ),
        "run": astro.run_solar_return,
    },
    "astro_profection_chart": {
        "description": (
            "Computes this year's PROFECTION ('профекция'/'profection') "
            "for someone's chart — a classical whole-sign technique: which "
            "house/sign of the natal chart is 'activated' this year "
            "(counting one whole sign per year of life from the natal "
            "Ascendant), and that sign's classical ruling planet (the "
            "year's 'time lord'). Builds no new ephemeris chart — pure "
            "calendar+rulership arithmetic over the existing natal chart. "
            "Use ONLY when the user explicitly asks about 'профекция'/"
            "'profection' — NOT for current transits, progressions, or "
            "returns (different tools/techniques). Requires the same "
            "birth data as astro_natal_chart, in the same free-text form "
            "(never invent it — skip this tool if it isn't in the "
            "conversation). Only add ';moment=YYYY-MM-DDTHH:MM' if a "
            "specific non-current age/date was asked about (default: "
            "right now)."
        ),
        "run": astro.run_profection,
    },
    "astro_rectification_trutine": {
        "description": (
            "Attempts to RECTIFY (narrow down) an UNCERTAIN/APPROXIMATE "
            "birth time using the classical 'Trutine of Hermes' method — "
            "searches a window of candidate birth times for the one where "
            "the birth Ascendant and Moon best mirror the Moon and "
            "Ascendant at conception. Use ONLY when the user explicitly "
            "asks to rectify/determine an uncertain birth time, or "
            "mentions 'трутина'/'trutine of hermes'/'ректификация' — NOT "
            "for any question where the birth time is already known "
            "exactly (use astro_natal_chart or another specific tool "
            "instead). Needs an approximate birth DATE, a place "
            "(coordinates), and either an approximate time (a "
            "+/-1-hour search window is used around it) or explicit "
            "'time_min=HH:MM;time_max=HH:MM' bounds — never invent a "
            "made-up exact time or window; skip this tool if none of "
            "that is present in the conversation. Optional overrides: "
            "'gestation_days=' (default 273), 'step_minutes=' (default 1, "
            "coarser = faster on a very wide window)."
        ),
        "run": rectification.run_rectification_trutine,
    },
    "astro_rectification_events": {
        "description": (
            "Attempts to RECTIFY (narrow down) an UNCERTAIN/APPROXIMATE "
            "birth time by testing candidate times against SEVERAL KNOWN "
            "LIFE EVENT dates (marriage, birth of a child, death, career "
            "change, illness/surgery, move, etc.) — for each candidate "
            "time it builds profections/progressions/directions/transits "
            "for every event and scores how well they confirm it; the "
            "candidate with the most confirmations across ALL events wins. "
            "Use this (not astro_rectification_trutine) when the user "
            "gives concrete life-event dates to rectify by, or explicitly "
            "asks for event-based/multi-technique rectification. Needs an "
            "approximate birth DATE, a place (coordinates), either an "
            "approximate time (+/-1-hour window) or explicit "
            "'time_min=HH:MM;time_max=HH:MM' bounds, AND at least one life "
            "event — never invent any of this, skip the tool if it's "
            "missing. Argument format: first the birth data as free text "
            "(same as astro_natal_chart, do NOT label it with a colon, "
            "e.g. NOT 'Дата рождения: ...'), THEN each event on its own "
            "line, in EITHER of two formats: the short 'description: date' "
            "(e.g. 'брак: 21.01.1983'), OR the fuller semicolon-separated "
            "'description; date; [time]; [place]; [lat]; [lon]; "
            "[comment]' (extra fields after the date are optional and are "
            "simply ignored, e.g. 'Первая любовь; 1.11.1986; 12:00; "
            "Одесса; 46n28; 30e44; заметка'). Date as DD.MM.YYYY or "
            "YYYY-MM-DD. Copy every event line VERBATIM, one per line, "
            "however many there are (even dozens) — never merge, "
            "summarize, drop, or invent one. Optional overrides: "
            "'step_minutes=' (default 10), 'window_minutes=' (default 120)."
        ),
        "run": rectification_events.run_rectification_events,
    },
    "astro_horary_question": {
        "description": (
            "Answers a HORARY question (хорар/хорарный вопрос) — a classical "
            "yes/or/no astrological judgment cast for the exact moment and "
            "place the question itself was asked (NOT a birth chart). Use "
            "this for any horary question — e.g. 'will she call me back', "
            "'should I take this job', 'will we sell the house this year', "
            "'who took my things' — including a follow-up asking to explain "
            "a verdict already given earlier in this conversation (this "
            "tool always gives a full reasoned explanation, not just a bare "
            "Да/Нет, so re-running it also works for 'почему'/'объясни'-"
            "style follow-ups). Never invent the moment/place; if the exact "
            "date, time, and place the question is being asked (or was "
            "asked) aren't given, ask for them instead of using this tool "
            "with guessed data. Argument: the question's date, time, and "
            "place/coordinates in free text (same tolerant parsing as "
            "astro_natal_chart — copy as written, don't reformat), plus the "
            "question itself in the same text (used only to guess which "
            "house the question is about — love/marriage, career, money, "
            "health, a lost/stolen object, etc.). Optional 'house=N' (1-12) "
            "to state the relevant house explicitly instead of relying on "
            "that guess. If no explicit moment is given, assume right now."
        ),
        "run": horary.run_horary_question,
    },
    "astro_electional_chart": {
        "description": (
            "Электива/элективная астрология — TWO modes, auto-detected "
            "from the wording (never decide this yourself, the tool's own "
            "model does): (1) EVALUATE a single PROPOSED moment already "
            "chosen by the user for a stated PURPOSE — e.g. 'подходит ли "
            "6 августа в 15:00 для подписания договора' (structurally the "
            "reverse of astro_horary_question: that reads an already-"
            "fixed moment against a QUESTION, this reads a user-PROPOSED "
            "moment against a stated ACTIVITY — do NOT use this for a "
            "horary yes/no question about something that will happen on "
            "its own); (2) SEARCH for the best moment when no specific "
            "candidate is named — e.g. 'когда лучше подписывать договор', "
            "'на какой день лучше запланировать переезд' — this tool "
            "scans forward from the nearest named date (or right now) "
            "and returns the single best moment it found, not a bare "
            "yes/no on one guess. Never invent the moment/place; if no "
            "place is given at all, ask for it instead of using this tool "
            "with guessed data (a specific moment IS optional for the "
            "search mode — only place and purpose are required there). "
            "Argument: whatever date/time/place was named, as written "
            "(same tolerant parsing as astro_natal_chart — copy as "
            "written, don't reformat; may be absent for a pure search "
            "request), plus the stated purpose/activity (used to classify "
            "which classical category — marriage, contract, travel, "
            "surgery, household chores, etc. — applies), plus, for a "
            "search, any explicitly named search window/end date if one "
            "was given (e.g. 'в течение месяца', 'до конца сентября', 'в "
            "диапазоне года') — copy that phrasing as written too, don't "
            "convert it yourself."
        ),
        "run": electional.run_electional_chart,
    },
}
