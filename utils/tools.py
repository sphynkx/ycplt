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
}
