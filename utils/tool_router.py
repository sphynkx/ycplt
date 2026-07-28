"""Decides whether a (non-image) user message needs one of the built-in
tools (utils/tools.py) before the chat model can answer it well — e.g. the
model has no notion of "now" and can't reliably do arithmetic on its own.

Uses the same lightweight zero-shot classifier pattern as utils/intent.py:
one extra small LLM call (temperature=0.0, a handful of tokens), defaulting
to "no tool" on any error or ambiguous output. A missed tool call just costs
a vaguer answer than ideal (recoverable by rephrasing); a wrongly-triggered
tool costs one extra fast classification call plus a short follow-up
generation, which is cheap compared to the risk of never using a tool that
was actually needed.

Adding a new tool doesn't require touching this file: the classifier prompt
is built from utils.tools.TOOL_REGISTRY's names and descriptions, so a new
entry there is picked up automatically.
"""
import asyncio
from dataclasses import dataclass
from typing import Optional

from utils import llm as llm_utils
from utils.tools import TOOL_REGISTRY


@dataclass
class ToolDecision:
    tool_name: Optional[str] = None
    tool_arg: str = ""


def _build_prompt(query: str) -> str:
    tool_lines = "\n".join(
        f'- {name}: {spec["description"]}' for name, spec in TOOL_REGISTRY.items()
    )
    return f"""You are a routing assistant for a chat application with access to these tools:
{tool_lines}

Decide whether answering the user's message requires one of these tools.
Reply with exactly one line, no other text:
  NONE                      — no tool needed, this can be answered directly
  TOOL:<name>               — for a tool that needs no argument
  TOOL:<name>|<argument>    — for a tool that needs an argument (e.g. the
                              expression to evaluate for "calculate")

User message: "{query}"
Answer:"""


def _parse(answer: str) -> ToolDecision:
    line = answer.strip().splitlines()[0].strip() if answer.strip() else ""
    if not line.upper().startswith("TOOL:"):
        return ToolDecision()
    body = line[len("TOOL:"):].strip()
    name, _, arg = body.partition("|")
    name = name.strip()
    if name not in TOOL_REGISTRY:
        return ToolDecision()
    return ToolDecision(tool_name=name, tool_arg=arg.strip())


def classify(query: str) -> ToolDecision:
    if llm_utils.get_llm() is None:
        return ToolDecision()
    try:
        answer = llm_utils.generate_sync(_build_prompt(query), max_tokens=20, temperature=0.0)
    except Exception:
        return ToolDecision()
    return _parse(answer)


async def classify_async(query: str) -> ToolDecision:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, classify, query)
