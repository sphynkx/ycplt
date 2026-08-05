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

history_context (optional, passed by routes/chat.py from recent messages
in the same conversation — BOTH roles, user and assistant, as a small
labelled transcript) exists for two related reasons: tools whose required
argument data — e.g. astro_natal_chart's birth data — was stated in an
earlier message, not necessarily the current one ("use the birth data I
gave you before"); and, just as importantly, recognizing a short follow-up
that continues something the ASSISTANT itself just said or suggested
("try a wider search window" -> user replies "давай пошире"). A real,
reported failure showed why the assistant's own side has to be included,
not just the user's: with only past USER messages visible, the classifier
had no way to know the assistant had just proposed widening a
rectification search window, so "давай окно пошире" looked like an
unrelated remark about resizing an application window and got classified
as needing no tool at all. This classifier only ever sees the CURRENT
message otherwise, so without this history block it has no way to notice
that earlier context is available at all. It's kept short and labelled
clearly as background, not as part of the current request, so the
classifier doesn't confuse "an earlier topic was mentioned" with "the
current message is asking about it."
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
    # The model's raw, unparsed answer — not used for any decision, kept
    # purely so callers can log it. A parsed-to-None decision is ambiguous
    # on its own: it means either "the model genuinely judged no tool is
    # needed" or "the model's answer didn't match the expected TOOL:/NONE
    # format and _parse gave up" — those are very different problems (one
    # is a classifier accuracy issue, the other is a prompt/parsing
    # brittleness issue against whatever model is currently loaded), and
    # without the raw text there was previously no way to tell them apart
    # from the server console after the fact.
    raw_answer: str = ""


# How much of the CURRENT message the classifier itself gets to see —
# separate from, and independent of, whatever routes/chat.py later builds
# as the actual tool_arg (which always uses the full, untruncated
# req.query — see _handle_tool_request). Added after a real, reported
# failure: a rectification request with 42 free-text life events (each a
# whole line of date/place/coordinates/comment) came back tool=None — the
# classifier judged "NONE" even though the very first line plainly said
# "ректификацию". This is the exact same failure class task #95's fix
# already addressed for HISTORY length (a small zero-shot classifier's
# judgment measurably degrades, not improves, with more raw text in its
# prompt) — that fix bounded history_context; this bounds the CURRENT
# query the same way, for the same reason. The intent-bearing part of a
# message is almost always near its start, so truncating from the end
# (keeping the head) preserves what the classifier actually needs.
_MAX_QUERY_CHARS = 1500


def _build_prompt(query: str, history_context: str = "") -> str:
    tool_lines = "\n".join(
        f'- {name}: {spec["description"]}' for name, spec in TOOL_REGISTRY.items()
    )
    display_query = query
    if len(query) > _MAX_QUERY_CHARS:
        display_query = (
            query[:_MAX_QUERY_CHARS]
            + f"\n...(сообщение обрезано для классификации, ещё {len(query) - _MAX_QUERY_CHARS} симв.)"
        )
    context_block = ""
    if history_context:
        context_block = (
            "\nEarlier in this same conversation (both the user's and your "
            "own — the assistant's — messages):\n"
            f'"""{history_context}"""\n'
            "(background only — use it to fill in a tool argument if the "
            "current message refers back to it, e.g. \"use what I told you "
            "before\"; ALSO use it to recognize a short follow-up that "
            "continues something the ASSISTANT itself just said or "
            "suggested, e.g. the assistant proposed trying a wider search "
            "window and the user now replies \"давай пошире\"/\"let's do "
            "that\" — route to the same tool that suggestion belongs to, "
            "don't treat an ambiguous short reply like that as a generic "
            "question with no tool needed; don't treat this block as the "
            "current request itself)\n"
        )
    return f"""You are a routing assistant for a chat application with access to these tools:
{tool_lines}
{context_block}
Decide whether answering the user's CURRENT message requires one of these tools.
Reply with exactly one line, no other text:
  NONE                      — no tool needed, this can be answered directly
  TOOL:<name>               — for a tool that needs no argument
  TOOL:<name>|<argument>    — for a tool that needs an argument (e.g. the
                              expression to evaluate for "calculate")

Current message: "{display_query}"
Answer:"""


def _parse(answer: str) -> ToolDecision:
    """Only the first line was checked here originally, on the assumption
    that a model told "reply with exactly one line, no other text" would
    actually do that. In practice this is model-dependent: a larger or
    more "chatty" model can preface the answer with a line or two of its
    own reasoning even when explicitly told not to, which pushed the real
    TOOL:/NONE line down and made this silently fall through to "no tool"
    every time — not a classifier accuracy problem, a parsing brittleness
    one. Scanning every line for the first one that actually looks like an
    answer survives that kind of preamble; it costs nothing when the model
    *does* follow the one-line instruction, since that's still just the
    first line checked."""
    raw = answer.strip()
    if not raw:
        return ToolDecision(raw_answer=answer)
    for line in raw.splitlines():
        line = line.strip().strip("*").strip()  # tolerate "**TOOL:...**"-style emphasis too
        upper = line.upper()
        if upper.startswith("TOOL:"):
            body = line[len("TOOL:"):].strip()
            name, _, arg = body.partition("|")
            name = name.strip()
            if name not in TOOL_REGISTRY:
                continue
            return ToolDecision(tool_name=name, tool_arg=arg.strip(), raw_answer=answer)
        if upper.startswith("NONE"):
            return ToolDecision(raw_answer=answer)
    return ToolDecision(raw_answer=answer)


def classify(query: str, history_context: str = "") -> ToolDecision:
    if llm_utils.get_llm() is None:
        return ToolDecision(raw_answer="<model not loaded>")
    try:
        # max_tokens raised from 60: some models add a short line or two of
        # their own reasoning before the actual TOOL:/NONE line despite
        # being told not to (see _parse's docstring) — if that preamble ate
        # most of the old budget, a long argument (a verbatim birth-data
        # quote) could get truncated before it was ever produced. This is a
        # generation-length allowance, not a cost control — there's no
        # per-token cost on a local model, only the fixed latency of
        # generating more tokens, which is worth it here to not silently
        # corrupt or drop the tool argument.
        answer = llm_utils.generate_sync(
            _build_prompt(query, history_context), max_tokens=120, temperature=0.0,
        )
    except Exception as e:
        return ToolDecision(raw_answer=f"<classifier call raised: {e}>")
    return _parse(answer)


async def classify_async(query: str, history_context: str = "") -> ToolDecision:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, classify, query, history_context)
