"""Detects whether a user message is asking to generate or edit an image, so
routes/chat.py can route it to the ycplt_img service instead of the regular
chat LLM.

Uses the already-loaded chat model itself as a small zero-shot classifier —
no extra model to load, no keyword list to maintain by hand, and it
generalizes across phrasing/language better than substring matching would.
"""
import asyncio

from utils import llm as llm_utils

_CLASSIFIER_PROMPT = """You are an intent classifier for a chat application.
Decide whether the user's message is asking to CREATE or EDIT an image,
picture, drawing, illustration, or photo (the message may be in any
language, e.g. "draw me...", "generate an image of...", "make a picture
of...", "edit this photo to...", "нарисуй", "сгенерируй картинку",
"отредактируй фото").

Reply with exactly one word: IMAGE or CHAT. No punctuation, no explanation.

User message: "{query}"
Answer:"""


def _parse_answer(answer: str) -> bool:
    normalized = answer.strip().strip(".\"'").upper()
    return normalized.startswith("IMAGE")


def is_image_request(query: str) -> bool:
    """Best-effort classification using the local LLM.

    Defaults to False (regular chat) on any error or ambiguous output — a
    false negative just costs a text answer instead of an image (recoverable
    by rephrasing), whereas a false positive would waste minutes on an
    unwanted image job. Safer default given the cost asymmetry.
    """
    if llm_utils.get_llm() is None:
        return False
    try:
        answer = llm_utils.generate_sync(
            _CLASSIFIER_PROMPT.format(query=query),
            max_tokens=5,
            temperature=0.0,
        )
    except Exception:
        return False
    return _parse_answer(answer)


async def is_image_request_async(query: str) -> bool:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, is_image_request, query)
