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


_EDIT_VS_QUESTION_PROMPT = """You are classifying a message that was sent
together with an attached image, for a chat application that can EDIT
images (change colors, remove/add things, apply a style, etc.) but has no
way to see or describe what's actually in an image.

Decide whether the message is an INSTRUCTION to edit/modify the attached
image (e.g. "make the background blue", "remove the person", "add a hat",
"сделай фон синим", "убери человека"), or anything else — a QUESTION about
the image's content, a request to describe/identify/analyze it, or
something unrelated to editing it (e.g. "what's in this picture?", "what
breed of dog is this?", "что изображено на картинке?", "опиши фото").

Reply with exactly one word: EDIT or QUESTION. No punctuation, no explanation.

Message: "{query}"
Answer:"""


def _parse_edit_answer(answer: str) -> bool:
    normalized = answer.strip().strip(".\"'").upper()
    return normalized.startswith("EDIT")


def is_edit_instruction(query: str) -> bool:
    """Only called when an image is attached (see routes/chat.py) — decides
    whether the accompanying text is an editing instruction versus a
    question or anything else about the image.

    Defaults to False (treat as "not an edit") on any error or ambiguous
    output. This is the opposite cost trade-off from is_image_request:
    here a false positive (wrongly treating a question as an edit
    instruction) sends nonsense text into an img2img job and burns CPU time
    on a meaningless result — exactly the failure mode this classifier
    exists to prevent — while a false negative just means a genuine edit
    instruction gets the honest "can't do that yet" reply instead of being
    acted on, recoverable by rephrasing more explicitly.
    """
    if llm_utils.get_llm() is None:
        return False
    try:
        answer = llm_utils.generate_sync(
            _EDIT_VS_QUESTION_PROMPT.format(query=query),
            max_tokens=5,
            temperature=0.0,
        )
    except Exception:
        return False
    return _parse_edit_answer(answer)


async def is_edit_instruction_async(query: str) -> bool:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, is_edit_instruction, query)
