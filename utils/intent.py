"""Detects whether a user message is asking to generate or edit an image, so
routes/chat.py can route it to the ycplt_img service instead of the regular
chat LLM.

Uses the already-loaded chat model itself as a small zero-shot classifier —
no extra model to load, no keyword list to maintain by hand, and it
generalizes across phrasing/language better than substring matching would.
"""
import asyncio
from typing import Optional

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
images (change colors, remove/add/replace things, apply a style, etc.) but
cannot itself see or describe what's in an image.

Rule: if the message is a COMMAND telling you to change something about
the image — remove/delete/add/replace/recolor/blur/crop something, apply a
filter or style, etc. — answer EDIT, even if the word "edit" is never used
and even if it only names an object to remove (e.g. "remove the cat from
this photo", "убери кота с фото", "убери кота с прикреплённого фото",
"удали фон", "сделай фон синим", "добавь шляпу" are all EDIT).

Otherwise — a QUESTION about the image's content, a request to describe,
identify, or analyze it, or anything not asking to change it (e.g. "what's
in this picture?", "what breed of dog is this?", "что изображено на
картинке?", "опиши фото", "кто на фото?") — answer QUESTION.

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
            max_tokens=6,
            temperature=0.0,
        )
    except Exception:
        return False
    return _parse_edit_answer(answer)


async def is_edit_instruction_async(query: str) -> bool:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, is_edit_instruction, query)


_REMOVAL_TARGET_PROMPT = """You already know the following message is an
instruction to edit an attached image. Now decide specifically: is it
asking to REMOVE or DELETE one particular object, person, or thing from
the image (e.g. "remove the cat", "убери кота", "delete the person in the
background", "удали фон", "get rid of the car")?

If yes, reply with just the English name of that object/thing (a short
noun phrase, e.g. "cat", "person", "background", "car" — translate it to
English if the message wasn't in English).

If the message is an edit instruction of any OTHER kind — changing a
color, adding something, applying a style or filter, anything that isn't
about removing one specific named thing — reply with exactly: NONE

Message: "{query}"
Answer:"""


def _parse_removal_target(answer: str) -> Optional[str]:
    normalized = answer.strip().strip(".\"'")
    if not normalized or normalized.upper().startswith("NONE"):
        return None
    return normalized


def get_removal_target(query: str) -> Optional[str]:
    """Only meaningful for messages already classified as an edit
    instruction (see is_edit_instruction) — determines whether the edit is
    specifically "remove this named object" and, if so, what the object is
    (translated to English, since the automatic-segmentation model used
    downstream, ycplt_img's CLIPSeg, has an English-trained text tower).

    This powers ycplt_img's automatic segmentation + inpainting path for
    object removal (see its README "Removing a named object") — plain
    img2img has no mechanism to understand or execute a removal
    instruction, it just partially re-renders the whole image guided by
    the prompt text, which is why "убери кота"/"remove the cat" used to
    just restyle the image while leaving the object in place.

    Returns None (treat as a general, non-removal edit) on any error or if
    it's ambiguous — the safe default: the edit still goes through plain
    img2img on ycplt_img's side, just without the extra precision
    auto-masking would add.
    """
    if llm_utils.get_llm() is None:
        return None
    try:
        answer = llm_utils.generate_sync(
            _REMOVAL_TARGET_PROMPT.format(query=query),
            max_tokens=10,
            temperature=0.0,
        )
    except Exception:
        return None
    return _parse_removal_target(answer)


async def get_removal_target_async(query: str) -> Optional[str]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_removal_target, query)
