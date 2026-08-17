"""Detects whether a user message is asking to generate or edit an image, so
routes/chat.py can route it to the ycplt_img service instead of the regular
chat LLM.

Uses the already-loaded chat model itself as a small zero-shot classifier —
no extra model to load, no keyword list to maintain by hand, and it
generalizes across phrasing/language better than substring matching would.
"""
import asyncio
import re
from typing import Optional, Tuple

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
        answer = llm_utils.classify_sync(
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
        answer = llm_utils.classify_sync(
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
        answer = llm_utils.classify_sync(
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


_RECONSTRUCTION_PROMPT = """You already know this message asks to remove
"{target}" from an attached image. Now decide: does the message ALSO
describe what should appear in the removed object's place — something
more specific than just "empty background" or "nothing there" (e.g.
"recreate the metal panel with holes that was behind it", "put a plain
wall there instead", "fill it with grass matching the rest of the lawn")?

If yes, reply with a short English phrase describing that content,
written as a text-to-image prompt fragment (e.g. "a section of a
perforated stainless steel panel with round holes, matching the
surrounding metal texture"). Translate to English if the message wasn't
in English.

If the message only says to remove/delete "{target}" with no further
description of what should replace it, reply with exactly: NONE

Message: "{query}"
Answer:"""


def _parse_reconstruction_prompt(answer: str) -> Optional[str]:
    normalized = answer.strip().strip(".\"'")
    if not normalized or normalized.upper().startswith("NONE"):
        return None
    return normalized


def get_reconstruction_prompt(query: str, target: str) -> Optional[str]:
    """Only meaningful once get_removal_target has already identified this
    as a "remove {target}" instruction — a second, narrow classification
    pass deciding whether the SAME message also describes what should be
    drawn in {target}'s place, as opposed to just wanting it gone.

    This exists because plain removal (ycplt_img's LaMa path, no text
    prompt at all — see that service's README "Removing a named object")
    and a described reconstruction are genuinely different jobs needing
    different tools: LaMa can only extend generic surrounding texture, it
    has no way to paint a SPECIFIC, describable thing (a particular
    pattern, material, or object) into the hole. When the user did
    describe something specific, ycplt_img routes through its
    StableDiffusion "inpaint" checkpoint instead (see its
    srv/worker.py's _generate_removal_edit) with this text as the prompt
    — the right tool once there's an actual description to work from.

    Returns None (plain removal, the common/default case) on any error,
    an ambiguous answer, or if the message simply doesn't describe
    anything beyond "remove it" — never raises, so a failure here just
    means the job falls back to LaMa instead of blocking the edit
    entirely."""
    if llm_utils.get_llm() is None:
        return None
    try:
        answer = llm_utils.classify_sync(
            _RECONSTRUCTION_PROMPT.format(target=target, query=query),
            max_tokens=60,
            temperature=0.0,
        )
    except Exception:
        return None
    return _parse_reconstruction_prompt(answer)


async def get_reconstruction_prompt_async(query: str, target: str) -> Optional[str]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_reconstruction_prompt, query, target)


_ENGLISH_EDIT_PROMPT = """You already know the message below is an
instruction to edit an attached image (NOT a "remove one named object"
instruction — that case is handled separately).

Rewrite it as a single, clear, concise English sentence suitable as a
prompt for an image-editing AI model. Preserve the meaning and intent
exactly — don't add details that aren't there, don't explain your answer,
don't add quotes around it. If the message is already in English, just
repeat it back, correcting only obvious typos.

Message: "{query}"
English instruction:"""


def _parse_english_edit_instruction(answer: str, fallback: str) -> str:
    normalized = answer.strip().strip("\"'")
    return normalized if normalized else fallback


def get_english_edit_instruction(query: str) -> str:
    """Translates/rephrases a general (non-remove_target) image edit
    instruction into English before it's sent to ycplt_img as the job's
    prompt.

    Real, reported failure this exists to fix: a Russian instruction
    ("Перерисуй фото женщины как если бы она была мужчиной") sent as-is
    produced a FLUX.1 Kontext edit that came back identical to the input
    photo — ycplt_img's own diagnostic logs showed its text encoders
    (CLIP's BPE vocabulary and T5's tokenizer) badly fragmenting or
    outright failing to represent Cyrillic text, leaving the model with
    no real instruction signal to condition on. Re-testing the identical
    setup with a clean English prompt produced correct tokenization.
    FLUX/Kontext (like the base Flux/CLIP/T5 stack it's built on) is
    trained overwhelmingly on English text — this mirrors why
    get_removal_target/get_reconstruction_prompt above already translate
    to English for CLIPSeg/the inpaint checkpoint, just applied to the
    general edit-instruction case now that ycplt_img can also route a
    plain img2img edit through Kontext (see its README "Instruction-based
    editing (FLUX.1 Kontext, experimental)").

    Deliberately NOT called for remove_target jobs: LaMa (the default
    object-removal path) ignores the prompt entirely, and
    reconstruct_prompt (the alternate SD-inpaint path) is already
    generated in English by get_reconstruction_prompt above — translating
    the raw query in that case would just be a wasted LLM call.

    Returns the ORIGINAL query unchanged (never None, never raises) on
    any error, an empty answer, or if the local LLM isn't loaded — always
    produces a usable prompt string, so a translation failure just means
    the edit proceeds with the original-language text exactly as before
    this function existed, not a blocked request.

    max_tokens=300 (raised from an initial 80): a "-Thinking-" style chat
    model (see utils/llm.py's generate_sync own <think> comment) can spend
    a large chunk of its budget on an internal reasoning trace before ever
    reaching the actual translated sentence — 80 was tight enough that a
    long/compound instruction risked getting cut off mid-<think>, in which
    case generate_sync's own stripping (which only removes a CLOSED
    <think>...</think> block) leaves the raw, unclosed trace as the
    "answer", and that garbled text would go straight to Kontext as its
    prompt. 300 gives real headroom for a thinking model's scratchpad
    to actually close before the visible answer, on top of the
    translated sentence itself (normally well under 50 tokens) — a real,
    but comparatively small, added cost against a job that already
    typically takes minutes to hours end to end (see ycplt_img's README
    "Instruction-based editing (FLUX.1 Kontext, experimental)")."""
    if llm_utils.get_llm() is None:
        return query
    try:
        answer = llm_utils.classify_sync(
            _ENGLISH_EDIT_PROMPT.format(query=query),
            max_tokens=300,
            temperature=0.0,
        )
    except Exception:
        return query
    return _parse_english_edit_instruction(answer, query)


async def get_english_edit_instruction_async(query: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_english_edit_instruction, query)


# --- synastry two-person text segmentation (LLM fallback) ----------------
#
# Why this exists: utils/astro.py's synastry free-text extraction is
# deliberately heuristic/regex-based, not LLM-based — the project's
# established, hard-won default (see astro.py's own module docstring: an
# earlier attempt at having a small model reformat/convert birth data
# proved unreliable, so plain regex parsing of the user's own wording
# replaced it). That heuristic (astro._split_two_person_text) finds the
# first two dates in the text and splits at the nearest comma/newline/
# whitespace between them — good enough for the phrasings tested so far,
# but a real, acknowledged limitation: sufficiently free-form phrasing
# (person labels and data interleaved in an order the heuristic doesn't
# expect) will keep finding new ways to break a purely positional
# splitting rule, and patching one specific phrasing at a time (already
# done twice this project) doesn't converge.
#
# Rather than replace the deterministic heuristic outright (rejected —
# it works fine for the common, already-tested phrasings, and a full LLM
# reformatting step reintroduces exactly the unreliable-conversion failure
# mode the project moved away from), this is a NARROW, single-purpose LLM
# call used only as a fallback when the heuristic split doesn't yield
# complete data for both people (see routes/chat.py's _handle_tool_request):
# the model's only job is SEGMENTATION — deciding which words belong to
# which person — not reformatting dates, converting coordinates, or
# transcribing anything. It's asked to quote the relevant portion of the
# original text back VERBATIM for each person, exactly the same
# "quoting back is reliable, reformatting is not" principle astro.py's own
# docstring already established for the tool-router's argument extraction.
# The quoted halves are then run through the exact same deterministic
# regex field-extraction (_fill_fields_from_text) the heuristic path
# already uses — this call only ever changes WHERE the text gets split,
# never what a date/time/coordinate/city string is taken to mean, so a
# hallucinated or mangled quote just produces "still missing data" (a
# clarifying question), not silently wrong chart data.
_TWO_PERSON_SPLIT_PROMPT = """Сообщение пользователя ниже описывает ДВУХ разных людей и их данные рождения (для сравнения гороскопов, синастрии). Раздели его на две части — одну на каждого человека.

В каждой части ДОСЛОВНО процитируй ту часть исходного сообщения, которая относится к этому человеку (имя или обозначение, дата, время, место рождения) — ничего не переводи, не переформатируй, не исправляй и не досочиняй, только скопируй соответствующий фрагмент исходного текста.

Ответь ровно в этом формате, без вступлений и пояснений:
ЧЕЛОВЕК 1: <дословная цитата из сообщения>
ЧЕЛОВЕК 2: <дословная цитата из сообщения>

Сообщение пользователя:
\"\"\"
{query}
\"\"\"
Ответ:"""

_TWO_PERSON_SPLIT_RE = re.compile(
    r"ЧЕЛОВЕК\s*1\s*:\s*(.*?)\s*\n\s*ЧЕЛОВЕК\s*2\s*:\s*(.*)", re.IGNORECASE | re.DOTALL
)


def _parse_two_person_split(answer: str) -> Optional[Tuple[str, str]]:
    m = _TWO_PERSON_SPLIT_RE.search(answer)
    if not m:
        return None
    part_a, part_b = m.group(1).strip(), m.group(2).strip()
    if not part_a or not part_b:
        return None
    return part_a, part_b


def split_two_person_text(query: str) -> Optional[Tuple[str, str]]:
    """Best-effort LLM fallback for astro.py's two-person synastry text
    split — see the module comment above for why this exists and why it's
    scoped to segmentation only, never reformatting. Returns None (caller
    falls back to whatever the deterministic heuristic already produced)
    on any error, an unparseable answer, or an empty half — this is purely
    additive: it can only help a case the heuristic already failed on, and
    can never make a working case worse, since routes/chat.py only calls
    this after confirming the heuristic split left required fields
    missing for at least one person."""
    if llm_utils.get_llm() is None:
        return None
    try:
        answer = llm_utils.classify_sync(
            _TWO_PERSON_SPLIT_PROMPT.format(query=query),
            max_tokens=400,
            temperature=0.0,
        )
    except Exception:
        return None
    return _parse_two_person_split(answer)


async def split_two_person_text_async(query: str) -> Optional[Tuple[str, str]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, split_two_person_text, query)
