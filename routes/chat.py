"""Chat routes: POST /chat (generates a reply, persisted to the DB) and GET /health.

Every /chat call:
  1. Creates a conversation if conversation_id isn't given (new chat).
  2. Saves the user's message with its send time. If an image was attached
     (ChatRequest.image_data), it's decoded and stored as a file attachment
     on this same user message (so it shows up on reload, same as any other
     image file — see db/repository.py add_file).
  3. If an image was attached, uses utils/intent.py's is_edit_instruction
     to decide whether the accompanying text is an editing instruction
     (routes to _handle_image_edit_request, mode="img2img"/"inpaint") or
     something else — a question about the image, a request to describe
     it, anything not about editing it (routes to _handle_image_question,
     mode="caption"). Either way this app only classifies intent and
     submits a job to ycplt_img (utils/image_client.py) — it has no vision
     or image-generation model of its own, only the chat LLM; ycplt_img
     hosts the moondream2 vision model used for captioning (see its
     README). Steps 4-5 below are skipped either way.
  4. Otherwise, uses utils/intent.py to decide whether this is a request to
     generate a brand new image. If so, submits a job to ycplt_img
     (utils/image_client.py) and stores a 'pending' placeholder message
     instead of calling the chat LLM — the background poller
     (utils/image_jobs.py) resolves it later, the same as for edits.
  5. Otherwise, uses utils/tool_router.py to decide whether one of the
     built-in tools (utils/tools.py — current date/time, a calculator, more
     can be registered there) is needed to answer well. If so, runs the
     tool and does a short follow-up generation using its result.
  6. Otherwise, generates a normal reply, measuring "thinking" time
     (thinking_ms).
  7. Saves the model's reply with its timestamp and thinking_ms.
  8. Extracts fenced code blocks from the reply as file attachments
     (utils/codeblocks.py) and stores them in the DB (db/repository.py add_file).
"""
import asyncio
import base64
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db import repository
from utils import astro
from utils import chart_draw
from utils import config
from utils import electional
from utils import horary
from utils import image_client
from utils import intent
from utils import interpret
from utils import llm as llm_utils
from utils import rag as rag_utils
from utils import rectification
from utils import rectification_events
from utils import tool_router
from utils import tools
from utils.codeblocks import extract_code_blocks

router = APIRouter()

# img2img default: 0 = ignore the prompt and return the input image
# unchanged, 1 = ignore the input image and generate from the prompt alone.
# 0.75 is stable-diffusion.cpp's usual middle ground — enough freedom to
# follow the instruction, while still recognizably starting from the
# uploaded image.
DEFAULT_EDIT_STRENGTH = 0.75


class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[int] = None
    # None = no artificial cap; the model generates until it stops on its own
    # or fills the context window (see utils/config.N_CTX). Pass an explicit
    # value only to deliberately shorten a particular reply.
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    use_rag: Optional[bool] = False
    # Set when the user armed the composer's ❓ help-mode toggle before
    # typing this message (see templates/index.html/static/js/app.js) —
    # force-routes straight to astro_help_assistant, bypassing tool_
    # router's own classification for THIS message. See chat()'s use of
    # this flag for why: classification is a small-model judgment call
    # that's least reliable exactly when the user themselves doesn't know
    # what they're asking for yet, which is precisely this feature's use
    # case — a deterministic UI toggle can't misclassify.
    force_help: Optional[bool] = False

    # Optional image attachment: presence of image_data means "edit this
    # image" (see chat() below). image_data is the raw file bytes, base64-
    # encoded, with no "data:...;base64," prefix.
    image_data: Optional[str] = None
    image_filename: Optional[str] = "upload.png"
    image_mime_type: Optional[str] = "image/png"
    # img2img strength override (0..1). None = DEFAULT_EDIT_STRENGTH.
    strength: Optional[float] = None


def _auto_title(text: str, limit: int = 40) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _prior_user_texts(conversation_id: int, exclude_message_id: int) -> List[str]:
    """Every prior USER message in this conversation, oldest first, as a
    plain list — feeds _extraction_history_context (utils/astro.py's
    regex-based field extraction) only. Deliberately user-only, unlike
    _prior_messages below: an assistant's own generated reply can contain
    plausible-looking dates/coordinates/degrees of its own (a chart's
    computed placements, a rectification report's "conception moment"
    line, ...) that a naive regex extractor could otherwise misparse as
    the user's own birth data — a real risk, not a hypothetical one, given
    how many dates a single astro answer can contain."""
    history = repository.list_messages(conversation_id)
    return [
        m["content"] for m in history if m["role"] == "user" and m["id"] != exclude_message_id
    ]


def _prior_messages(conversation_id: int, exclude_message_id: int) -> List[Dict]:
    """Every prior message (BOTH roles), oldest first — used only for
    tool_router's own classification context (_classifier_history_context)
    and _handle_chat_request's conversational-continuity context
    (_chat_history_block). Keeping the assistant's own replies here (unlike
    _prior_user_texts above) is the actual fix for a real, reported
    failure: the router used to see only past USER messages, so it had no
    way to notice that the ASSISTANT itself had just suggested something
    ("try a wider search window") — a short user follow-up ("давай окно
    пошире") referring back to that suggestion looked, to the router, like
    an unrelated message about resizing an application window, and got
    classified as needing no tool at all."""
    history = repository.list_messages(conversation_id)
    return [m for m in history if m["id"] != exclude_message_id]


# Deliberately small, for tool_router's own classification prompt
# specifically — NOT the same thing as "how much history is available for
# extracting a tool argument" (see _handle_tool_request, which uses the
# full, untruncated history instead). Tried feeding the entire
# conversation history into the classifier itself and it made routing
# measurably *worse* in practice: a small model's attention to the actual
# current message degrades once the prompt is dominated by a long history
# dump ("lost in the middle"), so a message that used to route correctly
# started coming back as "no tool needed" — the model just answered from
# its own (hallucinated) general knowledge instead. More context helps a
# deterministic regex extraction step; it does not reliably help an LLM's
# yes/no classification decision, so the two uses are kept separate on
# purpose rather than sharing one "give it everything" value.
#
# Counts BOTH roles now (previously 4 user messages only) — including the
# assistant's own last reply is the actual fix described in _prior_
# messages' docstring above, and keeping the total count the same (rather
# than 4 of each role, 8 total) keeps this consistent with the
# already-learned "don't dump too much into this specific small-model
# classification call" lesson.
_CLASSIFIER_HISTORY_MAX_MESSAGES = 4
_CLASSIFIER_HISTORY_MAX_CHARS_EACH = 300


def _label_role(role: str) -> str:
    return "Пользователь" if role == "user" else "Ассистент"


def _classifier_history_context(prior_messages: List[Dict]) -> str:
    recent = prior_messages[-_CLASSIFIER_HISTORY_MAX_MESSAGES:]
    return "\n".join(
        f"{_label_role(m['role'])}: {m['content'][:_CLASSIFIER_HISTORY_MAX_CHARS_EACH]}" for m in recent
    )


# For _handle_chat_request's plain (non-tool) reply path — separate budget
# from the classifier's own (above), since this feeds the actual answer
# generation rather than a cheap yes/no routing decision: a normal chat
# reply benefits from seeing more of the recent back-and-forth than a
# routing classifier does (see _CLASSIFIER_HISTORY_MAX_MESSAGES's own
# comment on why MORE history made THAT specific call worse, not better —
# that finding was about the routing decision specifically, not about
# conversational continuity in general).
_CHAT_HISTORY_MAX_TURNS = 6  # up to 6 user+assistant exchanges (12 messages)
_CHAT_HISTORY_MAX_CHARS_EACH = 800


def _chat_history_block(prior_messages: List[Dict]) -> str:
    """Plain-text "Пользователь: .../Ассистент: ..." transcript of the
    last few turns, or "" if this is the first message in the
    conversation — prepended to the generation prompt in
    _handle_chat_request so a short follow-up ("а сделай его короче",
    "давай другой вариант") isn't generated as if it were the very first
    message of a brand new conversation, which is what happened before
    this existed (every /chat call built its prompt from req.query alone,
    with no conversation history in it at all — not a context-SIZE
    problem, since N_CTX already comfortably fits far more than this;
    history simply was never included in the prompt to begin with)."""
    recent = prior_messages[-(_CHAT_HISTORY_MAX_TURNS * 2):]
    if not recent:
        return ""
    lines = []
    for m in recent:
        content = m["content"]
        if len(content) > _CHAT_HISTORY_MAX_CHARS_EACH:
            content = content[:_CHAT_HISTORY_MAX_CHARS_EACH].rstrip() + "…"
        lines.append(f"{_label_role(m['role'])}: {content}")
    return "\n".join(lines)


def _extraction_history_context(prior_texts: List[str]) -> str:
    """Untruncated — fed only into utils/astro.py's regex-based field
    extraction (via _handle_tool_request), never into an LLM prompt on its
    own, so there's no attention-dilution downside to keeping all of it."""
    return "\n".join(prior_texts)


async def _classify_tool_with_retry(
    req: ChatRequest, prior_messages: List[Dict]
) -> tool_router.ToolDecision:
    """The router's normal (non-forced) classification path — extracted
    out of chat() unchanged so req.force_help (see ChatRequest and chat()
    below) can skip straight past it instead of needing to be threaded
    through every line of it as an extra condition."""
    tool_decision = await tool_router.classify_async(
        req.query, _classifier_history_context(prior_messages)
    )
    # Deliberately unconditional (not just when a tool fires): "the router
    # decided no tool was needed" is exactly as important to see in the
    # log as which tool/argument it picked, when diagnosing a tool that
    # silently isn't being used for a message that clearly needed it.
    print(
        f"[tool_router] tool={tool_decision.tool_name!r} "
        f"arg={tool_decision.tool_arg!r} raw={tool_decision.raw_answer!r}"
    )

    if not tool_decision.tool_name and prior_messages:
        # Real, reported failure: a second, unrelated rectification request
        # (a different person's birth data/events) came back tool=None when
        # sent in a conversation that already had an earlier, unrelated
        # rectification exchange in it — the exact same brand-new message
        # routed correctly in a fresh conversation with no history at all.
        # This is the same "more raw text in the classifier's own prompt
        # measurably degrades its judgment" failure class already fixed
        # twice for OTHER causes (_CLASSIFIER_HISTORY_MAX_MESSAGES for
        # history length, tool_router._MAX_QUERY_CHARS for the current
        # message) — here the trigger is specifically a PRIOR tool
        # exchange's own substantial, data-heavy content sitting in
        # history_context. Rather than tune the history budget again (and
        # risk breaking the short-follow-up case _classifier_history_
        # context exists for — see that function's own docstring), retry
        # ONCE with no history at all before giving up: this can only ever
        # RECOVER a tool call the history diluted away, never break a case
        # that already worked (that returns above, before this block), and
        # can only regress a message that genuinely needs history to be
        # recognized as needing a tool at all (e.g. "давай окно пошире")
        # back to the same "no tool" outcome it would already have gotten
        # without this retry, since dropping history can't manufacture a
        # signal for a tool that isn't in the message on its own.
        retry_decision = await tool_router.classify_async(req.query, "")
        print(
            f"[tool_router] retry without history: tool={retry_decision.tool_name!r} "
            f"arg={retry_decision.tool_arg!r} raw={retry_decision.raw_answer!r}"
        )
        # Only trust the retry if it ALSO found real argument content in the
        # bare message itself (or the tool genuinely takes none, e.g.
        # get_current_datetime) — real, reported failure: a horary
        # follow-up message with no new moment/place in it at all ("Вердикт
        # Нет значит вещи не найдутся?") still matched astro_horary_question
        # on intent alone once history was dropped (that tool's own
        # description deliberately tells the classifier a "why"/"explain"-
        # style follow-up belongs to it too, so the SAME re-explanation
        # request routes back to it — see utils/horary.py), but came back
        # with an EMPTY tool_arg, and _handle_tool_request's concatenation
        # then silently pulled in whatever chart data happened to be
        # sitting in the surrounding conversation history instead — a
        # DIFFERENT, unrelated horary question from earlier in the same
        # chat, producing a confidently-worded but completely wrong reply.
        # The original rectification bug this retry exists for isn't broken
        # by this extra check: that failure was always a genuinely
        # self-contained new request (full birth data + event list quoted
        # in the message itself), so the retry's own tool_arg was already
        # non-empty in that case — this only rejects the specific pattern
        # where the retry re-derived a tool from bare intent with nothing
        # concrete behind it, which dropping history can never fix (see the
        # comment above this block for why retrying can only recover a
        # signal, never manufacture missing data).
        if retry_decision.tool_name and (
            retry_decision.tool_arg.strip() or retry_decision.tool_name == "get_current_datetime"
        ):
            tool_decision = retry_decision

    return tool_decision


@router.post("/chat")
async def chat(req: ChatRequest):
    if llm_utils.get_llm() is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    conversation_id = req.conversation_id
    if conversation_id is None:
        conversation_id = repository.create_conversation(_auto_title(req.query))
    elif repository.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Диалог не найден")

    image_bytes: Optional[bytes] = None
    if req.image_data:
        try:
            image_bytes = base64.b64decode(req.image_data, validate=True)
        except Exception:
            raise HTTPException(
                status_code=400, detail="Некорректные данные изображения (ожидается base64)"
            )

    sent_at = time.time()
    user_msg_id = repository.add_message(conversation_id, "user", req.query, sent_at)
    if image_bytes is not None:
        repository.add_file(
            user_msg_id,
            req.image_filename or "upload.png",
            req.image_mime_type or "image/png",
            image_bytes,
        )

    if image_bytes is not None:
        if await intent.is_edit_instruction_async(req.query):
            return await _handle_image_edit_request(conversation_id, req, image_bytes, sent_at)
        return await _handle_image_question(conversation_id, req, image_bytes, sent_at)

    if await intent.is_image_request_async(req.query):
        return await _handle_image_request(conversation_id, req.query, sent_at)

    prior_messages = _prior_messages(conversation_id, exclude_message_id=user_msg_id)

    if req.force_help:
        # The composer's ❓ toggle (see ChatRequest.force_help's own
        # comment): a deterministic override, not another classification
        # attempt. This intentionally skips tool_router entirely rather
        # than just biasing it, because the whole point of this mode is a
        # user who doesn't know what to ask for yet — exactly the case
        # where a small model's own judgment call is least trustworthy.
        # tool_arg is simply the user's own message; astro_help_assistant
        # needs no field extraction of its own (utils/tools.py's
        # astro_help_overview ignores its argument), and astro_help_
        # methodology.txt's own "not about astrology at all" section
        # covers a genuinely unrelated question typed in this mode —
        # answer it plainly rather than forcing an irrelevant technique
        # recommendation onto it.
        tool_decision = tool_router.ToolDecision(
            tool_name="astro_help_assistant",
            tool_arg=req.query,
            raw_answer="<forced: help-mode toggle, classifier not run>",
        )
        print(f"[tool_router] forced by help-mode toggle: tool={tool_decision.tool_name!r}")
    else:
        tool_decision = await _classify_tool_with_retry(req, prior_messages)

    if tool_decision.tool_name:
        prior_user_texts = _prior_user_texts(conversation_id, exclude_message_id=user_msg_id)
        return await _handle_tool_request(
            conversation_id,
            req,
            sent_at,
            tool_decision,
            _extraction_history_context(prior_user_texts),
            prior_user_texts,
        )

    return await _handle_chat_request(conversation_id, req, sent_at, prior_messages)


async def _handle_image_question(
    conversation_id: int, req: ChatRequest, image_bytes: bytes, sent_at: float
) -> dict:
    """Called when an image is attached but utils/intent.py's
    is_edit_instruction_async decided the accompanying text isn't an
    editing instruction — most commonly a question about the image's
    content ("what's in this picture?").

    Image understanding is a graphics-service capability, exactly like
    generation/editing: this chat app only classifies intent, it doesn't
    run any vision model itself. A mode="caption" job is submitted to
    ycplt_img (which hosts the moondream2 vision model — see its README),
    and a 'pending' placeholder is stored, same shape as
    _handle_image_request/_handle_image_edit_request; the background
    poller (utils/image_jobs.py) resolves it into the actual text answer
    once ready, or an error if the vision model isn't set up there.

    sent_at: the ORIGINAL time chat()'s own sent_at was captured, before
    any of the intent classification calls above it (is_edit_instruction_
    async here) ran — used as this placeholder's created_at instead of a
    fresh time.time() taken only now, after those calls already finished.
    A real, reported bug: previously this used its own late timestamp, so
    a message could visibly show a "sent" time strictly later than when
    the user actually pressed send — sometimes by well over a minute, if
    classification happened to be queued behind other model activity (see
    utils/llm.py's _FifoLock) — even though the pending state shows no
    clock at all in the meantime, so that late timestamp was the very
    first (and, until this fix, only) one ever displayed."""
    loop = asyncio.get_running_loop()
    try:
        job_id = await loop.run_in_executor(
            None,
            lambda: image_client.submit_job(req.query, mode="caption", init_image=image_bytes),
        )
    except image_client.ImageServiceError as e:
        raise HTTPException(status_code=502, detail=f"Сервис изображений недоступен: {e}")

    placeholder_text = "Распознаю изображение…"
    assistant_msg_id = repository.add_message(
        conversation_id,
        "assistant",
        placeholder_text,
        sent_at,
        status="pending",
        image_job_id=job_id,
    )
    repository.touch_conversation(conversation_id)

    return {
        "conversation_id": conversation_id,
        "query": req.query,
        "sent_at": int(sent_at * 1000),
        "response": placeholder_text,
        "responded_at": int(sent_at * 1000),
        "thinking_ms": None,
        "status": "pending",
        "message_id": assistant_msg_id,
        "contexts_used": 0,
        "files": [],
    }


async def _handle_chat_request(
    conversation_id: int, req: ChatRequest, sent_at: float, prior_messages: Optional[List[Dict]] = None
) -> dict:
    """prior_messages (both roles, oldest first — see chat()'s own
    _prior_messages call) is folded into the prompt via
    _chat_history_block below so a short follow-up in the SAME
    conversation ("а покороче", "давай другой вариант") is generated with
    actual awareness of what was just said, not as if it were the first
    message of a brand new conversation — a real, reported gap: every
    /chat call used to build this prompt from req.query alone, with no
    conversation history in it at all (not an N_CTX/context-SIZE problem;
    history simply was never included in the prompt to begin with).
    Optional/defaults to None (-> no history block) so any other caller
    of this function keeps working exactly as before."""
    contexts = []
    if req.use_rag and rag_utils.is_available():
        contexts = rag_utils.retrieve_context(req.query, config.TOP_K)
    prompt = rag_utils.build_prompt(req.query, contexts)

    history_block = _chat_history_block(prior_messages or [])
    if history_block:
        prompt = (
            "Недавняя часть этого же диалога (для контекста — текущий "
            "вопрос может быть коротким продолжением, ссылающимся на "
            "неё, например \"сделай короче\" или \"а другой вариант\"; "
            "это только фон, не отвечай на неё саму по себе):\n"
            f"{history_block}\n\n---\n\n{prompt}"
        )

    gen_start = time.time()
    try:
        resp_text = await llm_utils.generate_async(
            prompt, max_tokens=req.max_tokens, temperature=req.temperature or 0.7
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {e}")
    responded_at = time.time()
    thinking_ms = int((responded_at - gen_start) * 1000)

    assistant_msg_id = repository.add_message(
        conversation_id, "assistant", resp_text, responded_at, thinking_ms
    )

    files = []
    for block in extract_code_blocks(resp_text):
        content_bytes = block["content"].encode("utf-8")
        file_id = repository.add_file(
            assistant_msg_id, block["filename"], block["mime_type"], content_bytes
        )
        files.append(
            {
                "id": file_id,
                "filename": block["filename"],
                "mime_type": block["mime_type"],
                "size": len(content_bytes),
            }
        )

    repository.touch_conversation(conversation_id)

    return {
        "conversation_id": conversation_id,
        "query": req.query,
        "sent_at": int(sent_at * 1000),
        "response": resp_text,
        "responded_at": int(responded_at * 1000),
        "thinking_ms": thinking_ms,
        "status": "complete",
        "contexts_used": len(contexts),
        "files": files,
        "message_id": assistant_msg_id,
    }


# Tools whose raw result is meant to be *interpreted*, not just relayed —
# these route through the RAG methodology/reasoning-mode machinery (see
# below) instead of the generic "answer naturally" follow-up. A bare
# "answer naturally" prompt over the raw chart data alone produced
# shallow, generic descriptions in practice: it never saw the indexed
# interpretation-methodology document at all, since the tool-call path and
# the RAG path used to be entirely separate code paths.
#
# These also happen to be the tools whose argument is a free-text quote
# that utils/astro.py's own regex extraction parses (as opposed to e.g.
# calculate's argument, a math expression that must stay exactly what it
# is) — see the tool_arg handling below for why that matters.
_INTERPRETED_TOOL_NAMES = {
    "astro_natal_chart", "astro_transit_chart", "astro_synastry_chart", "astro_progression_chart",
    "astro_direction_chart", "astro_lunar_return_chart", "astro_solar_return_chart", "astro_profection_chart",
    # astro_rectification_trutine and astro_rectification_events are listed
    # here ONLY so their tool_arg gets the same generous query+router-quote
    # +prior-user-text concatenation as every other astro_* tool below
    # (harmless here too: utils/rectification_events.py's own event-line
    # parser only treats a line as an event if it matches "description:
    # date" AND isn't a birth-data label, so stray unrelated text from
    # concatenated prior messages is simply left as birth-field free
    # text). They do NOT go through the RAG-augmented reasoning-mode
    # follow-up below at all anymore — see _NO_FOLLOWUP_TOOL_NAMES and
    # _handle_tool_request's early-return branch for why (repeated real
    # testing showed the follow-up LLM call reliably contradicted the
    # tool's own computed best-candidate time somewhere in its own prose,
    # even after four separate layers of mitigation — a genuine small-
    # model reliability limit, not a wording problem worth continuing to
    # chase).
    "astro_rectification_trutine",
    "astro_rectification_events",
    # astro_horary_question — same tool_arg-concatenation reason, and it
    # DOES reach the RAG-augmented follow-up below every time (no
    # no-followup bypass, unlike the two rectification tools above): an
    # earlier two-tier design (an always-instant, never-interpreted "short
    # verdict" tool, with a second "give details" tool for follow-ups) was
    # reverted after real testing showed the short-verdict-only reply left
    # a genuinely rich, radical chart completely uninterpreted unless the
    # user knew to explicitly ask for more — see utils/horary.py's module
    # docstring for the full story.
    "astro_horary_question",
    # astro_electional_chart — same tool_arg-concatenation reason and the
    # same per-round place/purpose isolation as astro_horary_question just
    # below (two different elections asked back to back must not silently
    # share a place or purpose any more than two different horary
    # questions should) — DOES reach the RAG-augmented follow-up below
    # every time, same as horary, for the same reason: a bare "answer
    # naturally" prompt over the raw computed report alone would never see
    # electional_methodology.txt at all.
    "astro_electional_chart",
    # astro_help_assistant — conversational technique-selection/explanation
    # helper (utils.tools.astro_help_overview), added per explicit user
    # request for a "universal assistant" that doesn't assume any astrology
    # knowledge. Listed here for the same tool_arg-concatenation reason as
    # every entry above (harmless even though its "run" function ignores
    # tool_arg entirely — see astro_help_overview's own docstring) and
    # because it DOES need the RAG-augmented follow-up below: a bare
    # "answer naturally" prompt over the raw overview text alone would
    # never see astro_help_methodology.txt's fuller technique-comparison
    # and example-query material. Its computed_chunk wording is
    # special-cased just below — it's a reference overview, not a specific
    # person's computed chart data, and saying otherwise would be actively
    # misleading here.
    "astro_help_assistant",
}

# tool_name -> rag_data/ subfolder name, per README's "Recommended
# rag_data/ layout" table. Passed as retrieve_context's topic_hint so the
# right technique's methodology is guaranteed present in the RAG-augmented
# follow-up regardless of whether the user's free-text query happened to
# score a similarity hit against it — see retrieve_context's own docstring
# for why plain similarity search alone isn't enough here (the electional
# tool's "natal chart" mention bug: a mundane query shares no vocabulary
# with electional_methodology.txt's astrological terms, so it lost the
# similarity race to a different topic entirely, and that unrelated
# topic's own methodology got pulled in instead with no warning). Every
# _INTERPRETED_TOOL_NAMES entry that actually reaches the RAG follow-up
# below should have an entry here — the two rectification tools are
# deliberately omitted since they never reach it (see
# _NO_FOLLOWUP_TOOL_NAMES).
_TOOL_TOPIC: Dict[str, str] = {
    "astro_natal_chart": "astro_basics",
    "astro_transit_chart": "astro_transit",
    "astro_synastry_chart": "astro_synastry",
    "astro_progression_chart": "astro_progressions",
    "astro_direction_chart": "astro_progressions",
    "astro_lunar_return_chart": "astro_progressions",
    "astro_solar_return_chart": "astro_progressions",
    "astro_profection_chart": "astro_progressions",
    "astro_horary_question": "astro_horar",
    "astro_electional_chart": "astro_elect",
    "astro_help_assistant": "astro_help",
}

# These two tools' raw report IS the final answer by default — no
# follow-up LLM call, unless config.RECTIFICATION_LLM_FOLLOWUP is turned
# on in .env. Real testing across several rounds established this isn't
# fixable by better prompting: the follow-up call would correctly quote
# the tool's computed best-candidate time (after task #189's deterministic
# prepend), then go on to contradict it anyway somewhere in its own
# "Ответ" section (task #192's fix, then task #193's bookend fix, both
# still insufficient) — three consecutive real-world tests all showed the
# same contradiction pattern despite every mitigation tried. Skipping the
# follow-up call by default is strictly more reliable (nothing left to
# contradict — the reply IS the deterministic computation) and much
# faster (no generation call over a ~10000-character report). A side
# effect while the toggle is off: rectification_trutine_methodology.txt
# and rectification_events_methodology.txt aren't read by the app for
# these replies (no RAG retrieval happens without a follow-up LLM call) —
# both files are kept as standalone reference documentation under
# install/methodologies/, ready for when config.RECTIFICATION_LLM_FOLLOWUP
# is turned on for a more capable model (see utils/config.py's own comment
# on that setting) — at that point this set is simply not checked and
# every one of these two tools' requests falls through to the same
# RAG-augmented follow-up path every other astro_* tool already uses.
_NO_FOLLOWUP_TOOL_NAMES = {
    "astro_rectification_trutine",
    "astro_rectification_events",
}

# Used to visually bold the computed best-candidate line, in two different
# situations depending on config.RECTIFICATION_LLM_FOLLOWUP: with the
# follow-up call off (default), it's applied directly to the raw report so
# a human skimming it can spot the line at a glance; with the follow-up
# call turned back on, it's the same prepend/disclaimer/bookend safety net
# from tasks #189/#192/#193 (kept specifically for that case — see the
# tail of _handle_tool_request), since a more capable model is still worth
# double-checking against the same contradiction failure mode before
# trusting it unconditionally. Kept as a small dict of per-tool extractor
# functions since each tool's report has its own exact wording for this
# line (see each module's own extract_best_recommendation docstring).
_BEST_RECOMMENDATION_EXTRACTORS = {
    "astro_rectification_trutine": rectification.extract_best_recommendation,
    "astro_rectification_events": rectification_events.extract_best_recommendation,
    "astro_horary_question": horary.extract_best_recommendation,
    "astro_electional_chart": electional.extract_best_recommendation,
}

# --- chart-drawing wiring ----------------------------------------------
#
# tool_name -> spec-string function returning (text, subject, second,
# highlight_house) in ONE pass — the "cheap to recompute" 8 techniques
# (natal/transit/progression/direction/lunar_return/solar_return/profection
# in utils/astro.py, horary in utils/horary.py). Each of these functions IS
# the tool's normal computation (registered in TOOL_REGISTRY's plain
# run_* form is a thin wrapper around it) — calling it here instead of
# tool_spec["run"] means the chart-drawing subject comes from the exact
# same computation as the text reply, not a second rebuild. An earlier
# version called tool_spec["run"] for the text and a separate
# get_*_chart_subject(s) getter for the chart, silently DOUBLING every one
# of these 8 techniques' ephemeris/fixed-star computation on every reply —
# a real, reported performance regression, fixed by switching to this
# single-computation form instead.
#
# Deliberately excludes:
#   - astro_synastry_chart: needs the SAME split_hint used to build
#     tool_result — handled inline in the tool-dispatch block below
#     instead (astro.run_synastry_and_subject takes split_hint directly).
#   - astro_electional_chart, astro_rectification_trutine,
#     astro_rectification_events: each involves an expensive SEARCH
#     (electional's date-range scan, rectification's candidate-window
#     scan) — handled inline below via their own "_and_subject"/"_full"
#     functions (electional.run_electional_chart_and_subject,
#     rectification._run_rectification_trutine_full,
#     rectification_events.run_rectification_events_and_subject_async),
#     which return (text, subject) rather than this dict's 4-tuple shape.
_SIMPLE_AND_SUBJECT_FUNCS: Dict[str, Callable[[str], Tuple[str, Any, Any, Optional[int]]]] = {
    "astro_natal_chart": astro.run_natal_and_subject,
    "astro_transit_chart": astro.run_transit_and_subject,
    "astro_progression_chart": astro.run_progression_and_subject,
    "astro_direction_chart": astro.run_direction_and_subject,
    "astro_lunar_return_chart": astro.run_lunar_return_and_subject,
    "astro_solar_return_chart": astro.run_solar_return_and_subject,
    "astro_profection_chart": astro.run_profection_and_subject,
    "astro_horary_question": horary.run_horary_question_and_subject,
}

# Short Russian label for the chart's own header block (title_lines[0]) —
# see chart_draw.draw_wheel_svg's own title_lines param.
_CHART_TITLE_LABEL: Dict[str, str] = {
    "astro_natal_chart": "Натальная карта",
    "astro_transit_chart": "Транзитная карта",
    "astro_synastry_chart": "Синастрия",
    "astro_progression_chart": "Прогрессивная карта",
    "astro_direction_chart": "Карта директных положений",
    "astro_lunar_return_chart": "Лунное возвращение",
    "astro_solar_return_chart": "Солнечное возвращение",
    "astro_profection_chart": "Профекция",
    "astro_horary_question": "Хорарная карта",
    "astro_electional_chart": "Электив (наилучшая дата)",
    "astro_rectification_trutine": "Ректификация (Тритона Гермеса)",
    "astro_rectification_events": "Ректификация (по событиям)",
}

# What the outer ring represents, for the second line of the header block —
# only set for techniques that actually draw one (chart_subject built with
# second != None); omitted entirely for single-subject techniques.
_CHART_SECOND_LABEL: Dict[str, str] = {
    "astro_transit_chart": "Внешнее кольцо: транзитные планеты",
    "astro_synastry_chart": "Внешнее кольцо: планеты партнёра",
    "astro_progression_chart": "Внешнее кольцо: прогрессивные планеты",
    "astro_direction_chart": "Внешнее кольцо: директные положения",
    "astro_lunar_return_chart": "Внешнее кольцо: лунное возвращение",
    "astro_solar_return_chart": "Внешнее кольцо: солнечное возвращение",
}

# Label prefix for the SECOND subject's own date/time line in the header —
# only for techniques where chart_second is a real second kerykeion subject
# with its own birth-data-shaped fields (transit's moving moment,
# progression's symbolic progressed date, a return chart's exact return
# moment); omitted for astro_direction_chart (chart_second there is a plain
# list of synthetic overlay points, not a real subject — see
# astro.run_direction_and_subject's own docstring) and astro_synastry_chart
# (handled separately by _synastry_title_lines, since there both subjects
# are different PEOPLE, not one person plus a second moment). A real,
# reported gap: the header previously showed only the PRIMARY subject's own
# date/time/place, leaving no indication at all of what the second ring's
# moment actually was.
_CHART_SECOND_DATE_LABEL: Dict[str, str] = {
    "astro_transit_chart": "Момент транзита",
    "astro_progression_chart": "Прогрессивная дата",
    "astro_lunar_return_chart": "Дата возвращения",
    "astro_solar_return_chart": "Дата возвращения",
}


def _chart_datetime_line(subject: Any) -> Optional[str]:
    """DD.MM.YYYY HH:MM straight off the subject's own stored fields — no
    locale-dependent month-name formatting needed, and avoids the
    unreliable subject.iso_formatted_local_datetime string (its own
    quirks around historical/DST offsets aren't worth working around just
    for a header line) — for a subject built via
    AstrologicalSubjectFactory (natal, transit, progression, synastry).

    Lunar/solar return's own second subject is a DIFFERENT kerykeion model
    (PlanetReturnModel, from PlanetaryReturnFactory — see
    astro._build_return_subjects) that has no day/month/year/hour/minute
    fields at all, only iso_formatted_local_datetime — a real, reported
    gap: _CHART_SECOND_DATE_LABEL's "Дата возвращения" line silently never
    appeared for either return type because this function returned None
    every time for that subject shape. Falls back to parsing that ISO
    string ONLY when the day/month/etc. fields are genuinely absent
    (AttributeError), not for any other failure, so this stays a no-op
    for every other subject shape already working correctly."""
    try:
        return (
            f"{int(subject.day):02d}.{int(subject.month):02d}.{int(subject.year):04d} "
            f"{int(subject.hour):02d}:{int(subject.minute):02d}"
        )
    except AttributeError:
        pass
    except Exception:
        return None
    iso = getattr(subject, "iso_formatted_local_datetime", None)
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
        return f"{dt.day:02d}.{dt.month:02d}.{dt.year:04d} {dt.hour:02d}:{dt.minute:02d}"
    except Exception:
        return None


def _chart_place_line(subject: Any, spec_text: str) -> Optional[str]:
    """Best-effort place label for the header: a real city name if one
    can be found by name-matching spec_text against the same gazetteer
    utils/astro.py's own geocoding already uses (astro._lookup_city_exact
    — exact/alternate-name matches only, no fuzzy guessing for a header
    line), otherwise plain coordinates. subject.city can't be trusted for
    this — utils.astro._build_subject only ever passes lat/lon/tz into
    kerykeion's AstrologicalSubjectFactory, never a city name (fields["city"]
    is never populated anywhere in this app), so kerykeion's own default
    placeholder ("Greenwich") would show up instead of the real place."""
    city = None
    if spec_text:
        try:
            record = astro._lookup_city_exact(spec_text)
            if record:
                city = record.get("name")
        except Exception:
            city = None
    try:
        lat, lon = float(subject.lat), float(subject.lng)
    except Exception:
        return city
    coord = f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"
    return f"{city} ({coord})" if city else coord


def _chart_title_lines(
    tool_name: str, subject: Any, spec_text: str = "", second_subject: Any = None
) -> List[str]:
    """second_subject: the SAME chart_second passed to draw_wheel_svg's own
    `second` param, threaded through here too so the header can name the
    outer ring's own moment (see _CHART_SECOND_DATE_LABEL) — a real,
    reported gap: the header used to show ONLY the primary subject's own
    date/time/place, with nothing at all indicating what specific moment
    the second ring's positions were even for. Not used for
    astro_synastry_chart, which has its own _synastry_title_lines instead
    (there both subjects are different PEOPLE, not one person plus a
    second moment, so this function's single-person layout doesn't fit)."""
    label = _CHART_TITLE_LABEL.get(tool_name, tool_name)
    name = (getattr(subject, "name", "") or "").strip()
    lines = [label]
    if name and name.lower() not in ("subject", "electional"):
        lines.append(name)
    dt_line = _chart_datetime_line(subject)
    if dt_line:
        lines.append(dt_line)
    place_line = _chart_place_line(subject, spec_text)
    if place_line:
        lines.append(place_line)
    second_date_label = _CHART_SECOND_DATE_LABEL.get(tool_name)
    if second_date_label and second_subject is not None and not isinstance(second_subject, (list, tuple)):
        second_dt_line = _chart_datetime_line(second_subject)
        if second_dt_line:
            lines.append(f"{second_date_label}: {second_dt_line}")
    return lines


def _synastry_title_lines(subject_a: Any, subject_b: Any, spec_text: str) -> List[str]:
    """Synastry's own header layout, used instead of _chart_title_lines'
    single-person one — a real, reported gap: the generic layout showed
    only person A's own name/date/place, with nothing identifying person B
    beyond the generic "Внешнее кольцо: планеты партнёра" caption, even
    though a synastry chart is meaningless without knowing who BOTH people
    are. Each person's own subject already carries a usable name (either
    their real extracted name, or the "Человек A"/"Человек B" placeholder
    astro._build_synastry_subjects assigns — see that function's own
    docstring), so no separate lookup is needed here.

    spec_text is passed to _chart_place_line for BOTH people as a best-
    effort city-name lookup — since it's the combined two-person text, this
    can occasionally surface the wrong person's city if both are named
    exact matches in the gazetteer; each person's own lat/lng (always
    correct, from their own already-built subject) is included either way,
    so the coordinate shown is never wrong even when the city name guess
    is."""
    label = _CHART_TITLE_LABEL.get("astro_synastry_chart", "Синастрия")
    lines = [label]
    for tag, subj in (("A", subject_a), ("B", subject_b)):
        name = (getattr(subj, "name", "") or "").strip() or f"Человек {tag}"
        dt_line = _chart_datetime_line(subj) or ""
        place_line = _chart_place_line(subj, spec_text) or ""
        detail = ", ".join(part for part in (dt_line, place_line) if part)
        lines.append(f"{tag}: {name}" + (f" — {detail}" if detail else ""))
    return lines


async def _attach_chart_if_applicable(
    loop: asyncio.AbstractEventLoop,
    assistant_msg_id: int,
    tool_name: str,
    query: str,
    spec_text: str,
    chart_subject: Any,
    chart_second: Any,
    chart_highlight_house: Optional[int],
) -> List[dict]:
    """Draws and attaches the wheel-chart SVG for this tool's reply, unless
    there's no subject to draw (lookup failed somewhere upstream) or the
    user's own message explicitly declined an image (see
    utils/chart_draw.should_draw_chart's own docstring on why it defaults
    to drawing). Mirrors the exact sync file-attachment shape the plain-
    chat code-block path already uses (db.repository.add_file + a
    {"id", "filename", "mime_type", "size"} dict) so the frontend needs no
    changes at all — see utils/chart_draw.py's own module docstring.

    spec_text is the same birth-info text used to build chart_subject
    (tool_arg, not req.query) — passed through to _chart_title_lines so
    the header's place label is looked up against the actual text that
    produced this specific chart, not just the current message."""
    if chart_subject is None:
        return []
    should_draw = await loop.run_in_executor(None, chart_draw.should_draw_chart, query)
    if not should_draw:
        return []
    if tool_name == "astro_synastry_chart" and chart_second is not None:
        title_lines = _synastry_title_lines(chart_subject, chart_second, spec_text)
    else:
        title_lines = _chart_title_lines(tool_name, chart_subject, spec_text, second_subject=chart_second)
    second_label = _CHART_SECOND_LABEL.get(tool_name) if chart_second is not None else None
    # horary_methodology.txt section 4 (electional_methodology.txt reuses
    # the same rule) specifies a genuinely different aspect set/orb scheme
    # from every other technique here — see astro._CLASSICAL_ASPECT_NAMES'
    # own comment and chart_draw.draw_wheel_svg's classical_aspects
    # docstring for the full story. Every other tool_name keeps drawing
    # with the general aspect table, unchanged.
    classical_aspects = tool_name in ("astro_horary_question", "astro_electional_chart")
    # Cross-chart orb profile for the non-classical case (see
    # chart_draw.draw_wheel_svg's own dual_orb_profile docstring):
    # synastry gets its own flat-per-aspect table (two real people), every
    # other dual technique with a real second subject (transit,
    # progression, lunar/solar return) shares the "transit" profile —
    # direction needs no entry here at all, since its chart_second is a
    # list of overlay points, not a real subject, so chart_draw.py's own
    # is_real_second_subject check already skips cross-chart aspects for
    # it regardless of what's passed here.
    dual_orb_profile = "synastry" if tool_name == "astro_synastry_chart" else "transit"
    try:
        svg_bytes = await loop.run_in_executor(
            None,
            lambda: chart_draw.draw_wheel_svg(
                chart_subject,
                second=chart_second,
                title_lines=title_lines,
                highlight_house=chart_highlight_house,
                second_label=second_label,
                classical_aspects=classical_aspects,
                dual_orb_profile=dual_orb_profile,
            ),
        )
    except Exception as e:
        print(f"[chart_draw] rendering failed for {tool_name}: {e!r}")
        return []
    filename = chart_draw.unique_chart_filename(prefix=tool_name.replace("astro_", ""))
    file_id = repository.add_file(assistant_msg_id, filename, "image/svg+xml", svg_bytes)
    return [
        {
            "id": file_id,
            "filename": filename,
            "mime_type": "image/svg+xml",
            "size": len(svg_bytes),
        }
    ]


async def _handle_tool_request(
    conversation_id: int,
    req: ChatRequest,
    sent_at: float,
    decision: tool_router.ToolDecision,
    history_context: str = "",
    prior_user_texts: Optional[List[str]] = None,
) -> dict:
    """Runs the tool utils/tool_router.py picked, then does one more LLM
    call that turns the raw tool result into a natural-language answer to
    the user's original question. Two model calls total for this path (the
    router's classification, plus this one) — acceptable, since both are
    small compared to a full chat generation.

    For _INTERPRETED_TOOL_NAMES, that follow-up call is the same RAG
    reasoning-mode prompt used by _handle_chat_request (see utils/rag.py),
    with the tool's computed result folded in as an always-include context
    chunk alongside whatever real methodology/fact chunks the query itself
    retrieves — this is what actually lets rag_data/astrology's
    interpretation-methodology document reach the model for these
    questions, instead of a generic "summarize this data" instruction with
    no awareness that document exists. Other tools (datetime, calculator)
    keep the original simple prompt — there's no interpretive depth to add
    for those, so retrieving RAG context for them would just be wasted
    work."""
    tool_spec = tools.TOOL_REGISTRY[decision.tool_name]

    if decision.tool_name in _INTERPRETED_TOOL_NAMES:
        # decision.tool_arg is the router's own transcription of the
        # birth-info text, and testing showed it isn't reliably complete —
        # identical requests sometimes came back with the date/time
        # dropped and only the coordinates kept, even at temperature=0.0
        # (small-model sampling isn't perfectly deterministic in practice).
        # Rather than continue trying to make the router's transcription
        # more reliable, sidestep the problem: utils/astro.py's
        # _extract_fields() finds each field (date/time/coordinates)
        # wherever it appears and takes the first match, so it's harmless
        # to just hand it every text source that might contain the birth
        # info — the current message, the router's (possibly incomplete)
        # quote, and any earlier-conversation context — concatenated. A
        # field missing from one source is simply found in another.
        tool_arg = "\n".join(filter(None, [req.query, decision.tool_arg, history_context]))
    else:
        tool_arg = decision.tool_arg

    loop = asyncio.get_running_loop()

    if decision.tool_name == "astro_horary_question":
        # Horary is uniquely location-sensitive in a way none of the other
        # _INTERPRETED_TOOL_NAMES are: for natal/transit/synastry, reusing
        # the SAME person's birth data across an entire conversation is
        # correct and intended (that's exactly why the generic union above
        # hands every source to _extract_fields at once), but each horary
        # question has its OWN distinct casting moment AND place — two
        # different horary questions asked minutes apart in the same
        # conversation must NOT silently share a city.
        #
        # Three real, reported bugs shaped this before it landed here.
        # Round 1: a follow-up that gave its own new date/time but no place
        # fell back to the full current+history union, and the free-text
        # city search silently resolved an unrelated earlier question's
        # city — fixed by adding horary._classify_new_horary_round (the
        # model, shown the current message ALONE, decides NEW vs
        # CONTINUATION) so a genuinely new-but-incomplete question reports
        # its missing field instead of silently borrowing one. Round 2: a
        # correctly-classified CONTINUATION ("Одесса, Украина", nothing
        # else) still fell back to the FULL union, which is the entire
        # UNBOUNDED conversation history — pulling in a stray time and a
        # stray city from a COMPLETELY DIFFERENT, much earlier horary
        # question. Narrowing that to "current message + the single
        # immediately preceding one" fixed the reported case, but the user
        # raised the real underlying point: a round boundary should mean
        # everything before it is genuinely forgotten, and a continuation
        # should be free to look back as far as ITS OWN round's start —
        # never further, but also never artificially capped at exactly one
        # message if a round runs to two or three follow-ups.
        #
        # Fix: horary._collect_current_round_texts walks backward through
        # prior user messages, classifying each one the SAME way the
        # current message already was, and stops at the most recent one
        # that's itself a genuine NEW-question boundary — so a
        # CONTINUATION only ever pulls in exactly its own round, however
        # many messages that is, and a fresh round's own missing fields are
        # never filled in from anything on the other side of that boundary.
        current_turn = "\n".join(filter(None, [req.query, decision.tool_arg]))
        prior_texts = prior_user_texts or []
        is_new = await loop.run_in_executor(None, horary._classify_new_horary_round, req.query)
        if is_new is True:
            tool_arg = current_turn
        elif is_new is False:
            round_texts = await loop.run_in_executor(
                None, horary._collect_current_round_texts, prior_texts
            )
            tool_arg = "\n".join(filter(None, round_texts + [current_turn]))
        else:
            # Classifier unavailable/unparsable — fall back to a
            # deterministic check, preferring the narrowest arg that's
            # already self-sufficient, only widening one step at a time
            # (current message alone, then +1 immediately preceding
            # message, before finally accepting the full union already
            # built above as a last resort).
            _, missing_current = await loop.run_in_executor(None, astro._extract_fields, current_turn)
            if not missing_current:
                tool_arg = current_turn
            else:
                immediate_prior = prior_texts[-1] if prior_texts else ""
                narrow_arg = "\n".join(filter(None, [current_turn, immediate_prior]))
                _, missing_narrow = await loop.run_in_executor(None, astro._extract_fields, narrow_arg)
                if not missing_narrow:
                    tool_arg = narrow_arg
                # else: keep the full union already built above as a last resort.

    if decision.tool_name == "astro_electional_chart":
        # Same round-isolation mechanism as astro_horary_question just
        # above, for the same reason: each election has its OWN moment and
        # purpose, and reusing horary's proven _classify_new_horary_round-
        # style approach here (utils/electional.py's own
        # _classify_new_electional_round/_collect_current_round_texts)
        # avoids repeating that class of bug rather than waiting to
        # rediscover it independently for this tool too.
        current_turn = "\n".join(filter(None, [req.query, decision.tool_arg]))
        prior_texts = prior_user_texts or []
        is_new = await loop.run_in_executor(None, electional._classify_new_electional_round, req.query)
        if is_new is True:
            tool_arg = current_turn
        elif is_new is False:
            round_texts = await loop.run_in_executor(
                None, electional._collect_current_round_texts, prior_texts
            )
            tool_arg = "\n".join(filter(None, round_texts + [current_turn]))
        else:
            _, missing_current = await loop.run_in_executor(None, astro._extract_fields, current_turn)
            if not missing_current:
                tool_arg = current_turn
            else:
                immediate_prior = prior_texts[-1] if prior_texts else ""
                narrow_arg = "\n".join(filter(None, [current_turn, immediate_prior]))
                _, missing_narrow = await loop.run_in_executor(None, astro._extract_fields, narrow_arg)
                if not missing_narrow:
                    tool_arg = narrow_arg
                # else: keep the full union already built above as a last resort.

        # User-requested improvement: if the querent's OWN natal chart was
        # built earlier in this same conversation (e.g. an astro_natal_chart
        # request before this election), utils/electional.py can also check
        # real transits from each candidate moment to that person's own
        # natal Sun/Moon/Ascendant, on top of the generic significators
        # every election already checks. That lookup needs the FULL prior
        # conversation (an earlier natal-chart round is a round boundary
        # for _collect_current_round_texts above, so it's excluded from
        # tool_arg by design) — appended here, past a clearly-delimited
        # marker, rather than widening the round-scoped tool_arg itself
        # (see utils/electional.py's HISTORY_MARKER for why). A no-op when
        # there's no prior conversation at all (the common case for a
        # first-message election).
        if prior_texts:
            tool_arg = tool_arg + electional.HISTORY_MARKER + "\n".join(prior_texts)

    # Synastry-only hybrid extraction: try the plain deterministic
    # heuristic first (astro.run_synastry's own default), and only fall
    # back to a narrow, single-purpose LLM call when that heuristic left
    # required fields missing for either person — a real, reported
    # limitation of the pure date-position heuristic (free-form phrasing
    # it doesn't expect keeps finding new ways to break a purely
    # positional split). The LLM's only job is deciding which words
    # belong to which person (utils.intent.split_two_person_text_async
    # quotes the original text back verbatim, never reformats a date/
    # time/coordinate/city itself) — see astro._extract_two_person_fields'
    # own docstring for why this stays safe. Every OTHER tool (natal,
    # transit, datetime, calculator, ...) is untouched by this branch and
    # keeps using the generic tool_spec["run"] dispatch below.
    split_hint = None
    chart_subject: Any = None
    chart_second: Any = None
    chart_highlight_house: Optional[int] = None

    if decision.tool_name == "astro_synastry_chart":
        heuristic_missing = await loop.run_in_executor(
            None, astro.synastry_fields_missing, tool_arg
        )
        if heuristic_missing:
            split_hint = await intent.split_two_person_text_async(tool_arg)
            print(
                f"[tool_request] synastry heuristic split incomplete "
                f"(missing={heuristic_missing!r}); LLM segmentation hint: "
                f"{split_hint!r}"
            )
        # Single computation for both the text reply and the chart's two
        # subjects (astro.run_synastry_and_subject) — no longer a second,
        # separate rebuild of both people's charts just to get something
        # to draw (see _SIMPLE_AND_SUBJECT_FUNCS' own comment on why that
        # doubling was a real performance regression for every technique
        # it applied to, synastry included).
        tool_result, chart_subject, chart_second = await loop.run_in_executor(
            None, lambda: astro.run_synastry_and_subject(tool_arg, split_hint=split_hint)
        )
    elif decision.tool_name == "astro_electional_chart":
        # Reuses the ONE expensive range-search call for both tool_result
        # and the winning candidate's subject, instead of re-running the
        # search — see electional.run_electional_chart_and_subject.
        tool_result, chart_subject = await loop.run_in_executor(
            None, electional.run_electional_chart_and_subject, tool_arg
        )
    elif decision.tool_name == "astro_rectification_trutine":
        tool_result, chart_subject = await loop.run_in_executor(
            None, rectification._run_rectification_trutine_full, tool_arg
        )
    elif decision.tool_name == "astro_rectification_events":
        tool_result, chart_subject = await rectification_events.run_rectification_events_and_subject_async(
            tool_arg
        )
    elif decision.tool_name in _SIMPLE_AND_SUBJECT_FUNCS:
        # The remaining 8 "cheap" techniques — one call does double duty
        # (text reply + chart subject), see _SIMPLE_AND_SUBJECT_FUNCS' own
        # comment for why this replaced the old tool_spec["run"] + separate
        # get_*_chart_subject(s) two-call pattern.
        func = _SIMPLE_AND_SUBJECT_FUNCS[decision.tool_name]
        tool_result, chart_subject, chart_second, chart_highlight_house = await loop.run_in_executor(
            None, func, tool_arg
        )
    else:
        tool_result = await loop.run_in_executor(None, tool_spec["run"], tool_arg)
    # Same rationale as the [tool_router] print above: the model's final
    # answer is a paraphrase of tool_result, one more LLM call removed from
    # what the tool actually computed — printing the raw result here is
    # what makes "why did it say X" diagnosable at all.
    print(f"[tool_request] {decision.tool_name} raw result: {tool_result!r}")

    if decision.tool_name in _NO_FOLLOWUP_TOOL_NAMES and not config.RECTIFICATION_LLM_FOLLOWUP:
        # See _NO_FOLLOWUP_TOOL_NAMES's own comment for the full rationale
        # (three consecutive real tests, four mitigation layers, still
        # contradicted) and utils/config.py's RECTIFICATION_LLM_FOLLOWUP
        # comment for how to turn this back on for a more capable model.
        # astro_horary_question is NOT gated by this branch at all (see
        # _INTERPRETED_TOOL_NAMES' own comment on it) — it always reaches
        # the RAG-augmented follow-up below instead.
        # tool_result already IS the deterministic report,
        # written for a human reader by rectification.py/rectification_
        # events.py (Russian prose, headers, per-candidate breakdowns) —
        # no further LLM call, no RAG retrieval, nothing left to
        # potentially disagree with it. The only touch here is bolding the
        # best-candidate line(s) so a human skimming a long report can
        # still find them at a glance.
        resp_text = str(tool_result)
        extractor = _BEST_RECOMMENDATION_EXTRACTORS.get(decision.tool_name)
        if extractor:
            best_line = extractor(resp_text)
            if best_line:
                resp_text = resp_text.replace(best_line, f"**{best_line}**")

        responded_at = time.time()
        thinking_ms = int((responded_at - sent_at) * 1000)
        assistant_msg_id = repository.add_message(
            conversation_id, "assistant", resp_text, responded_at, thinking_ms
        )
        chart_files = await _attach_chart_if_applicable(
            loop, assistant_msg_id, decision.tool_name, req.query, tool_arg,
            chart_subject, chart_second, chart_highlight_house,
        )
        repository.touch_conversation(conversation_id)
        return {
            "conversation_id": conversation_id,
            "query": req.query,
            "sent_at": int(sent_at * 1000),
            "response": resp_text,
            "responded_at": int(responded_at * 1000),
            "thinking_ms": thinking_ms,
            "status": "complete",
            "contexts_used": 0,
            "files": chart_files,
            "tool_used": decision.tool_name,
            "message_id": assistant_msg_id,
        }

    if (
        decision.tool_name in _INTERPRETED_TOOL_NAMES
        and rag_utils.is_available()
        and not astro.is_error_result(tool_result)
    ):
        # is_error_result guards against retrieving and injecting the full
        # methodology/context for a placeholder like "не хватает данных" —
        # there's no chart data to interpret yet, so this would just be
        # wasted (and, on a small context window, potentially
        # budget-breaking) work for a question that isn't ready to be
        # answered yet anyway.
        #
        # topic_hint=_TOOL_TOPIC.get(...) guarantees THIS tool's own
        # methodology is included regardless of whether req.query (the
        # user's free-text wording) happened to score a similarity hit
        # against it — see retrieve_context's own docstring and
        # _TOOL_TOPIC's comment above for why that guarantee is needed,
        # not just nice-to-have.
        rag_contexts = rag_utils.retrieve_context(
            req.query, topic_hint=_TOOL_TOPIC.get(decision.tool_name)
        )
        if decision.tool_name == "astro_help_assistant":
            # astro_help_assistant has no specific person's chart behind
            # it at all — tool_result is utils.tools.astro_help_overview's
            # fixed technique cheat-sheet, reference material for ANY
            # user, not a computed fact about the current question. The
            # generic wording above ("точная...карта конкретного
            # человека...не утверждать, что этих данных не хватает") would
            # be actively misleading here (there's no missing-data case to
            # deny, and no specific chart to insist is real) — a distinct,
            # accurately-scoped chunk instead.
            computed_chunk = {
                "text": (
                    "СПРАВОЧНЫЙ МАТЕРИАЛ О МЕТОДИКАХ ПРИЛОЖЕНИЯ (общий "
                    "обзор для любого пользователя, не данные конкретного "
                    "человека):\n"
                    f"{tool_result}"
                ),
                "topic": "astrology",
                "always_include": True,
            }
        else:
            computed_chunk = {
                "text": (
                    "ДАННЫЕ ДЛЯ ЭТОГО ЗАПРОСА (уже вычислены и предоставлены — "
                    "это не пример и не общий случай, а точная натальная/"
                    "транзитная/прогрессивная/синастрическая карта конкретного "
                    "человека (или двух людей) из вопроса пользователя; не "
                    "пересчитывать, не менять и не утверждать, что этих данных "
                    "не хватает или что они не были даны):\n"
                    f"{tool_result}"
                ),
                "topic": "astrology",
                # Always included regardless of retrieval ranking, for two
                # reasons: the model needs the actual computed data no matter
                # what, and marking it always_include=True guarantees
                # build_prompt's step-by-step reasoning-mode prompt activates
                # for every astro answer, even on a run where no real
                # methodology chunk happened to rank in the top-k on its own.
                "always_include": True,
            }

        # Multi-stage RAG: rank the chart's own significant facts/points,
        # run a targeted retrieval query per one, and have the model
        # "digest" those raw fragments into already-reasoned notes in one
        # extra call, before the final answer call below even starts —
        # see utils/interpret.py's module docstring for the full rationale
        # (a plain top-k search against the user's free-text question
        # essentially never surfaces reference material organized by
        # specific placement/aspect, however large that reference corpus
        # is). Natal, transit, and synastry charts each get their own
        # profile extractor (astro.get_planet_profiles /
        # astro.get_transit_profiles / astro.get_synastry_profiles — the
        # latter two read a point's house against a DIFFERENT chart's
        # cusps than its own, via astro._house_of_degree) and their own
        # digest framing (interpret.digest_facts_async's chart_kind
        # parameter) — each reads its aspects with a different meaning
        # (permanent character vs. current activation/timing vs.
        # relationship dynamics between two people), so none of the three
        # can share one digest prompt.
        digested = ""
        name_a = name_b = ""
        if decision.tool_name == "astro_natal_chart":
            profiles = astro.get_planet_profiles(tool_arg)
            digested = await interpret.digest_facts_async(profiles)
        elif decision.tool_name == "astro_transit_chart":
            profiles = astro.get_transit_profiles(tool_arg)
            digested = await interpret.digest_facts_async(profiles, chart_kind="transit")
        elif decision.tool_name == "astro_progression_chart":
            profiles = astro.get_progression_profiles(tool_arg)
            digested = await interpret.digest_facts_async(profiles, chart_kind="progression")
        elif decision.tool_name == "astro_direction_chart":
            profiles = astro.get_direction_profiles(tool_arg)
            digested = await interpret.digest_facts_async(profiles, chart_kind="direction")
        elif decision.tool_name == "astro_lunar_return_chart":
            profiles = astro.get_lunar_return_profiles(tool_arg)
            digested = await interpret.digest_facts_async(profiles, chart_kind="lunar_return")
        elif decision.tool_name == "astro_solar_return_chart":
            profiles = astro.get_solar_return_profiles(tool_arg)
            digested = await interpret.digest_facts_async(profiles, chart_kind="solar_return")
        elif decision.tool_name == "astro_profection_chart":
            profiles = astro.get_profection_profiles(tool_arg)
            digested = await interpret.digest_facts_async(profiles, chart_kind="profection")
        elif decision.tool_name == "astro_synastry_chart":
            # Both people's profiles are digested together in one combined
            # list — each profile's own "text" already names which person
            # it belongs to (see astro.get_synastry_profiles), so the
            # digest prompt can tell them apart without needing two
            # separate LLM calls. split_hint (computed above, possibly
            # None) is passed through so the profiles reflect the exact
            # same person-A/person-B split that produced tool_result —
            # without this they'd redo the plain heuristic split
            # independently and could come back empty even when the LLM
            # hint just fixed tool_result above.
            profiles_a, profiles_b, name_a, name_b = astro.get_synastry_profiles(
                tool_arg, split_hint=split_hint
            )
            digested = await interpret.digest_facts_async(
                profiles_a + profiles_b, chart_kind="synastry"
            )

        if digested and decision.tool_name == "astro_natal_chart":
            # Sectioned prompt instead of build_prompt's generic reasoning-
            # mode template: repeated real testing showed the generic
            # template's final answer collapsing to one short paragraph no
            # matter how strongly it was told to elaborate. Only used when
            # a digest actually succeeded — the digest step is what supplies
            # the already-reasoned material each section weaves together;
            # without it (digest failed) fall back to the plain generic
            # prompt.
            followup_prompt = interpret.build_sectioned_answer_prompt(
                req.query, str(tool_result), digested, rag_contexts
            )
        elif digested and decision.tool_name == "astro_transit_chart":
            # Same mechanism, transit-specific section list/framing (see
            # interpret.build_transit_answer_prompt) — brings transit
            # answers up to the same quality bar natal charts already
            # have, instead of the generic reasoning-mode fallback below.
            followup_prompt = interpret.build_transit_answer_prompt(
                req.query, str(tool_result), digested, rag_contexts
            )
        elif digested and decision.tool_name == "astro_progression_chart":
            # Same mechanism again, progression-specific section list/
            # framing (see interpret.build_progression_answer_prompt) —
            # reframed around a slow, decades-long unfolding instead of
            # transit's short current-period timescale.
            followup_prompt = interpret.build_progression_answer_prompt(
                req.query, str(tool_result), digested, rag_contexts
            )
        elif digested and decision.tool_name == "astro_direction_chart":
            # Same mechanism again, direction-specific section list/
            # framing (see interpret.build_direction_answer_prompt) —
            # every point moves by the SAME solar arc, unlike
            # progression's per-point speeds.
            followup_prompt = interpret.build_direction_answer_prompt(
                req.query, str(tool_result), digested, rag_contexts
            )
        elif digested and decision.tool_name == "astro_lunar_return_chart":
            # Same mechanism again (see interpret.build_lunar_return_
            # answer_prompt) — a real independent monthly return chart,
            # read both on its own terms and via aspects to natal.
            followup_prompt = interpret.build_lunar_return_answer_prompt(
                req.query, str(tool_result), digested, rag_contexts
            )
        elif digested and decision.tool_name == "astro_solar_return_chart":
            # Same mechanism again (see interpret.build_solar_return_
            # answer_prompt) — annual counterpart to the lunar return.
            followup_prompt = interpret.build_solar_return_answer_prompt(
                req.query, str(tool_result), digested, rag_contexts
            )
        elif digested and decision.tool_name == "astro_profection_chart":
            # Same mechanism again (see interpret.build_profection_
            # answer_prompt) — deliberately short section list, since
            # astro.get_profection_profiles only ever returns two profiles
            # (the year/ruler summary fact plus the ruler's own natal
            # profile), not a whole chart's worth of points.
            followup_prompt = interpret.build_profection_answer_prompt(
                req.query, str(tool_result), digested, rag_contexts
            )
        elif digested and decision.tool_name == "astro_synastry_chart":
            # Same mechanism again, relationship-framed section list (see
            # interpret.build_synastry_answer_prompt) — name_a/name_b let
            # the prompt tell the model which generic label ("Человек A"/
            # "Человек B") corresponds to which person, and to prefer the
            # user's own wording if they named both people by name.
            followup_prompt = interpret.build_synastry_answer_prompt(
                req.query, str(tool_result), digested, rag_contexts, name_a, name_b
            )
        elif decision.tool_name == "astro_help_assistant":
            # NOT rag_utils.build_prompt below — a real, reported failure:
            # asked "На каком материке расположен Кейптаун?" through the
            # help-mode toggle, the model answered with a rundown of
            # astrology techniques instead of the actual geography
            # question. Root cause: build_prompt's reasoning-mode template
            # opens by asserting its context IS relevant to the question
            # and tells the model to enumerate "the specific facts from
            # the context that matter" — true by construction for every
            # OTHER _INTERPRETED_TOOL_NAMES entry (their computed_chunk is
            # always this exact person's real chart), but only sometimes
            # true here, since this tool must also handle a question with
            # nothing to do with astrology at all (see astro_help_
            # methodology.txt). interpret.build_help_answer_prompt makes
            # that relevance check an explicit first step instead of an
            # assumed premise — see its own docstring for the full story.
            followup_prompt = interpret.build_help_answer_prompt(
                req.query, computed_chunk["text"], rag_contexts
            )
        else:
            # Order matters: raw computed data first, then general
            # retrieved/methodology context — a real failure was observed
            # where the model's "Рассуждение:" correctly used the raw data
            # but "Ответ:" then claimed no data was given; putting the
            # actual person's data immediately after "Context:", ahead of
            # generic reference material, plus the strengthened wording
            # above and build_prompt's explicit consistency instruction,
            # are the mitigations for that. (No digested-notes chunk here:
            # this branch is only reached when the digest step above
            # failed — either way there's nothing digested to include.)
            followup_prompt = rag_utils.build_prompt(
                req.query, [computed_chunk] + rag_contexts
            )
        # Interpretation benefits from a bit more creative latitude than
        # the default chat temperature, and from not being nudged toward
        # brevity the way the generic tool prompt below is ("...concisely").
        default_temperature = 0.5
    else:
        followup_prompt = (
            f'The user asked: "{req.query}"\n'
            f"Tool used: {decision.tool_name}\n"
            f"Tool result: {tool_result}\n\n"
            "Using this result, answer the user's question naturally and "
            "concisely, in the same language the user wrote in. Don't mention "
            "that a tool was used unless it's relevant to the answer."
        )
        default_temperature = 0.7

    gen_start = time.time()
    try:
        resp_text = await llm_utils.generate_async(
            followup_prompt, max_tokens=req.max_tokens, temperature=req.temperature or default_temperature
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {e}")

    # Reachable for the two rectification tools only when
    # config.RECTIFICATION_LLM_FOLLOWUP is turned on (see
    # _NO_FOLLOWUP_TOOL_NAMES's own comment), and unconditionally for
    # astro_horary_question (which has no no-followup bypass at all — see
    # _INTERPRETED_TOOL_NAMES' own comment on it). Restores the exact
    # prepend/disclaimer/bookend safety net from tasks #189/#192/#193 for
    # whichever tool this ends up being: even a more capable model is
    # worth double-checking against the same "small model contradicts its
    # own tool's computed result" failure mode before trusting it
    # unconditionally, and this costs nothing when it isn't needed.
    extractor = _BEST_RECOMMENDATION_EXTRACTORS.get(decision.tool_name)
    if extractor:
        best_line = extractor(str(tool_result))
        if best_line:
            resp_text = (
                f"**{best_line}**\n"
                "_(это точный вычисленный результат; если рассуждение ниже "
                "почему-то называет другой ответ как верный — доверяй "
                "именно этой строке, а не рассуждению под ней)_\n\n"
                f"{resp_text}\n\n"
                "---\n"
                "**Напоминание** (если рассуждение выше в итоге назвало "
                "другой ответ — это ошибка модели, а не пересчёт; верен "
                f"именно вычисленный результат):\n**{best_line}**"
            )

    responded_at = time.time()
    thinking_ms = int((responded_at - gen_start) * 1000)

    assistant_msg_id = repository.add_message(
        conversation_id, "assistant", resp_text, responded_at, thinking_ms
    )
    chart_files = await _attach_chart_if_applicable(
        loop, assistant_msg_id, decision.tool_name, req.query, tool_arg,
        chart_subject, chart_second, chart_highlight_house,
    )
    repository.touch_conversation(conversation_id)

    return {
        "conversation_id": conversation_id,
        "query": req.query,
        "sent_at": int(sent_at * 1000),
        "response": resp_text,
        "responded_at": int(responded_at * 1000),
        "thinking_ms": thinking_ms,
        "status": "complete",
        "contexts_used": 0,
        "files": chart_files,
        "tool_used": decision.tool_name,
        "message_id": assistant_msg_id,
    }


async def _handle_image_request(conversation_id: int, query: str, sent_at: float) -> dict:
    """sent_at: see _handle_image_question's own docstring — the real
    moment the user's message was received, captured before the
    is_image_request_async classification call that runs ahead of this
    function, used as this placeholder's created_at instead of a fresh
    (and potentially much later) time.time()."""
    loop = asyncio.get_running_loop()
    try:
        job_id = await loop.run_in_executor(None, image_client.submit_job, query)
    except image_client.ImageServiceError as e:
        raise HTTPException(status_code=502, detail=f"Сервис изображений недоступен: {e}")

    placeholder_text = "Генерирую изображение… это может занять до нескольких десятков минут."
    assistant_msg_id = repository.add_message(
        conversation_id,
        "assistant",
        placeholder_text,
        sent_at,
        status="pending",
        image_job_id=job_id,
    )
    repository.touch_conversation(conversation_id)

    return {
        "conversation_id": conversation_id,
        "query": query,
        "sent_at": int(sent_at * 1000),
        "response": placeholder_text,
        "responded_at": int(sent_at * 1000),
        "thinking_ms": None,
        "status": "pending",
        "message_id": assistant_msg_id,
        "contexts_used": 0,
        "files": [],
    }


async def _handle_image_edit_request(
    conversation_id: int, req: ChatRequest, image_bytes: bytes, sent_at: float
) -> dict:
    """Submits an img2img job to ycplt_img using the attached image as the
    starting point.

    Plain img2img has no way to execute a "remove X" instruction — it just
    partially re-renders the whole image guided by the prompt text, so the
    named object doesn't actually disappear, it just gets restyled (see the
    garbled results before this was added). For that specific case,
    utils/intent.get_removal_target_async extracts the object being
    removed (translated to English) and passes it along as
    remove_target; ycplt_img uses it to automatically segment and inpaint
    just that region instead (see its README "Removing a named object").
    Any other kind of edit (color, style, additions, ...) leaves
    remove_target unset and goes through plain img2img as before.

    sent_at: see _handle_image_question's own docstring — captured before
    is_edit_instruction_async AND get_removal_target_async below (two
    classification calls, both real model generations), used as this
    placeholder's created_at so the displayed "sent" time reflects when
    the user actually sent the message, not when these calls happened to
    finish.
    """
    loop = asyncio.get_running_loop()
    strength = req.strength if req.strength is not None else DEFAULT_EDIT_STRENGTH
    remove_target = await intent.get_removal_target_async(req.query)
    try:
        job_id = await loop.run_in_executor(
            None,
            lambda: image_client.submit_job(
                req.query,
                mode="img2img",
                strength=strength,
                init_image=image_bytes,
                remove_target=remove_target,
            ),
        )
    except image_client.ImageServiceError as e:
        raise HTTPException(status_code=502, detail=f"Сервис изображений недоступен: {e}")

    placeholder_text = "Редактирую изображение… это может занять до нескольких десятков минут."
    assistant_msg_id = repository.add_message(
        conversation_id,
        "assistant",
        placeholder_text,
        sent_at,
        status="pending",
        image_job_id=job_id,
    )
    repository.touch_conversation(conversation_id)

    return {
        "conversation_id": conversation_id,
        "query": req.query,
        "sent_at": int(sent_at * 1000),
        "response": placeholder_text,
        "responded_at": int(sent_at * 1000),
        "thinking_ms": None,
        "status": "pending",
        "message_id": assistant_msg_id,
        "contexts_used": 0,
        "files": [],
    }


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": llm_utils.get_llm() is not None,
        "rag_index": rag_utils.is_available(),
        "image_service_url": config.IMAGE_SERVICE_URL,
        "astro_engine": astro.status(),
    }
