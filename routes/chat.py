"""Chat routes: POST /chat (generates a reply, persisted to the DB) and GET /health.

Every /chat call:
  1. Creates a conversation if conversation_id isn't given (new chat).
  2. Saves the user's message with its send time. If an image was attached
     (ChatRequest.image_data), it's decoded and stored as a file attachment
     on this same user message (so it shows up on reload, same as any other
     image file — see db/repository.py add_file).
  3. If an image was attached, unconditionally treats the message as a
     request to edit that image (an attachment is an unambiguous signal —
     there's no captioning model yet to otherwise interpret "why did you
     send me this picture") and routes to _handle_image_edit_request,
     skipping steps 4-5 below.
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
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db import repository
from utils import config
from utils import image_client
from utils import intent
from utils import llm as llm_utils
from utils import rag as rag_utils
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
        return await _handle_image_edit_request(conversation_id, req, image_bytes)

    if await intent.is_image_request_async(req.query):
        return await _handle_image_request(conversation_id, req.query)

    tool_decision = await tool_router.classify_async(req.query)
    if tool_decision.tool_name:
        return await _handle_tool_request(conversation_id, req, sent_at, tool_decision)

    return await _handle_chat_request(conversation_id, req, sent_at)


async def _handle_chat_request(conversation_id: int, req: ChatRequest, sent_at: float) -> dict:
    contexts = []
    if req.use_rag and rag_utils.is_available():
        contexts = rag_utils.retrieve_context(req.query, config.TOP_K)
    prompt = rag_utils.build_prompt(req.query, contexts)

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
    }


async def _handle_tool_request(
    conversation_id: int, req: ChatRequest, sent_at: float, decision: tool_router.ToolDecision
) -> dict:
    """Runs the tool utils/tool_router.py picked, then does one more (short)
    LLM call that turns the raw tool result into a natural-language answer
    to the user's original question. Two model calls total for this path
    (the router's classification, plus this one) — acceptable, since both
    are small compared to a full chat generation."""
    tool_spec = tools.TOOL_REGISTRY[decision.tool_name]
    loop = asyncio.get_running_loop()
    tool_result = await loop.run_in_executor(None, tool_spec["run"], decision.tool_arg)

    followup_prompt = (
        f'The user asked: "{req.query}"\n'
        f"Tool used: {decision.tool_name}\n"
        f"Tool result: {tool_result}\n\n"
        "Using this result, answer the user's question naturally and "
        "concisely, in the same language the user wrote in. Don't mention "
        "that a tool was used unless it's relevant to the answer."
    )

    gen_start = time.time()
    try:
        resp_text = await llm_utils.generate_async(
            followup_prompt, max_tokens=req.max_tokens, temperature=req.temperature or 0.7
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {e}")
    responded_at = time.time()
    thinking_ms = int((responded_at - gen_start) * 1000)

    assistant_msg_id = repository.add_message(
        conversation_id, "assistant", resp_text, responded_at, thinking_ms
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
        "files": [],
        "tool_used": decision.tool_name,
    }


async def _handle_image_request(conversation_id: int, query: str) -> dict:
    loop = asyncio.get_running_loop()
    try:
        job_id = await loop.run_in_executor(None, image_client.submit_job, query)
    except image_client.ImageServiceError as e:
        raise HTTPException(status_code=502, detail=f"Сервис изображений недоступен: {e}")

    placeholder_at = time.time()
    placeholder_text = "Генерирую изображение… это может занять до нескольких десятков минут."
    assistant_msg_id = repository.add_message(
        conversation_id,
        "assistant",
        placeholder_text,
        placeholder_at,
        status="pending",
        image_job_id=job_id,
    )
    repository.touch_conversation(conversation_id)

    return {
        "conversation_id": conversation_id,
        "query": query,
        "sent_at": int(placeholder_at * 1000),
        "response": placeholder_text,
        "responded_at": int(placeholder_at * 1000),
        "thinking_ms": None,
        "status": "pending",
        "message_id": assistant_msg_id,
        "contexts_used": 0,
        "files": [],
    }


async def _handle_image_edit_request(
    conversation_id: int, req: ChatRequest, image_bytes: bytes
) -> dict:
    """Submits an img2img job to ycplt_img using the attached image as the
    starting point. utils/image_client.submit_job already accepts mode/
    init_image/strength (mode="img2img" + an image encodes to
    init_image_b64 in the job payload) — see its docstring — so no client
    changes were needed for this.

    NOTE: as of this writing, ycplt_img itself only implements plain
    txt2img and ignores the extra job fields, so an edit job currently
    generates from the prompt alone rather than actually editing the
    image — see README's "API contract for ycplt_img (next phase)" section.
    This handler is the ycplt-side half of the feature; ycplt_img needs a
    matching update before edits behave as expected end to end.
    """
    loop = asyncio.get_running_loop()
    strength = req.strength if req.strength is not None else DEFAULT_EDIT_STRENGTH
    try:
        job_id = await loop.run_in_executor(
            None,
            lambda: image_client.submit_job(
                req.query, mode="img2img", strength=strength, init_image=image_bytes
            ),
        )
    except image_client.ImageServiceError as e:
        raise HTTPException(status_code=502, detail=f"Сервис изображений недоступен: {e}")

    placeholder_at = time.time()
    placeholder_text = "Редактирую изображение… это может занять до нескольких десятков минут."
    assistant_msg_id = repository.add_message(
        conversation_id,
        "assistant",
        placeholder_text,
        placeholder_at,
        status="pending",
        image_job_id=job_id,
    )
    repository.touch_conversation(conversation_id)

    return {
        "conversation_id": conversation_id,
        "query": req.query,
        "sent_at": int(placeholder_at * 1000),
        "response": placeholder_text,
        "responded_at": int(placeholder_at * 1000),
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
    }
