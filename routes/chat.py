"""Chat routes: POST /chat (generates a reply, persisted to the DB) and GET /health.

Every /chat call:
  1. Creates a conversation if conversation_id isn't given (new chat).
  2. Saves the user's message with its send time.
  3. Uses utils/intent.py to decide whether this is an image request. If so,
     submits a job to ycplt_img (utils/image_client.py) and stores a
     'pending' placeholder message instead of calling the chat LLM — the
     background poller (utils/image_jobs.py) resolves it later.
  4. Otherwise, generates a normal reply, measuring "thinking" time
     (thinking_ms).
  5. Saves the model's reply with its timestamp and thinking_ms.
  6. Extracts fenced code blocks from the reply as file attachments
     (utils/codeblocks.py) and stores them in the DB (db/repository.py add_file).
"""
import asyncio
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
from utils.codeblocks import extract_code_blocks

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[int] = None
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    use_rag: Optional[bool] = False


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

    sent_at = time.time()
    repository.add_message(conversation_id, "user", req.query, sent_at)

    if await intent.is_image_request_async(req.query):
        return await _handle_image_request(conversation_id, req.query)

    return await _handle_chat_request(conversation_id, req, sent_at)


async def _handle_chat_request(conversation_id: int, req: ChatRequest, sent_at: float) -> dict:
    contexts = []
    if req.use_rag and rag_utils.is_available():
        contexts = rag_utils.retrieve_context(req.query, config.TOP_K)
    prompt = rag_utils.build_prompt(req.query, contexts)

    gen_start = time.time()
    try:
        resp_text = await llm_utils.generate_async(
            prompt, max_tokens=req.max_tokens or 256, temperature=req.temperature or 0.7
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


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": llm_utils.get_llm() is not None,
        "rag_index": rag_utils.is_available(),
        "image_service_url": config.IMAGE_SERVICE_URL,
    }
