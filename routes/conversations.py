"""Conversation routes: list, create, message history, rename, delete,
export.

Used by the browser sidebar to switch between parallel chats, restore
history on page reload, and manage the accumulated chat list (rename,
download a full archive, delete).
"""
import io
import json
import zipfile
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from db import repository

router = APIRouter(prefix="/api/conversations")


class CreateConversationRequest(BaseModel):
    title: Optional[str] = "Новый чат"


class RenameConversationRequest(BaseModel):
    title: str


@router.get("")
async def list_conversations():
    return repository.list_conversations()


@router.post("")
async def create_conversation(req: CreateConversationRequest):
    conv_id = repository.create_conversation(req.title or "Новый чат")
    return repository.get_conversation(conv_id)


@router.get("/{conversation_id}/messages")
async def get_messages(conversation_id: int):
    if repository.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    return repository.list_messages(conversation_id)


@router.patch("/{conversation_id}")
async def rename_conversation(conversation_id: int, req: RenameConversationRequest):
    if repository.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Название не может быть пустым")
    repository.rename_conversation(conversation_id, title)
    return repository.get_conversation(conversation_id)


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: int):
    repository.delete_conversation(conversation_id)
    return {"status": "ok"}


def _ascii_filename_part(title: str) -> str:
    """ASCII-only fallback for the plain `filename=` parameter of
    Content-Disposition — that header value must be latin-1-encodable
    (Starlette raises UnicodeEncodeError otherwise), which a Cyrillic
    title (the normal case here — conversation titles default to Russian
    text) never is. The real, readable title is carried separately via
    `filename*=UTF-8''...` (see export_conversation below), which every
    current browser prefers over this fallback when present — this one
    only matters for very old user agents that ignore filename*."""
    cleaned = "".join(
        c if (c.isalnum() and ord(c) < 128) or c in " ._-" else "_" for c in title
    )
    return cleaned.strip()[:60] or "chat"


@router.get("/{conversation_id}/export")
async def export_conversation(conversation_id: int):
    """A full, self-contained archive of one conversation: conversation.json
    (metadata + every message, in order) plus every file attachment's raw
    bytes under files/, so the archive is restorable/inspectable without the
    app's own database. Message entries reference their attachments by
    archive_path rather than embedding bytes inline (e.g. as base64) — that
    keeps conversation.json plain, readable text even when a conversation
    has several images attached, at the (accepted) cost of the reader having
    to open two files to see a message with an image."""
    conv = repository.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Диалог не найден")

    messages = repository.list_messages(conversation_id)

    dump = {
        "id": conv["id"],
        "title": conv["title"],
        "created_at": conv["created_at"],
        "updated_at": conv["updated_at"],
        "messages": [],
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in messages:
            msg_entry = {
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "created_at": m["created_at"],
                "thinking_ms": m["thinking_ms"],
                "status": m["status"],
                "files": [],
            }
            for stub in m["files"]:
                full = repository.get_file(stub["id"])
                if full is None:
                    continue
                archive_path = f"files/{m['id']}_{full['id']}_{full['filename']}"
                zf.writestr(archive_path, full["content"])
                msg_entry["files"].append(
                    {
                        "filename": full["filename"],
                        "mime_type": full["mime_type"],
                        "size": full["size"],
                        "archive_path": archive_path,
                    }
                )
            dump["messages"].append(msg_entry)

        zf.writestr("conversation.json", json.dumps(dump, ensure_ascii=False, indent=2))

    buf.seek(0)
    ascii_name = f"chat_{conversation_id}_{_ascii_filename_part(conv['title'])}.zip"
    utf8_name = f"chat_{conversation_id}_{conv['title']}.zip"

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{ascii_name}"; '
                f"filename*=UTF-8''{quote(utf8_name)}"
            )
        },
    )
