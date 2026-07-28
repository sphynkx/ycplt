"""Conversation routes: list, create, message history, delete.

Used by the browser sidebar to switch between parallel chats and to restore
history on page reload.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db import repository

router = APIRouter(prefix="/api/conversations")


class CreateConversationRequest(BaseModel):
    title: Optional[str] = "Новый чат"


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


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: int):
    repository.delete_conversation(conversation_id)
    return {"status": "ok"}
