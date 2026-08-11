"""Exports a single chat message (text + its image attachments) as a PDF
— see utils/pdf_export.py for the actual rendering. One message per PDF
by design: a whole-conversation export was considered but a single
message (the user's own explicit ask: "the button on each reply") is a
much simpler, always-fresh unit — no risk of a stale export if the
conversation keeps growing after the button was clicked."""
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from db import repository
from utils import pdf_export

router = APIRouter(prefix="/api/messages")

_ROLE_LABEL_RU = {"user": "Пользователь", "assistant": "Ассистент"}


@router.get("/{message_id}/pdf")
async def export_message_pdf(message_id: int):
    message = repository.get_message(message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="Сообщение не найдено")

    file_stubs = repository.list_files_for_messages([message_id]).get(message_id, [])
    images = []
    for stub in file_stubs:
        if not stub["mime_type"].startswith("image/"):
            continue
        full = repository.get_file(stub["id"])
        if full:
            images.append((full["content"], full["mime_type"]))

    role_label = _ROLE_LABEL_RU.get(message["role"], message["role"])
    try:
        when = time.strftime("%d.%m.%Y %H:%M", time.localtime(message["created_at"]))
    except Exception:
        when = ""
    meta_line = f"{role_label} · {when}" if when else role_label

    try:
        pdf_bytes = pdf_export.build_message_pdf(message["content"] or "", images, meta_line)
    except Exception as e:
        # Most likely cause: weasyprint's system libraries (Pango/cairo/
        # gdk-pixbuf) aren't installed on this host — see
        # install/requirements.txt's own comment on the weasyprint entry.
        raise HTTPException(status_code=500, detail=f"Не удалось собрать PDF: {e}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        # "inline" (not "attachment") so opening the URL in a new tab
        # shows the PDF in the browser's own viewer — same reasoning as
        # routes/files.py's image handling — while the viewer's own
        # download button still lets the person save it.
        headers={"Content-Disposition": f'inline; filename="message_{message_id}.pdf"'},
    )
