"""Downloading files extracted from model replies (code, generated images,
etc.) — see utils/codeblocks.py and routes/chat.py."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from db import repository

router = APIRouter(prefix="/api/files")


@router.get("/{file_id}")
async def download_file(file_id: int):
    f = repository.get_file(file_id)
    if f is None:
        raise HTTPException(status_code=404, detail="Файл не найден")
    # "attachment" forces a download unconditionally, even when the
    # browser is just displaying an <img> or the user explicitly opened
    # the file's own URL in a new tab to view it — the inline <img> tag
    # already worked before (browsers render an <img> resource inline
    # regardless of this header), but a real, reported gap was that
    # opening the SAME chart image in its own tab/window (e.g. via
    # target="_blank") triggered a download instead of showing it.
    # "inline" for image/* fixes that while leaving every other
    # attachment (extracted code files, etc.) downloading as before —
    # those have no in-browser viewer worth showing inline anyway.
    disposition = "inline" if f["mime_type"].startswith("image/") else "attachment"
    return Response(
        content=f["content"],
        media_type=f["mime_type"],
        headers={"Content-Disposition": f'{disposition}; filename="{f["filename"]}"'},
    )
