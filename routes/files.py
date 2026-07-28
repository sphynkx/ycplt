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
    return Response(
        content=f["content"],
        media_type=f["mime_type"],
        headers={"Content-Disposition": f'attachment; filename="{f["filename"]}"'},
    )
