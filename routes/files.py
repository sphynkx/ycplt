"""Downloading files extracted from model replies (code, generated images,
etc.) — see utils/codeblocks.py and routes/chat.py."""
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from db import repository

router = APIRouter(prefix="/api/files")


def _content_disposition(disposition: str, filename: str) -> str:
    """Builds a Content-Disposition header value that's always safe to
    send — real, reported crash: HTTP header values must be latin-1
    encodable, and a filename containing non-Latin-1 characters (e.g. a
    Cyrillic original upload name like "без названия.jpg" — see
    routes/chat.py's ChatRequest.image_filename, which stores whatever
    name the browser's file picker reported) made Starlette's Response
    raise UnicodeEncodeError while building the response, turning a
    perfectly fine file into a 500 for the single specific attachment
    that happened to have a non-ASCII name — every other download
    (ASCII-named uploads, generated charts, extracted code files) worked
    fine, which is why this went unnoticed until a real Cyrillic filename
    was downloaded.

    Sends both forms: a plain ASCII-sanitized filename="..." (non-ASCII
    bytes replaced with "?") for any client that only understands the
    old-style parameter, and the RFC 5987 filename*=UTF-8''... form (percent-
    encoded UTF-8, itself pure ASCII so it can never trigger this same
    crash) that every modern browser prefers and will display with the
    real name intact.
    """
    ascii_fallback = filename.encode("ascii", errors="replace").decode("ascii")
    encoded = quote(filename, safe="")
    return f"{disposition}; filename=\"{ascii_fallback}\"; filename*=UTF-8''{encoded}"


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
        headers={"Content-Disposition": _content_disposition(disposition, f["filename"])},
    )
