"""Birth-profile routes: list/get/edit/delete stored birth profiles, plus
import/export against the AstroZet (.zbs) file format — see
utils/astrozet.py for the format itself and db/connection.py's
birth_profiles table for how it's stored internally (our own choice,
independent of the .zbs wire format).

Deliberately API-only for now: there's no chat-UI integration for actually
*using* a saved profile inside a conversation yet — that UX (how a profile
would get referenced from a chat, e.g. a picker, a slash-command, a button
next to the composer) hasn't been designed. This only covers the bounded,
concrete ask: get birth data into this app from a real AstroZet .zbs file,
and back out again. A plain JSON text field is used for the .zbs content
(not multipart/form-data) to match this app's existing convention of doing
file handling client-side and sending plain/base64 text in the request body
(see routes/chat.py's own image_data field).
"""
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from db import repository
from utils import astrozet

router = APIRouter(prefix="/api/profiles")


class BirthProfileIn(BaseModel):
    name: str
    date: str            # 'YYYY-MM-DD'
    time: Optional[str] = "12:00"
    utc_offset: Optional[str] = None
    place: Optional[str] = None
    lat: float
    lon: float
    sex: Optional[str] = None
    comment: Optional[str] = None
    photo_path: Optional[str] = None


class BirthProfilePatch(BaseModel):
    name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    utc_offset: Optional[str] = None
    place: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    sex: Optional[str] = None
    comment: Optional[str] = None
    photo_path: Optional[str] = None


class ZbsImportRequest(BaseModel):
    content: str  # raw .zbs file text, read client-side (e.g. FileReader.readAsText())


def _ascii_filename_part(name: str) -> str:
    """Same reasoning as routes/conversations.py's _ascii_filename_part —
    Content-Disposition's plain filename= must be latin-1-encodable, which
    a Cyrillic name (the normal case here) never is."""
    cleaned = "".join(
        c if (c.isalnum() and ord(c) < 128) or c in " ._-" else "_" for c in name
    )
    return cleaned.strip()[:60] or "profile"


@router.get("")
async def list_profiles():
    return repository.list_birth_profiles()


@router.post("")
async def create_profile(req: BirthProfileIn):
    profile_id = repository.create_birth_profile(req.model_dump())
    return repository.get_birth_profile(profile_id)



# NOTE ON ROUTE ORDER: FastAPI/Starlette match routes in registration order,
# and a typed path converter (e.g. "/{profile_id}" with profile_id: int)
# that matches a literal segment like "/export" but fails to convert it
# returns a 422 immediately — it does NOT fall through to a later, more
# specific route. So "/import" and the bulk "/export" MUST be registered
# above "/{profile_id}" and its siblings below, or a request to
# GET /api/profiles/export would 422 instead of ever reaching
# export_all_zbs(). Confirmed with a real failing test during development.
@router.post("/import")
async def import_zbs(req: ZbsImportRequest):
    """Parses every record in a .zbs file's text and stores each as a new
    birth profile. Malformed individual lines don't abort the whole
    import — every line is attempted independently, and both the created
    profiles and any per-line errors are returned so the caller can see
    exactly what happened (a real interchange file with dozens of records
    and one typo shouldn't lose the other 40)."""
    parsed, parse_errors = astrozet.parse_zbs(req.content)
    if not parsed and not parse_errors:
        raise HTTPException(status_code=400, detail="Файл пуст или не содержит записей")

    created: List[Dict[str, Any]] = []
    for profile in parsed:
        profile_id = repository.create_birth_profile(profile)
        created.append(repository.get_birth_profile(profile_id))

    return {
        "imported": len(created),
        "profiles": created,
        "errors": [
            {"line": e.line_number, "raw": e.raw_line, "reason": e.reason}
            for e in parse_errors
        ],
    }


@router.get("/export")
async def export_all_zbs():
    """All stored profiles as one .zbs file, in the same order as
    list_profiles (name, case-insensitive)."""
    profiles = repository.list_birth_profiles()
    text = astrozet.export_zbs(profiles)
    return Response(
        content=text.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="birth_profiles.zbs"'},
    )


@router.get("/{profile_id}")
async def get_profile(profile_id: int):
    profile = repository.get_birth_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Профиль не найден")
    return profile


@router.patch("/{profile_id}")
async def patch_profile(profile_id: int, req: BirthProfilePatch):
    if repository.get_birth_profile(profile_id) is None:
        raise HTTPException(status_code=404, detail="Профиль не найден")
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    if fields:
        repository.update_birth_profile(profile_id, fields)
    return repository.get_birth_profile(profile_id)


@router.delete("/{profile_id}")
async def delete_profile(profile_id: int):
    repository.delete_birth_profile(profile_id)
    return {"status": "ok"}


@router.get("/{profile_id}/export")
async def export_profile_zbs(profile_id: int):
    """A single profile as a one-line .zbs file — for re-importing into
    AstroZet itself, or sharing one record without the rest of the list."""
    profile = repository.get_birth_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Профиль не найден")
    text = astrozet.export_zbs([profile])

    ascii_name = f"profile_{profile_id}_{_ascii_filename_part(profile['name'])}.zbs"
    utf8_name = f"profile_{profile_id}_{profile['name']}.zbs"
    return Response(
        content=text.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{ascii_name}"; '
                f"filename*=UTF-8''{quote(utf8_name)}"
            )
        },
    )
