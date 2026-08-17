"""CRUD operations on conversations, messages, and file attachments.

Time units:
  - in the DB, created_at is stored in seconds (time.time()), as usual for
    sqlite/Python;
  - list_messages() returns created_at already in milliseconds — for JS
    Date() on the frontend, and so the format matches what POST /chat
    returns (see routes/chat.py).

Message status:
  - 'complete' — a normal, finished message (the vast majority).
  - 'pending'  — an assistant placeholder for an image job still running on
    ycplt_img; image_job_id points at the remote job. Resolved by the
    background poller in utils/image_jobs.py.
  - 'error'    — an image job that failed remotely; content holds the error.
"""
import time
from typing import Any, Dict, List, Optional

from db.connection import get_conn


# ---------- Conversations ----------
def create_conversation(title: str = "Новый чат") -> int:
    now = time.time()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        return cur.lastrowid


def list_conversations() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_conversation(conversation_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        return dict(row) if row else None


def touch_conversation(conversation_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (time.time(), conversation_id)
        )


def rename_conversation(conversation_id: int, title: str) -> None:
    """Deliberately does not touch updated_at — a rename shouldn't jump the
    conversation to the top of the sidebar's most-recently-active ordering
    the way an actual new message does (see list_conversations' ORDER BY)."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id)
        )


def delete_conversation(conversation_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


# ---------- Messages ----------
def add_message(
    conversation_id: int,
    role: str,
    content: str,
    created_at: float,
    thinking_ms: Optional[int] = None,
    status: str = "complete",
    image_job_id: Optional[int] = None,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at, thinking_ms, "
            "status, image_job_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, role, content, created_at, thinking_ms, status, image_job_id),
        )
        return cur.lastrowid


def list_messages(conversation_id: int) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, role, content, created_at, thinking_ms, status, image_job_id "
            "FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()

    messages = [dict(r) for r in rows]
    for m in messages:
        m["created_at"] = int(m["created_at"] * 1000)  # seconds -> milliseconds

    files_map = list_files_for_messages([m["id"] for m in messages])
    for m in messages:
        m["files"] = files_map.get(m["id"], [])
    return messages


def get_message(message_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, conversation_id, role, content, created_at, thinking_ms, status, "
            "image_job_id FROM messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        return dict(row) if row else None


def list_pending_image_messages() -> List[Dict[str, Any]]:
    """All messages currently waiting on a ycplt_img job — what the
    background poller (utils/image_jobs.py) needs to check on each tick."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, conversation_id, image_job_id, created_at FROM messages "
            "WHERE status = 'pending' AND image_job_id IS NOT NULL ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]


def complete_image_message(message_id: int, content: str, thinking_ms: Optional[int] = None) -> None:
    """Marks a pending image message as done (image bytes are stored
    separately as a file attachment via add_file)."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE messages SET status = 'complete', content = ?, thinking_ms = ?, "
            "image_job_id = NULL WHERE id = ?",
            (content, thinking_ms, message_id),
        )


def fail_image_message(message_id: int, error_message: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE messages SET status = 'error', content = ?, image_job_id = NULL WHERE id = ?",
            (error_message, message_id),
        )


# ---------- File attachments ----------
def add_file(message_id: int, filename: str, mime_type: str, content: bytes) -> int:
    now = time.time()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO files (message_id, filename, mime_type, content, size, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, filename, mime_type, content, len(content), now),
        )
        return cur.lastrowid


def get_file(file_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, filename, mime_type, content, size FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
        return dict(row) if row else None


# ---------- Birth profiles (AstroZet .zbs import/export) ----------
# Storage shape is this app's own choice (see db/connection.py's
# birth_profiles table comment) — utils/astrozet.py is only the boundary
# format used to import/export these rows as .zbs text.
def create_birth_profile(profile: Dict[str, Any]) -> int:
    now = time.time()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO birth_profiles "
            "(name, date, time, utc_offset, place, lat, lon, sex, comment, "
            "photo_path, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                profile["name"],
                profile["date"],
                profile.get("time") or "12:00",
                profile.get("utc_offset"),
                profile.get("place"),
                profile["lat"],
                profile["lon"],
                profile.get("sex"),
                profile.get("comment"),
                profile.get("photo_path"),
                now,
                now,
            ),
        )
        return cur.lastrowid


def list_birth_profiles() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM birth_profiles ORDER BY name COLLATE NOCASE"
        ).fetchall()
        return [dict(r) for r in rows]


def get_birth_profile(profile_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM birth_profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        return dict(row) if row else None


def update_birth_profile(profile_id: int, fields: Dict[str, Any]) -> None:
    """Partial update — only columns present in `fields` are touched.
    `updated_at` is always bumped."""
    allowed = {
        "name", "date", "time", "utc_offset", "place", "lat", "lon",
        "sex", "comment", "photo_path",
    }
    columns = [k for k in fields if k in allowed]
    if not columns:
        return
    set_clause = ", ".join(f"{c} = ?" for c in columns)
    values = [fields[c] for c in columns]
    with get_conn() as conn:
        conn.execute(
            f"UPDATE birth_profiles SET {set_clause}, updated_at = ? WHERE id = ?",
            (*values, time.time(), profile_id),
        )


def delete_birth_profile(profile_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM birth_profiles WHERE id = ?", (profile_id,))


def list_files_for_messages(message_ids: List[int]) -> Dict[int, List[Dict[str, Any]]]:
    """{message_id: [{id, filename, mime_type, size}, ...]} — no BLOB content."""
    if not message_ids:
        return {}
    placeholders = ",".join("?" * len(message_ids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT id, message_id, filename, mime_type, size FROM files "
            f"WHERE message_id IN ({placeholders}) ORDER BY id",
            message_ids,
        ).fetchall()

    result: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        result.setdefault(r["message_id"], []).append(
            {
                "id": r["id"],
                "filename": r["filename"],
                "mime_type": r["mime_type"],
                "size": r["size"],
            }
        )
    return result
