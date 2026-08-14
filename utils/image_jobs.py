"""Background poller that resolves pending image jobs.

Runs as an asyncio task for the lifetime of the app process — independent of
any open browser tab, so image generation keeps going even if the browser is
closed; the result is simply there next time the conversation is opened.

For every message with status = 'pending' (see
db/repository.py:list_pending_image_messages), this checks the job's status
on ycplt_img (utils/image_client.py) on a fixed interval
(config.IMAGE_POLL_INTERVAL_SEC). GET /jobs/{id} always reports the job's
`mode`, which is what distinguishes how "done" is resolved — no separate
tracking is needed on this side for that:
  - done, mode="caption"     -> writes the text answer (result_text,
                                already included in the status response)
                                as the message content directly — no file
                                attachment, unlike every other mode.
  - done, any other mode     -> downloads the image, stores it as a file
                                attachment (mime_type image/png), marks the
                                message complete.
  - error                    -> marks the message as failed with the
                                remote error (phrased per-mode).
  - queued/processing        -> left alone, checked again next tick.
  - service unreachable      -> left pending, retried next tick.

Either way, once resolved (done or error) the job is acknowledged (DELETE)
so ycplt_img can drop it from its queue.

Duration ("thinking_ms"): a real, reported gap — image-job messages never
showed how long they took at all (unlike a normal chat reply's "думал
X.X с"), and their displayed "sent" timestamp could visibly jump forward
past when the user actually sent the request (routes/chat.py used to
stamp created_at only after intent classification + job submission
finished, not at the moment the message was received). Both are fixed
together: routes/chat.py now stores the message's created_at as the true
early sent_at, and this module computes thinking_ms as wall-clock time
from that same created_at to job completion — full "how long since you
hit send", the number a user actually wants for something that can take
tens of minutes, not just the image model's own generation time.
"""
import asyncio
import time
from typing import Optional

from db import repository
from utils import config
from utils import image_client
from utils import llm as llm_utils

_task: Optional[asyncio.Task] = None


def start_background_poller() -> None:
    """Starts the poller task once. Safe to call more than once (a second
    call is a no-op if the task is already running)."""
    global _task
    if _task is not None:
        return
    _task = asyncio.create_task(_poll_loop())


async def _poll_loop() -> None:
    while True:
        await asyncio.sleep(config.IMAGE_POLL_INTERVAL_SEC)
        try:
            await _poll_once()
        except Exception as e:
            print(f"[image_jobs] poll error: {e}")


async def _poll_once() -> None:
    pending = repository.list_pending_image_messages()
    if not pending:
        return

    loop = asyncio.get_running_loop()
    for msg in pending:
        job_id = msg["image_job_id"]
        message_id = msg["id"]
        # Still in seconds here (unlike repository.list_messages' own
        # created_at, which converts to ms for the frontend) — this is the
        # true moment the user's request was received (see routes/chat.py's
        # _handle_image_request/_handle_image_edit_request/_handle_image_
        # question, which now use chat()'s own early sent_at rather than a
        # fresh, much-later timestamp). Used below to compute a
        # thinking_ms-equivalent duration once the job resolves — full
        # wall-clock time since the request was sent, matching what a user
        # actually wants to know ("how long did this take"), not just the
        # image model's own generation time.
        created_at = msg["created_at"]
        try:
            status = await loop.run_in_executor(None, image_client.get_status, job_id)
        except image_client.ImageServiceError as e:
            print(f"[image_jobs] status check failed for job {job_id}: {e}")
            continue

        state = status.get("status")
        is_caption = status.get("mode") == "caption"
        if state == "done":
            if is_caption:
                await _resolve_caption_done(message_id, job_id, status, created_at, loop)
            else:
                await _resolve_done(message_id, job_id, created_at, loop)
        elif state == "error":
            error_text = status.get("error_message") or "unknown error"
            label = "Ошибка распознавания изображения" if is_caption else "Ошибка генерации изображения"
            repository.fail_image_message(message_id, f"{label}: {error_text}")
            await loop.run_in_executor(None, image_client.delete_job, job_id)
        # queued / processing: nothing to do yet, check again next tick


async def _resolve_done(
    message_id: int, job_id: int, created_at: float, loop: asyncio.AbstractEventLoop
) -> None:
    try:
        image_bytes = await loop.run_in_executor(None, image_client.get_result, job_id)
    except image_client.ImageServiceError as e:
        print(f"[image_jobs] failed to fetch result for job {job_id}: {e}")
        return

    thinking_ms = int((time.time() - created_at) * 1000)
    filename = f"image_{job_id}.png"
    repository.add_file(message_id, filename, "image/png", image_bytes)
    repository.complete_image_message(message_id, "Готово!", thinking_ms)
    await loop.run_in_executor(None, image_client.delete_job, job_id)


async def _resolve_caption_done(
    message_id: int, job_id: int, status: dict, created_at: float, loop: asyncio.AbstractEventLoop
) -> None:
    """mode="caption": the raw answer is already included in the status
    response as result_text (see ycplt_img's db.get_job_status) — no
    separate result download and no file attachment, unlike every other
    job mode.

    moondream2 (the vision model, hosted in ycplt_img) answers in English
    regardless of the question's language — it's a small, English-centric
    captioning model, not a general multilingual one. Rather than teach
    ycplt_img to translate (which would mean giving it its own LLM, when
    the main chat LLM already lives right here), a short follow-up
    generation on the main chat model rephrases the raw caption into a
    natural answer in the same language the user asked in — the same
    "tool result -> natural-language answer" pattern _handle_tool_request
    (routes/chat.py) already uses for datetime/calculator results. Falls
    back to the raw caption if this step fails for any reason, rather than
    failing the whole message over what's ultimately a phrasing nicety.
    """
    raw_caption = (status.get("result_text") or "").strip()
    if not raw_caption:
        text = "(пустой ответ)"
    else:
        question = status.get("prompt") or "Describe this image."
        followup_prompt = (
            f'The user asked about an attached image: "{question}"\n'
            f"Visual analysis of the image (from a vision model): {raw_caption}\n\n"
            "Using this analysis, answer the user's question naturally and "
            "concisely, in the same language the user wrote in. Don't "
            "mention that a separate analysis or tool was used."
        )
        try:
            text = await llm_utils.generate_async(followup_prompt, temperature=0.3)
        except Exception as e:
            print(f"[image_jobs] caption rephrase failed for job {job_id}: {e}")
            text = raw_caption

    thinking_ms = int((time.time() - created_at) * 1000)
    repository.complete_image_message(message_id, text, thinking_ms)
    await loop.run_in_executor(None, image_client.delete_job, job_id)
