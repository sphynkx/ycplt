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
"""
import asyncio
from typing import Optional

from db import repository
from utils import config
from utils import image_client

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
        try:
            status = await loop.run_in_executor(None, image_client.get_status, job_id)
        except image_client.ImageServiceError as e:
            print(f"[image_jobs] status check failed for job {job_id}: {e}")
            continue

        state = status.get("status")
        is_caption = status.get("mode") == "caption"
        if state == "done":
            if is_caption:
                await _resolve_caption_done(message_id, job_id, status, loop)
            else:
                await _resolve_done(message_id, job_id, loop)
        elif state == "error":
            error_text = status.get("error_message") or "unknown error"
            label = "Ошибка распознавания изображения" if is_caption else "Ошибка генерации изображения"
            repository.fail_image_message(message_id, f"{label}: {error_text}")
            await loop.run_in_executor(None, image_client.delete_job, job_id)
        # queued / processing: nothing to do yet, check again next tick


async def _resolve_done(message_id: int, job_id: int, loop: asyncio.AbstractEventLoop) -> None:
    try:
        image_bytes = await loop.run_in_executor(None, image_client.get_result, job_id)
    except image_client.ImageServiceError as e:
        print(f"[image_jobs] failed to fetch result for job {job_id}: {e}")
        return

    filename = f"image_{job_id}.png"
    repository.add_file(message_id, filename, "image/png", image_bytes)
    repository.complete_image_message(message_id, "Готово!")
    await loop.run_in_executor(None, image_client.delete_job, job_id)


async def _resolve_caption_done(
    message_id: int, job_id: int, status: dict, loop: asyncio.AbstractEventLoop
) -> None:
    """mode="caption": the answer is text, already included in the status
    response as result_text (see ycplt_img's db.get_job_status) — no
    separate result download and no file attachment, unlike every other
    job mode."""
    text = (status.get("result_text") or "").strip() or "(пустой ответ)"
    repository.complete_image_message(message_id, text)
    await loop.run_in_executor(None, image_client.delete_job, job_id)
