"""HTTP client for ycplt_img (https://github.com/sphynkx/ycplt_img) — the
image generation service running on a separate machine.

Deliberately standard-library only (urllib), no requests/httpx — the client
is tiny (4 operations), an extra dependency isn't worth it.

The service is passive: submit_job() returns a job_id right away (generation
runs for minutes, nothing waits for it here), while get_status()/get_result()
are polled separately — normally from the background asyncio task in
utils/image_jobs.py, driven by routes/chat.py's intent detection (see
utils/intent.py) for whether a message is an image request.
"""
import base64
import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from utils import config


class ImageServiceError(Exception):
    """The service is unreachable or returned an error."""


def _request(method: str, path: str, body: Optional[dict] = None) -> Dict[str, Any]:
    url = f"{config.IMAGE_SERVICE_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=config.IMAGE_HTTP_TIMEOUT_SEC) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {"error": raw.decode("utf-8", errors="replace")}
        raise ImageServiceError(f"{e.code}: {payload.get('error', payload)}") from e
    except urllib.error.URLError as e:
        raise ImageServiceError(f"ycplt_img unreachable ({config.IMAGE_SERVICE_URL}): {e}") from e


def submit_job(
    prompt: str,
    mode: str = "txt2img",
    negative_prompt: Optional[str] = None,
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg_scale: float = 7.5,
    seed: Optional[int] = None,
    strength: Optional[float] = None,
    init_image: Optional[bytes] = None,
    mask_image: Optional[bytes] = None,
) -> int:
    """Queues a job on ycplt_img, returns job_id. Does not wait for the result."""
    body: Dict[str, Any] = {
        "prompt": prompt,
        "mode": mode,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
    }
    if negative_prompt:
        body["negative_prompt"] = negative_prompt
    if seed is not None:
        body["seed"] = seed
    if strength is not None:
        body["strength"] = strength
    if init_image is not None:
        body["init_image_b64"] = base64.b64encode(init_image).decode("ascii")
    if mask_image is not None:
        body["mask_image_b64"] = base64.b64encode(mask_image).decode("ascii")

    result = _request("POST", "/jobs", body)
    return result["job_id"]


def get_status(job_id: int) -> Dict[str, Any]:
    """{"id", "status": queued|processing|done|error, "error_message", ...}"""
    return _request("GET", f"/jobs/{job_id}")


def get_result(job_id: int) -> bytes:
    """Downloads the finished image (PNG bytes). Only call once status == 'done'."""
    url = f"{config.IMAGE_SERVICE_URL}/jobs/{job_id}/result"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=config.IMAGE_HTTP_TIMEOUT_SEC) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise ImageServiceError(f"result for job {job_id} unavailable: {e.code}") from e
    except urllib.error.URLError as e:
        raise ImageServiceError(f"ycplt_img unreachable ({config.IMAGE_SERVICE_URL}): {e}") from e


def delete_job(job_id: int) -> None:
    """Acknowledges the result was retrieved — the service will remove the job from its queue."""
    _request("DELETE", f"/jobs/{job_id}")
