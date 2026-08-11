"""Page routes: the root path serves the browser chat page."""
import time

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Cache-busting query string for static assets (?v=...) — set once per
# process start, not per request. A real, reported problem: static/js/app.js
# has no versioning at all in its <script src>, so after a deploy that
# changes it (new buttons, bugfixes, ...) a browser that already cached the
# old file (its own heuristic cache, or an aggressive reverse-proxy cache in
# front of uvicorn) keeps serving the stale script indefinitely — the
# person sees old behavior with no error of any kind, since nothing failed,
# it just silently never re-fetched. Restarting the process (which every
# real deploy already does) changes this value, forcing a fresh fetch.
_STATIC_VERSION = int(time.time())


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Modern Starlette/FastAPI's TemplateResponse signature is (request, name, ...),
    # not (name, {"request": request}) as in older tutorials. With the old
    # signature, the first positional argument is treated as request and the
    # second as name, so a dict context ends up where a template name string
    # is expected (hence "TypeError: unhashable type: 'dict'" when looking up
    # the template in Jinja2's cache).
    return templates.TemplateResponse(
        request=request, name="index.html", context={"static_version": _STATIC_VERSION}
    )
