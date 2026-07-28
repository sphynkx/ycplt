"""Page routes: the root path serves the browser chat page."""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Modern Starlette/FastAPI's TemplateResponse signature is (request, name, ...),
    # not (name, {"request": request}) as in older tutorials. With the old
    # signature, the first positional argument is treated as request and the
    # second as name, so a dict context ends up where a template name string
    # is expected (hence "TypeError: unhashable type: 'dict'" when looking up
    # the template in Jinja2's cache).
    return templates.TemplateResponse(request=request, name="index.html")
