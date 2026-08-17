"""
Application entry point.

This file only wires things together: creates the FastAPI app, includes the
routers, and kicks off DB/model/RAG initialization plus the background image
job poller on startup. All actual logic lives elsewhere:
  - routes/    — HTTP routes (chat.py — chat API, conversations.py — chat
                 threads, files.py — attachment downloads, pages.py — the
                 browser page)
  - db/        — SQLite: connection, schema, CRUD (conversations/messages/files)
  - utils/     — config, LLM loading/inference, RAG helpers, code block
                 parsing, image-request intent detection, the ycplt_img client
  - templates/ — HTML page templates
  - static/js/ — browser-side JavaScript (served at /static)

Run:
    python app.py
    (or: uvicorn app:app --host <HOST> --port <PORT>)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import asyncio

from db.connection import init_db
from routes import chat, conversations, export, files, pages, profiles
from utils import astro
from utils import config
from utils import image_jobs
from utils import llm as llm_utils
from utils import rag as rag_utils


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------- startup ----------
    config.log_effective_config()  # prints actual effective values of the
                                    # settings most prone to silent typo/
                                    # wrong-file mix-ups (see its own
                                    # docstring) — always the very first
                                    # thing in the console on every restart
    init_db()               # creates DB tables if they don't exist yet
    rag_utils.load_rag()    # optional: not fatal if no index is present
    llm_utils.load_llm()    # required: startup fails if this raises
    llm_utils.load_router_llm()  # optional: no-op unless config.ROUTER_MODEL_PATH is
                                  # set, and non-fatal even if it is but fails to load
                                  # (see that function's own docstring)

    # Optional: not fatal if kerykeion/timezonefinder/geonamescache aren't
    # installed. Runs in a worker thread (not awaited synchronously here)
    # so it doesn't delay serving the first request behind its ~18s cost —
    # it just needs to finish running once before the first astro chart
    # question comes in, which in practice it comfortably will.
    asyncio.get_running_loop().run_in_executor(None, astro.warmup)

    # Background task: polls ycplt_img for pending image jobs and resolves
    # them into the chat history once ready. Runs independently of any open
    # browser tab (see utils/image_jobs.py).
    image_jobs.start_background_poller()

    yield

    # ---------- shutdown ----------
    # Real, reported crash this fixes: with no shutdown handler at all
    # (the previous @app.on_event("startup")-only setup had none), the
    # embedded LLM was only ever released by Python's own garbage collector
    # during interpreter shutdown, which on at least one real deployment
    # (Python 3.14, a recent llama-cpp-python build) produced
    # `TypeError: 'NoneType' object is not callable` deep inside
    # llama_cpp's own free_model — see utils/llm.close_llm's own docstring
    # for the full explanation. Closing it here, explicitly, while uvicorn
    # is still gracefully shutting down (not yet at interpreter teardown),
    # avoids that ordering problem.
    llm_utils.close_llm()
    llm_utils.close_router_llm()


app = FastAPI(title="ycplt", lifespan=lifespan)

app.include_router(pages.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(files.router)
app.include_router(export.router)
app.include_router(profiles.router)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=config.HOST, port=config.PORT, reload=False)
