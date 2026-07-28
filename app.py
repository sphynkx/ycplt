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
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from db.connection import init_db
from routes import chat, conversations, files, pages
from utils import config
from utils import image_jobs
from utils import llm as llm_utils
from utils import rag as rag_utils

app = FastAPI(title="ycplt")

app.include_router(pages.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(files.router)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    init_db()               # creates DB tables if they don't exist yet
    rag_utils.load_rag()    # optional: not fatal if no index is present
    llm_utils.load_llm()    # required: startup fails if this raises

    # Background task: polls ycplt_img for pending image jobs and resolves
    # them into the chat history once ready. Runs independently of any open
    # browser tab (see utils/image_jobs.py).
    image_jobs.start_background_poller()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=config.HOST, port=config.PORT, reload=False)
