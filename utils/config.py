"""Shared application configuration.

All settings are overridable via environment variables (see install/.env.example),
loaded via python-dotenv. Priority order (highest first): a real process
environment variable > a value from .env in the project root > the
hardcoded default below. python-dotenv's load_dotenv() never overrides
variables already present in the environment, so this ordering is free.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root (the directory containing app.py) — used to anchor every
# relative path default below (MODEL_PATH, DB_PATH, RAG_DATA_DIR, ...).
#
# Without this, "data/chat.sqlite3" would be resolved against the process's
# current working directory at launch time, not against the project itself.
# That's fragile: running "python app.py" from a different directory (or a
# systemd unit whose WorkingDirectory doesn't match how you tested manually)
# silently creates/reads a *different* database file at some other path,
# which looks exactly like "all my chat history disappeared after a
# restart" even though nothing was actually deleted. Anchoring to this
# file's location makes path resolution independent of the launch method.
BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_path(env_name: str, default_relative: str) -> str:
    """Reads env_name (falling back to default_relative) and, if the result
    is a relative path, resolves it against BASE_DIR instead of the current
    working directory. An absolute value in .env is always used as-is."""
    value = os.environ.get(env_name, default_relative)
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path)


# ---------- App server ----------
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "4010"))

# ---------- LLM ----------
MODEL_PATH = _resolve_path("MODEL_PATH", "models/model.gguf")

# 2 physical cores / 4 threads on the reference i7-5500U — more than 4
# threads buys nothing, there are no more physical cores.
N_THREADS = int(os.environ.get("N_THREADS", "4"))
N_CTX = int(os.environ.get("N_CTX", "2048"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "0"))  # no usable GPU acceleration on this hardware

# ---------- Chat history (conversations, messages, file attachments — see db/) ----------
DB_PATH = _resolve_path("DB_PATH", "data/chat.sqlite3")

# ---------- RAG (optional) ----------
RAG_DATA_DIR = _resolve_path("RAG_DATA_DIR", "rag_data")     # source documents (*.txt, *.pdf)
INDEX_PATH = _resolve_path("INDEX_PATH", "data/faiss_index.bin")  # built index (build_index.py)
META_PATH = _resolve_path("META_PATH", "data/meta.pkl")      # chunk metadata (build_index.py)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("TOP_K", "3"))

# ---------- Image generation service (ycplt_img, on a separate machine) ----------
# Passive queue — see utils/image_client.py and https://github.com/sphynkx/ycplt_img
IMAGE_SERVICE_HOST = os.environ.get("IMAGE_SERVICE_HOST", "192.168.7.7")
IMAGE_SERVICE_PORT = int(os.environ.get("IMAGE_SERVICE_PORT", "4011"))
IMAGE_SERVICE_URL = f"http://{IMAGE_SERVICE_HOST}:{IMAGE_SERVICE_PORT}"

IMAGE_POLL_INTERVAL_SEC = int(os.environ.get("IMAGE_POLL_INTERVAL_SEC", "10"))  # how often the app polls ycplt_img
IMAGE_HTTP_TIMEOUT_SEC = int(os.environ.get("IMAGE_HTTP_TIMEOUT_SEC", "10"))    # short-request timeout (not generation itself)
