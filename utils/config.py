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
# 32768 — Qwen2.5's native trained context length, not an arbitrary large
# number picked to be safe. The previous default here (2048) was sized for
# plain short chat turns and repeatedly wasn't enough once RAG context,
# retrieved facts/methodology, and the astro tool's chart data all landed
# in the same prompt at once ("Requested tokens (2774) exceed context
# window of 2048", confirmed in practice, not a hypothetical). On
# CPU-only, fully local hardware there's no cost pressure that justifies
# trading that off against still hitting this error occasionally — the
# real cost of a bigger N_CTX is RAM for the KV cache and somewhat slower
# prompt processing on long prompts, not anything scarcer. Lower this back
# down if your hardware genuinely can't spare the RAM.
N_CTX = int(os.environ.get("N_CTX", "32768"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "0"))  # no usable GPU acceleration on this hardware

# ---------- Chat history (conversations, messages, file attachments — see db/) ----------
DB_PATH = _resolve_path("DB_PATH", "data/chat.sqlite3")

# ---------- RAG (optional) ----------
RAG_DATA_DIR = _resolve_path("RAG_DATA_DIR", "rag_data")     # source documents (*.txt, *.pdf) — may use topic subfolders, see build_index.py
INDEX_PATH = _resolve_path("INDEX_PATH", "data/faiss_index.bin")  # built index (build_index.py)
META_PATH = _resolve_path("META_PATH", "data/meta.pkl")      # chunk metadata (build_index.py)
# Multilingual by default — RAG source documents are commonly in Russian,
# and all-MiniLM-L6-v2 (the previous default) is English-only, giving poor
# retrieval quality on non-English text. paraphrase-multilingual-MiniLM-L12-v2
# is the multilingual sibling of that same MiniLM family (same sentence-
# transformers ecosystem, similar size/speed on CPU, ~50 languages incl.
# Russian). Changing this requires rebuilding the index from scratch
# (python build_index.py) — embeddings from different models aren't
# compatible even when the vector dimension happens to match.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.environ.get("TOP_K", "3"))
# Caps how much text retrieve_context's "always_include" (methodology)
# expansion can add on top of the ordinary top-k hits (see utils/rag.py).
# Without this, a long methodology document gets pulled in *entirely* the
# moment any chunk from its topic is retrieved, regardless of length. This
# is still a real bound, not a value chosen to be stingy: even with
# N_CTX raised well past what a small local model needs for one plain
# chat turn, an unbounded expansion could still keep growing as
# methodology documents grow, and a bound that scales with "however big
# the document happens to be" isn't really a bound. 6000 chars (~1500
# tokens) comfortably fits a substantial methodology document alongside
# retrieved facts, the astro tool's chart data, and the model's own answer
# within the current N_CTX default (32768) with a lot of headroom to
# spare — raise it further if you have a genuinely large methodology
# corpus and the context budget for it.
RAG_ALWAYS_INCLUDE_MAX_CHARS = int(os.environ.get("RAG_ALWAYS_INCLUDE_MAX_CHARS", "6000"))

# ---------- Image generation service (ycplt_img, on a separate machine) ----------
# Passive queue — see utils/image_client.py and https://github.com/sphynkx/ycplt_img
IMAGE_SERVICE_HOST = os.environ.get("IMAGE_SERVICE_HOST", "192.168.7.7")
IMAGE_SERVICE_PORT = int(os.environ.get("IMAGE_SERVICE_PORT", "4011"))
IMAGE_SERVICE_URL = f"http://{IMAGE_SERVICE_HOST}:{IMAGE_SERVICE_PORT}"

IMAGE_POLL_INTERVAL_SEC = int(os.environ.get("IMAGE_POLL_INTERVAL_SEC", "10"))  # how often the app polls ycplt_img
IMAGE_HTTP_TIMEOUT_SEC = int(os.environ.get("IMAGE_HTTP_TIMEOUT_SEC", "10"))    # short-request timeout (not generation itself)

# Note: no vision/captioning model config here on purpose — this app only
# hosts the chat LLM. Image understanding (mode="caption") is a graphics-
# service capability, submitted as a job to ycplt_img like generation/
# editing; see utils/image_client.py and routes/chat.py._handle_image_question.
