"""Shared application configuration.

All settings are overridable via environment variables (see install/.env.example),
loaded via python-dotenv. Priority order (highest first): a real process
environment variable > a value from .env in the project root > the
hardcoded default below. python-dotenv's load_dotenv() never overrides
variables already present in the environment, so this ordering is free.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# ---------- App server ----------
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "4010"))

# ---------- LLM ----------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.gguf")

# 2 physical cores / 4 threads on the reference i7-5500U — more than 4
# threads buys nothing, there are no more physical cores.
N_THREADS = int(os.environ.get("N_THREADS", "4"))
N_CTX = int(os.environ.get("N_CTX", "2048"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "0"))  # no usable GPU acceleration on this hardware

# ---------- Chat history (conversations, messages, file attachments — see db/) ----------
DB_PATH = os.environ.get("DB_PATH", "data/chat.sqlite3")

# ---------- RAG (optional) ----------
RAG_DATA_DIR = os.environ.get("RAG_DATA_DIR", "rag_data")     # source documents (*.txt, *.pdf)
INDEX_PATH = os.environ.get("INDEX_PATH", "data/faiss_index.bin")  # built index (build_index.py)
META_PATH = os.environ.get("META_PATH", "data/meta.pkl")      # chunk metadata (build_index.py)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("TOP_K", "3"))

# ---------- Image generation service (ycplt_img, on a separate machine) ----------
# Passive queue — see utils/image_client.py and https://github.com/sphynkx/ycplt_img
IMAGE_SERVICE_HOST = os.environ.get("IMAGE_SERVICE_HOST", "192.168.7.7")
IMAGE_SERVICE_PORT = int(os.environ.get("IMAGE_SERVICE_PORT", "4011"))
IMAGE_SERVICE_URL = f"http://{IMAGE_SERVICE_HOST}:{IMAGE_SERVICE_PORT}"

IMAGE_POLL_INTERVAL_SEC = int(os.environ.get("IMAGE_POLL_INTERVAL_SEC", "10"))  # how often the app polls ycplt_img
IMAGE_HTTP_TIMEOUT_SEC = int(os.environ.get("IMAGE_HTTP_TIMEOUT_SEC", "10"))    # short-request timeout (not generation itself)
