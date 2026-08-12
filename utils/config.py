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

# Governs ONLY the token-generation phase (sequential, one token at a
# time) — see N_THREADS_BATCH below for the separate prompt-processing
# phase. Originally set to 4 for the old reference i7-5500U (2 physical
# cores/4 threads — more than 4 bought nothing there). Counterintuitively,
# raising this on the current, much beefier i7-8700 (6 physical/12
# logical) did NOT help in real testing — 4 measured faster end-to-end
# than both 6 and 10 on the same box, same prompt. This isn't a fluke:
# generation is memory-bandwidth-bound and pays a synchronization barrier
# once per generated token across every thread involved, so past some
# point (which depends on memory bandwidth, not core count) adding threads
# increases that per-token overhead faster than it adds useful parallel
# work — a well-documented llama.cpp behavior on desktop-class CPUs with
# a handful of memory channels, not the server-class hardware this kind
# of setting is often tuned for. If you change this, re-measure a real
# request end-to-end rather than assuming a higher number is better.
N_THREADS = int(os.environ.get("N_THREADS", "4"))
# The prompt-processing/"batch" phase (parallelizable across positions,
# unlike token generation above) has its own, independent thread count in
# llama-cpp-python — NOT the same knob as N_THREADS, and NOT set at all
# before this was added: llama_cpp.Llama's own source defaults
# n_threads_batch to `multiprocessing.cpu_count()` (all logical cores)
# whenever it isn't passed explicitly, regardless of whatever N_THREADS
# is. That silent default is exactly why every logical CPU could be seen
# pegged at ~100% during a request even with N_THREADS turned down — the
# batch phase was never governed by N_THREADS in the first place. Exposed
# here so it can be tuned independently instead of guessing from N_THREADS
# alone. Defaults to the full logical core count (matching the library's
# own prior implicit behavior) so leaving this unset changes nothing.
N_THREADS_BATCH = int(os.environ.get("N_THREADS_BATCH", str(os.cpu_count() or 4)))
# How many prompt tokens llama.cpp processes per parallel batch during the
# prompt/prefill phase — a size, not a thread count (see N_THREADS_BATCH
# above for that). Also never set explicitly before this; llama-cpp-python
# itself defaults to 512 when omitted, which is what stays in effect here
# unless overridden. Larger values can speed up prefill on a long RAG-heavy
# prompt at the cost of more RAM for the batch buffer — worth trying 1024
# or 2048 on this project's typically long (methodology-document-heavy)
# prompts, but re-measure rather than assume, same as N_THREADS above.
N_BATCH = int(os.environ.get("N_BATCH", "512"))
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
# llama-cpp-python's own default is already 1.1, not 1.0 — this is a
# slightly higher explicit floor, set after a real, reproducible glitch: a
# long generation (the astro sectioned answer, which repeats concepts like
# "practical/practicality" across several paragraphs) started emitting
# stray Chinese characters mid-sentence (Qwen models' vocabulary includes
# CJK tokens, and a small quantized model under-penalized for repetition
# can apparently fall back to one instead of rephrasing in Russian). This
# is a real trade-off, not a free fix: too high a value can make the model
# avoid necessary repeated words (planet/sign names have to repeat
# legitimately in this app's output) or produce less coherent text — raise
# further only if the glitch recurs, and lower it back toward 1.1 if
# answers start reading as unnaturally avoidant of repeating chart terms.
REPEAT_PENALTY = float(os.environ.get("REPEAT_PENALTY", "1.15"))

# ---------- Chat history (conversations, messages, file attachments — see db/) ----------
DB_PATH = _resolve_path("DB_PATH", "data/chat.sqlite3")

# ---------- RAG (optional) ----------
RAG_DATA_DIR = _resolve_path("RAG_DATA_DIR", "rag_data")     # source documents (*.txt, *.pdf) — may use topic subfolders, see build_index.py
# Per-corpus index layout: build_index.py writes one faiss_index.bin+meta.pkl
# pair per corpus (a rag_data/ topic subfolder, or the loose files directly
# in rag_data/ itself) under INDEX_DIR/<topic>/ — rebuilding one corpus
# never touches, re-reads, or re-OCRs any other corpus's documents. This
# replaced an earlier single-combined-index design once a real corpus
# (several rag_data/ subfolders, some containing OCR-heavy .djvu/.pdf
# scans) made a single from-scratch rebuild take hours with no visibility
# into progress or a way to redo just the one corpus that actually changed.
# utils/rag.py loads every corpus found under INDEX_DIR and merges their
# retrieval results at query time (see utils/rag.py's module docstring),
# so this is transparent to anything reading RAG results — only
# build_index.py's own invocation changed (see its module docstring for
# the new `python build_index.py [topic_or_path]` usage).
INDEX_DIR = _resolve_path("INDEX_DIR", "data/rag_index")
# Legacy single-file index — still supported as a fallback by utils/rag.py
# if INDEX_DIR doesn't exist (e.g. an index built before per-corpus
# indexing existed), so an already-deployed server keeps working without
# forcing an immediate rebuild. New indexes are written under INDEX_DIR,
# not here — these two settings exist purely for that migration path.
INDEX_PATH = _resolve_path("INDEX_PATH", "data/faiss_index.bin")  # legacy built index (pre-per-corpus)
META_PATH = _resolve_path("META_PATH", "data/meta.pkl")      # legacy chunk metadata (pre-per-corpus)
# How many external processes (antiword, djvutxt, ddjvu, pdftoppm,
# tesseract, extract_chmLib, patool, ...) build_index.py runs at once while
# reading a corpus's source documents. This is the setting that actually
# turns "one slow external tool after another" into overlapping work —
# reading/OCR'ing is dominated by these subprocess calls, not by Python or
# by the (single, already-batched) embedding step, so this is where
# concurrency actually pays off. Matches N_THREADS' reasoning: the
# reference i7-5500U has 2 physical cores/4 threads, so 4 concurrent
# external processes is a reasonable default; raise it on faster/more-core
# hardware, lower it if the machine struggles to stay responsive while
# indexing runs.
INDEX_CONCURRENCY = int(os.environ.get("INDEX_CONCURRENCY", "4"))
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
# is NOT a cost/quota setting — there's no billing here, everything runs
# locally — it's a safety margin under N_CTX (above), which is itself a
# hard technical ceiling, not a dial to economize on: llama.cpp simply
# cannot process more tokens than N_CTX in one call, and going past what
# the model was actually trained to attend over (32768 for Qwen2.5)
# degrades output quality even on hardware that could technically hold
# more in RAM. So the real limits here are the model's own architecture
# and however much RAM you can give the KV cache — not a preference for
# keeping this number small. The previous default (6000) was confirmed in
# practice to be too tight: after interpretation_methodology.txt grew past
# ~6000 chars (adding the fabrication-guardrail and unicode-symbol
# sections), a simulation of build_index.py's own chunking showed the cap
# silently dropping the symbol-legend table, the whole "how to format the
# answer" section, and the worked example — the model was never even
# seeing that content, no matter how the wording was tuned. 16000 chars was
# raised to fix that at the time, but the SAME failure mode recurred later,
# twice at once: horary_methodology.txt grew to ~20400 chars (radicality
# nuance, derived-house notes, the verdict-to-practical-meaning section,
# then the multi-school comparison section) and astro_progressions'
# bundled methodology docs (progression+direction+lunar_return+
# solar_return+profection, all sharing one topic and therefore one shared
# budget — see README's RAG topic-layout table) summed to ~18300 chars —
# both silently over the old 16000 cap, confirmed by rerunning the exact
# same chunking simulation described above. In horary's case this dropped
# the entire "Обозначения"/"Порядок изложения ответа"/"Пример" sections and
# most of the new comparison section, which is exactly why the model
# stopped mentioning any methodology comparison at all despite the section
# being added to the source document — the document was edited correctly,
# it just never reached the model. 28000 chars (~7000-10000 tokens
# depending on tokenizer efficiency on Cyrillic) comfortably covers both of
# today's largest methodology payloads with real headroom (~35-55%) for
# further growth, alongside retrieved facts, the astro tool's chart data,
# and the model's own answer, within the current N_CTX default (32768) —
# raise it further (and N_CTX and available RAM alongside it) if any
# methodology corpus grows past that, or check for silent truncation
# yourself with a quick chunking simulation, which is also exactly what
# build_index.py now warns about automatically at build time. Don't treat
# "the model isn't using content that's clearly in the source document" as
# a prompt-wording problem before ruling this out first — it looks
# identical from the outside and wastes time chasing the wrong fix.
RAG_ALWAYS_INCLUDE_MAX_CHARS = int(os.environ.get("RAG_ALWAYS_INCLUDE_MAX_CHARS", "28000"))

# ---------- Rectification tools: optional LLM follow-up ----------
# Whether astro_rectification_trutine/astro_rectification_events get a
# follow-up LLM call (RAG-augmented reasoning over the tool's own computed
# report, same mechanism every other astro_* tool uses) on top of the
# report itself. Defaults to OFF: real testing with this project's
# reference small model (see routes/chat.py's _NO_FOLLOWUP_TOOL_NAMES
# comment for the full history) showed the follow-up call reliably ending
# up CONTRADICTING the tool's own computed best-candidate time somewhere
# in its own prose — three separate real-world tests, four layers of
# mitigation added one at a time (prepend the correct line, a disclaimer
# sentence, an explicit "don't contradict this" instruction in both
# methodology documents, bookending the line at both ends of the reply),
# none of it held up. That's a genuine reliability limit of this specific
# small model on this specific task (consistently transcribing one exact
# number out of a large technical report), not a wording problem worth
# continuing to chase — so the follow-up call was removed rather than
# mitigated further. This toggle exists so the capability isn't lost
# outright: swap in a more capable model later, set this to true in
# .env, and the RAG-augmented reasoning step (plus the same prepend/
# disclaimer/bookend safety net, kept in the code specifically for this
# toggle) comes back with no code changes needed — just re-verify the new
# model doesn't repeat the same contradiction before trusting it.
RECTIFICATION_LLM_FOLLOWUP = os.environ.get(
    "RECTIFICATION_LLM_FOLLOWUP", "false"
).strip().lower() in ("1", "true", "yes", "on")

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
