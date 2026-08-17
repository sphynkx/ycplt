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

# ---------- Optional tiny router/classifier model ----------
# Empty by default = disabled: every classifier call (utils/tool_router.py,
# utils/intent.py's several classifiers, and similar field-extraction/
# mode-detection calls in utils/horary.py, utils/electional.py,
# utils/astro.py, utils/chart_draw.py, utils/rectification_events.py) then
# falls back to the main chat model (see utils/llm.classify_sync's own
# docstring), i.e. today's existing behavior, unchanged.
#
# Real, reported problem this exists to fix: this app makes MANY short,
# temperature=0.0, tight-max_tokens classifier calls per user message, and
# every one of them previously rode on the SAME big model (e.g.
# Qwen3.5-9B) as the actual long-form answer — visibly adding latency to
# every single message even though none of these calls need that model's
# writing quality, just enough language understanding to output one short,
# structured decision. Point this at a deliberately tiny instruction model
# (0.5B-1.5B parameters — a heavily quantized one, even Q2/Q1, is fine;
# classification doesn't need writing quality) — see README's "Tiny router
# model" section for a concrete recommendation and download link.
#
# Always loaded embedded (llama-cpp-python directly), regardless of
# LLM_BACKEND above — a model this small gains nothing from llama-server's
# concurrency machinery (it's already close to instant), so it isn't
# worth this module's dual-backend complexity for a component whose whole
# point is being fast and lightweight.
#
# Defaults to the exact path README's "Tiny router model" section tells
# you to download the recommended model to — a fresh checkout that
# follows the documented install steps (download the model, drop it at
# this path) gets the router active with zero .env editing required, the
# same way MODEL_PATH's own default assumes the documented model has been
# downloaded to models/. This is NOT the same thing as inventing a hidden
# default for an unrelated/undocumented setting — the file this points at
# is spelled out in the docs, so "the default is accurate" only holds
# because the install steps make it true. If the file isn't there yet
# (not downloaded, or a completely from-scratch checkout with no models/
# populated), load_router_llm() below degrades gracefully exactly like any
# other missing/bad router model file: a clear startup warning, then a
# transparent fallback to the main model — never fatal.
#
# Set YCPLT_ROUTER_MODEL_PATH="" explicitly (present in .env but empty) to
# opt back OUT of the router model entirely — distinct from leaving the
# variable unset, which now means "use the documented default path above".
_raw_router_model_path = os.environ.get("YCPLT_ROUTER_MODEL_PATH")
if _raw_router_model_path is not None and _raw_router_model_path.strip() == "":
    ROUTER_MODEL_PATH = ""
else:
    ROUTER_MODEL_PATH = _resolve_path(
        "YCPLT_ROUTER_MODEL_PATH", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    )
# A few classifier calls (tool_router's own tool-selection call in
# particular) include real conversation history and can reach several
# thousand tokens in practice — keep enough headroom that switching to a
# tiny model doesn't just trade one context-overflow bug for another (see
# README's own "Context-size overflow" story for llama-server, a related
# but distinct bug). Deliberately smaller than the main model's N_CTX
# (32768) since classifier prompts are still far shorter than a full
# RAG-heavy final answer.
ROUTER_N_CTX = int(os.environ.get("YCPLT_ROUTER_N_CTX", "8192"))
# Full logical core count by default, unlike the main model's N_THREADS
# (deliberately low at 4 — see that setting's own comment on being
# memory-bandwidth-bound). A model this small is more likely to fit in
# cache and actually benefit from more threads rather than fight over
# memory bandwidth, but this is a reasonable starting guess, not something
# measured yet on real hardware — re-test once a real router model is in
# place, the same discipline already applied to N_THREADS/N_THREADS_BATCH.
ROUTER_N_THREADS = int(os.environ.get("YCPLT_ROUTER_N_THREADS", str(os.cpu_count() or 4)))

# ---------- LLM backend: embedded (default) vs a separately hosted llama-server ----------
#
# "embedded" (default): the ORIGINAL behavior — utils/llm.py loads a single
# llama-cpp-python Llama object directly in this process, serialized behind
# a hand-rolled FIFO lock (see that module's _FifoLock) so two concurrent
# /chat requests can't race on the same model instance's KV cache. This
# works, but has a real, reported limitation: since there's only one
# request in flight at a time system-wide, a long generation for one
# conversation (an image-edit job's classifier calls, a long RAG-heavy
# astro answer, ...) makes every OTHER conversation's own message —
# even a trivial "hi, how are you" — wait in strict FIFO order behind it,
# confirmed in practice by testing two chats at once.
#
# "server": routes every generate_sync/generate_async call to a
# SEPARATELY hosted llama-server instance (the native C++ server binary
# from the llama.cpp project itself — ggml-org/llama.cpp's tools/server,
# NOT the same thing as the llama-cpp-python pip package's own
# `llama_cpp.server` module, which was checked directly against its
# source and found to serialize requests behind a single anyio.Lock just
# like the embedded backend above — switching to THAT would gain nothing
# but a network hop). llama-server supports real parallel request
# handling via multiple "slots" (--parallel/-np) with continuous batching
# (-cb, on by default) — on CPU-only hardware this does NOT meaningfully
# raise total throughput (no idle parallel matrix units to exploit the
# way a GPU has), but it DOES fix the actual reported problem: a short
# request arriving mid-generation gets folded into the next decode step
# across all active slots, instead of waiting for one long generation to
# finish outright. See README's "Separate llama-server backend" section
# for the full setup (building llama-server, systemd unit, choosing a
# --parallel count for your CPU's core count).
#
# Defaults to "embedded" so nothing changes for anyone who hasn't set up
# a separate llama-server instance — exactly the same off-by-default,
# single-flag toggle pattern already used for ycplt_img's
# RECONSTRUCT_ENABLED/KONTEXT_ENABLED, so this can be tried and reverted
# just as easily if llama-server doesn't work out.
LLM_BACKEND = os.environ.get("YCPLT_LLM_BACKEND", "embedded").strip().lower()

# Only used when LLM_BACKEND=="server". Defaults to localhost — unlike
# ycplt_img (a genuinely separate machine), llama-server runs on the SAME
# box as this app: it's a second local process talking to the same GGUF
# model file, not a remote service, so the plain loopback address is the
# right default here. Override only if llama-server ever moves elsewhere.
LLAMA_SERVER_HOST = os.environ.get("YCPLT_LLAMA_SERVER_HOST", "127.0.0.1")
# 4012, not llama-server's own upstream default of 8080 — this project
# reserves 4010+ for the whole ycplt family (ycplt=4010, ycplt_img=4011,
# llama-server=4012, ...), one port each, specifically so they don't
# collide with other unrelated services that commonly squat on 8080. See
# README's "Port allocation across the ycplt family" table.
LLAMA_SERVER_PORT = int(os.environ.get("YCPLT_LLAMA_SERVER_PORT", "4012"))
LLAMA_SERVER_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"

# 0 (disabled) by default — matching the embedded backend, which has NO
# timeout at all (it's a plain in-process Python call). Real, confirmed
# consequence of the previous 1200s (20 min) default: a genuine heavy
# RAG answer (astro_natal_chart, full methodology + digest + sectioned
# prompt) was still generating at n_gen=1388+ tokens, tg=2.58 tok/s, when
# this app's own client-side timeout fired and aborted it mid-answer —
# llama-server's own log showed "cancel task" / "stop processing", i.e.
# the model was making real progress, just not fast enough to finish
# inside 1200s. This app's own N_CTX/max_tokens policy is already "no
# artificial cap" (see generate_sync's own docstring in utils/llm.py),
# and a bigger --ctx-size (see LLAMA_SERVER_PORT's own comment and
# install/llama-server.service) tends to slow tg further via a larger
# KV cache to scan per token — so a fixed timeout here fights the app's
# own design elsewhere. Set to a positive number of seconds only if you
# specifically want a hard ceiling (e.g. to fail fast on a genuinely
# stuck request) and are fine with heavy techniques being cut off before
# they finish.
LLAMA_SERVER_TIMEOUT_SEC = int(os.environ.get("YCPLT_LLAMA_SERVER_TIMEOUT_SEC", "0"))

# ---------- Remote LLM provider (optional, off by default) ----------
# Off by default (empty) — every generate_sync/generate_async call (the
# actual long-form chat reply — NOT the classifier calls covered by
# ROUTER_MODEL_PATH above, which deliberately stay local regardless of
# this setting: they're already fast/free, and this app's real pain point
# is the MAIN answer's speed on local CPU hardware — a full RAG-heavy
# technique like a natal chart or rectification can take 18-49+ minutes
# at ~3-4 tok/s) is instead sent to an external cloud API first, with an
# UNCONDITIONAL, automatic fallback to whatever local backend
# (LLM_BACKEND above) is already configured on ANY failure — network
# error, missing/invalid key, rate limit, malformed response — logged as
# a warning, never raised to the caller. Mirrors ROUTER_MODEL_PATH's own
# "off/broken = transparently fall back, never crash" philosophy.
#
# Values: "" (default, local only), "openai" (OpenAI's own
# /v1/chat/completions API — see utils/llm.py's _generate_remote_openai),
# or "claude" (Anthropic's own Messages API — a genuinely different
# request/response shape from OpenAI's: content as blocks, a separate
# top-level "system" field, x-api-key/anthropic-version headers instead of
# a bearer token, and max_tokens is REQUIRED rather than optional — see
# utils/llm.py's _generate_remote_claude). REMOTE_LLM_PROVIDER is named
# generically (not e.g. USE_OPENAI) specifically so this is just one more
# accepted value here, not a second, differently-named setting.
REMOTE_LLM_PROVIDER = os.environ.get("REMOTE_LLM_PROVIDER", "").strip().lower()
# Required whenever REMOTE_LLM_PROVIDER is set; ignored otherwise.
REMOTE_LLM_API_KEY = os.environ.get("REMOTE_LLM_API_KEY", "").strip()
# Provider-specific model name. If REMOTE_LLM_MODEL isn't set explicitly,
# the default depends on which provider is selected: gpt-4o-mini for
# OpenAI, claude-haiku-4-5 for Claude — in both cases the provider's own
# fastest/cheapest current model, a reasonable default for this app's
# actual need (a capable enough writer for astrology interpretation text,
# not the single most powerful model available), and a free/rate-limited
# key is far more likely to hold up against it than against a larger,
# pricier model.
_REMOTE_LLM_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-4-5",
}.get(REMOTE_LLM_PROVIDER, "gpt-4o-mini")
REMOTE_LLM_MODEL = os.environ.get("REMOTE_LLM_MODEL", "").strip() or _REMOTE_LLM_DEFAULT_MODEL
# <= 0 disables the timeout (waits indefinitely) — same reasoning as
# LLAMA_SERVER_TIMEOUT_SEC's own comment: this covers a real generation,
# not just a liveness probe, so a short fixed timeout would fight this
# app's own "no artificial cap" policy elsewhere.
REMOTE_LLM_TIMEOUT_SEC = int(os.environ.get("REMOTE_LLM_TIMEOUT_SEC", "0"))

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


def log_effective_config() -> None:
    """Prints a short summary of the settings most likely to silently
    misconfigure — every one of these has, in real use, either used the
    YCPLT_ prefix inconsistently (LLM_BACKEND/LLAMA_SERVER_*/ROUTER_*, see
    their own comments above) or been edited in install/.env.example
    instead of the real .env this app actually reads. os.environ.get()
    has no way to warn about a typo'd or misplaced variable name on its
    own — a wrong name just silently returns the hardcoded default, with
    zero error — so this exists purely to make the ACTUAL effective value
    of each one impossible to miss at startup, every single restart,
    instead of having to guess or grep .env by hand. Called first thing
    in app.py's lifespan, before anything else initializes."""
    lines = ["=" * 60, "ycplt: effective configuration", "=" * 60]
    lines.append(f"HOST:PORT            = {HOST}:{PORT}")
    lines.append(f"MODEL_PATH           = {MODEL_PATH}")
    lines.append(f"N_CTX / N_THREADS    = {N_CTX} / {N_THREADS} (batch threads: {N_THREADS_BATCH})")
    lines.append(f"LLM_BACKEND          = {LLM_BACKEND!r}"
                 + (" (main model in-process)" if LLM_BACKEND != "server" else ""))
    if LLM_BACKEND == "server":
        lines.append(f"  LLAMA_SERVER_URL   = {LLAMA_SERVER_URL}")
        lines.append(f"  LLAMA_SERVER_TIMEOUT_SEC = {LLAMA_SERVER_TIMEOUT_SEC} (0 = disabled)")
    if ROUTER_MODEL_PATH:
        _router_file_state = "" if os.path.exists(ROUTER_MODEL_PATH) else " — FILE NOT FOUND, will fall back to the main model"
        lines.append(
            f"ROUTER_MODEL_PATH    = {ROUTER_MODEL_PATH} (n_ctx={ROUTER_N_CTX}, "
            f"n_threads={ROUTER_N_THREADS}){_router_file_state}"
        )
    else:
        lines.append("ROUTER_MODEL_PATH    = (explicitly disabled) — classifier calls use the main model")
    if REMOTE_LLM_PROVIDER:
        key_state = "set" if REMOTE_LLM_API_KEY else "MISSING"
        lines.append(
            f"REMOTE_LLM_PROVIDER  = {REMOTE_LLM_PROVIDER!r} (model={REMOTE_LLM_MODEL!r}, "
            f"API key {key_state}) — main answers only, falls back to local on any failure"
        )
    else:
        lines.append("REMOTE_LLM_PROVIDER  = (not set) — main answers use the local model only")
    lines.append(f"DB_PATH              = {DB_PATH}")
    lines.append(f"RAG_DATA_DIR         = {RAG_DATA_DIR}")
    lines.append(f"IMAGE_SERVICE_URL    = {IMAGE_SERVICE_URL}")
    lines.append("=" * 60)
    print("\n".join(lines))
