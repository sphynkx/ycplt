"""Loading and calling the chat LLM — either embedded directly in this
process (llama-cpp-python, the original/default behavior) or via a
separately-hosted llama-server instance on the SAME machine that supports
real concurrent-request handling (continuous batching / parallel slots).
See config.LLM_BACKEND ("embedded" default, or "server").

Every caller (utils/intent.py's classifiers, utils/tool_router.py,
utils/interpret.py, routes/chat.py's final answer, ...) goes through
generate_sync/generate_async/get_llm/load_llm exactly as before — the
backend split is entirely internal to this module, so nothing outside it
needed to change when the "server" backend was added.
"""
import json
import os
import re
import asyncio
import threading
import urllib.error
import urllib.request
from typing import Optional

from utils import config

# Holds either a real llama_cpp.Llama instance (backend="embedded") or the
# sentinel string "server" once config.LLM_BACKEND=="server" and
# load_llm() has confirmed llama-server is reachable — get_llm() callers
# throughout the codebase only ever check "is this None" (is a model
# available at all), never call methods on it directly, so a plain
# truthy sentinel is a safe stand-in for the server backend.
_llm = None


class _FifoLock:
    """A mutex that grants access in strict first-come-first-served order —
    unlike a plain threading.Lock, whose underlying OS mutex makes no
    ordering promise among several blocked waiters (whichever thread the
    OS/runtime happens to wake next gets it, which in practice is "roughly
    fair" but not guaranteed, and isn't even the point: a plain Lock only
    guarantees mutual exclusion, not ordering).

    ONLY used by the "embedded" backend (see _generate_embedded) — the
    "server" backend has no lock at all, since llama-server itself handles
    concurrent requests safely via its own parallel slots (see that
    backend's own comments below for why this genuinely fixes the
    limitation this lock works around, rather than just moving it).

    This matters here specifically because a single /chat request already
    makes several SEQUENTIAL calls into this module of its own accord (see
    generate_sync's own docstring: intent/tool-router classification, a
    tool's own field/round classifiers, digest_facts_async, the final
    answer) — every one of those previously raced independently for
    _llm_lock. With two conversations' requests genuinely running at once
    (see README's own "Concurrency: one model, many chats" section), a
    plain Lock gave no guarantee that either conversation's OWN sequence of
    calls would even complete in order relative to the other's, let alone
    that one conversation wouldn't be starved indefinitely by a second one
    that happens to keep winning the race. A ticket-based FIFO lock fixes
    both: every caller — a quick classifier call and a long final-answer
    generation alike — draws a ticket the instant it asks for the lock and
    is served strictly in that arrival order, regardless of which
    conversation it belongs to or how long any one call takes.

    A plain threading.Lock (not asyncio.Lock) underneath the ticketing on
    purpose, for the same reason the old comment here gave: most callers
    reach generate_sync from a worker thread (nearly every utils/*.py
    caller dispatches its whole tool/handler function via
    loop.run_in_executor(None, ...), not from inside a coroutine, where an
    asyncio.Lock isn't usable at all), and this needs to work identically
    from a plain sync call and from generate_async's own
    run_in_executor-dispatched thread."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._next_ticket = 0
        self._now_serving = 0

    def __enter__(self) -> "_FifoLock":
        with self._cv:
            my_ticket = self._next_ticket
            self._next_ticket += 1
            if my_ticket != self._now_serving:
                # Only printed when there's real contention (someone else
                # is already being served or already queued ahead) — a
                # quiet no-op the rest of the time, so normal single-chat
                # use generates no extra log noise.
                print(f"[llm] request queued: {my_ticket - self._now_serving} ahead of it, waiting for a turn")
            while my_ticket != self._now_serving:
                self._cv.wait()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        with self._cv:
            self._now_serving += 1
            self._cv.notify_all()
        return False


# See _FifoLock's own docstring above for why this replaced a plain
# threading.Lock — only used by the "embedded" backend.
_llm_lock = _FifoLock()

# Some GGUF models (e.g. Qwen3's "-Thinking-" variants) are trained to
# always emit an internal chain-of-thought scratchpad wrapped in
# <think>...</think> before the real answer — by design/convention this
# reasoning trace is far less language-constrained than the model's actual
# final answer, and reliably comes out in English even when everything
# else (prompt, expected answer) is in Russian, confirmed by real testing
# after switching models. This app has no use for that scratchpad (every
# caller wants the final answer only, in the requested language), and
# nothing downstream expects it, so it's stripped here — the one shared
# place every caller's output passes through, regardless of which backend
# produced it, rather than in each of the many individual prompts/callers.
# A no-op for any model that doesn't emit think-tags at all, so this is
# safe to leave in regardless of which model/backend is configured.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _load_embedded():
    """The original behavior: loads config.MODEL_PATH directly into this
    process via llama-cpp-python. Raises RuntimeError with a clear message
    if the file is missing or not in GGUF format."""
    from llama_cpp import Llama

    global _llm

    if not os.path.exists(config.MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {config.MODEL_PATH}. "
            f"Download a GGUF model and place it at this path (see README)."
        )
    if not config.MODEL_PATH.lower().endswith(".gguf"):
        raise RuntimeError(
            f"Expected a GGUF (.gguf) file, got: {config.MODEL_PATH}. "
            f"Old .bin/.ggml files (ggmlv2/ggmlv3) are not supported by modern llama.cpp."
        )
    try:
        _llm = Llama(
            model_path=config.MODEL_PATH,
            n_ctx=config.N_CTX,
            n_threads=config.N_THREADS,
            # Previously left unset, which meant llama-cpp-python silently
            # substituted its own default (all logical cores) for both of
            # these regardless of N_THREADS above — see config.py's
            # N_THREADS_BATCH/N_BATCH comments for the full explanation and
            # why that made every CPU core look pegged even when N_THREADS
            # was turned down.
            n_threads_batch=config.N_THREADS_BATCH,
            n_batch=config.N_BATCH,
            n_gpu_layers=config.N_GPU_LAYERS,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{config.MODEL_PATH}': {e}")

    print("Model loaded successfully (embedded backend).")
    return _llm


def _check_server_reachable() -> None:
    """Fails fast at startup if config.LLM_BACKEND=="server" but
    llama-server isn't actually reachable at config.LLAMA_SERVER_URL —
    the same guarantee _load_embedded() already gives for a missing model
    file, rather than only discovering the misconfiguration on the first
    real /chat request."""
    req = urllib.request.Request(f"{config.LLAMA_SERVER_URL}/health", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"llama-server at {config.LLAMA_SERVER_URL} responded with HTTP {resp.status} on /health"
                )
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"llama-server unreachable at {config.LLAMA_SERVER_URL} (config.LLM_BACKEND='server'): {e}. "
            "Start llama-server first (see README's 'Separate llama-server backend' section), "
            "or set YCPLT_LLM_BACKEND=embedded to use the built-in model instead."
        ) from e


def load_llm():
    """Loads/connects the chat model according to config.LLM_BACKEND.
    Raises RuntimeError with a clear message on failure either way — a
    missing model file for "embedded", or an unreachable llama-server for
    "server" — so a misconfiguration is caught at startup, not on the
    first real chat message."""
    global _llm

    if config.LLM_BACKEND == "server":
        _check_server_reachable()
        _llm = "server"  # truthy sentinel — see this module's own top comment
        print(f"Using llama-server backend at {config.LLAMA_SERVER_URL}.")
        return _llm

    return _load_embedded()


def get_llm():
    return _llm


def close_llm() -> None:
    """Explicitly releases the embedded model before process shutdown —
    call this from app.py's own lifespan shutdown phase, not just when it
    feels convenient.

    Real, reported crash this exists to fix: with no explicit close
    anywhere in this app's own code (no shutdown handler existed at all
    before this), the embedded Llama instance was only ever cleaned up by
    Python's own garbage collector during interpreter shutdown
    (llama_cpp's own Llama.__del__). On at least one real deployment
    (Python 3.14, a recent llama-cpp-python build), that teardown ordering
    produced `TypeError: 'NoneType' object is not callable` deep inside
    llama_cpp's free_model — a known class of bug where a C extension's
    __del__ references another module-level global that Python's own
    interpreter shutdown may already have cleared by the time __del__
    actually runs, not something specific to this app's own code (previous
    Python/llama-cpp-python combinations on the same code apparently
    didn't hit this ordering issue, hence "used to close cleanly").
    Closing the model explicitly and early — during uvicorn's own graceful
    shutdown, while the whole process (including every llama_cpp module
    global) is still fully intact — avoids that ordering problem
    entirely, rather than leaving cleanup to whatever order Python's own
    finalizer happens to run in at some later, less predictable point.

    Safe to call regardless of config.LLM_BACKEND: a no-op for the
    "server" backend (there's no local model object to release there —
    llama-server, a separate process, manages its own lifecycle), and
    safe to call even if load_llm() was never reached (startup failed
    before it)."""
    global _llm
    if config.LLM_BACKEND != "server" and _llm is not None:
        try:
            _llm.close()
        except Exception as e:
            print(f"[llm] error while closing model: {e}")
    _llm = None


def _generate_embedded(prompt: str, max_tokens: Optional[int], temperature: float) -> str:
    with _llm_lock:
        out = _llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            # See config.REPEAT_PENALTY's comment: set explicitly (rather than
            # relying on llama-cpp-python's own 1.1 default) after a real
            # generation glitch — a long, repetition-heavy answer started
            # emitting stray Chinese characters instead of rephrasing in
            # Russian.
            repeat_penalty=config.REPEAT_PENALTY,
        )
    return out["choices"][0]["message"]["content"]


def _generate_server(prompt: str, max_tokens: Optional[int], temperature: float) -> str:
    """Posts to llama-server's OpenAI-compatible /v1/chat/completions.
    llama-server extends the standard OpenAI request body with extra
    llama.cpp-specific sampling fields (repeat_penalty among them), passed
    through directly here — no separate client library needed, same
    stdlib-urllib-only convention utils/image_client.py already uses for
    ycplt_img, for the same reason (a tiny, occasional HTTP client isn't
    worth a new dependency).

    No lock here at all, unlike _generate_embedded above — this is the
    entire point of the "server" backend: llama-server's own --parallel
    slots + continuous batching (-cb) handle multiple simultaneous
    requests safely on its own side, so a second call landing here while
    the first is still in flight is not a race condition to guard against,
    it's the intended, fixed behavior (see config.LLM_BACKEND's own
    comment for why this was worth doing at all).

    reasoning_effort="none": a real, confirmed second bug beyond the
    reasoning_content one --reasoning-format none already fixes.
    --reasoning-format only controls WHERE an already-generated <think>
    block ends up in the JSON response, not WHETHER the model generates
    one at all — confirmed in practice: even with --reasoning-format
    none in place, tool_router's tight max_tokens=120 budget was still
    entirely consumed by an in-progress reasoning trace (visible,
    unclosed, cut off mid-thought - e.g. "...I need to" - instead of
    landing on the actual tool name), never reaching a usable answer.
    llama-server's own docs (tools/server/README.md) are explicit that
    the per-request "reasoning_effort": "none" field disables
    reasoning/thinking outright, not just where it's reported - exactly
    what's needed, since this app never reads the reasoning trace on
    EITHER backend (see _THINK_BLOCK_RE below, which exists purely to
    discard it from the embedded backend's own inline output). Setting
    this here, in the request body itself, means it's enforced by this
    app's own code regardless of how llama-server happens to be
    started, rather than relying on a CLI flag someone could forget."""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "repeat_penalty": config.REPEAT_PENALTY,
        "reasoning_effort": "none",
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{config.LLAMA_SERVER_URL}/v1/chat/completions",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    # <= 0 disables the timeout entirely (urlopen(timeout=None) waits
    # indefinitely) — see config.LLAMA_SERVER_TIMEOUT_SEC's own comment for
    # why the default is generous rather than short: this covers a full
    # generation, not just a liveness probe.
    timeout = config.LLAMA_SERVER_TIMEOUT_SEC if config.LLAMA_SERVER_TIMEOUT_SEC > 0 else None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"llama-server returned HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"llama-server unreachable ({config.LLAMA_SERVER_URL}): {e}") from e

    message = payload["choices"][0]["message"]

    # Real, confirmed bug this guards against: llama-server's own
    # --reasoning-format defaults to "auto", which for a "thinking" model
    # (this app's Qwen3.5-9B included) routes the <think> block into a
    # SEPARATE message.reasoning_content field instead of leaving it
    # inline in message.content (see llama.cpp's own
    # tools/server/README.md). install/llama-server.service already
    # passes --reasoning-format none specifically to avoid this — that
    # puts reasoning back inline in content, matching the embedded
    # backend's own behavior exactly, so _THINK_BLOCK_RE below strips it
    # identically either way. This fallback exists only as a second line
    # of defense (a future llama-server upgrade changing defaults, or a
    # deployment that forgot the flag): confirmed in practice that a
    # tight max_tokens budget (utils/tool_router.py's classifier calls)
    # can be entirely consumed by reasoning under "auto", leaving content
    # empty with no error — tool_router silently treated that as "no
    # tool", not a crash, so this failure mode is easy to miss without
    # looking at the raw response. Falling back to reasoning_content
    # rather than returning an empty string at least surfaces SOME text
    # instead of silently discarding a real (if incomplete) answer.
    content = message.get("content") or ""
    if not content and message.get("reasoning_content"):
        content = message["reasoning_content"]
    return content


def generate_sync(prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
    """Runs one generation. max_tokens=None (the default) means "no artificial
    cap" — the model then generates until it emits a stop token or the
    context window (config.N_CTX, or llama-server's own --ctx-size) is
    full, whichever comes first. Pass an explicit max_tokens only when a
    caller genuinely wants a shorter answer (e.g. utils/intent.py's
    one-word classifiers).

    Dispatches to _generate_embedded or _generate_server per
    config.LLM_BACKEND — see this module's own top comment and
    config.LLM_BACKEND's comment for the full rationale. Every existing
    caller (there are many, across utils/intent.py, utils/tool_router.py,
    utils/interpret.py, routes/chat.py, ...) is unaffected by which
    backend is active: this function's signature and return value are
    identical either way.

    embedded backend: serialized on _llm_lock — a real, reported gap: this
    app handles more than one conversation at once (separate chats in the
    sidebar), and nothing before this stopped two independent /chat
    requests from both reaching this function at the same time on two
    different threads. llama-cpp-python's own Llama class has no internal
    thread-safety guard at all (checked directly against its source — no
    lock, no threading import anywhere in it), so two concurrent
    create_chat_completion calls against the SAME _llm instance would race
    on its internal context/KV cache with no protection — not just
    "slower", but a real risk of garbled output or a crash. The single
    loaded model is a hard limit either way — genuinely SIMULTANEOUS
    generation of two different answers isn't possible on one CPU-bound
    model instance without a second one resident in memory — what this
    lock actually buys is turning an unsafe race into a safe, correct,
    strictly-ordered FIFO queue.

    server backend: no lock — real, reported limitation of the embedded
    backend's FIFO queue this exists to fix: a long generation for one
    conversation made every OTHER conversation's own message, even a
    trivial "hi, how are you", wait in strict order behind it, confirmed
    in practice. llama-server's own --parallel slots + continuous batching
    interleave a short request into the next decode step across all active
    slots instead of forcing it to wait for one long generation to finish
    outright — see config.LLM_BACKEND's own comment for the honest caveat
    that this improves fairness/interleaving on CPU-only hardware, not raw
    throughput (no idle parallel matrix units to exploit the way a GPU
    would give)."""
    if _llm is None:
        raise RuntimeError("Model is not loaded")

    if config.LLM_BACKEND == "server":
        content = _generate_server(prompt, max_tokens, temperature)
    else:
        content = _generate_embedded(prompt, max_tokens, temperature)

    # A model that emits an unclosed <think> (truncated by max_tokens before
    # it ever reached </think>) would otherwise have its ENTIRE answer
    # eaten by a greedy strip — only strip a properly closed block, and
    # leave an unclosed one as-is (a real but rare failure mode: the caller
    # gets a visibly weird answer that at least isn't silently empty,
    # rather than this function hiding the fact that generation ran out of
    # budget mid-thought). Applies identically regardless of which backend
    # produced content.
    if "<think>" in content and "</think>" in content:
        content = _THINK_BLOCK_RE.sub("", content).strip()
    return content


async def generate_async(prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, generate_sync, prompt, max_tokens, temperature)
