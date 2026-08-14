"""Loading and calling the GGUF model via llama-cpp-python.

Keeps a single Llama instance per process (module-level singleton),
available through get_llm() after load_llm() has been called at app startup.
"""
import os
import re
import asyncio
import threading
from typing import Optional

from llama_cpp import Llama

from utils import config

_llm: Optional[Llama] = None


class _FifoLock:
    """A mutex that grants access in strict first-come-first-served order —
    unlike a plain threading.Lock, whose underlying OS mutex makes no
    ordering promise among several blocked waiters (whichever thread the
    OS/runtime happens to wake next gets it, which in practice is "roughly
    fair" but not guaranteed, and isn't even the point: a plain Lock only
    guarantees mutual exclusion, not ordering).

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
# threading.Lock — every generate_sync call (the one place every caller,
# sync or async, funnels through) now goes through here.
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
# place every caller's output passes through — rather than in each of the
# many individual prompts/callers. A no-op for any model that doesn't emit
# think-tags at all (today's default, Qwen2.5-3B-instruct, doesn't), so
# this is safe to leave in regardless of which model is configured.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def load_llm() -> Llama:
    """Loads the model from config.MODEL_PATH. Raises RuntimeError with a
    clear message if the file is missing or not in GGUF format."""
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

    print("Model loaded successfully.")
    return _llm


def get_llm() -> Optional[Llama]:
    return _llm


def generate_sync(prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
    """Runs one generation. max_tokens=None (the default) means "no artificial
    cap" — llama-cpp-python then generates until the model emits a stop token
    or the context window (config.N_CTX) is full, whichever comes first.
    Pass an explicit max_tokens only when a caller genuinely wants a shorter
    answer (e.g. utils/intent.py's one-word classifier).

    Serialized on _llm_lock — a real, reported gap: this app handles more
    than one conversation at once (separate chats in the sidebar), and
    nothing before this stopped two independent /chat requests from both
    reaching this function at the same time on two different threads.
    llama-cpp-python's own Llama class has no internal thread-safety guard
    at all (checked directly against its source — no lock, no threading
    import anywhere in it), so two concurrent create_chat_completion calls
    against the SAME _llm instance would race on its internal context/KV
    cache with no protection — not just "slower", but a real risk of
    garbled output or a crash, since nothing here previously prevented it.
    The single loaded model is a hard limit either way — genuinely
    SIMULTANEOUS generation of two different answers isn't possible on one
    CPU-bound model instance without a second one resident in memory (a
    real option, just a much bigger RAM/ops cost, not something this fixes
    by itself) — what this lock actually buys is turning an unsafe race
    into a safe, correct, strictly-ordered FIFO queue (see _FifoLock's own
    docstring for why "strictly ordered" needed calling out separately
    from just "safe"): whichever request asked for the lock first is
    guaranteed to be served first, and finishes before the next one
    starts, instead of both corrupting each other or racing in whatever
    order the OS happened to wake them. Chart/data computation for a
    different conversation is NOT affected — that work doesn't touch _llm
    at all and genuinely can run in parallel; only the actual model call
    is serialized (and now queued) here."""
    if _llm is None:
        raise RuntimeError("Model is not loaded")
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
    content = out["choices"][0]["message"]["content"]
    # A model that emits an unclosed <think> (truncated by max_tokens before
    # it ever reached </think>) would otherwise have its ENTIRE answer
    # eaten by a greedy strip — only strip a properly closed block, and
    # leave an unclosed one as-is (a real but rare failure mode: the caller
    # gets a visibly weird answer that at least isn't silently empty,
    # rather than this function hiding the fact that generation ran out of
    # budget mid-thought).
    if "<think>" in content and "</think>" in content:
        content = _THINK_BLOCK_RE.sub("", content).strip()
    return content


async def generate_async(prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, generate_sync, prompt, max_tokens, temperature)
