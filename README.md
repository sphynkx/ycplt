# ycplt — local astrology chat assistant (FastAPI + llama-cpp-python)

A local, self-hosted assistant for Western tropical astrology: natal charts,
transits, secondary progressions, solar-arc directions, lunar/solar returns,
profections, synastry, horary questions, electional astrology, and birth-time
rectification (two methods) — all computed deterministically (Swiss
Ephemeris via [kerykeion](https://github.com/g-battaglia/kerykeion), fully
offline, no API keys), then explained in natural language by a local GGUF
model, grounded against a project-authored, per-technique reasoning
methodology via RAG rather than left to the model's own unaided judgment. A
**universal help assistant** (`astro_help_assistant`) is built in for anyone
who isn't sure which technique they need, doesn't know astrology
terminology, or just wants a plain-language question answered — see
"Universal help assistant" below. Every technique's result can be rendered as
an SVG wheel chart and exported to PDF alongside the text explanation.

The project started as a general-purpose local ChatGPT-like chat app and
still has that foundation underneath: a FastAPI server running a GGUF model
on CPU (llama-cpp-python), a browser UI (sidebar with parallel chats,
persisted history), a small extensible built-in tool layer (date/time,
calculator, and the astrology tools above), code extraction from model
replies as downloadable file attachments, and automatic image
generation/editing requests routed to a companion service
([ycplt_img](https://github.com/sphynkx/ycplt_img)) based on the meaning of
the user's message (no manual toggle). All of that general-purpose machinery
is still there and still works standalone (plain chat, image generation/
editing, RAG over any documents on any subject) — astrology is simply what
this deployment's own tool layer and RAG corpus are currently built out
for.

## Project layout

```
app.py                    — entry point: FastAPI app, routers, startup wiring
routes/
  chat.py                   — POST /chat, GET /health
  conversations.py          — /api/conversations — list/create/history/delete
  files.py                  — /api/files/{id} — download file attachments
  pages.py                  — GET / (browser chat page)
  profiles.py                — /api/profiles — birth-profile CRUD + AstroZet .zbs import/export
db/
  connection.py              — SQLite connection, schema, init_db(), migrations
  repository.py               — CRUD: conversations, messages, files, birth_profiles
utils/
  astrozet.py                 — AstroZet .zbs format: parse/export (see routes/profiles.py)
  config.py                  — all settings, loaded from .env (python-dotenv)
  llm.py                      — loads and calls the GGUF model (llama-cpp-python)
  rag.py                      — optional RAG (FAISS + sentence-transformers)
  codeblocks.py                — extracts fenced code blocks from model replies
  intent.py                    — LLM-based classifiers: image request? edit vs question?
  image_client.py               — HTTP client for the ycplt_img job queue
  image_jobs.py                 — background poller resolving pending image jobs
                                   (generation/edit results, and caption text answers)
  tools.py                      — registry of built-in tools (datetime, calculator, astro chart, ...)
  tool_router.py                 — LLM-based classifier: does this need a tool?
  astro.py                       — natal/transit/progression/direction/return/profection/
                                   synastry chart computation via kerykeion (optional)
  rectification.py               — birth-time rectification (Trutine of Hermes search) via kerykeion (optional)
  rectification_events.py        — multi-technique, multi-event birth-time rectification via kerykeion (optional)
templates/
  index.html                  — sidebar + chat UI markup (fetch to /chat and /api/*)
static/
  js/app.js                    — browser-side JavaScript, served at /static/js/app.js
build_index.py             — builds one FAISS index per corpus from RAG source documents
install/
  requirements.txt            — Python dependencies
  .env.example                 — template for .env (copy and adjust)
  ycplt.service                — systemd unit file
  methodologies/                — master copies of every project-authored `*_methodology.txt`
                                  reasoning doc (interpretation, transit, synastry, progression,
                                  direction, lunar/solar return, profection, both rectification
                                  techniques, horary). Not read directly by the app — copy the file(s)
                                  relevant to a topic into that topic's `rag_data/<topic>/`
                                  subfolder to actually activate it (see "Recommended rag_data/
                                  layout" below); kept centrally here so there's one canonical
                                  copy per technique instead of duplicates drifting across
                                  several rag_data/ subfolders.
models/model.gguf          — the model file (you provide it, see below)
rag_data/                  — RAG source documents (.txt/.html/.rtf/.pdf/.doc/.djvu/.chm/.zip/.rar/.arj/.7z/.ha),
                             organized into topic subfolders (methodology files copied in from
                             install/methodologies/, plus your own selected reference corpus per
                             topic — see "Recommended rag_data/ layout" below) — you provide them
data/                      — generated data:
  rag_index/<topic>/          — one faiss_index.bin+meta.pkl pair per corpus (build_index.py)
  chat.sqlite3                — conversations/messages/files (created at startup)
```

## Hardware and why this stack

Current reference hardware: Intel Core i7-8700 @ 3.20GHz (6 physical cores /
12 threads), 16 GB RAM, no discrete GPU used for inference — at least 25 GB
free disk space (the chat model alone is ~6 GB; add RAM-cache headroom, the
embedding model, and RAG index data on top). Currently running
**Qwen3.5-9B, `UD-Q4_K_XL` quant** (~6 GB) as the chat model — see "3. Chat
model" above for the full download link and quant alternatives — which is
fine for plain chat on this hardware, but the heaviest single-request
workloads (a full RAG-interpreted horary, electional, or progression
answer — several sequential LLM calls per request, each fed a large
methodology document) measured 10-20 minutes in real testing, not seconds.
See "Tuning CPU inference threads" right below before assuming this needs
better hardware — the two thread-count settings involved were previously
never independently tunable, and the naive "more threads = faster"
assumption measured FALSE on this exact machine.

### Tuning CPU inference threads

llama-cpp-python exposes two independent thread-count settings that are
easy to conflate, and this project only ever set one of them until real
multi-hour testing on the current reference hardware surfaced the gap:

- **`N_THREADS`** — governs only the token-*generation* phase (sequential,
  one token at a time). Real measurement on the i7-8700 (6 physical/12
  logical cores): `N_THREADS=4` finished a progression request end-to-end
  faster than both `N_THREADS=6` and `N_THREADS=10` on the same prompt —
  6 was in fact the slowest of the three, over 30 minutes before the test
  was aborted. This isn't a fluke: generation is memory-bandwidth-bound
  and pays a synchronization barrier across every thread once per
  generated token, so past some point that depends on your memory
  bandwidth (not your core count), more threads add more of that overhead
  than they add useful parallel work. Don't assume a higher number is
  better here — re-measure a real request end-to-end.
- **`N_THREADS_BATCH`** — governs the prompt-*processing*/prefill phase,
  which is more parallelizable than generation and was never set
  explicitly in this project before — llama-cpp-python's own default
  (`multiprocessing.cpu_count()`, i.e. every logical core) silently took
  over instead. This is why `htop` could show all 12 logical cores at
  ~100% even while `N_THREADS` was turned down to 6 — that observation
  was about this setting, not the one being changed. Now independently
  configurable; see `install/.env.example`.
- **`N_BATCH`** — a batch *size* (prompt tokens processed per parallel
  step during prefill), not a thread count; also never set explicitly
  before (library default: 512).

Tried `N_THREADS=4` + `N_THREADS_BATCH=8` + `N_BATCH=2048` together on a
natal request next: the tool-dispatch and chart-computation steps were
each genuinely fast (under a minute apiece, better than any earlier run),
but the final answer still took 31 minutes end-to-end — clearly the
generation phase itself, not prefill. Reverting all three to their plain
defaults (`N_THREADS=4`, `N_THREADS_BATCH`/`N_BATCH` unset) measured 835
seconds on the same technique — worse than the very first 601s baseline,
but far better than 31 minutes. No combination of these three settings
tried so far has beaten the plain defaults, and single-run wall-clock
comparisons across different techniques (each pulling in different-sized
methodology text, generating a different actual answer length) are noisy
enough — background load, thermal behavior over a 20-30 minute run, etc.
— that a one-off worse or better result doesn't reliably prove a setting
change is the cause. Bottom line: leave `N_THREADS_BATCH`/`N_BATCH` unset
(library defaults) unless you're prepared to test a change with several
repeated runs of the *same* technique/query, not a single comparison
across different requests — that's the only way to tell a real effect
from run-to-run noise on this kind of long-running CPU workload.

This replaces an earlier, noticeably weaker reference machine: i7-5500U (2
physical cores / 4 threads), 12 GB RAM, GeForce 940M (2 GB, no usable GPU
acceleration for LLM inference) — on that hardware, even a 7B model ran at
around 1 token/sec, too slow for comfortable chat, which is why the
project's shipped default started at 3B and only moved up to today's 9B
default after this hardware upgrade (see "3. Chat model" above for the
full history of that change and the measured latency difference between
the two). If you're deploying on hardware closer to the OLD reference specs
above, drop back down to a 3B-class (or smaller) model instead — see the
lighter alternatives listed in "3. Chat model".

Hence, independent of whichever specific machine this runs on:

- Inference via **llama-cpp-python** — native GGUF support, CPU inference,
  actively maintained.
- Model size/quantization should be picked to match the actual CPU
  available, not assumed — a model that's comfortably fast on a modern
  6-core/12-thread desktop can be unusably slow on an older 2-core/4-thread
  laptop CPU, and vice versa there's no reason to stay on a smaller/weaker
  model than the hardware can actually support (see the small-vs-large
  model trade-off discussion in "3. Chat model" above).
- GPU is not used (`N_GPU_LAYERS=0` by default) on either reference machine
  — raise it if you do have a GPU with enough VRAM for the model you're
  running; llama-cpp-python supports GPU offload, this project simply
  hasn't needed it on either machine documented here.
- Chat history lives in **SQLite** (`data/chat.sqlite3`) — enough for a
  single-user local app, no separate database server needed.

## Concurrency: one model, many chats

The sidebar supports any number of parallel conversations, and the app
runs as one `uvicorn` process (no `workers=` argument) with one process-
wide `Llama` instance (`utils/llm.py`) — so what actually happens when two
chats are used at the same time is worth being precise about.

**What genuinely runs in parallel.** FastAPI dispatches each `/chat`
request's blocking work (astro chart computation, RAG retrieval, tool
routing) onto a thread pool via `loop.run_in_executor(...)`, and none of
that touches the shared model — two different conversations' chart/data
prep can and does run at the same time on separate threads, using whatever
spare CPU cores are available.

**What does not, and why.** Actual answer generation is a different
story: `llama-cpp-python`'s `Llama` class has no internal thread-safety
guard at all (checked directly against its source) — two threads calling
`create_chat_completion` on the same instance at the same time would race
on its internal context/KV cache with nothing stopping them. Before this
was noticed, nothing in this app prevented that either: two chats
answered close together in time could have corrupted each other's output
or crashed, not just been slow. `utils/llm.py`'s `generate_sync` (the one
function every LLM call in this app funnels through, sync or async) now
serializes every call through a module-level FIFO queue.

**Why a queue and not just a lock.** A single `/chat` request already
makes several *sequential* calls into this module on its own — intent
classification, tool routing, a tool's own field/round extraction,
digest/fact-summarization, and finally the answer itself (anywhere from
2 to 7 calls depending on the request, see `generate_sync`'s own
docstring). A plain `threading.Lock` only guarantees that two calls never
run at the same instant — it makes no promise about *order* among
several waiting threads, so with two conversations genuinely in flight
at once, one conversation's own back-to-back calls had no guarantee of
finishing in a sane order relative to the other's, and there was no
guarantee against one conversation being repeatedly outraced by another.
`utils/llm.py`'s `_FifoLock` fixes this: every caller — a quick
classifier call and a long final-answer generation alike — draws a
ticket the instant it asks for the lock and is served strictly in that
arrival order, regardless of which conversation it belongs to. It's a
plain `threading.Lock`-based implementation under the hood (not
`asyncio.Lock`), for the same reason as before: most callers reach it
from a worker thread via `run_in_executor`, not from inside a coroutine,
where `asyncio.Lock` isn't usable at all. Verified directly: eight
threads with staggered start times, mixing the fake model's calls,
completed in exactly their arrival order every time, with zero mutual-
exclusion violations and the expected fully-serialized total time. When
there's real contention, the app logs a line noting how many requests
are already ahead in the queue — silent the rest of the time.

**The honest limit.** This turns an unsafe, unordered race into a safe,
strictly-ordered, one-request-at-a-time queue for generation — it does
NOT make two chats' answers generate *simultaneously*. That's a real
hardware/architecture limit, not a bug: there is exactly one CPU-bound
model instance, and genuinely simultaneous generation of two different
answers would need a second one resident in memory (a real option — see
"Hardware and why this stack" above for what that costs in RAM — just a
much bigger change than this fix). In practice: send a message in a
second chat while the first is still working, and it queues safely and
fairly behind it rather than racing with it or being silently dropped —
the reply lands in its own conversation's history once its turn comes,
correctly, regardless of which chat happens to be open on screen by then
(see "Parallel chats" above for the matching frontend fix, which already
guards against a reply ever being misapplied to whichever chat is open
when it arrives).

### Separate llama-server backend (optional, off by default)

The FIFO queue above is safe and correct, but it's still strictly one
request at a time, system-wide — real, reported consequence: a long
generation for one conversation (an image-edit job's classifier calls, a
long RAG-heavy astro answer, ...) makes every OTHER conversation's own
message, even a trivial "hi, how are you", visibly wait its turn behind
it, confirmed in practice with two chats open at once.

`utils/config.LLM_BACKEND` ("embedded", the default, or "server") switches
`utils/llm.py` between the original in-process `Llama` object above and a
separately-running **llama-server** instance — the native C++ server
binary from the `llama.cpp` project itself (`ggml-org/llama.cpp`'s
`tools/server`), run as its own process on the SAME machine as this app.
It supports real concurrent request handling via multiple "slots"
(`--parallel`/`-np`) with continuous batching (`-cb`, on by default).

**Important: this is NOT the same thing as `llama-cpp-python`'s own
bundled `llama_cpp.server` module** (`python3 -m llama_cpp.server`) —
checked directly against its source (`llama_cpp/server/app.py`): it wraps
a single `Llama` object behind one `anyio.Lock`, functionally identical
to this app's own `_FifoLock` above, just with a network hop added and no
real concurrency gained. The native `llama-server` binary (built from
`llama.cpp`'s own source, or a matching prebuilt release for your CPU) is
what actually has parallel slots.

**Honest expectation on CPU-only hardware:** continuous batching's usual
GPU benefit is throughput, from bigger fused matrix multiplies filling
otherwise-idle parallel compute — CPU has no equivalent idle capacity to
exploit; the physical cores are already the bottleneck either way. The
real, still-genuine benefit here is **fairness/interleaving, not raw
throughput**: with N slots, a short classifier call arriving mid-generation
gets folded into the *next* decode step across all active slots, instead
of waiting for one long generation to finish outright — directly fixing
the "hi, how are you gets stuck" problem above, at little to no cost to
total tokens/sec.

**Setup:**

1. Build or download `llama-server` from
   [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (matching
   your CPU's instruction set — AVX2/AVX512/NEON, same consideration as
   any llama.cpp build) — a separate binary from the `llama-cpp-python`
   pip package already used elsewhere in this project, though it reads
   the exact same GGUF model file.
2. Run it against the same `MODEL_PATH` this app already uses, choosing a
   `--parallel` slot count for your CPU's core count (a starting point:
   2-3 slots on a modest machine, 3-4 with 12+ cores and RAM to spare —
   more slots isn't free, each needs its own share of `--ctx-size` for the
   KV cache):

   ```bash
   llama-server \
     --model models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
     --host 127.0.0.1 --port 4012 \
     --parallel 3 --ctx-size 98304 \
     --threads 4 --threads-batch $(nproc) \
     --reasoning-format none
   ```

   Port `4012`, not llama-server's own upstream default of `8080` — see
   "Port allocation across the ycplt family" at the top of Installation &
   Setup.

   **`--ctx-size` must be `N_CTX * --parallel`, not just `N_CTX`.**
   llama-server splits `--ctx-size` evenly across `--parallel` slots
   (`n_ctx_slot` in its own startup log) — the embedded backend, by
   contrast, gives one request the WHOLE of `config.N_CTX` (32768).
   Real, confirmed consequence of leaving `--ctx-size` at 32768 with
   `--parallel 3`: each slot only got ~11008 tokens, and a real
   `astro_natal_chart` request (full RAG methodology + per-planet digest
   + sectioned-answer prompt) needing 14314 tokens was flatly rejected:
   `HTTP 400 exceed_context_size_error`. `98304` = `32768 * 3` restores
   the same full-`N_CTX` headroom per slot that the embedded backend
   already relies on for this app's heaviest techniques (horary,
   electional, and rectification can run just as large). This roughly
   **triples the KV cache's memory footprint** versus the old 32768 —
   if that doesn't fit in available RAM, reduce `--parallel` instead
   (e.g. `--parallel 2` with `--ctx-size 32768` still gives ~16384/slot,
   more headroom than the original 11008, at the cost of one fewer
   concurrent conversation) rather than lowering `--ctx-size` below
   `N_CTX`.

   **`--reasoning-format none` is required, not optional — but it's only
   half the fix.** Real, confirmed bug: llama-server's own default
   (`--reasoning-format auto`) routes a "thinking" model's `<think>`
   block into a separate `message.reasoning_content` JSON field instead
   of leaving it in `message.content` (see llama.cpp's own
   `tools/server/README.md`) — this app's `utils/llm.py` only reads
   `message.content`. A classifier call with a small `max_tokens` budget
   (`utils/tool_router.py`, `utils/intent.py`) can have its ENTIRE budget
   consumed by reasoning under `auto`, coming back completely empty —
   confirmed in practice: `tool_router` logged `raw=''` and silently
   failed to route an otherwise-correct `astro_natal_chart` request.
   `none` leaves thoughts unparsed in `content` instead, matching the
   embedded backend's own behavior, so `_THINK_BLOCK_RE` strips a
   properly-closed `<think>` block the same way regardless of backend.

   **But `--reasoning-format` only controls WHERE an already-generated
   think block ends up, not WHETHER the model generates one at all** —
   confirmed in practice a second time: even with `--reasoning-format
   none` in place, `tool_router`'s 120-token budget was STILL entirely
   consumed by an in-progress, unclosed reasoning trace (visible this
   time, cut off mid-thought instead of hidden in `reasoning_content`),
   never reaching the actual tool name. The real fix is
   `utils/llm.py`'s own `_generate_server()`, which now sends
   `"reasoning_effort": "none"` in every request body —
   llama-server's docs are explicit this disables reasoning/thinking
   outright, not just where it's reported. This app never reads the
   reasoning trace on either backend (see `_THINK_BLOCK_RE`'s own
   comment — it exists purely to discard it), so there's no downside to
   turning it off entirely. Being set in the request body itself (not
   just a CLI flag) means it applies no matter how llama-server was
   started. `_generate_server()` also falls back to `reasoning_content`
   if `content` ever comes back empty anyway, as a last-resort second
   line of defense.

   **`--threads-batch` matters, don't skip it.** llama.cpp's own CLI
   default makes `--threads-batch` equal to `--threads` if left unset —
   i.e. the same small thread count (4 above) used for token GENERATION
   would also apply to PROMPT PROCESSING, unlike the embedded backend,
   which already defaults `N_THREADS_BATCH` to the full logical core
   count for exactly this reason (see that setting's own comment in
   `utils/config.py`). Real, reported consequence of leaving this unset: a
   trivial one-line test message took several minutes end-to-end after
   switching to this backend — most likely this (prefix/prompt processing
   for a history- and system-prompt-heavy request, now running on only 4
   cores instead of every core), though not yet confirmed with a controlled
   before/after measurement on that specific machine.

   Put this behind its own systemd unit for the same reasons `app.py`
   and `ycplt_img` already have one (survives reboots, restarts on
   crash) — see `install/llama-server.service` for a ready-to-copy unit
   (adjust `WorkingDirectory`/model path/flags to match your own layout).
3. Set in `.env`:

   ```bash
   YCPLT_LLM_BACKEND=server
   YCPLT_LLAMA_SERVER_HOST=127.0.0.1
   YCPLT_LLAMA_SERVER_PORT=4012
   ```

   Note the `YCPLT_` prefix on all three — unlike every other setting in
   `.env`. A real, reported mix-up: writing `LLM_BACKEND=server` (matching
   the unprefixed style everything else uses) is silently ignored, with no
   warning, and this app quietly keeps using the embedded backend.
   `install/.env.example` keeps `YCPLT_LLM_BACKEND` commented out
   (embedded stays the default there) after the `reasoning_content` bug
   above was found — uncomment and set these three exactly once
   llama-server is running with `--reasoning-format none`.
4. Restart this app. `utils/llm.load_llm()` checks `llama-server`'s
   `/health` endpoint at startup and fails fast with a clear error if it
   isn't reachable, the same guarantee a missing `MODEL_PATH` file already
   gives for the embedded backend.

Revert at any time by setting `YCPLT_LLM_BACKEND=embedded` (or removing
the line — that's `install/.env.example`'s own default again) and
restarting — nothing else needs to change, matching the same
off-by-default, single-flag-revert pattern already used for
`ycplt_img`'s `RECONSTRUCT_ENABLED`/`KONTEXT_ENABLED`.

**Real-hardware verdict, after three rounds of fixes:** `tool_router`
now correctly triggers `astro_natal_chart` via llama-server once
`reasoning_effort: "none"` is set (see `_generate_server()`'s own
comment) — confirmed with a fresh natal-chart test. Two further, real
issues surfaced immediately after that fix, both now addressed above:

1. **Context overflow**: the full RAG-heavy prompt for a real technique
   needed 14314 tokens, but `--ctx-size 32768` split across
   `--parallel 3` only gave each slot ~11008 — fixed by raising
   `--ctx-size` to `98304` (`N_CTX * --parallel`, see step 2 above).
2. **Client-side timeout**: with the larger `--ctx-size` (a bigger KV
   cache to scan per token), generation slowed further (~2.6 tok/s
   observed) and a long answer was still mid-generation, making real
   progress, when this app's own `LLAMA_SERVER_TIMEOUT_SEC` (previously
   1200s) cut the connection — llama-server's own log showed the task
   being cancelled, not failing on its own. Now `0` (disabled) by
   default, matching the embedded backend's own total absence of a
   timeout (see `utils/config.py`'s own comment).

A full RAG-heavy answer (e.g. a natal chart reading) takes a comparable
~18-20+ minutes on this hardware whether generated via the embedded
backend or via llama-server, once the technique actually gets triggered
correctly on both — that's this model's (Qwen3.5-9B) real per-token
generation speed on this CPU for a long, multi-section answer, not
something either backend can fix. The concurrency benefit (not blocking
other chats behind one long answer) is still the entire point of this
backend; it was never meant to make any single answer faster.

### Tiny router model for classification calls (optional, off by default)

Roughly 19 of the 23 call sites that talk to the LLM in this app aren't
generating a user-facing answer at all — they're one-shot classifiers:
`tool_router` deciding which tool (if any) to call, `intent` deciding
whether a message is an edit vs. a question, `horary`/`electional`
extracting a date/field from free text, and so on. Every one of these
already used `temperature=0.0` and a tight `max_tokens` (5-400) long
before this feature existed, because they only ever need a single short
decision, not prose. Running all of them through the same big answer
model (Qwen3.5-9B on this hardware) means every tool call pays that
model's full per-token latency just to produce a one-word or
one-sentence classification — this is the "router overhead" the app's
own logs make very visible while a chat is waiting for `tool_router` to
finish before it can even start the real answer.

`utils/llm.py` now has a second, independent model slot for exactly
this: `classify_sync()` / `classify_async()`, backed by `_router_llm`,
loaded by `load_router_llm()` at startup and torn down by
`close_router_llm()` at shutdown, both called from `app.py`'s lifespan
next to the main model's own `load_llm()`/`close_llm()`. All 19
classifier-style call sites now call `classify_sync()`/`classify_async()`
instead of `generate_sync()`/`generate_async()`; the 4 genuinely
answer-style sites (the actual chat reply in `routes/chat.py`, the
per-planet interpretation prose in `utils/interpret.py`, and the image
caption rephrase in `utils/image_jobs.py`) were deliberately left
untouched — they need the big model's actual writing quality.

**Off by default, zero risk**: `ROUTER_MODEL_PATH` (see below) is empty
by default. Whenever it's unset, or set but the file doesn't load for
any reason, `classify_sync()` transparently falls back to calling
`generate_sync()` — i.e. today's exact behavior, unchanged. Unlike the
main model, a bad/missing router model is **never fatal**: `load_router_llm()`
just logs a warning and moves on. This means the feature can be added to
a checkout with no configuration changes at all, and turned on later by
just pointing `ROUTER_MODEL_PATH` at a real file and restarting.

**Always embedded, regardless of `LLM_BACKEND`**: the router model is
loaded in-process via `llama_cpp.Llama` even when the main model is
using the separate `llama-server` backend above. A model this small
gains nothing from llama-server's multi-slot concurrency machinery (the
whole point of that backend is letting several *long* answers run in
parallel without blocking each other) — a classification call is single,
short, and already fast enough that adding a second network hop to a
separate server process would likely cost more than it saves.

**Choosing a model**: this needs to be small and fast above all else —
accuracy only matters insofar as it still reliably makes the same
narrow decisions these prompts already ask for (which tool, edit vs.
question, yes/no). The recommended, verified model is
[Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
— quantized directly by the Qwen team itself (NOT `unsloth` — an earlier
draft of this section pointed at a non-existent `unsloth/...` repo; this
is the real one, confirmed by fetching the page directly). A
0.49B-parameter model, small enough that the `Q4_K_M` quant
(`qwen2.5-0.5b-instruct-q4_k_m.gguf`) is only 491 MB and loads/runs
almost instantly on CPU. The repo's own available quants are `Q2_K`
(415 MB), `Q3_K_M` (432 MB), `Q4_0`/`Q4_K_M` (~430-491 MB), `Q5_0`/`Q5_K_M`
(~490-522 MB), `Q6_K` (650 MB), `Q8_0` (676 MB) — there's no `Q1` quant
offered at all (confirmed on the model page), and a model this small has
very little redundancy left to cut before it stops reliably following
instructions, so `Q4_K_M` is a reasonable default rather than chasing the
smallest file. Download the specific `.gguf` file from that repo's
"Files and versions" tab and point `YCPLT_ROUTER_MODEL_PATH` at it.

```bash
YCPLT_ROUTER_MODEL_PATH=models/qwen2.5-0.5b-instruct-q4_k_m.gguf
YCPLT_ROUTER_N_CTX=8192
YCPLT_ROUTER_N_THREADS=12
```

**Important — this line must go in your real `.env`, not just
`install/.env.example`** (that file is only a template this app never
reads directly — see "Separate llama-server backend" above for the exact
same class of mix-up with `YCPLT_LLM_BACKEND`). After setting it, restart
the app; `utils/llm.py`'s `load_router_llm()` now always prints one of
three lines at startup so this is never ambiguous again: "ROUTER_MODEL_PATH
not set" (feature off, using the main model), "ROUTER_MODEL_PATH is set
but not found" (typo'd path), or "Router model loaded successfully" —
if you don't see any of these, the app didn't restart with the new
setting picked up.

`YCPLT_ROUTER_N_CTX` defaults to `8192` — smaller than the main model's
`N_CTX` (32768), but generous enough for `tool_router`'s own
history-heavy prompts, which in practice can reach several thousand
tokens once recent chat history is included. `YCPLT_ROUTER_N_THREADS`
defaults to this machine's full logical core count (unlike the main
model's deliberately low `N_THREADS=4` — see "Tuning CPU inference
threads" above); this is a reasonable starting guess for a model this
tiny, not something measured on real hardware yet the way the main
model's thread count was.

Restart the app after changing any of these three — same as every other
`.env` setting.

### Remote LLM provider for the main answer (optional, off by default)

Separate from the tiny router model above, this lets the **main chat
answer** (the actual long-form reply — `routes/chat.py`, plus the
per-planet interpretation prose and image-caption rephrase) be generated
by an external API instead of the local model, when the local model's
generation time is a problem (a full RAG-heavy answer can take 18-20+
minutes on modest CPU hardware — see "Separate llama-server backend"
above).

**This does NOT affect classifier calls** — `tool_router`, `intent`,
field extraction, and every other `classify_sync()`/`classify_async()`
call site (see "Tiny router model" above) always run locally regardless
of this setting, since routing a one-word classification through a
network API would only add latency for no benefit.

```bash
REMOTE_LLM_PROVIDER=openai
REMOTE_LLM_API_KEY=sk-...
REMOTE_LLM_MODEL=gpt-4o-mini
REMOTE_LLM_TIMEOUT_SEC=0
```

- `REMOTE_LLM_PROVIDER` — empty by default (feature off, local model
  only). Two supported values: `openai` (OpenAI's `/v1/chat/completions`)
  or `claude` (Anthropic's own Messages API, `utils/llm.py`'s
  `_generate_remote_claude` — a genuinely different request/response
  shape: `x-api-key`/`anthropic-version` headers instead of a bearer
  token, `max_tokens` is required rather than optional, and the reply
  comes back as a list of content blocks rather than one message
  string). The name is deliberately generic (a provider-choice string,
  not a boolean) so supporting a second provider was just one more
  accepted value here, not a second, differently-named setting.
- `REMOTE_LLM_API_KEY` — your OpenAI or Anthropic API key, matching
  whichever `REMOTE_LLM_PROVIDER` you set. For Claude, generate one at
  [console.anthropic.com](https://console.anthropic.com) → API Keys; a
  key created for general inference use works fine here (no special
  scope is required for the Messages API).
- `REMOTE_LLM_MODEL` — if unset, defaults to the provider's own fast/cheap
  model: `gpt-4o-mini` for `openai`, `claude-haiku-4-5` for `claude`.
  Override explicitly if you want a different model for either provider.
- `REMOTE_LLM_TIMEOUT_SEC` — `0` (default) means no timeout, matching
  this app's other network calls.

**Off by default, transparent fallback, never fatal**: exactly the same
philosophy as the router model. If `REMOTE_LLM_PROVIDER` is unset, the
main answer is generated locally, same as always. If it's set but the
call fails for any reason (missing/invalid `REMOTE_LLM_API_KEY`, network
error, HTTP error from the provider), `utils/llm.py` prints a warning to
the console and silently falls back to the local model (`LLM_BACKEND`-based,
embedded or llama-server, whichever is already configured) for that
generation — the request never errors out to the user because of a
remote-API problem.

Restart the app after changing any of these — same as every other `.env`
setting. Startup logging (`log_effective_config()`) prints whether
`REMOTE_LLM_PROVIDER` is set and whether `REMOTE_LLM_API_KEY` is present
(as "set"/"MISSING" — the key's actual value is never printed).

**Troubleshooting: OpenAI `429 insufficient_quota`** — if the console
shows `REMOTE_LLM_PROVIDER=openai call failed (OpenAI API returned HTTP
429: ... "insufficient_quota" ...)`, this is an account-side billing/quota
condition on OpenAI's end, not a bug in this app or a malformed request —
the same error occurs calling OpenAI's own `openai` Python client directly
with the same key. Check
[platform.openai.com/settings/billing](https://platform.openai.com/settings/billing)
for the key's plan/usage tier; a previously-working free-tier key can stop
working if OpenAI changes free-tier availability. Either way,
`generate_sync` already falls back to the local model automatically when
this happens, so a bad/exhausted remote key degrades gracefully rather
than breaking chat.

## Installation & Setup

Everything needed to go from a fresh checkout to a running app, in order.
Steps 1-5 are required; 6-8 are each independently optional (astrology
charts, PDF export, and RAG document search respectively) — skip any you
don't need, and come back to a given one later any time without redoing the
others.

### Port allocation across the ycplt family

Read this before configuring anything below — it's the one thing that's
easy to get wrong once more than one of these services is running.

| Service | Default port | Runs where | Set via |
|---|---|---|---|
| `ycplt` (this app) | `4010` | — | `PORT` |
| `ycplt_img` (image generation/editing) | `4011` | separate machine | `IMAGE_SERVICE_PORT` (on ycplt_img's own side), `IMAGE_SERVICE_HOST`/`IMAGE_SERVICE_PORT` (here, pointing at it) |
| `llama-server` (optional concurrent-LLM backend) | `4012` | SAME machine as `ycplt` | `--port` on its own command line, `YCPLT_LLAMA_SERVER_PORT` (here, pointing at it) |

Convention: every service in this family gets its own port starting at
`4010`, one each, rather than reusing `8080`/`5000`/other common defaults
that are more likely to collide with something else already running on
the same box. `llama-server`'s own upstream default (`8080`) is
deliberately NOT used here for that reason — if you add another service
to this family later, give it `4013`, and so on.

**A real, reported gotcha this exact numbering ran into**: after changing
`YCPLT_LLM_BACKEND`/`YCPLT_LLAMA_SERVER_PORT` in `.env` and starting
`llama-server` on the new port, `ycplt` itself kept behaving as if
nothing had changed — both were configured correctly, but `ycplt`'s
already-running process was still holding whatever config it read at ITS
OWN last startup (`utils/config.py` reads every `YCPLT_*` environment
variable exactly once, at import time — see that file's own module
docstring). **Restarting `llama-server` alone is not enough — `ycplt`
itself must ALSO be stopped and restarted** any time `.env` changes,
including just for this backend toggle.

### 1. System packages and Python environment

Verified end-to-end on a fresh Fedora install. System packages up front cover
everything RAG source ingestion needs (`antiword`/`.doc`, `p7zip`+
`p7zip-plugins`/`unrar-free` for archives, `djvulibre`/`.djvu` scans,
`poppler-utils`/rendering scanned PDF pages, `tesseract`+its Russian
language pack for OCR'ing anything with no embedded text layer (djvu or
PDF), `chmlib`/`.chm` help files, `gcc`/`cmake`/`python-devel` to build the
`ha` archiver from source — see "8. Optional: RAG document search" below
for what each package is for):

```bash
dnf install gcc cmake python-devel antiword p7zip p7zip-plugins unrar-free \
  djvulibre poppler-utils tesseract tesseract-langpack-rus chmlib
cd /tmp
git clone https://github.com/val-khokhlov/ha
cd ha
cmake . -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make all
cp ha /usr/local/bin
cd /var/www
git clone https://github.com/sphynkx/ycplt
cd ycplt
python -m venv .venv
source .venv/bin/activate
pip install -r install/requirements.txt
```

On Debian/Ubuntu, substitute the first line with:
```bash
apt install gcc cmake python3-dev antiword p7zip-full unrar \
  djvulibre-bin poppler-utils tesseract-ocr tesseract-ocr-rus libchm-bin
```
(building `ha` from source is the same either way — it isn't packaged for
either distro).

Already have the project installed and just need `.djvu`/scanned-PDF-OCR/`.chm`
support added after the fact? Only the new system packages are needed — no
venv/pip changes:
```bash
# Fedora
dnf install djvulibre poppler-utils tesseract tesseract-langpack-rus chmlib
# Debian/Ubuntu
apt install djvulibre-bin poppler-utils tesseract-ocr tesseract-ocr-rus libchm-bin
```

`val-khokhlov/ha` is a modern buildable reimplementation of the old
early-1990s DOS `HA` archiver — the format itself is dead and unpackaged
everywhere, but this gives `build_index.py` a real way to read `.ha` source
archives instead of skipping them outright.

### 2. Configuration (.env)

Copy the template and adjust as needed:

```bash
cp install/.env.example .env
```

All settings are read via `python-dotenv` in `utils/config.py`. Priority
(highest first): a real process environment variable > a value from `.env` >
the hardcoded default.

Path-valued settings (`MODEL_PATH`, `DB_PATH`, `RAG_DATA_DIR`, `INDEX_DIR`,
`INDEX_PATH`, `META_PATH`) may be relative or absolute. A relative value is resolved
against the project root (the directory containing `app.py`), not against
the current working directory the app happens to be launched from — so
`data/chat.sqlite3` always means the same file whether you run
`python app.py` from inside the project, from `cron`, or via the systemd
unit below. (Earlier versions resolved these against the launch-time cwd,
which could silently point at a different, empty database if the app was
ever started a different way — looking exactly like "my chat history
disappeared after a restart" even though nothing was deleted.)

Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `127.0.0.1` | App bind address |
| `PORT` | `4010` | App port |
| `MODEL_PATH` | `models/model.gguf` | Path to the GGUF chat model |
| `N_THREADS` | `4` | Inference threads |
| `N_CTX` | `32768` | Context window (comfortably below the current default model's own native window — lower it if RAM is tight) |
| `N_GPU_LAYERS` | `0` | Layers offloaded to GPU (0 = CPU only) |
| `REPEAT_PENALTY` | `1.15` | Generation repetition penalty — raise if the model glitches into repeated or foreign-script text on long answers, lower toward llama-cpp-python's own default (1.1) if answers start avoiding necessary repeated terms (planet/sign names) |
| `DB_PATH` | `data/chat.sqlite3` | Chat history database |
| `RAG_DATA_DIR` | `rag_data` | RAG source documents folder |
| `INDEX_DIR` | `data/rag_index` | One `faiss_index.bin`+`meta.pkl` pair per corpus lives under here (build_index.py) |
| `INDEX_PATH` | `data/faiss_index.bin` | Legacy single-index fallback only (pre-per-corpus) — see utils/rag.py |
| `META_PATH` | `data/meta.pkl` | Legacy single-index metadata fallback only — see utils/rag.py |
| `INDEX_CONCURRENCY` | `4` | Concurrent external processes (tesseract, antiword, ddjvu, ...) while indexing one corpus |
| `EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformers embedding model |
| `TOP_K` | `3` | Number of RAG chunks retrieved per query |
| `RAG_ALWAYS_INCLUDE_MAX_CHARS` | `28000` | Cap on methodology-doc auto-inclusion size (see utils/rag.py) — raised from 16000 after horary's and astro_progressions' methodology payloads grew past it, silently losing their tail sections; re-check with a chunking simulation (build_index.py warns automatically) if you grow a methodology doc further |
| `RECTIFICATION_LLM_FOLLOWUP` | `false` | Re-enable the RAG-augmented follow-up LLM call for the two rectification tools (off by default — see "Rectification tools" above) |
| `HF_TOKEN` | (unset) | Optional Hugging Face Hub token (rate limit / warning) |
| `HF_HUB_OFFLINE` | (unset) | Set to `1` once models are cached, to skip Hub network checks entirely |
| `IMAGE_SERVICE_HOST` | `192.168.7.7` | ycplt_img host |
| `IMAGE_SERVICE_PORT` | `4011` | ycplt_img port |
| `IMAGE_POLL_INTERVAL_SEC` | `10` | How often the background poller checks ycplt_img |
| `IMAGE_HTTP_TIMEOUT_SEC` | `10` | Timeout for short status/submit requests (not generation itself) |

### 3. Chat model

Download a GGUF model and place it at `models/` (or point `MODEL_PATH` at it):

```bash
wget https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf -O models/Qwen3.5-9B-UD-Q4_K_XL.gguf
```

**Current default: Qwen3.5-9B, `UD-Q4_K_XL` quant (~6 GB)** —
https://huggingface.co/unsloth/Qwen3.5-9B-GGUF (Unsloth's own GGUF build of
the official Alibaba Qwen3.5-9B release — "UD" is Unsloth's own "Dynamic"
quantization scheme, their own recommended default across that repo's
tooling examples; a plain K-quant like `Qwen3.5-9B-Q4_K_M.gguf` from the
same repo is a safer fallback if `UD-Q4_K_XL` ever fails to load on an
older llama-cpp-python build — swap the filename, nothing else changes).
Ignore the `mmproj-*.gguf` files in that repo — those are for the model's
image-understanding half, unused here (this app only calls
`create_chat_completion` for text).

Replaced the earlier Qwen2.5-3B-instruct default after real testing on the
OLD reference hardware from "Hardware and why this stack" above (i7-5500U,
2c/4t, 12 GB RAM, no usable GPU) — since superseded by the current
i7-8700/16 GB reference machine, see that section for the full story:
noticeably fewer of the small model's characteristic failures
(contradicting a tool's own computed verdict, inventing facts not present
in the supplied data, garbling non-Russian text mid-answer) at a real but
tolerable latency cost — a full RAG-interpreted horary answer (the
heaviest single-request workload in this app) ran roughly 400-550s on
this hardware, versus ~250s for the old 3B default. Note this model is a
"-Thinking-"-heritage model — it reasons via an internal `<think>...</think>`
scratchpad before its real answer, by design (see `utils/llm.py`'s
`generate_sync`, which strips that block automatically before returning
any answer to a caller — this matters for ANY thinking-style model you
might swap in later, not just this one). Its own native context window is
262,144 tokens (extensible further) — this app's own `N_CTX=32768` default
is unrelated to that ceiling and doesn't need raising just because of it.

Earlier, lighter alternatives (still supported, just no longer the
shipped default — useful on genuinely constrained hardware, or if 9B's
latency doesn't fit your use case):

- **Qwen2.5-3B-Instruct-GGUF** (file `qwen2.5-3b-instruct-q4_k_m.gguf`, ~2 GB) —
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF — the original
  default on the OLD reference hardware (see "Hardware and why this stack"
  above), ~250s there for a full horary answer vs 9B's 400-550s, at the
  cost of the small-model failure modes above. On the current i7-8700
  reference machine both would run considerably faster than either of
  those figures.
- If too slow even at 3B — **Qwen2.5-1.5B-Instruct-GGUF** (faster, lower quality).
- **Qwen2.5-7B-Instruct, Q4_K_M** (~4.7 GB):
  https://huggingface.co/paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF
  (direct file: https://huggingface.co/paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf)
- **Qwen2.5-14B-Instruct, Q4_K_M** (~9 GB) — comparable size/latency
  ballpark to the current Qwen3.5-9B default, from an older model
  generation:
  https://huggingface.co/TheRains/Qwen2.5-14B-Instruct-Q4_K_M-GGUF
  (direct file: https://huggingface.co/TheRains/Qwen2.5-14B-Instruct-Q4_K_M-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf)
- If a bigger model feels tight on RAM together with the full
  32768-token context, lower `N_CTX` (e.g. to 8192-16384) rather than
  dropping back to a smaller model — a single astro answer's actual
  prompt rarely needs the full window.

A word of caution on model-hunting on Hugging Face generally: community
repos with nonstandard version numbers, mixed-vendor names (e.g. a model
claiming lineage from two unrelated labs at once), or self-reported
benchmarks against models that don't otherwise show up anywhere are common
enough to be worth real scrutiny — verify a repo's base model against the
publisher's own official org page before trusting its README's claims at
face value.

### 4. Running the app

```bash
python app.py
# or: uvicorn app:app --host <HOST> --port <PORT>
```

The chat UI is served at `http://<HOST>:<PORT>/` (default
`http://127.0.0.1:4010/`). On first run, `data/chat.sqlite3` is created
automatically with the current schema; on later runs, `init_db()` migrates
an existing database in place if new columns were added (no data loss).

#### Running as a systemd service

```bash
sudo cp install/ycplt.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ycplt
```

The unit file uses `EnvironmentFile=/var/www/ycplt/.env` and intentionally
has no `User=` line (this deployment runs as root). Adjust
`WorkingDirectory` and `ExecStart` if the app lives somewhere other than
`/var/www/ycplt`.

### 5. Optional: astrology chart engine

Every `astro_*` tool (natal, transit, synastry, progressions, directions,
returns, profection, horary, electional, both rectification techniques, and
the universal help assistant's technique-selection logic) needs this — skip
it only if you want ycplt as a plain general-purpose chat app with no
astrology features at all.

```bash
pip install kerykeion timezonefinder geonamescache
```

- **kerykeion** (https://github.com/g-battaglia/kerykeion) does the actual
  chart computation — Swiss Ephemeris under the hood, fully offline, no API
  key. It's AGPL-3.0 — see `utils/astro.py`'s own docstring before
  redistributing this project if that matters for your situation.
- **timezonefinder** resolves the correct timezone automatically from
  birth/event coordinates — without it, that lookup is simply skipped
  (kerykeion still works from explicit coordinates, you'd just need to also
  supply the timezone yourself).
- **geonamescache** (~34k world cities, bundled with the package, no
  separate download) resolves a bare city name with no coordinates given —
  without it, only explicit coordinates work.

All three are optional independently of each other and of the base app —
each missing package's own feature is skipped gracefully (best-effort
imports inside the tool functions), not a hard failure. See "Built-in
tools" below for how the free-text date/coordinate/city parsing this
enables actually works.

### 6. Optional: PDF export

Needed for the per-message "PDF" export button (see "Exporting a message to
PDF" below). `install/requirements.txt` already includes `weasyprint`
itself, but weasyprint renders through real system-level libraries (Pango,
cairo, gdk-pixbuf) and fonts, not just a pip package:

```bash
# Debian/Ubuntu
apt install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 \
  libcairo2 fonts-dejavu-core
# Fedora
dnf install pango cairo gdk-pixbuf2 dejavu-sans-fonts
```

The stylesheet asks for `DejaVu Sans` explicitly (Cyrillic + reasonable
symbol coverage) — weasyprint renders through whatever fonts are actually
installed on the host, same as a browser would, so without an equivalent
font present, Cyrillic text in an exported PDF would come out as tofu boxes.
If `pip install weasyprint` succeeds but PDF generation fails at runtime,
it's almost always one of these system libraries missing, not a Python-level
problem — see weasyprint's own docs for the full troubleshooting list.

### 7. Optional: RAG document search

Needed only for `use_rag: true` / the `astro_help_assistant`'s and every
astro tool's methodology-grounded reasoning step (see "RAG — search over
your own documents" below for the full setup walkthrough — building a
corpus, indexing, and enabling it). The dependencies:

```bash
pip install sentence-transformers faiss-cpu numpy   # core RAG (required for any of this)
pip install pypdf                 # only needed for *.pdf sources
pip install charset-normalizer    # auto-detects .txt/.html encoding (cp1251, koi8-r, ...)
pip install beautifulsoup4        # only needed for *.html/*.htm sources
pip install striprtf              # only needed for *.rtf sources (pure Python, no system tool)
pip install patool                # only needed for *.rar/*.arj/*.7z source archives
```

All covered in one go by `install/requirements.txt` from step 1. The
system-level tools for `.doc`/`.djvu`/`.chm`/scanned-PDF/archive source
formats (`antiword`, `djvulibre`, `poppler-utils`, `tesseract`, `chmlib`,
`ha`, `patool`'s own backends) are the same ones already installed in step
1 above — nothing further needed there. Without any of these, the app keeps
working as a normal chat; `use_rag` simply has no effect (see
`utils/rag.py`).

## Interface

- **Sidebar** — "+ New chat" button and a list of conversations (ChatGPT-style).
  Clicking a conversation switches to it. Hovering a row reveals three small
  icon buttons: "✎" renames it in place (an inline text field replaces the
  title; Enter/blur saves via `PATCH /api/conversations/{id}`, Escape
  discards — a dedicated button rather than double-clicking the title itself,
  since that title is already the row's click target for opening the chat
  and would race against a second gesture on the same element), "⬇"
  downloads a full archive of that conversation (`GET
  /api/conversations/{id}/export` — a `.zip` with the message dump as JSON
  plus every file attachment's raw bytes), and "✕" deletes it (with
  confirmation, cascades to its messages and files in the database).
- **Parallel chats** — each conversation is stored separately in the database;
  any number can be kept simultaneously and switched between via the sidebar.
  Switching away from a chat mid-request does NOT cancel it — the backend
  keeps computing regardless of what's on screen, and its answer lands in
  the right conversation's history whether or not that conversation is
  still the one open when it finishes (`sentForConversationId` in
  `static/js/app.js`'s submit handler, fixing a real bug where the reply
  used to unconditionally splice into whatever chat happened to be open at
  the moment it arrived). Sending a message in a second chat while the
  first is still working is safe but NOT simultaneous generation — see
  "Concurrency: one model, many chats" below.
- **Session persistence** — the current conversation id is stored in the
  browser's `localStorage`; on page reload, history is restored from
  `GET /api/conversations/{id}/messages`.
- **Timestamps** — shown in small text under each message: send time (user
  messages) or reply time and "thinking" duration (model replies), e.g.
  "Ответ 14:32:05 · думал 8.4 с".
- **Code as file attachments** — if a model reply contains fenced code blocks
  (` ```lang ... ``` `), they're additionally shown as separate attachment
  cards with a download link (`utils/codeblocks.py` + `/api/files/{id}`), not
  just as text inside the message.
- **Image requests** — no toggle needed. If a message asks to draw, generate,
  or edit an image (in any language), it's automatically detected and routed
  to ycplt_img instead of the chat model; see below.
- **Attach an image** — the 📎 button next to the composer picks a local
  image file; sending a message with an image attached always skips the
  generate/chat/tool routing and instead classifies the accompanying text
  as either an edit instruction ("make the background blue") or a
  question about the image's content ("what's in this picture?") — see
  "Editing an uploaded image" and "Understanding an uploaded image" below.
- **Copy button** — hovering a message card reveals a 📋 button in its
  corner that copies the message's raw text; each fenced code block gets
  its own 📋 button too, copying just that block instead of the whole
  message. Falls back to the legacy `execCommand('copy')` when the page
  isn't a secure context (e.g. opened over plain `http://` from a LAN
  address), since the modern Clipboard API refuses to work there.
- **Help-mode toggle** — a small ❓ button stacked under the 📎 attach
  button, for a user with no astrology background who isn't sure which
  technique applies (or whether one does at all) — or who just wants to
  ask the assistant something, on any topic, without being force-fit into
  a specific technique. Clicking it *arms* help mode (the button
  highlights and a banner appears above the composer); nothing is sent —
  the user still types and sends their own question, on whatever topic
  they want, exactly as normal. That one message is submitted with
  `force_help: true` (`ChatRequest.force_help`, `routes/chat.py`), which
  routes it straight to `astro_help_assistant` WITHOUT running
  `tool_router`'s own classification at all — a deliberate bypass, not
  just a bias, because the router's small-model judgment call is least
  reliable exactly when the user themselves doesn't know what they're
  asking for yet (the whole reason this mode exists). One-shot: the toggle
  disarms itself once that message is sent (click it again for another
  help-mode question); the small banner also has its own "✕" to disarm it
  before sending. An earlier version of this feature sent a fixed canned
  question instead of letting the user type their own — replaced after
  feedback that a single hardcoded prompt can't represent "ask anything,
  possibly unrelated to astrology entirely". See "Universal help
  assistant" below for how `astro_help_assistant` itself handles a
  genuinely unrelated question once it gets one.

## API

| Method/path | Description |
|---|---|
| `POST /chat` | Send a message. Body: `{query, conversation_id?, use_rag?, max_tokens?, temperature?, image_data?, image_filename?, image_mime_type?, strength?, force_help?}`. Without `conversation_id`, creates a new conversation. `max_tokens` defaults to `null` — no artificial cap; the model generates until it stops on its own or fills the context window (`N_CTX`). `image_data` is a base64-encoded image (no `data:...;base64,` prefix); if present, `utils/intent.py` classifies the accompanying text as an edit instruction (submits an img2img job — `strength`, 0..1, default 0.75, controls how much the result may diverge from the input) or a question about the image (submits a caption job) — see "Editing"/"Understanding an uploaded image" below. `force_help` (default `false`) skips `tool_router`'s classification entirely and routes straight to `astro_help_assistant` — set by the composer's ❓ help-mode toggle (see "Interface" and "Universal help assistant" below), not meant to be combined with `image_data`. Any of image-generation, image-edit, or image-caption requests return a `pending` placeholder immediately instead of chat text; otherwise returns the model's reply with `sent_at`/`responded_at` (ms), `thinking_ms`, and a `files` list. |
| `GET /health` | Model/RAG-index status and the configured `image_service_url`. For vision/generation model diagnostics, see ycplt_img's own `GET /health` — this app doesn't hold any of those models. |
| `GET /api/conversations` | List conversations (id, title, updated_at), sorted by last activity. |
| `POST /api/conversations` | Manually create an empty conversation (usually unnecessary — `/chat` creates one lazily). |
| `GET /api/conversations/{id}/messages` | Message history for a conversation, including files, timestamps, and `status`. |
| `PATCH /api/conversations/{id}` | Rename a conversation. Body: `{title}`; 400 on an empty/whitespace-only title, 404 if the conversation doesn't exist. Doesn't touch `updated_at` — a rename shouldn't jump the chat to the top of the sidebar's most-recently-active ordering the way an actual new message does. |
| `GET /api/conversations/{id}/export` | Download a full archive (`.zip`) of one conversation: `conversation.json` (title, timestamps, every message in order) plus every file attachment's raw bytes under `files/`, referenced from the JSON by `archive_path` rather than embedded inline — keeps the JSON dump plain, readable text even for conversations with several images attached. |
| `DELETE /api/conversations/{id}` | Delete a conversation (cascades to its messages and files). |
| `GET /api/files/{id}` | Download a file attachment (extracted code, or a generated image). |
| `GET /api/profiles` | List stored birth profiles (see "Birth profiles: AstroZet .zbs import/export" below). |
| `POST /api/profiles` | Create one birth profile directly (not via a .zbs file). Body: `{name, date, time?, utc_offset?, place?, lat, lon, sex?, comment?, photo_path?}`. |
| `GET /api/profiles/{id}` | Fetch one birth profile. 404 if it doesn't exist. |
| `PATCH /api/profiles/{id}` | Partially update a birth profile — only the fields present in the body are changed. |
| `DELETE /api/profiles/{id}` | Delete a birth profile. |
| `POST /api/profiles/import` | Import every record from a .zbs file's text. Body: `{content}` (the raw file text — read client-side, not a multipart upload). Malformed lines don't abort the whole import; response includes both the created profiles and a per-line error list. |
| `GET /api/profiles/export` | Download every stored profile as one `.zbs` file. |
| `GET /api/profiles/{id}/export` | Download a single profile as a one-line `.zbs` file. |

## Birth profiles: AstroZet .zbs import/export

**API-only for now** — there's no chat-UI integration for actually *using* a
saved profile inside a conversation yet (e.g. referencing one by name
instead of retyping full birth data). That UX wasn't clear enough to design
yet (a picker? a button next to the composer? a slash-command?) and was
explicitly parked pending further discussion. What's implemented is the
bounded, concrete half of the ask: getting real birth data into this app
from an AstroZet `.zbs` file, and back out again.

**AstroZet** is a third-party Windows astrology program. Its `.zbs` format
is a semicolon-delimited, one-record-per-line birth-data interchange file —
plain text, not this app's own storage shape. Per an explicit design
choice: birth profiles are stored however is convenient for this app (see
`db/connection.py`'s `birth_profiles` table), and `.zbs` is used only at the
import/export boundary (`utils/astrozet.py`), not as the internal
representation.

Line shape, confirmed against a real example:

```
Name; DD.MM.YYYY; HH:MM:SS; UTC_offset; Place; Lat; Lon; Sex; Comment;
```

e.g.:

```
Иван Петров; 15.08.1985; 12:00:00; +4; Винница, Винницкая обл., Украина; 49n14; 28e29; M; Далее комментарий в свободной форме|Значок пайпа обозначает перевод строки|PHOTO: ClosePeople\plysyi.jpg|строка начинающаяся с "PHOTO: " и далее относительный путь к фото;
```

Field conversions, both directions:

- **Date**: `DD.MM.YYYY` <-> this app's own `'YYYY-MM-DD'` (the shape
  `utils/astro.py`'s `_build_subject()` expects).
- **Time**: `HH:MM:SS` on import (seconds are read but not kept — this app
  only stores `HH:MM`); always re-emitted as `HH:MM:00` on export.
- **UTC offset** (e.g. `+4`): kept verbatim as a plain string, only so a
  re-exported `.zbs` round-trips faithfully. It is **never** used to
  resolve a timezone for the astro engine — that's always
  `astro._resolve_timezone(lat, lon)`, an offline lookup from coordinates,
  independent of whatever bare offset the source program recorded (which
  doesn't account for DST, historical offset changes, etc.).
- **Lat/Lon**: degrees + hemisphere letter + minutes, no separator (e.g.
  `49n14` = 49°14'N, `28e29` = 28°29'E) <-> plain signed decimal degrees.
- **Comment**: `.zbs` uses `|` as a display-newline separator; stored
  internally as plain text with real newline characters. A segment
  starting with `PHOTO: ` followed by a relative path is AstroZet's own
  convention for an attached photo reference and may appear anywhere among
  the `|`-separated segments (not necessarily last) — `utils/astrozet.py`
  scans every segment for it rather than assuming a fixed position, and
  stores it in its own `photo_path` column, separate from the plain
  comment text. On export, if `photo_path` is set, a `PHOTO: <path>`
  segment is appended back onto the comment (a legal position per the
  format's own "may be located anywhere" rule, not necessarily its
  original one).

Import (`POST /api/profiles/import`) parses every line independently — one
malformed record (e.g. a typo'd date) doesn't discard the rest of an
otherwise-valid file; the response returns both the successfully created
profiles and a per-line `{line, raw, reason}` error list for anything that
didn't parse. Both a single profile and the full stored list can be
exported back out as `.zbs` text (`GET /api/profiles/{id}/export` /
`GET /api/profiles/export`) for re-importing into AstroZet itself.

**Using it today (no browser UI yet — `curl` or any HTTP client):**

Import a `.zbs` file (its raw text goes in the JSON body's `content` field,
not as a file upload — read the file client-side first):

```bash
python3 -c "
import json, sys
print(json.dumps({'content': open(sys.argv[1], encoding='utf-8').read()}))
" profiles.zbs > /tmp/import_body.json
curl -X POST http://localhost:4010/api/profiles/import \
  -H "Content-Type: application/json" \
  -d @/tmp/import_body.json
```

(the small Python one-liner just does the JSON-escaping correctly —
trying to inline a multi-line `.zbs` file's text directly into a shell
string is error-prone with real quoting/newlines)

List everything stored:

```bash
curl http://localhost:4010/api/profiles
```

Export everything back to one `.zbs` file (e.g. to re-import into AstroZet):

```bash
curl http://localhost:4010/api/profiles/export -o birth_profiles.zbs
```

Export a single profile:

```bash
curl http://localhost:4010/api/profiles/5/export -o profile_5.zbs
```

### Attaching a .zbs file directly to a chat message

A separate, more direct use of the format: AstroZet users commonly keep
ONE `.zbs` file per person that holds both that person's own birth-data
record AND, on separate lines in the same file, their real life events
(dates of a marriage, a job change, an accident, ...) — comment fields
hold free-text explanations. That's exactly the input
`astro_rectification_events` already wants (an approximate birth time to
refine, plus a list of life events to test candidate times against), so
such a file can be attached directly to a chat message instead of typing
all of that out as prose.

`ChatRequest` accepts an optional `zbs_data` field (the raw `.zbs` file
TEXT — not base64, since it's already plain text) alongside the normal
`query`:

```bash
curl -X POST http://localhost:4010/chat \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json
print(json.dumps({
    'query': 'Сделай ректификацию для этого человека по приложенным событиям',
    'zbs_data': open('ivan_with_events.zbs', encoding='utf-8').read(),
}))
")"
```

**Heuristic for telling the subject apart from events within one file**
(confirmed with the user — AstroZet's own files don't mark this
explicitly): the FIRST record in the file is the subject; every record
after it is either a life event (rectification) or a second person
(synastry) — see `utils/astrozet.zbs_profiles_to_spec_text`'s own
docstring. Which interpretation actually applies depends entirely on
which tool `tool_router` picks for the message's own typed instruction —
this function doesn't try to guess that itself; it emits data for BOTH
interpretations at once (a plain `name=...;date=...;lat=...;lon=...` line
for the first profile, the same data again as `_a`/`_b`-suffixed keys for
the first two profiles, and one semicolon-formatted event line per
remaining profile) and lets whichever technique's own extraction code
pick out only what it understands — the same "hand every candidate
source to extraction, let each field's own resolver find what it needs"
approach `astro._extract_fields`'s own docstring already describes for
combining the typed message + the router's own transcription + prior
conversation history. A `.zbs` attachment is simply one more candidate
source, folded in the same way (`routes/chat.py`'s `zbs_context_text`) —
it does NOT alter the user's own typed `query`, which is still what's
shown/stored in the chat history; the raw `.zbs` text is instead stored
as its own downloadable file attachment on that message, the same
visibility an attached image already gets.

Works today via this JSON field — there's no dedicated "attach a .zbs"
button in the browser UI yet (the composer's 📎 button is still
image-only); that's a natural next step if this proves useful in
practice.

## Automatic image request routing

Instead of a manual "generate image" toggle, every `/chat` message is passed
through `utils/intent.py`, which asks the already-loaded chat model itself
(zero-shot, one-word answer, `temperature=0.0`) whether the message is asking
to create or edit an image. On error or ambiguity it defaults to regular
chat — a missed image request just costs a text reply (recoverable by
rephrasing), while a false positive would waste minutes on an unwanted
generation job.

If classified as an image request:

1. `routes/chat.py` submits a job to ycplt_img (`utils/image_client.py`) and
   immediately stores an assistant message with `status = "pending"` and the
   job id, returning that placeholder to the browser without waiting.
2. A background task (`utils/image_jobs.py`, started from `app.py` on
   startup) polls ycplt_img every `IMAGE_POLL_INTERVAL_SEC` seconds for any
   message still pending. This runs independently of any open browser tab —
   generation keeps going even if the tab is closed; the result is simply
   there next time the conversation is opened.
3. Once ycplt_img reports the job as done, the poller downloads the image,
   stores it as a `image/png` file attachment on the original message, marks
   the message `status = "complete"`, and acknowledges the job (`DELETE`) so
   ycplt_img drops it from its queue. On a reported error, the message is
   marked `status = "error"` with the failure reason.

   **Timestamp and duration, fixed.** The pending placeholder's stored
   `created_at` is the moment `chat()` itself received the request (its own
   early `sent_at`, from *before* any intent-classification calls run), not
   a fresh timestamp taken only after those calls and the job submission had
   already finished. This used to be a real, reported bug: since the
   "pending" state shows no clock at all in the UI, the very first (and,
   before this fix, only) timestamp a message ever displayed was that late
   one — visibly *later* than when the user actually pressed send, by
   however long classification took (worse under real model contention,
   since those calls share the same `_FifoLock`-serialized queue as every
   other generation — see "Concurrency: one model, many chats"). Once
   resolved, `utils/image_jobs.py` now also computes a `thinking_ms`
   equivalent — full wall-clock time from that same `created_at` to
   completion — and passes it to `complete_image_message`, so an image
   reply's meta line reads "Ответ HH:MM:SS · думал X.X с" exactly like a
   normal chat reply, instead of never showing a duration at all (`thinking_ms`
   previously stayed `NULL` for every image job, silently). No frontend
   changes were needed for this — `static/js/app.js`'s existing rendering
   already displays `thinking_ms` whenever it's non-null.
4. In the browser, a message with `status = "pending"` is shown in a
   distinct (dimmed/italic) style, and the page polls
   `GET /api/conversations/{id}/messages` every few seconds while any message
   in the open conversation is still pending, so the finished image appears
   without a manual reload. Image attachments render inline; other file
   attachments (e.g. extracted code) still show as a download card.

ycplt_img itself is a separate daemon with its own SQLite job queue,
processing one job at a time on a persistently-loaded image model — see its
own README at https://github.com/sphynkx/ycplt_img for setup and hardware
notes.

## Editing an uploaded image

Attaching an image via the 📎 button and sending a message routes the whole
request differently from a plain-text one:

1. The browser reads the picked file with `FileReader`, base64-encodes it,
   and sends it as `image_data` in the same `/chat` call as the typed
   instruction — no separate upload endpoint.
2. `routes/chat.py` decodes it and stores it as a file attachment on the
   user's own message (so it's visible in the chat history on reload, the
   same generic image rendering as a generated result).
3. `utils/intent.py`'s `is_edit_instruction_async` then classifies the
   accompanying text: is it an instruction to edit the image ("make the
   background blue", "remove the person"), or something else — most
   commonly a question about the image's content ("what's in this
   picture?"). A genuine edit instruction submits an img2img job to
   ycplt_img via `utils/image_client.submit_job(prompt, mode="img2img",
   strength=..., init_image=image_bytes)`; anything else is routed to
   image *understanding* instead (see below) rather than sent into an
   img2img job as a meaningless prompt.
4. For edit instructions specifically, `utils/intent.get_removal_target_async`
   answers one more question: is this asking to *remove one specific named
   object* ("remove the cat", "убери кота"), as opposed to a color/style/
   addition edit? Plain img2img has no way to execute a removal command —
   it just partially re-renders the whole image guided by the prompt text,
   so the object doesn't actually disappear, it just gets restyled (this
   was a real bug: "убери кота с фото" produced a warped image with the
   cat still in it). If it *is* a removal instruction, the object's name
   (translated to English) is sent along as `remove_target`; ycplt_img
   then automatically segments and inpaints just that region instead of
   running plain img2img — see its README "Removing a named object". Any
   other kind of edit leaves `remove_target` unset.
5. If (and only if) a `remove_target` was found, a second, narrower
   classification runs: `utils/intent.get_reconstruction_prompt_async`
   decides whether the SAME message also describes what should appear in
   the removed object's place, not just "gone" (e.g. "убери кота с фото.
   На месте кота воссоздай участок металлического барабана с
   отверстиями" describes a specific replacement; plain "убери кота"
   does not). This exists because ycplt_img's default removal path
   (LaMa, prompt-free — see its README) can only extend generic
   background into the hole; it has no way to paint something SPECIFIC
   there. When a description is found (translated to an English
   text-to-image prompt fragment), it's sent along as `reconstruct_prompt`,
   which routes the job through ycplt_img's prompt-guided `INPAINT_MODEL`
   checkpoint instead of LaMa for that one job — see ycplt_img's README
   "Describing what should replace the removed object". A plain "remove
   X" with no further description leaves `reconstruct_prompt` unset and
   keeps using LaMa exactly as before, unaffected by this extra check.
6. From there, an edit job follows the identical pending →
   background-poller → complete flow as image generation
   (`utils/image_jobs.py` doesn't distinguish generate jobs from edit jobs
   — it just polls a job id and resolves it), so no changes were needed
   there.

## Understanding an uploaded image

This app has no vision model of its own — only the chat LLM. Image
understanding, exactly like generation and editing, is a graphics-service
capability: `_handle_image_question` (`routes/chat.py`) submits a
`mode="caption"` job to ycplt_img (`prompt` = the question, `init_image_b64`
= the attached image) and stores a `pending` placeholder, the same shape
as a generation/edit job; `utils/image_jobs.py`'s background poller
resolves it once ready — for `mode="caption"` that means writing the text
answer straight into the message content, with no file attachment (see
that module's docstring for the mode-aware resolution logic).

ycplt_img hosts the actual vision model
([moondream2](https://huggingface.co/vikhyatk/moondream2), via
`llama-cpp-python`) and its GGUF files live in *its* `models/` directory,
not this app's — see
[its README](https://github.com/sphynkx/ycplt_img#understanding-an-uploaded-image-modecaption)
for the download links and setup. It's optional there: if it isn't set
up, a caption job simply comes back as an `error` status with a clear
message, resolved the same way any other failed job would be
(`repository.fail_image_message`) — this app doesn't need to know why a
caption job failed, only that it did.

moondream2 answers in English regardless of the question's language, so
`_resolve_caption_done` does one short follow-up generation on the main
chat LLM — the raw caption plus the original question in, a natural
answer in the same language the user asked in out — the same "tool result
→ natural-language answer" pattern already used for datetime/calculator
results (see "Built-in tools" below). Falls back to the raw (English)
caption if that step itself fails, rather than losing the answer entirely.

If image questions keep failing, check ycplt_img's own `GET /health`
(`vision` field: `files_found`, `loaded`, `load_error`) on that machine —
not this app's `/health`, which has nothing to report about vision since
it holds no vision model.

## ycplt_img API contract (as implemented)

`utils/image_client.py`'s `submit_job()` builds the request shape below
for all three job kinds this app sends; see ycplt_img's own README for the
authoritative API reference (this is a quick summary from the client side):

```jsonc
// Plain generation:
{"prompt": "a cat wearing a hat", "mode": "txt2img",
 "width": 512, "height": 512, "steps": 20, "cfg_scale": 7.5
 // + optional: negative_prompt, seed
}

// Editing an uploaded image:
{"prompt": "make the background blue", "mode": "img2img",
 "width": 512, "height": 512, "steps": 20, "cfg_scale": 7.5,
 "strength": 0.75, "init_image_b64": "<base64-encoded source image>"
 // + optional: negative_prompt, seed; mask_image_b64 for mode="inpaint"
 //   (not currently sent by this app's UI, but the client already
 //   supports it if a masked-inpainting UI is added later)
}

// Removing a named object (also mode="img2img", but with remove_target
// set — see utils/intent.get_removal_target_async and ycplt_img's README
// "Removing a named object"):
{"prompt": "убери кота с фото", "mode": "img2img", "remove_target": "cat",
 "width": 512, "height": 512, "steps": 20, "cfg_scale": 7.5,
 "init_image_b64": "<base64-encoded source image>"
 // strength/negative_prompt are ignored for this path — ycplt_img sets
 // its own for the auto-mask + inpaint step
}

// Removing a named object AND describing its replacement (also
// mode="img2img" + remove_target, plus an optional reconstruct_prompt —
// see utils/intent.get_reconstruction_prompt_async and ycplt_img's README
// "Describing what should replace the removed object"):
{"prompt": "убери кота с фото. На месте кота воссоздай участок металлического барабана с отверстиями",
 "mode": "img2img", "remove_target": "cat",
 "reconstruct_prompt": "a section of perforated stainless steel panel with round holes, matching the surrounding metal texture",
 "width": 512, "height": 512, "steps": 20, "cfg_scale": 7.5,
 "init_image_b64": "<base64-encoded source image>"
 // reconstruct_prompt is only sent when get_reconstruction_prompt_async
 // found a real description; omitted (falls back to LaMa) for a plain
 // "remove X" with nothing further said about the replacement
}

// Understanding an uploaded image:
{"prompt": "what's in this picture?", "mode": "caption",
 "width": 512, "height": 512, "steps": 20, "cfg_scale": 7.5,
 // width/height/steps/cfg_scale are ignored for this mode but still
 // required by the request shape — any value works
 "init_image_b64": "<base64-encoded source image>"}
```

`GET /jobs/{id}` (status), `GET /jobs/{id}/result` (image modes only),
and `DELETE /jobs/{id}` are shared by all job kinds. For `mode="caption"`,
the answer comes back as `result_text` directly on the `GET /jobs/{id}`
status response instead of through `/result` — see
`utils/image_jobs.py._resolve_caption_done`.

## Built-in tools (date/time, calculator, extensible)

A local LLM has two well-known blind spots: it has no notion of "now" (its
training data has a fixed cutoff and it can't tell you today's date), and it
often gets arithmetic wrong past a few digits. Rather than hardcoding
special cases for these two questions, `/chat` routes through a small,
generic tool layer:

1. After the image-intent check, `utils/tool_router.py` asks the model
   (zero-shot, one line, `temperature=0.0`) whether answering the message
   needs one of the tools registered in `utils/tools.py`, and if so, which
   one (plus an argument, for tools that need one — e.g. the expression to
   evaluate for the calculator). On error or ambiguity it answers "no tool
   needed", same fail-safe default as the image classifier.
2. If a tool is picked, `routes/chat.py` runs it directly (no LLM call —
   `get_current_datetime` reads the system clock, `calculate` evaluates the
   expression) and does one more short generation: the tool's result is
   handed to the model with a prompt asking it to phrase a natural answer
   to the user's original question using that result.
3. The reply is then saved and returned exactly like a normal chat message
   (`status: "complete"`) — no schema or frontend changes were needed for
   this.

**Conversation memory for plain (non-tool) replies.** A real, reported gap:
every `/chat` call used to build its generation prompt from the current
message alone — no prior turns of the SAME conversation were ever included,
so a short follow-up ("а покороче", "давай другой вариант") was generated
as if it were the very first message of a brand new conversation. This was
never a context-window SIZE problem (`N_CTX` is already generous, see
below) — history simply wasn't being placed into the prompt at all.
`routes/chat.py`'s `_handle_chat_request` now prepends a short transcript
of the last few turns (`_CHAT_HISTORY_MAX_TURNS`, default 6 exchanges,
`_CHAT_HISTORY_MAX_CHARS_EACH`, default 800 chars per message) labelled by
role ("Пользователь"/"Ассистент") ahead of the normal prompt. This is
deliberately separate from — and more generous than — the tool router's
own history budget just above: an earlier, real finding was that dumping
a lot of history into the ROUTER's cheap yes/no classification call made
routing measurably *worse* (a small model's attention to the actual
current message degrades once the prompt is dominated by history, "lost
in the middle"), but that finding was specific to that one classification
decision, not to conversational continuity in a normal generated reply —
so the two budgets are kept independent rather than sharing one value.

Built-in tools:

- **`get_current_datetime`** — current date, time, and day of week.
- **`calculate`** — arithmetic expressions (`+ - * / // % **`, parentheses).
  Evaluated via a restricted `ast` walk, not `eval()` — it can only ever
  do arithmetic on numbers, never call functions, import anything, or
  access names, so a malformed or adversarial expression just returns an
  error string instead of executing.
- **`astro_natal_chart`** / **`astro_transit_chart`** / **`astro_progression_chart`** /
  **`astro_direction_chart`** / **`astro_lunar_return_chart`** /
  **`astro_solar_return_chart`** / **`astro_profection_chart`** / **`astro_synastry_chart`** /
  **`astro_rectification_trutine`** / **`astro_rectification_events`**
  — computes an astrological chart (planet signs/houses, aspects) via
  [kerykeion](https://github.com/g-battaglia/kerykeion) (`utils/astro.py`),
  fully offline (Swiss Ephemeris, no API key). `astro_natal_chart` computes
  a birth chart; `astro_transit_chart` computes current (or a given
  moment's) planetary positions and their aspects to a natal chart — for
  "what's happening right now" style questions; `astro_progression_chart`
  computes secondary progressions ("day for a year") — a slow, symbolic,
  decades-long unfolding read the same way a transit is (natal-house
  overlay via `_house_of_degree`), just for a computed "progressed" moment
  instead of the real current one (`_secondary_progressed_datetime`) — for
  "what stage of life/development" style questions, as opposed to
  transit's short-term "what's happening now"; `astro_direction_chart`
  computes solar arc directions — EVERY natal point shifted by the SAME
  precise arc (the angular distance the progressed Sun has moved from its
  own natal position, reusing the progression machinery above), unlike
  progression's per-point speeds — a precise, calculable timing technique
  (historically used for rectification); since there's no independent
  kerykeion chart for a "directed" moment at all, its aspects are matched
  directly against a table of standard aspect angles
  (`astro._ASPECT_ANGLES`) rather than via kerykeion's own `AspectsFactory`;
  `astro_lunar_return_chart` / `astro_solar_return_chart` compute the
  person's lunar (~monthly) / solar (~annual) return — a REAL independent
  chart cast for the moment the Moon/Sun returns to its exact natal degree,
  via kerykeion's own `PlanetaryReturnFactory`, read both on its own terms
  (its own house system) and via aspects back to the natal chart, the same
  two-sided reading synastry uses for two people's charts; both use ONLY
  the natal birth location for the return chart (no relocation support —
  deliberately left out for now, a real more specialized technique);
  `astro_profection_chart` computes the current year's profection — a
  classical technique that builds NO new ephemeris chart at all, just
  whole-sign-per-year arithmetic from the natal Ascendant plus a classical
  (7-planet, no outer planets) rulership lookup for the resulting sign's
  ruling "time lord" planet — deliberately using WHOLE-SIGN houses rather
  than the quadrant house system every other technique here uses (kerykeion's
  natal cusps), a real, explicit fork for this one technique specifically;
  `astro_synastry_chart` compares TWO people's natal charts — for
  compatibility/relationship questions (see its own paragraph further down
  for the two-person-specific parts). All eight require birth date, time,
  and place (as coordinates) to already be present in the conversation —
  the tool descriptions explicitly tell the router never to invent
  placeholder birth data, so if it's missing the model just asks the user
  for it in the follow-up answer instead of guessing.

  **`astro_rectification_trutine`** (`utils/rectification.py`) is
  different from the eight above in one fundamental way: it doesn't take
  one exact birth time at all — it takes an UNCERTAIN one (a window) and
  SEARCHES it for the candidate birth time that best satisfies the
  classical "Trutine of Hermes" rectification rule (as the Moon at birth,
  so the Ascendant at conception; as the Ascendant at birth, so the Moon
  at conception — conception estimated as a fixed number of days before
  each candidate birth time, `gestation_days=`, default 273). It builds
  TWO charts (birth + estimated conception) per candidate time across the
  whole window (`step_minutes=`, default 1) and reports the several most
  distinctly-different (not just adjacent-minute) candidates ranked by how
  closely each one satisfies the rule, plus an explicit warning if the
  best candidate sits right at the edge of the search window (meaning the
  true best result is likely OUTSIDE the window that was actually
  checked). This tool deliberately skips the digest/profile machinery
  every other technique above goes through (see its own module docstring
  — there's no single chart's "significant points" to rank here, just a
  ranked list of candidate times); it goes straight into the generic
  RAG reasoning-mode prompt (`rag_utils.build_prompt`) alongside whatever
  the user's own indexed rectification reference corpus retrieves, which
  fits well since that prompt already asks the model to reason step by
  step over given facts before answering. This is deliberately the
  simplest, fastest, single-method first step of a much larger
  rectification vision — the second, bigger step of that vision is now
  implemented too, see `astro_rectification_events` right below. See
  `rectification_trutine_methodology.txt` for how to interpret
  this tool's numeric output (what the "total mismatch" figures mean,
  why several candidates are a normal result rather than an error, and
  what this specific implementation simplifies away) — the classical
  theory/history of the method itself lives in the user's own indexed
  corpus, not in that file.

  **`astro_rectification_events`** (`utils/rectification_events.py`) is the
  multi-technique, multi-event rectification search: instead of one
  classical rule, it takes an uncertain birth-time WINDOW plus a list of
  known LIFE EVENTS (marriage, birth of a child, death, career change,
  illness/surgery, move, etc. — free text, one per line, either as
  `description: date` or the richer real-world `description; date; [time];
  [place]; [lat]; [lon]; [comment]` — see `_try_parse_semicolon_event`;
  fields after the date are optional and simply ignored except an optional
  time, which IS used if given) and, for EACH candidate birth time in the window,
  builds a profection, secondary progression, solar-arc direction, and
  transit chart for EACH event, scoring how well each technique's moving
  points aspect the classical "elements" (occupant planets + ruler +
  co-ruler, per REKTIF.TXT's worked example) of the houses that event type
  belongs to (`_EVENT_HOUSE_KEYWORDS`, a small static Russian keyword
  dictionary — marriage -> 7th/1st house, career -> 10th/6th house, etc.).
  Transits are weighted most heavily, profections least, matching REKTIF.TXT's
  own emphasis ("транзиты — более мощное указание"). The candidate with
  the highest aggregate CONFIRMATION score wins — the opposite polarity
  from Trutine's mismatch-error score (higher is better here, not lower) —
  and, exactly like Trutine, several genuinely different candidates are
  reported (`_diverse_top_candidates`), never just one answer, plus the
  same edge-of-search-window warning. Each candidate's report is broken
  down per event and per technique, so the model (and the user) can see
  *which* events/techniques support *which* candidate, not just a single
  opaque score. Event lines are parsed OUT of the free-text argument
  before the usual birth-field regex extraction runs (any line matching
  `description: date`, unless the description looks like a birth-data
  label such as "Дата рождения: ..." — see `_BIRTH_LABEL_EXCLUSIONS`), so
  birth data and events can be given in any order in the same message.
  Runs the per-candidate evaluation concurrently via `asyncio.gather` +
  a dedicated `ThreadPoolExecutor` (`run_rectification_events_async`,
  wrapped in a plain sync `run_rectification_events` for
  `TOOL_REGISTRY`'s string-in/string-out contract via `asyncio.run()`) —
  worthwhile here since a real run builds roughly `1 + 2*n_events` charts
  per candidate, unlike Trutine's fixed two — `_effective_max_candidates`
  shrinks the candidate count (search resolution) as the event count
  grows instead of capping the event list itself (raised to 80 after real
  usage), keeping total runtime bounded either way. Same
  deliberately-no-digest, straight-into-the-generic-RAG-prompt
  architecture as Trutine, and same documented-simplifications convention
  — see `rectification_events_methodology.txt` for what this tool's
  numbers mean and what it simplifies away (quadrant houses instead of
  Koch, event-time defaults to local noon unless a per-event time was
  given, profection's proxy scoring; event-house classification is now
  primarily model-based rather than keyword-only — see below).

  Report verbosity is ALSO adaptive (`_adaptive_report_limits` +
  `_MAX_REPORT_CHARS` hard safety net) — fixing a real, reported crash: at
  a fixed verbosity, a 42-event request produced a raw report alone
  equivalent to roughly 89000 tokens, blowing straight through the
  model's 32768-token context (`Requested tokens (89155) exceed context
  window of 32768`) once routes/chat.py injects it as an always-include
  RAG chunk (which has no size cap of its own for a tool's raw result —
  see `_handle_tool_request`'s `computed_chunk`). Fewer top candidates and
  less per-event match detail are shown as the event count grows, so the
  report stays a bounded few-thousand-tokens regardless of how many
  events were given, while the single most important line — the best
  candidate's actual recommended date/time — is always present and
  phrased as prominently as possible (`rectification_events_methodology.
  txt` explicitly requires the model to quote it as the first sentence of
  its answer, after a separate real failure where the model discussed
  which house/planet mattered at length without ever stating a concrete
  rectified time at all).

  A real 42-event life history surfaced two more gaps, both fixed:
  `_EVENT_HOUSE_KEYWORDS` was missing groups for relationship-meeting
  ("знаком"/"встрет"/"встреч"/"познаком" -> 7th/5th), romance ("любов"/
  "влюб"/"роман" -> 5th/7th), and broader breakup phrasing ("ушел от"/
  "ушла от"/"разошли"/"бросил", folded into the existing divorce group),
  plus a genuine keyword-collision bug where "Ограбление квартиры" (an
  apartment robbery) matched the move/housing group purely because it
  contains "квартир" — fixed by adding a dedicated theft/robbery group
  ("ограблен"/"кража"/"грабеж"/"обокра"/"украл" -> 2nd/8th/12th) and
  placing it BEFORE the move group so it wins the first-match check.
  Separately, real event lists often describe one multi-stage life event
  as several lines sharing a colon-prefixed label (e.g. "5я работа
  (судьбоносно важна): Request to vacancy" / "...: дал свое согласие" /
  "...: Успешное завершение испытательного срока") where only some
  phrasings hit a keyword. `_propagate_prefix_categories` (called at the
  end of `_extract_events_and_birth_text`) groups events by the text
  before their `:`, and if ANY sibling in a group matched confidently, its
  houses are propagated to the group's unmatched siblings too (tagged
  with a `category_note` so the report shows *why* a house was assigned —
  "определён по аналогии..." — distinct from both a direct keyword match
  and a genuine "[неопределённая категория событий]" fallback, in both
  `_format_candidate_block` and the event echo list). Verified against
  the real 42-event dataset: all 42 events now classify via keyword or
  prefix-analogy, zero fall back to the generic 1st/10th-house default.

  The "ИТОГОВЫЙ ЛУЧШИЙ ВАРИАНТ" line was also repositioned: it used to
  sit AFTER the (potentially 40+ line) event echo list, which real
  testing showed the small local model could lose track of; it's now the
  first substantive line of the report (right after the title/window
  lines, before the event list) and is repeated again, verbatim, at the
  very end of the report — bookending both ends of a potentially long
  report is more reliable against "lost in the middle" attention behavior
  than either position alone.

  Event-house classification then moved from keyword-only to model-based:
  the user explicitly considered keyword/substring matching unreliable in
  principle ("нельзя предусмотреть весь перечень событий, а модель
  справится"), accepting the added runtime cost. `_classify_event_houses_
  llm`/`_classify_event_houses_llm_async` now ask the already-loaded chat
  model, ONE event description at a time, which house(s) (1-12) that event
  semantically belongs to (the prompt spells out all 12 houses' classical
  meanings so the model reasons from real domain knowledge, not pattern-
  matching a handful of examples), and `_apply_llm_event_classification`
  (called once, right after the `_MAX_EVENTS` truncation, before the
  candidate search starts — so its cost is `O(n_events)`, independent of
  how large the search window is) overrides the keyword-based result
  whenever the model succeeds. Runs sequentially, not concurrently — the
  process has exactly one loaded Llama instance, so parallel dispatch
  would only contend for it, not add throughput. The keyword dictionary
  and prefix-propagation are NOT removed — they're the automatic fallback
  whenever the model is unavailable or its answer doesn't parse into a
  valid house number, so this change can only ever improve classification
  over the old keyword-only baseline, never regress it.

  Separately, a real failure showed the follow-up model doesn't just
  sometimes omit the "ИТОГОВЫЙ ЛУЧШИЙ ВАРИАНТ"/"Лучший найденный вариант"
  line — it can actively INVENT a different, physically implausible
  rectified time in its own prose (observed case: the tool's actual best
  candidate was well inside its search window, but the model's final
  answer stated a time several hours away, outside anything the tool even
  searched). No amount of prompt reinforcement in either methodology
  document fully prevented this. Rather than keep tuning prompts,
  `rectification.extract_best_recommendation` and `rectification_events.
  extract_best_recommendation` (simple regexes over each tool's own report
  text) let `routes/chat.py`'s `_handle_tool_request` pull the actually-
  computed best-candidate line back out verbatim and prepend it to
  `resp_text`, ahead of whatever the model itself generated
  (`_BEST_RECOMMENDATION_EXTRACTORS`) — this guarantees the correct number
  always reaches the user, independent of the small model's own
  reliability at transcribing or reasoning about it. This guarantees the
  number is PRESENT, but a further real test showed it doesn't by itself
  guarantee the model won't CONTRADICT it further down — one reply
  correctly showed the prepended line (08:52) but then concluded, in its
  own "Ответ:", that a different time (07:52, the starting/medical time)
  was actually best. Two more layers were added on top: a one-line Russian
  disclaimer is appended right after the prepended best-candidate line
  ("это точное вычисленное значение; если рассуждение ниже почему-то
  называет другое время как лучшее — доверяй именно этой строке"), so the
  user isn't left guessing which of two disagreeing numbers to trust; and
  both methodology documents gained an explicit "never conclude with a
  different time than this line" instruction, as defense in depth even
  though the disclaimer doesn't depend on the model actually following it.

  That methodology-doc reinforcement did NOT hold up under further
  testing: a following real test showed the exact same contradiction again
  (prepended line correctly showed 08:41, model's own "Ответ:" concluded
  "8 часов 30 минут" instead) — two reinforcement rounds failing the same
  way means this is genuinely a limit of this small model's own
  reliability, not a wording problem left to keep tuning. `resp_text` now
  also bookends the correct line at the very END of the reply (after a
  `---` separator), not just the start — mirroring the same "don't trust
  one position, repeat it" principle already used inside the tool's own
  report (`rectification_events.run_rectification_events_async`'s
  `summary_line` bookending). This doesn't depend on the model behaving
  any better than it already does; it just guarantees the actual LAST
  thing the user reads is the correct number again, directly countering
  "it looked right at the top, then contradicted itself at the end"
  instead of hoping further prompt wording fixes it.

  A THIRD real test still contradicted the prepended/bookended line
  (computed 08:41, model's "Ответ:" concluded "8 часов 30 минут" instead)
  — three consecutive real-world tests, four separate mitigation layers
  (prepend, disclaimer, methodology-doc reinforcement, bookend), all still
  insufficient. At this point the conclusion changed from "keep
  mitigating the follow-up call's unreliability" to "stop making the
  follow-up call at all": `_NO_FOLLOWUP_TOOL_NAMES` (both rectification
  tools) makes `_handle_tool_request` return the tool's own deterministic
  report as `resp_text` directly, right after it's computed — no RAG
  retrieval, no digest, no `llm_utils.generate_async` call, nothing left
  to potentially contradict the computed number, since the reply no
  longer contains any model-generated prose at all for these two tools.
  Only touch: the best-candidate line(s) are bolded (string `.replace()`,
  reusing the same `_BEST_RECOMMENDATION_EXTRACTORS` functions) so a human
  reading the raw report can still spot them at a glance. This is also
  meaningfully faster — no generation call over what can be a ~10000-
  character report — and `thinking_ms` for these replies now reflects the
  actual tool computation time, not a separate (removed) generation step.
  One real side effect, discussed with the user before making this
  change: `rectification_trutine_methodology.txt` and `rectification_
  events_methodology.txt` are no longer read by the app at all for these
  two tools while the follow-up stays off (no follow-up LLM call means no
  RAG retrieval happens) — both files stay in `install/methodologies/` as
  standalone reference documentation. Every OTHER astro tool (natal,
  transit, synastry, ...) is unaffected — this contradiction failure mode
  is specific to a task that demands transcribing one exact computed
  number consistently out of a large report, not a general problem with
  RAG-augmented interpretation.

  This is a default, not a permanent removal: `config.RECTIFICATION_LLM_
  FOLLOWUP` (env var `RECTIFICATION_LLM_FOLLOWUP`, off by default — see
  `install/.env.example`) gates the early-return in `_NO_FOLLOWUP_TOOL_
  NAMES`'s check — set it to `true` and both rectification tools fall
  through to the exact same RAG-augmented follow-up path every other
  astro_* tool already uses, methodology documents included, with no other
  code changes needed. The prepend/disclaimer/bookend safety net from
  tasks #189/#192/#193 wasn't deleted, only moved to run specifically on
  this now-optional path (right after the follow-up `generate_async`
  call) — even a future, more capable model is worth double-checking
  against the exact same contradiction failure mode before trusting it
  unconditionally, and the check costs nothing when the toggle is off.
  Turn this on only after separately re-verifying a new/larger model
  doesn't repeat the contradiction — nothing about this toggle guarantees
  a different model will behave better, it just makes trying one, and
  reverting instantly if it doesn't help, a one-line `.env` change instead
  of a code change.

  Separately, a real test also showed a SECOND rectification request in a
  conversation that already had an earlier, unrelated rectification
  exchange in it come back `tool=None` — the identical message routed
  correctly in a brand-new conversation with no history. `routes/chat.py`'s
  `chat()` handler now retries the classification ONCE with `history_
  context=""` whenever the first attempt (with history) returns no tool
  and there IS prior conversation history, using that retry's result only
  if it actually found a tool. This can only ever recover a tool call the
  history diluted away — a message that already routed correctly returns
  before this retry runs, and a short follow-up that genuinely depends on
  history to be recognized (e.g. "давай окно пошире") would just get
  "no tool" again on the retry too, exactly as before this fix existed.

  Separately, `utils/tool_router.py`'s classifier now also caps how much
  of the CURRENT message it sees (`_MAX_QUERY_CHARS`, 1500 chars, keeping
  the head where intent usually is) — the same "less raw text in the
  classifier's own prompt measurably improves its judgment" finding
  `_CLASSIFIER_HISTORY_MAX_MESSAGES` already established for history,
  applied to the current message too, after a real 42-event rectification
  request came back `tool=None` (the classifier judged "NONE" despite the
  very first line plainly saying "ректификацию"). This only affects the
  classifier's own prompt — the actual tool argument construction in
  `_handle_tool_request` always uses the full, untruncated `req.query`.

  The argument the router extracts for the other eight tools is meant to be a
  verbatim quote of the birth info from the user's own message, not a
  reformatted one — `utils/astro.py` parses
  common date formats (including "5 июля 1976"), times, and coordinates
  (decimal or degree-minute-second with N/S/E/W) itself, and resolves the
  timezone automatically from the coordinates via `timezonefinder` — this
  turned out to matter in practice: asking the small router model to
  itself convert a date/coordinate format and look up an IANA timezone
  name in one short completion was unreliable, quoting the original
  wording back is a much easier task for it. A bare city name with no
  coordinates at all also resolves, via `geonamescache` (~34k world
  cities, bundled with the package — no download or file to place
  anywhere): exact name match first (checked against every alternate-
  language name too, so Cyrillic city names generally work), then a
  same-first-few-letters "stem" fallback (checking a couple of prefix
  lengths, not just one fixed length) for when the name appears in some
  Russian grammatical case rather than the gazetteer's nominative form
  ("в Одессе" vs "Одесса") — not real morphological analysis, just a cheap
  approximation, but works for most city names. Ambiguous matches (a name
  that exists in more than one country, or an ordinary word that
  coincidentally happens to also be some obscure place's alternate name)
  are resolved by picking the most populous match ACROSS every exact and
  stem match together in one pool — fixed from an earlier version that
  let an exact match win outright without ever comparing it against a
  stem match's population, a real bug found via testing: the Russian word
  "года" ("of the year", present in nearly every birth-info sentence)
  happens to be listed as an alternate name for a ~19k-population Japanese
  town, which used to silently win over a same-sentence stem match on
  Kyiv (pop. ~2.95M) purely because "года" was an *exact* match and Kyiv's
  declined form ("Киеве") was only a *stem* one — the two were never
  actually compared. A second, independent bug fixed alongside it: a base
  city name shorter than the stem-comparison length (e.g. "Киев", 4
  letters) was previously only ever indexed under its own full-length
  bucket, but a declined form in the text ("Киеве", 5 letters) only ever
  checked its own equally-long bucket — which could never match a
  shorter one — so a short city name's declined form couldn't become a
  match candidate at all before this fix, regardless of the population
  comparison above. See "Installation & Setup" above ("5. Optional:
  astrology chart engine") for the exact packages and what each one's
  absence gracefully falls back to.

  Birth data mentioned earlier in the conversation (not just the current
  message) is also picked up, within limits: `routes/chat.py` hands
  `utils/tool_router.py` a short excerpt of the last few messages — BOTH
  roles, user and assistant — as background context, specifically so "use
  the birth data I gave you before" can still route correctly instead of
  silently failing because the classifier only ever saw the current
  message in isolation. Including the assistant's own recent replies (not
  just the user's) here matters for a second, related reason: a real,
  reported failure showed a short user follow-up ("давай окно пошире")
  continuing something the ASSISTANT itself had just suggested (widen a
  rectification search window) getting misrouted as "no tool needed" —
  with only past user messages visible, the classifier had no way to know
  what its own previous reply had proposed at all.

  Multi-stage RAG for natal-chart answers (`utils/interpret.py`): plain RAG
  (retrieve chunks similar to the user's whole question, paste them into
  the prompt) turned out to be a poor fit for "what does this specific
  placement mean" reference material, which is normally written and titled
  by placement ("Солнце в Раке", "Юпитер в 12 доме") — very different
  wording from a birth-data question, so a single top-k search rarely
  surfaces it however large the indexed corpus is. Instead, for
  `astro_natal_chart` answers: `utils/astro.py`'s `get_planet_profiles()`
  builds one PROFILE per significant point — its sign, house, retrograde
  state, its OWN aspects to other points (each aspect carrying the *other*
  point's sign/house too, so the digest step can judge how strong that
  aspecting influence actually is), and any fixed-star conjunction —
  ranked by the same qualitative priority rules the methodology document
  states in prose (angularity, orb precision, retrogradation, aspect
  count), reimplemented in plain Python rather than left for the model to
  apply on the fly. Each profile gets several targeted retrieval queries
  (sign, house, one per aspect, one per star conjunction) via
  `rag.retrieve_similarity_only()`; one additional LLM call then "digests"
  every profile's raw fragments into one synthesized note each — not a
  list of separate facts — explicitly weighing how each aspect colors the
  placement (favorable aspects reinforce only the compatible qualities of
  the aspecting planet, tense aspects create friction, strength scales
  with orb tightness and the aspecting planet's own placement) before the
  final answer-synthesis call gets to see any of it. This replaced an
  earlier flat, disconnected planet/aspect/house-fact design after real
  end-to-end review of a full answer showed its actual failure: aspects
  were being digested in total isolation from the placement they
  modified, so a hard square was never distinguished from a supportive
  trine, and a 12th-house placement's normal muting effect was ignored
  entirely — bundling sign+house+aspects into one profile is what lets a
  single digest note actually synthesize them together. Standalone
  "what does house N mean" facts were dropped for the same reason from
  the other direction: they produced their own free-floating paragraphs
  that reviewed as unwanted — a house's cusp sign is already visible in
  the raw computed chart text every answer gets, which is enough context
  on its own once it's no longer competing as a first-class fact of its
  own. This is a real latency trade — one more model call per natal-chart
  answer — made deliberately for interpretive accuracy over speed. A
  per-profile (rather than one-call-for-all) digest pass was considered
  for even deeper synthesis but ruled out for now given how much it would
  multiply an already multi-minute CPU-only generation — worth revisiting
  if hardware/latency allow later.

  Transit-chart answers go through the same digest/sectioned-answer
  pipeline now too, via a shared two-chart layer
  (`utils/astro.py`'s `get_dual_chart_profiles()`, consumed by
  `get_transit_profiles()`), not a lesser-quality fallback anymore. Two
  things worth knowing if you're extending this further (e.g. towards
  synastry): first, a transiting planet's **house is computed against
  the natal chart's own cusps** (`astro._house_of_degree`), not from an
  independent house system built for the transit moment/location itself
  — this is the standard transit-astrology convention (a transit reading
  is about which of *your* houses a planet is currently moving through),
  and fixes a real bug where the raw chart text used to report the
  transit moment's own houses instead; `_format_transit_text` and
  `get_dual_chart_profiles` both now go through the same
  `_house_of_degree` helper so the two can never disagree with each
  other. Second, the digest prompt (`interpret._build_digest_prompt`)
  and the final-answer prompt (`interpret.build_transit_answer_prompt`)
  are each parameterized/duplicated rather than reusing the natal
  versions unmodified — a transiting planet's aspects are read as
  current activation/timing, not permanent character, and
  `TRANSIT_ANSWER_SECTIONS` (general picture / support / tension /
  timing / summary) is accordingly a different breakdown from
  `ASTRO_ANSWER_SECTIONS`, built specifically to make use of each
  aspect's `movement_ru` ("сходящийся, усиливается" / "расходящийся,
  ослабевает") for its dedicated timing section — natal charts have no
  equivalent of "is this fading or intensifying," so that data existed
  before this rewrite but nothing used it.

  Synastry (`astro_synastry_chart`) is `get_dual_chart_profiles()`'s
  second consumer, exactly as planned when the transit rewrite above
  introduced it: `astro.get_synastry_profiles()` calls it TWICE, once per
  direction (person A's points overlaid onto person B's houses, then
  person B's onto person A's) — a synastry reading is genuinely
  bidirectional in a way a transit reading isn't (a moment doesn't have
  "its own" chart to profile the way a second person does), so both
  directions matter and neither is a redundant restatement of the other.
  `get_dual_chart_profiles()` gained a handful of new parameters
  (`other_point_label`, `reference_house_label`, `query_prefix`, `kind`)
  purely so synastry could relabel its generic transit-oriented
  "транзитный .../натальный N дом" phrasing into person-specific
  phrasing ("Венера ♀ (Мария)" / "7 дом у Ивана") — every parameter
  defaults to the exact original transit wording, so `run_transit`'s
  behavior is unchanged unless a caller explicitly overrides them.

  The one genuinely new problem synastry introduced: pulling TWO
  independent sets of birth data out of one free-text message, which the
  existing single-person `_extract_fields()` was never built for.
  `astro._extract_two_person_fields()` handles this with the same
  fast-path/fallback structure as the single-person case: explicit
  `date_a=`/`time_a=`/`lat_a=`/... plus `_b` key=value pairs (parsed for
  free by the already-generic key=value parser), or — the expected common
  case — free text naming two people back to back ("Иван, 5 июля 1976 в
  4:30 в Одессе, и Мария, 12 марта 1980 в 9:15 в Киеве"), split into two
  independent halves by `astro._split_two_person_text()`: it finds the
  first two recognizable dates in the text and splits between them at the
  last comma found in between (falling back to the plain midpoint if
  there's no comma there) — a best-effort heuristic, not a real parser,
  in the same accepted-approximation spirit as the city-name stem
  matching described above. Each half is then resolved via the *exact*
  same single-person field-resolution logic `_extract_fields()` uses (now
  shared as `_fill_fields_from_text()`, pulled out specifically so the
  two-person path can't silently drift from the single-person one),
  entirely independently per half. Person names are deliberately NOT
  guessed from the free text (only taken from explicit `name_a=`/`name_b=`
  key=value input) — a fragile "nearest capitalized word" heuristic risks
  being *wrong* in a way a safe generic default ("Человек A"/"Человек B")
  isn't; `interpret.build_synastry_answer_prompt()` instead tells the
  model to substitute the real names in its own answer if the user's
  question named both people, using first-mentioned = person A's data,
  second-mentioned = person B's — a much easier task for the model
  (loose natural-language mapping) than for a regex in Python.

  The final synastry answer uses its own section list
  (`interpret.SYNASTRY_ANSWER_SECTIONS`: overall compatibility / emotional
  connection / communication / friction points / summary) — deliberately
  framed around the *relationship*, not either person's character in
  isolation, which is also why `interpret.build_synastry_answer_prompt()`
  explicitly instructs the model to describe connections between the two
  charts rather than either person's placements on their own. A
  project-authored reasoning-methodology document for this technique
  (mirroring `interpretation_methodology.txt`'s own role — HOW to weigh
  house overlays/aspects/angularity into a synthesized reading, not a
  factual planet-meaning corpus) was written separately (master copy under
  `install/methodologies/synastry_methodology.txt`); copy it into the
  `rag_data/astro_synastry/` topic subfolder (see "Recommended `rag_data/`
  layout" below and "RAG — search over your own documents") alongside
  whatever factual reference material you assemble for this topic.

  Chart coverage beyond the classical 10 planets + Ascendant/MC: houses
  (all 12 cusps, used as placement context, not their own fact), Chiron,
  mean/true Lilith, Part of Fortune, Vertex, conjunctions (≤1.5°) to six
  commonly-used fixed stars (Regulus, Aldebaran, Antares, Fomalhaut,
  Spica, Algol), and six minor aspects (semi-sextile, semi-square,
  quintile, sesquiquadrate, biquintile, quincunx — orbs 2-3°, much
  tighter than the five majors' 5-8°, per standard convention that minor
  aspects only matter close to exact) alongside the five major aspects,
  are all computed and eligible for `get_planet_profiles()`. The extended
  point set required switching from kerykeion's `AstrologicalSubject`
  class to the lower-level
  `AstrologicalSubjectFactory.from_birth_data(..., active_points=[...])` —
  the former is a deprecated compatibility wrapper hardcoding a fixed
  18-point list with no way to ask for anything more, which is why these
  points always silently came back empty before. Profile selection always
  includes the Sun, Moon, Part of Fortune, and any star-conjunct point
  regardless of score — a pure angularity/aspect-count score has no
  notion of "fundamental" or "rare/notable" and was confirmed (in a real
  chart with several fixed-star conjunctions) to otherwise crowd them out
  entirely; everything else fills the remaining budget by score.

  Astrological terms (signs, points, aspects) are rendered with their
  Unicode symbols (☉ ☽ ♈ ☌ □ △ ⚺ ⚻ ⚼ etc.) directly in the Russian text
  `utils/astro.py` produces, rather than relying on an instruction asking
  the model to recall or insert them — more reliable in practice, since
  the model already has the real computed data to copy the symbol from
  instead of having to generate it correctly from scratch. Signs without a
  widely-standard single-glyph symbol (semi-square, quintile, biquintile)
  are deliberately left without one rather than an invented glyph.

  The final natal-chart answer is generated from a dedicated sectioned
  prompt (`interpret.build_sectioned_answer_prompt()`) instead of the
  generic reasoning-mode template `rag.build_prompt()` uses elsewhere: real
  repeated testing showed the generic template's answer collapsing into a
  single short paragraph regardless of how strongly it was told to
  elaborate, since an abstract "write in detail" instruction isn't a strong
  forcing function on its own. The sectioned prompt instead names seven
  concrete section headers (identity, home/emotional life, mind, love,
  work, growth/challenges, summary), requested as markdown headers, that
  the model must fill in turn, using the computed chart data plus the
  digested per-profile notes above — with an explicit instruction to
  always reuse a point's sign/house exactly as given rather than
  re-deriving it (added after a real answer stated two different houses
  for the same planet in two different sections) and a rule against
  non-Russian/CJK script leakage (added after a real answer, generated by
  a small quantized model under repetitive phrasing, glitched into stray
  Chinese characters mid-sentence — see `config.REPEAT_PENALTY` below for
  the generation-side mitigation for the same issue). This section list is
  a proposed breakdown, not a fixed schema — `interpret.ASTRO_ANSWER_SECTIONS`
  (transit's `interpret.TRANSIT_ANSWER_SECTIONS`, progression's
  `interpret.PROGRESSION_ANSWER_SECTIONS`, direction's `interpret.
  DIRECTION_ANSWER_SECTIONS`, the returns' `interpret.LUNAR_RETURN_
  ANSWER_SECTIONS`/`interpret.SOLAR_RETURN_ANSWER_SECTIONS`, profection's
  deliberately short (4, not 6-7) `interpret.PROFECTION_ANSWER_SECTIONS`
  — astro.get_profection_profiles only ever returns two profiles, so a
  longer section list would just pressure the model into padding thin
  material — and synastry's `interpret.SYNASTRY_ANSWER_SECTIONS`) are
  plain lists, easy to add to, rename, or drop sections from. Falls back
  to the generic reasoning-mode template if the digest step produced
  nothing for any of the eight chart types.

  `utils/astro.py`'s `ASTRO_OPERATIONS` registry (`natal`/`transit`/
  `synastry`/`progression`/`direction`/`lunar_return`/`solar_return`/
  `profection`) is the extension point for further chart types (composite
  charts, event-time/electional search, birth-time rectification): write
  one function, add one registry entry, wire a new `TOOL_REGISTRY`
  description when there's a concrete need for it. Directions have no
  independent kerykeion chart to build at all (every natal point is just
  shifted by the same solar arc — see `astro._build_direction_subjects`),
  so their aspects are matched directly against `astro._ASPECT_ANGLES`
  instead of kerykeion's `AspectsFactory`, and their profiles are built by
  bespoke code (`astro.get_direction_profiles`) rather than a call to
  `get_dual_chart_profiles`. Lunar/solar returns DO get a real independent
  chart (kerykeion's `PlanetaryReturnFactory` — `astro._build_return_
  subjects` walks forward from a safely-early probe date until it finds
  the return period actually active right now, rather than trusting a
  single fixed search margin, since the real return period isn't exactly
  365.25/27.3 days), so they reuse `get_dual_chart_profiles` unchanged
  aside from a new `include_angles=True` option (added specifically for
  this — a return chart's own Ascendant is a real, independently
  meaningful point, unlike a moving transit moment's) plus
  `force_include_labels` including the Ascendant. Profections build no new
  ephemeris chart at all — see `astro._profection_house_and_ruler` and
  `astro._CLASSICAL_RULERS_RU` — and deliberately use WHOLE-SIGN houses
  counted from the natal Ascendant's own sign, a real, explicit fork from
  the quadrant house system (kerykeion's natal cusps) every other
  technique here uses, confirmed as the intended classical convention for
  this one technique specifically rather than an inconsistency.

  **`astro_horary_question`** (`utils/horary.py`) implements HORARY
  astrology — a classical yes/no judgment cast for the exact moment and
  place a QUESTION is asked, not a birth chart at all (radicality/validity
  check, significators via ruler-of-house-I/quesited-house, essential
  dignity, aspects, Moon void-of-course, translation/collection of light —
  per Masenkov's "Построение хорарной карты" plus the general-technique
  chapters of Lavoie's "Lose This Book..." reviewed for this feature,
  excluding that book's lost-object-specific chapters and any Vedic
  material). Computes the full chart deterministically, then that report
  becomes the computed-facts context for the same RAG-augmented follow-up
  every other astro_* tool already uses, reasoning against
  `horary_methodology.txt` (master copy under `install/methodologies/`;
  copy it into `rag_data/astro_horar/` — see "Recommended `rag_data/`
  layout" below — alongside whatever horary reference material you
  assemble for that topic) — the same one-tool-always-interpreted pattern
  as natal/transit/synastry, no special-casing.

  This replaced an earlier two-tier design (a separate always-instant,
  never-interpreted "short verdict" tool, plus a second "give details" tool
  the router had to be talked into picking for a follow-up) — reverted
  after real testing (a real "who took my things" horary question) showed
  the short-verdict-only reply left a genuinely rich, radical chart
  completely uninterpreted unless the user knew to explicitly ask for
  more. Real testing also showed the hard radicality veto was too blunt:
  per Lavoie, even a non-radical chart still carries situational detail
  worth explaining (why the matter can't/won't proceed), so a failed
  radicality check no longer skips computation — it's carried alongside
  the verdict as an explicit caveat for the model to present cautiously,
  not a blank refusal to interpret the chart at all.

  The rectification tools' own "small local model contradicts its own
  tool's computed result if asked to reason freely" lesson still applies
  here — `horary_methodology.txt`'s "authority of the computed verdict"
  section is the prompt-side mitigation, and `utils/horary.py`'s own
  `ИТОГОВЫЙ ВЕРДИКТ` bookend (`extract_best_recommendation`, registered in
  `_BEST_RECOMMENDATION_EXTRACTORS`) is the matching code-side prepend/
  disclaimer/bookend safety net — always active here (unlike
  rectification's equivalent, which is gated behind
  `RECTIFICATION_LLM_FOLLOWUP`), since the user explicitly wants this
  follow-up call on for horary regardless of model capability.

  The "quesited house" is resolved either by an explicit `house=N`
  override, or by keyword classification: `_TOPIC_HOUSE_KEYWORDS` (what
  the question is about, same accepted-approximation spirit as
  `astro_rectification_events`' `_EVENT_HOUSE_KEYWORDS`), combined —
  when a named third party is detected via `_PERSON_HOUSE_KEYWORDS`
  (daughter/son, spouse, sibling, parent, friend, cousin, boss...) — with
  Masenkov's classical chart-turning arithmetic (`_derived_house`): the
  third party's own house becomes house I for them, and the topic house is
  then counted from there. Added after real testing (a "will my daughter
  choose French at university" question) showed the single-hop version
  silently answering as if it were the QUERENT's own house 9, and worse,
  the model started inventing its own uncomputed derived-house arithmetic
  in prose — a real fabrication-rule violation, caused by the real
  computation not being surfaced to it at all. The chain (who + their
  topic house + the resulting derived house) is now spelled out explicitly
  in the computed-facts block whenever a third party is detected, so the
  model narrates a real number instead of guessing one.

  Known v1 scope limits (see `utils/horary.py`'s own module docstring for
  the full list): only a fixed set of common 2-hop relations is covered —
  an arbitrarily deep chain ("my cousin's dog's health") still falls back
  to plain single-hop topic classification; essential dignity only covers
  the 7 classical planets (Uranus/Neptune/Pluto predate the system and
  have none, by design, not a gap); the direct-aspect/translation/
  collection search is restricted to the six classical aspects
  (conjunction/sextile/square/trine/quincunx/opposition) rather than the
  wider minor-aspect set `astro.py` uses for natal/transit/synastry
  reading — quintile, biquintile, etc. have no place in classical horary
  doctrine and, found via real testing, could otherwise get picked as the
  "significant" aspect and force a negative verdict by default (neither
  favorable nor hard in this module's own classification).

  **`astro_electional_chart`** (`utils/electional.py`) implements ELECTIONAL
  astrology — the inverse of horary: instead of reading a fixed moment
  against a QUESTION, it reads a user-PROPOSED moment against a stated
  PURPOSE (sign a contract, marry, travel, undergo surgery, launch a
  business, ...), judging whether that moment is a good time to start the
  thing. Per Andreeva's "Элективная астрология" (planetary-hour doctrine,
  per-hour recommend/avoid lists per activity, Moon-by-sign/-house transit
  meanings) and Scofield's "Being On Time" (lunar-phase rules, Mercury
  retrograde doctrine, diurnal-cycle concept) — deliberately excludes the
  Vedic muhurta (panchanga: tithi/nakshatra/yoga/karana) system and Pavel
  Globa's lecture material entirely, per explicit instruction, not merely
  out of scope; don't reintroduce either even as supplementary reference
  material.

  It reuses horary's proven apparatus — querent/quesited significators,
  essential dignity, the classical-aspects-only restriction, Via Combusta/
  combustion/besiegement — duplicated rather than imported from
  `utils/horary.py` (same "avoid coupling two independently-evolving,
  separately-tested modules" reasoning already applied to `astro.py`'s own
  copy of the LLM-first field-extraction pattern), and adds electional-
  specific computation: real planetary-hour calculation (Chaldean order,
  via `swe.rise_trans` sunrise/sunset, correctly spanning the sunset/
  sunrise boundary and anchored to the correct local weekday), Moon-phase
  scoring, and a purpose category classified by the model (LLM-first,
  keyword fallback) into a fixed enum — the category-to-house/planet
  mapping itself is a plain Python dict lookup (`_CATEGORY_TABLE`), never
  left to the model to compute. Unlike horary's binary radical/non-radical
  cascade, the verdict is a signed point-tally across all these factors,
  with Moon void-of-course as a hard override to "неблагоприятно"
  regardless of score — same unconditional-negative precedent as horary's
  own Moon-void handling.

  Two request modes, classified by `_classify_request_mode` (LLM-first,
  no regex fallback of its own — see below): a "single" request
  ("подходит ли момент X для дела Y") judges exactly the named moment
  directly; a "range" request ("на какой день лучше...") scans forward
  hour-by-hour from the nearest named moment (or right now, if none was
  given) across a window — 30 days by default, or an explicit length/end
  date the user names ("в течение 60 дней", "до конца сентября" —
  `_extract_window_days_llm` has the model extract only a plain unit
  count or a literal calendar date and does the actual date-difference
  ARITHMETIC in Python, never trusting the model with it, same
  never-let-the-LLM-compute-a-fact-it-could-get-wrong rule as everywhere
  else in this app) — evaluating every candidate hour and keeping the
  single best one. Ranking is three-tiered, not flat score: never a
  Moon-void candidate over a non-void one (hard override, same as the
  single-moment path); then by VERDICT CATEGORY first (благоприятно beats
  смешанно beats неблагоприятно — `_VERDICT_RANK`); only within the same
  verdict does the raw point score break a tie, with earliest-of-equal
  winning after that. This was a real correction: ranking by flat score
  alone let a "смешанно" moment with an unusually high score outrank a
  genuinely "благоприятно" one — the qualitative presence of favorable
  indicators has to decide first, exactly because (per real corpus
  research — see below) the point-tally itself has no classical citation
  for comparing calendar dates against each other; it's this module's own
  engineering device for ranking candidates, and the report/methodology
  text says so explicitly rather than letting the model attribute it to
  a source.

  The range-mode report headlines a deterministic `ИТОГОВЫЙ ЛУЧШИЙ
  МОМЕНТ: <date> <time>` bookend line (top AND bottom of the report,
  extracted by `extract_best_recommendation`) — mirroring
  `rectification_events.py`'s own "ИТОГОВЫЙ ЛУЧШИЙ ВАРИАНТ ВРЕМЕНИ
  РОЖДЕНИЯ" bookend exactly, same reasoning: the headline fact a search
  produces is a concrete DATE, not a favorable/unfavorable label, and a
  small local model can otherwise omit or invent a different one in its
  own free prose. Real testing caught the underlying bug this whole
  design replaced: a "на какой день лучше" question used to have whatever
  date/time was in the message (typically just the moment the question
  was asked) silently evaluated as if it were the user's own proposed
  moment, producing a confident-looking but meaningless single-moment
  verdict instead of a real search.

  A real corpus-research pass (OCR'ing three previously-unread scanned
  PDFs — Li Liman, Robson, Tsypin's "Основы элективной астрологии") also
  corrected the category→house table: household/domestic chores
  (cleaning, laundry, small repairs) now map to house IV, not house I —
  Tsypin names IV explicitly "Дом результатов" for household-type
  elections, and Robson's general principle is "house = ruled by the
  election's subject matter", not "house I for any beginning". House I
  remains the true fallback for a genuinely uncategorizable "начинание",
  and is separately noted (per Li Liman) as a universal supporting
  significator alongside whichever topical house applies, not a
  replacement for one.

  Every `astro_*` tool's RAG-augmented follow-up now also passes a
  `topic_hint` into `rag_utils.retrieve_context` (see `_TOOL_TOPIC` in
  `routes/chat.py`) naming its own `rag_data/` subfolder, guaranteeing
  that tool's own methodology is included even when the user's free-text
  wording doesn't happen to score a similarity hit against it — plain
  top-k similarity search alone isn't topic-scoped (by design — see
  `utils/rag.py`'s own docstring) and a mundane electional query like "на
  какой день лучше убираться в комнате" was found in practice to lose
  the similarity race entirely to an unrelated topic's methodology
  (surfacing as a spurious "natal chart" mention in an electional
  answer), even with `rag_data/astro_elect/` fully indexed.

  Two later, explicitly user-requested additions on top of the above,
  both purely additive:

  - **Day-of-week ruler checklist.** `compute_planetary_hour` already
    computed BOTH the day ruler and the hour ruler, but only the hour
    ruler was scored against each category's sympathetic/avoid sets — the
    day ruler was displayed and otherwise ignored. It now gets the exact
    same +/-1 check the hour ruler already got (Andreeva's own per-planet
    recommendations cover both the day and the hour, not two separate
    schemes), applied independently, not as a fallback for a missing hour
    ruler.
  - **Querent's own natal chart, when one exists earlier in the SAME
    conversation.** If the querent already had an `astro_natal_chart`
    request answered earlier in this conversation, `run_electional_chart`
    now also checks real transits from EVERY candidate moment to that
    specific person's own natal Sun/Moon/Ascendant (favorable Jupiter/
    Venus aspects score +1, hard Mars/Saturn squares/oppositions score
    -1) — on top of, never instead of, the generic house-I/house-of-the-
    matter significators every election already checks. This needed the
    FULL prior conversation, not just the current election's own
    round-scoped text (an earlier natal-chart request is its own separate
    round by `_classify_new_electional_round`'s own design, so it's
    normally invisible to this tool) — `routes/chat.py` appends it after
    a dedicated `electional.HISTORY_MARKER` delimiter, past whatever
    round-scoped `tool_arg` it already built, rather than widening the
    round-scoped extraction prompts themselves (same "single string,
    clearly-delimited sections" convention `rectification_events.py`
    already uses for birth data + event lines). A dedicated LLM lookup
    (`_extract_querent_natal_fields_llm`) over that history section looks
    specifically for the QUERENT's own birth data — not the election's
    own moment/place, and not another person's data — and silently does
    nothing when none is found, which is the common case. Doubles the
    per-candidate ephemeris cost for a range search when it does apply
    (an extra `AspectsFactory.dual_chart_aspects` call per hour), accepted
    as a real but bounded cost the same way the window-size ceiling
    already accepts an up-to-370-day hourly scan as worthwhile.

  Known v1 scope limits (see `utils/electional.py`'s own module docstring):
  one house per activity category; no Vronsky-style marriage degree-
  tables, no lunar-day (тити) system, no body-part-specific Moon-sign
  check for medical elections, no muhurta, no religious/church calendar.
  Master methodology copy under `install/methodologies/`; copy it into
  `rag_data/astro_elect/` — again, without the muhurta or Globa source
  files, even as reference material for the corpus.

  **Universal help assistant (`astro_help_assistant`, `utils/tools.py`'s
  `astro_help_overview`).** Every technique above assumes the user already
  knows which one they need. This tool is for the opposite case — a user
  with no astrology background who isn't sure what to ask for, wants
  techniques explained/compared, or wants help phrasing a request — added
  per explicit user request for a conversational "universal assistant"
  rather than a static help page. Reuses the exact same machinery every
  other `astro_*` tool already uses (`_INTERPRETED_TOOL_NAMES` +
  `_TOOL_TOPIC` + the RAG-augmented reasoning-mode follow-up in
  `routes/chat.py`) instead of a bespoke code path: `astro_help_overview`
  returns a fixed, deterministic one-line-per-technique cheat sheet (no
  birth data or free-text extraction of its own — it ignores its
  argument), which becomes the computed-facts context for the same
  follow-up call, reasoning against `astro_help_methodology.txt` (master
  copy under `install/methodologies/`; copy it into `rag_data/astro_help/`
  — see "Recommended `rag_data/` layout" below). That methodology
  document is deliberately NOT another factual astrology corpus — it's
  written for THIS decision (which technique fits a plain-language need),
  covering: a decision tree from a bytovoy description to a specific tool;
  an explicit resolution for the most common real confusion (transit vs.
  progressions vs. directions vs. lunar/solar return vs. profection — all
  "what's happening now/soon" questions with different horizons and
  mechanics); the horary-vs-electional-vs-return distinction (who controls
  the moment — an event that just happens to you vs. one you can schedule
  yourself); and guidance on phrasing a request in plain language, one
  concrete example per technique. `routes/chat.py` special-cases this
  tool's own `computed_chunk` wording (not the shared "точная карта
  конкретного человека" phrasing every other tool's computed_chunk uses —
  there's no specific person's chart behind this one, just reference
  material, and claiming otherwise would mislead the model into treating a
  generic overview as this user's own data).

  `TOOL_REGISTRY`'s description for this tool is deliberately narrow, since
  a router entry this conversational-sounding risks over-triggering: it's
  scoped to genuine uncertainty about which technique applies, technique
  explanation/comparison requests, and request-phrasing help — explicitly
  NOT for a message that already names a specific technique or already
  carries enough data to run one directly (that should still route to the
  specific tool), and NOT for anything unrelated to this app's own
  techniques (general knowledge, small talk — e.g. "когда родился
  Пушкин", "на каком материке Кейптаун" — falls through to the plain,
  non-tool chat path completely unaffected, same as it always did before
  this tool existed). This also means the assistant is reachable
  automatically mid-conversation, in the middle of any other technique's
  own dialogue, whenever the router judges a message this way — no extra
  plumbing needed for that, since the router already re-classifies every
  new message against every registered tool regardless of what the
  previous reply used.

  Reachable two ways: naturally, by typing a question about which
  technique to use (goes through the router logic just described), or via
  the composer's ❓ help-mode toggle (see "Interface" above), which
  bypasses that router logic entirely for one message —
  `ChatRequest.force_help` short-circuits `chat()` straight to
  `astro_help_assistant` with the user's own typed text as `tool_arg`, no
  classification call made at all. The forced path exists precisely
  because router classification is a small-model judgment call, least
  reliable exactly when the user doesn't know what they're asking for yet
  — a UI toggle can't misclassify. It also means a forced message can be
  about literally anything, not just this app's techniques (the toggle
  doesn't pre-filter what the user types) — section 6 of `astro_help_
  methodology.txt` exists specifically for that case: a genuinely
  unrelated question gets a plain, direct answer instead of a forced
  technique recommendation.

  **Recommended `rag_data/` layout.** Each technique's own methodology
  write-up is maintained centrally under `install/methodologies/` (see the
  project layout tree above) and only takes effect once copied into the
  matching `rag_data/<topic>/` subfolder — the filename itself isn't fixed
  or hardcoded (see "RAG — search over your own documents" below, "The
  filename itself isn't fixed or hardcoded anywhere"), only the
  `_methodology` suffix matters. The current recommended subfolder set —
  fewer, broader folders than a strict one-subfolder-per-technique split —
  is:

  | `rag_data/` subfolder | methodology file(s) from `install/methodologies/` | covers |
  |---|---|---|
  | `astro_basics` | `interpretation_methodology.txt` | natal chart reading |
  | `astro_transit` | `transit_methodology.txt` | transits |
  | `astro_synastry` | `synastry_methodology.txt` | two-person compatibility |
  | `astro_progressions` | `progression_methodology.txt`, `direction_methodology.txt`, `lunar_return_methodology.txt`, `solar_return_methodology.txt`, `profection_methodology.txt` | every timing/prognostic technique except transits |
  | `astro_rectif` | `rectification_trutine_methodology.txt`, `rectification_events_methodology.txt` | both rectification tools |
  | `astro_horar` | `horary_methodology.txt` | horary questions (`astro_horary_question`'s reasoning step) |
  | `astro_elect` | `electional_methodology.txt` | electional astrology (`astro_electional_chart`'s reasoning step) — deliberately excludes Vedic muhurta and Globa's lecture material from the source corpus, per explicit design decision; don't add those files to this subfolder even as reference texts |
  | `astro_help` | `astro_help_methodology.txt` | technique selection/explanation help (`astro_help_assistant`'s reasoning step) — not a factual astrology corpus, a decision-guide for which technique fits a plain-language need |

  In each subfolder, put the methodology file(s) first, then whatever
  factual reference corpus you've selected for that topic (planet/house/
  aspect meaning texts, classical sources, etc.) — same "methodology +
  factual corpus, one topic subfolder" structure the rest of this section
  describes.

  One real trade-off worth knowing before adopting this layout:
  `always_include` expansion (see "RAG — search over your own documents"
  below) pulls in *every* methodology chunk from a topic the instant *any*
  chunk from that topic is retrieved — so `astro_progressions` bundling
  five techniques together means a question that only concerns, say,
  solar returns will also inject profection's, progression's, direction's,
  and lunar return's full methodology text alongside it (bounded by
  `RAG_ALWAYS_INCLUDE_MAX_CHARS`, so this can't overflow the context, just
  dilutes it with less relevant material). This was an explicit,
  deliberate simplification over the earlier one-subfolder-per-technique
  recommendation — split `astro_progressions` back into per-technique
  subfolders instead if that cross-activation ever proves confusing to the
  model in practice.

Adding a new tool is meant to be a small, self-contained change:

1. Write a function in `utils/tools.py` with signature `(arg: str) -> str`.
2. Register it in `TOOL_REGISTRY` with a clear one-line description — the
   router builds its classifier prompt directly from these descriptions, so
   a vague description leads to vague routing.

Nothing else needs to change: `utils/tool_router.py` and `routes/chat.py`
read the registry, they don't name individual tools.

Ideas for further tools (not implemented): a persistent notes/reminders
store (the app already has SQLite + a DB layer, so this is mostly a new
table + two tool functions); a "search past conversations" tool once the
sidebar search mentioned below exists; a unit/currency converter; exposing
RAG document search as an explicit tool (`use_rag` is currently automatic-
only via a request flag, not a chat-time decision) so the model can decide
mid-conversation whether to consult the indexed documents. A web search
tool is possible in principle but a bigger lift — it needs outbound internet
access, a search API/key, and more thought about what an offline-first local
app should reach out to the network for.

## Astrological wheel-chart rendering

Every `astro_*` tool reply is now accompanied by a rendered wheel chart —
the same circle-with-houses-and-planets image any desktop astrology
program draws — attached as a file on the assistant's message, alongside
its text answer. Pure SVG (`utils/chart_draw.py`), not PNG or a raster
library: a wheel chart is just circles, radial lines, wedge paths, and
text, all of which SVG expresses natively as a short XML string, rendered
in milliseconds with no model or GPU involved. `image/svg+xml` renders
inline in the chat UI with zero frontend changes — the existing file-
attachment pipeline (`db.repository.add_file` + `GET /api/files/{id}` +
the frontend's `mime_type.startsWith('image/')` check) already handles
any image MIME type, SVG included. Filenames are unix-time-based
(`utils.chart_draw.unique_chart_filename`, millisecond resolution) so two
charts attached to the same reply can never collide.

**Wheel convention** (matches mainstream Western astrology software, refined
against the user's own reference screenshots from a real desktop
astrology program): Ascendant at 9 o'clock (screen left), houses numbered
counterclockwise from there (IV/IC at 6 o'clock, VII/Descendant at 3
o'clock, X/MC at 12 o'clock). Three concentric bands, outer to inner: the
12 zodiac-sign wedges (pastel by element — fire/earth/air/water), a
dedicated light-blue **house band** just inside it holding the house
cusp ticks and Roman-numeral labels, then the planet band. Splitting the
house band out from the zodiac band is deliberate — an earlier version
ran house-cusp lines all the way from the edge to the center, which
visually collided with the aspect web; now the 8 non-angular cusps are
short ticks confined to the outer bands only, never reaching the
planet/aspect area. The two angle axes (Ascendant-Descendant,
Midheaven-Imum Coeli) are drawn as single diameters (each pair is exactly
180° apart by definition) that poke out past every ring and end in a
marker — an arrowhead at the Ascendant, a small circle at the
Midheaven — so the four cardinal points stay identifiable even where
they'd otherwise be lost among aspect lines nearer the center. Every
point additionally gets a small filled dot at its TRUE (un-nudged) degree
right on the zodiac ring's inner edge — necessary because the glyphs
themselves use a collision-avoiding "lane" system (crowded points nudged
onto concentric rings within the planet band, each with a thin tick line
back to its dot) — without that dot, an unaspected planet whose glyph got
nudged away from its real position would be impossible to locate exactly.
A retrograde point shows a small "R" riding below the glyph's baseline
(SVG's `baseline-shift="sub"`), not the traditional "℞" character, which
read poorly at this size.

**Aspect lines**, per the explicit brief this was built against: trine/
sextile = green, square = red, conjunction/opposition = blue; every minor
aspect (semisextile, semisquare, quintile, sesquiquadrate, biquintile,
quincunx) = purple/lilac, with a lighter or darker shade depending on
whether `utils/astro.py`'s own harmonious/tense classification calls it
one or the other. Line width is interpolated between thick (an exact, 0°-
orb aspect) and thin (right at that aspect's own configured max orb) —
"how close to exact" drives thickness rather than a fixed width per
aspect type. Applying aspects (still forming) are solid; separating ones
are dashed. Every aspect line is anchored to a planet's true-degree dot
(the same dot described above), never to a separate floating inner point —
an earlier version anchored aspects to a fixed inner circle with nothing
marking it, which looked like the lines were "hanging" in empty space.
Aspect lines on the wheel itself are restricted to the 10 classical
planets plus Ascendant/MC (`chart_draw._WHEEL_ASPECT_POINTS`) — a real
visual test with the full active-point set (which also includes the
lunar node, Chiron, Lilith, and Part of Fortune) produced an unreadable
web of ~100 possible pairs; those extra points are still placed on the
ring as glyphs, just without aspect lines of their own.

**Per-technique aspect set/orb.** Wheel-chart drawing (`utils/chart_draw.py`,
used by both chat and PDF export, which just reuses the same stored SVG
bytes) now uses four separate orb profiles instead of one flat table for
everything — real astrology software conventionally uses different orbs
depending on what's being compared, and this app previously didn't.

- *Classical* (horary/electional, `draw_wheel_svg`'s `classical_aspects=True`,
  set from `tool_name` in `routes/chat.py._attach_chart_if_applicable`):
  only the six classical aspect types (conjunction/sextile/square/trine/
  quincunx/opposition — not the five extra minors), with an orb that
  depends on which bodies are involved — 8-10° when either party is the
  Sun or Moon, 6-7° between other planets, a flat 5° for quincunx — per
  `horary_methodology.txt` section 4 (`electional_methodology.txt` reuses
  the same rule explicitly). This was the first fix made here and remains
  untouched by the three profiles below (deliberately — the two
  methodology docs are the ground truth for this pair, not any of the
  screenshots that shaped the other three).
- *Natal* (`astro.natal_orb_limit`): a chart's own INTERNAL aspects —
  applies no matter which technique produced that chart (a real natal
  chart, a progressed or return chart read on its own, ...). Orb is a
  genuine per-body table (the classical "moiety" convention): each body
  has its own allowance per aspect type, and the real orb between two
  specific bodies is the average of their two allowances (e.g. Sun's
  conjunction orb 12° + Moon's 10° → 11° for a Sun-Moon conjunction).
- *Transit-family* (`astro.transit_orb_limit`): cross-chart aspects
  between one real chart and a technique-derived moment — transit,
  progression, lunar return, solar return (direction never reaches this
  at all: its "second" is a list of synthetic overlay points, not a real
  subject, so cross-chart aspects are never computed for it in the first
  place). Same per-body-average rule as natal, just far tighter overall
  and only differentiating Sun/Moon from everything else on the four
  tightest minor-aspect rows.
- *Synastry* (`astro.synastry_orb_limit`): cross-chart aspects for
  `astro_synastry_chart` specifically — two real people conventionally
  get a different, wider orb than a single derived moment. Flat per-aspect
  orb, no per-body variation (the reference table had none). Semi-sextile
  and quincunx have no dedicated row at all and fall back to the generic
  `astro._ALL_ASPECTS` default, matching the reference software's own gap.

All four numeric tables (`astro._NATAL_ORB_BY_BODY`/`_TRANSIT_ORB_BY_BODY`/
`_SYNASTRY_ORB_BY_BODY`, plus the pre-existing `_CLASSICAL_ASPECTS_WIDE`)
were transcribed from the user's own reference astrology software's
per-technique aspect/orb configuration screens, deliberately excluding
aspect families (septile/novile/decile) this app never computed before
and has no methodology text prescribing.

This used to be a real, reported bug: every chart was always drawn with
one flat general table regardless of technique, so e.g. a horary wheel
could show an aspect line (a quintile) that doesn't exist under horary's
own doctrine at all, and a synastry chart used the same tight orb as a
transit overlay despite the two techniques conventionally using very
different widths. Investigating the horary/electional half of this also
surfaced that `utils/horary.py`'s own verdict computation had the
identical gap — worse, a previous fix attempt (a local `_HORARY_ASPECTS`
list, correctly reasoned) was written but never actually passed to
kerykeion, so it did nothing — and `utils/electional.py`'s own
`_ELECTIONAL_ASPECTS` correctly restricted the aspect *types* but never
added the luminary-aware orb either; both are now fixed and share the
classical implementation with the wheel.

kerykeion's own `active_aspects` parameter can only express one orb per
aspect *name*, not per pair of bodies (and its own pydantic validation
requires that orb to be a whole-degree int) — so every non-classical
chart is computed against `astro._PER_TECHNIQUE_ASPECTS_WIDE`, a table
widened (and rounded up) to whatever the widest natal/transit/synastry
allowance for that aspect could ever be, then filtered back down in
Python to the real, narrower, per-pair cutoff via
`astro.natal_orb_limit`/`transit_orb_limit`/`synastry_orb_limit` before
anything is drawn — the same "wide table, then post-filter" pattern the
classical profile already used. `chart_draw.draw_wheel_svg` takes a new
`dual_orb_profile` parameter ("transit" by default, "synastry" for
`astro_synastry_chart`) to pick which of the two cross-chart functions
applies; the inner chart's own aspects always use the natal profile
regardless of that flag, since a chart's own internal aspects don't
change meaning just because it's being compared to something else.

**Header block.** Each chart's top-left text block now includes the
technique label, the subject's name (when a real one was given, not the
generic "Subject"/"electional" placeholder), a `DD.MM.YYYY HH:MM` line
read straight off the kerykeion subject's own stored fields, and a place
line — a real city name if one can be matched in the same birth-info text
via `utils.astro._lookup_city_exact` (the same gazetteer the free-text
parser already uses), falling back to plain coordinates otherwise.
`subject.city` itself can't be used for this — `utils.astro._build_subject`
only ever passes lat/lon/tz into kerykeion, never a city string, so it
would just show kerykeion's own placeholder ("Greenwich") instead of the
real place.

**Viewing a chart full-size.** `GET /api/files/{id}` used to always send
`Content-Disposition: attachment`, which forces a download even when a
person deliberately opens the file's own URL in a new tab to see it
larger — routes/files.py now sends `inline` for every `image/*` MIME
type (still `attachment` for everything else, e.g. extracted code files),
and the chat UI wraps each image attachment in a `target="_blank"` link
so clicking it opens the full chart in its own tab instead of triggering
a download.

**Single vs. dual charts.** Every technique ends up with either one
kerykeion subject (natal, horary, profection, electional's winning
moment, each rectification tool's winning candidate) or two (transit,
synastry, progression, solar/lunar return) — `draw_wheel_svg`'s own
`(subject, second=...)` signature mirrors that split. When there's a
second subject, its own planets are drawn in an outer ring at their real
degree, read against the FIRST (reference) subject's own houses/signs —
the same interpretive convention this app's dual-chart text reports
already use (see `astro.get_dual_chart_profiles`), just drawn instead of
narrated. Solar arc directions are the one exception with no real second
chart at all (every natal point is shifted by one shared arc, not
independently cast) — `astro.run_direction_and_subject` returns a plain
list of overlay points (its 3rd tuple slot) instead of a second subject,
and the renderer draws that ring with no cross-chart aspect lines
(there's nothing to compute them against). Profection instead shades its
one "activated" house wedge distinctly (`highlight_house`, its 4th tuple
slot).

**Deciding whether to draw at all.** `chart_draw.should_draw_chart`
default is to draw — the user's own stated requirement is "draw unless
told not to" — via a one-line LLM classification of the user's message
for an explicit opt-out ("без картинки", "не рисуй карту", "только
текст", ...). This is the one LLM-first classifier in the app with a
*permissive* default: every other classifier here (image intent, new-
round detection, etc.) defaults to the conservative/narrower behavior
when no model is loaded or on any exception, but this one defaults to
`True` (draw) in both of those cases, since drawing is the expected
normal outcome, not the exceptional one.

**Where the subject comes from, per tool** (`routes/chat.py`,
`_SIMPLE_AND_SUBJECT_FUNCS` and the surrounding dispatch code in
`_handle_tool_request`): for 8 of the 10 chart techniques (natal,
transit, progression, direction, lunar/solar return, profection) plus
horary, the tool's own `run_*` function has a sibling `run_*_and_subject`
that does the full field-extraction/ephemeris computation exactly ONCE
and returns `(text, subject, second, highlight_house)` (a uniform 4-tuple,
padded with `None` for whichever slots a given technique doesn't use) —
`run_*` itself is now a thin one-line wrapper (`return
run_x_and_subject(spec)[0]`). `_SIMPLE_AND_SUBJECT_FUNCS` maps all 9 of these tool names (the 8 above
plus horary) straight to its `_and_subject` function so
`_handle_tool_request` can call it once, generically, and unpack the
same 4-tuple shape for every one of them — including horary's
`run_horary_question_and_subject`, which pads its own `(text, subject)`
result out to `(text, subject, None, None)` for exactly this reason (an
earlier version of this function returned a bare 2-tuple, which crashed
`_handle_tool_request`'s generic unpack with "not enough values to
unpack" the first time a live horary request actually hit that code
path — fixed by conforming to the same 4-tuple shape as every other
entry in the dict, rather than special-casing horary's dispatch).
This replaced an earlier design where a separate
`get_*_chart_subject(s)` getter rebuilt the same subject from scratch
*after* the text reply had already computed it once — harmless-looking
but a real, reported regression (every one of these 9 techniques was
doing its full ephemeris/fixed-star/aspect computation twice per
request); the `_and_subject` pattern eliminates that by construction,
since the SAME call now produces both the text and the chart data.
Synastry follows the identical idea via `run_synastry_and_subject`
(returning `(text, person_a, person_b)`), called with the exact same
`split_hint` (LLM-assisted person-A/person-B text split) that produced
the text reply — without that, a rare case where the plain heuristic
split failed could draw a different pairing than the one described in
the answer. The three search-based techniques — electional
(date-range scan) and both rectification tools (candidate-window scan) —
are the one deliberate exception to "just recompute it": their search is
too expensive to run twice, so the winning candidate's subject is
threaded straight out of the SAME search call that produces the text
report, via sibling functions added specifically for this
(`electional.run_electional_chart_and_subject`,
`rectification._run_rectification_trutine_full`,
`rectification_events.run_rectification_events_and_subject_async`) —
`run_electional_chart`, `run_rectification_trutine`, and
`run_rectification_events` (the plain string-returning versions
`TOOL_REGISTRY` still calls directly) are now thin wrappers around these.
For an electional range search, this means the chart drawn is the winning
candidate's own chart, not the querent's natal chart or any rejected
candidate.

## Exporting a message to PDF

A 📄 button next to each message's copy button (`GET /api/messages/{id}/pdf`,
`routes/export.py`) renders that one message — its text plus every image
attachment it has (a chart SVG, most commonly) — as a standalone PDF,
opened `inline` in a new browser tab (the viewer's own controls cover
save/print; no separate download endpoint needed). One message per PDF by
design, not a whole-conversation export: the person's own ask was "a
button on each reply", and a single message is a simpler, always-fresh
unit — nothing to go stale if the conversation keeps growing after the
button was clicked.

`utils/pdf_export.py` renders via **weasyprint** (HTML+CSS → PDF) rather
than a low-level drawing library like reportlab, for two reasons specific
to this app's content: reportlab's built-in fonts have no Cyrillic glyphs
at all (every Russian character would come out as tofu boxes unless a TTF
font were manually registered), and reportlab has no SVG support at all
(the rendered wheel charts are SVG — weasyprint embeds them natively via
a `data:image/svg+xml;base64,...` `<img>`, no separate rasterization
step). Needs system-level libraries and fonts beyond the `pip install`
itself — see "Installation & Setup" above ("6. Optional: PDF export") for
the exact packages.

The message-text rendering (bold via `**`, headings via `#`, fenced code
blocks) is a deliberate line-for-line port of `static/js/app.js`'s own
`renderProseText`/`renderInlineFormatted`/`renderMessageBody` — the PDF
is meant to look like the same message already shown in the chat UI, not
a differently-formatted document. If that JS ever changes, mirror the
change in `utils/pdf_export.py` too (each function there names its exact
JS counterpart in its own docstring).

A freshly-received reply needs its own DB message id before the button
can point anywhere real — `routes/chat.py`'s response dicts now include
`"message_id": assistant_msg_id` on every complete-status reply (previously
only the async image-job placeholder path returned this), and the chat
UI's submit handler reads it into `record.id` so the button works
immediately, not just after a page reload.

## Why file attachments are stored as BLOBs in SQLite

Code extracted from model replies, and images resolved from ycplt_img, are
stored as BLOBs in the `files` table — alongside the conversation and message
records, in one database file. For a local single-user app this is the
simplest option: one database = one backup, no need to keep files on disk in
sync with database records. If large binary files become common enough to
matter, storing only a path/metadata in the DB and the file itself on disk
would be worth revisiting — but for code snippets and individual images this
is unnecessary complexity for now.

## RAG — search over your own documents (optional)

1. Install dependencies — see "Installation & Setup" above ("7. Optional: RAG
   document search") for the exact `pip install` list (all covered in one go
   by `install/requirements.txt`) and which system tool each source format
   needs (already installed in step 1 of that section: `antiword`,
   `djvulibre`, `poppler-utils`, `tesseract`, `chmlib`, `ha`).

   `.zip` archives are read directly (stdlib, no extra package). `.rar`/`.arj`/`.7z`,
   legacy binary `.doc` (MS Word 97-2003, NOT `.docx`), `.ha` and `.chm` archives, `.djvu`/`.djv`
   scans, and scanned (no-text-layer) `.pdf` pages all need matching **system** tools — see the
   "Installation & Setup" section above for the package names and the `ha` build steps
   (Debian/Ubuntu and Fedora both covered there).

   `.djvu`/`.djv` specifically needs `djvulibre` (provides the `djvutxt`, `ddjvu`, and
   `djvused` CLI tools) always, plus `tesseract` with a Russian language pack
   (`tesseract-ocr-rus` on Debian/Ubuntu, `tesseract-langpack-rus` on Fedora) *only* for
   scans that have no embedded OCR text layer — `build_index.py` tries the fast direct
   extraction first (`djvutxt`) and only falls back to rendering pages and OCR'ing them
   (`ddjvu` + `tesseract`) if that comes back empty, so tesseract is only actually invoked
   for scans that genuinely need it.

   `.pdf` gets the same OCR treatment, page by page rather than whole-document: pypdf's
   normal text extraction is tried first for every page, and only pages that come back
   (near-)empty — genuinely scanned pages with no text layer, which do turn up mixed into
   otherwise "born-digital" PDFs — are rendered via `pdftoppm` (**`poppler-utils`**) and
   OCR'd via the same `tesseract` as `.djvu`. A fully text-based PDF never touches
   `pdftoppm`/`tesseract` at all.

   `.chm` (Compiled HTML Help, WinHelp's successor) needs `extract_chmLib` — package
   `chmlib` on Fedora, `libchm-bin` on Debian/Ubuntu — which unpacks it into a plain
   directory of HTML "chapter" files, each then read exactly like any other `.html` file
   (no CHM-specific parsing of its own).

   Without these system tools installed, `build_index.py` doesn't fail: `.doc` files fall
   back to a cruder built-in text scrape, scanned `.pdf` pages and `.djvu`/`.djv` files
   without their OCR tools are indexed with whatever text pypdf/djvutxt *could* extract
   (possibly none, for a fully scanned document), and archives (including `.chm`) it can't
   open are skipped with a printed warning — one missing tool never aborts the whole
   indexing run.

2. Put your files (`.txt`, `.html`/`.htm`, `.rtf`, `.pdf`, `.doc`, `.djvu`/`.djv`, or
   `.zip`/`.rar`/`.arj`/`.7z`/`.ha`/`.chm` archives of any of those) in `rag_data/` (created automatically on first run of
   the script if missing). Archives are extracted and their contents indexed the same way as
   loose files, recursively (including archives nested inside archives, up to a small depth
   limit). Group documents into topic subfolders if you have more than one subject —
   `rag_data/astrology/planets.txt`, `rag_data/cooking/pasta.txt`, and so on (for this
   project's astro subject specifically, see "Recommended `rag_data/` layout" above for the
   current 8-subfolder scheme and which `install/methodologies/*_methodology.txt` master copy
   goes in each). Each topic
   subfolder (plus the loose files directly in `rag_data/`, if any) is its own **corpus**,
   with its own index file (see step 3) — not one combined index for everything. Retrieval
   at query time still searches across every corpus in one pass regardless of topic (there's
   no per-request topic selector in the app), but every chunk is tagged with its topic in its
   corpus's metadata, which is what the methodology mechanism below relies on. Files placed
   directly in `rag_data/` (no subfolder) form their own "root" corpus with topic=None.

3. Build the index(es):

   ```bash
   python build_index.py                  # build every corpus found under rag_data/
   python build_index.py <topic>           # build only rag_data/<topic>/
   python build_index.py root              # build only the loose files directly in rag_data/
   python build_index.py path/to/folder    # build only that folder, wherever it is
   ```

   Each corpus gets its own `faiss_index.bin`+`meta.pkl` pair under `data/rag_index/<topic>/`
   (`data/rag_index/_root/` for the no-topic case) — `utils/rag.py` loads every corpus found
   there at app startup and merges their retrieval results, so this is transparent to
   anything reading RAG results. The practical payoff: re-run this for a single topic any
   time only that topic's documents changed, instead of re-reading (and re-OCR'ing) every
   other topic's documents too just to pick up one new file. This matters in practice — a
   real multi-topic corpus with several OCR-heavy `.djvu`/`.pdf` scans could previously take
   hours to rebuild from scratch with no way to redo just the one corpus that changed, and no
   visibility into which corpus indexing was even on.

   Reading a corpus's documents (as opposed to embedding them) is dominated by external
   processes — antiword, djvutxt, ddjvu, pdftoppm, tesseract, extract_chmLib, patool — one
   per document, or one pair per OCR'd page. These run **concurrently**, not one at a time:
   `INDEX_CONCURRENCY` (default 4, `utils/config.py`) caps how many run at once. Measured on
   a genuinely 2-core sandbox, OCR'ing an 8-page scanned PDF took 8.4s at
   `INDEX_CONCURRENCY=1` and 4.1s at `INDEX_CONCURRENCY=2` — a ~2x speedup matching the core
   count exactly, with no further gain past it on that hardware. Raise `INDEX_CONCURRENCY` on
   a faster/more-core machine; lower it if indexing makes the machine unresponsive for other
   work while it runs. This splits documents into chunks, computes embeddings (one batched
   call per corpus — concurrency doesn't apply here, it's not the bottleneck), and saves the
   FAISS index and metadata. Re-run for a given corpus any time its source documents change,
   or for everything after changing `EMBED_MODEL` (see below — embeddings from a different
   model aren't compatible with an existing index even if the vector dimension happens to
   match).

4. Restart the server so it picks up the index(es) at startup (the log will show
   `RAG index loaded: N corpus/corpora, M chunks total`).

5. Add `"use_rag": true` to a `/chat` request (no toggle in the browser UI
   yet — API/curl only):

   ```bash
   curl -X POST http://127.0.0.1:4010/chat -H "Content-Type: application/json" \
     -d '{"query": "What do the documents say about X?", "use_rag": true}'
   ```

If no index has been built, or the RAG dependencies aren't installed, the
server keeps working as a normal chat; `use_rag` simply has no effect (see
`utils/rag.py`).

**Embedding model.** `EMBED_MODEL` defaults to
`paraphrase-multilingual-MiniLM-L12-v2`, not the more commonly-referenced
`all-MiniLM-L6-v2` — the latter is English-only and gives noticeably worse
retrieval on non-English documents (Russian included). Both are
sentence-transformers models of similar size/speed on CPU; the
multilingual one just also understands ~50 other languages. Override
`EMBED_MODEL` in `.env` if you'd rather use a different one (e.g. a
larger multilingual model for better quality at the cost of slower
indexing/queries) — just remember to rebuild the index afterward.

This is **not** a file you download and place under `models/` yourself,
unlike `MODEL_PATH` (the GGUF chat model). `SentenceTransformer(EMBED_MODEL)`
fetches it automatically from the Hugging Face Hub
(https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,
~470MB) the first time it's needed — either the first `python build_index.py`
run or the first app startup — and caches it locally afterward (default
cache dir `~/.cache/huggingface/hub`), the same auto-download pattern
already used for CLIPSeg in ycplt_img. The only implication for a
privacy-focused local setup: the machine needs internet access once, for
that first download; every run after that is fully offline.

**Methodology documents — always included, not just similarity-matched.**
Ordinary chunk-similarity search is good at finding isolated facts ("what
does Mars in the 7th house mean"), but bad at surfacing a document that
describes *how to combine* facts into a conclusion — its wording rarely
resembles a specific question closely enough to rank in the top-k. This
matters for a genuinely synthesis-oriented topic like astrological
interpretation, where the individual planet/house/aspect facts retrieve
fine on their own, but the rules for weighing and combining several of
them into a new, specific conclusion might not surface at all.

Name such a document with a `_methodology` suffix before the extension —
e.g. `rag_data/astrology/interpretation_methodology.txt` — and
`utils/rag.py`'s `retrieve_context` will always include its chunks
whenever ordinary similarity search already found other chunks from the
same topic, regardless of the methodology document's own similarity rank.
**The filename itself isn't fixed or hardcoded anywhere** — only that
`_methodology` suffix (checked by `build_index.py`'s `_is_methodology()`)
matters; call it `synastry_methodology.txt`, `natal_methodology.pdf`,
`метод_транзитов_methodology.txt`, whatever fits your own naming scheme.
This also means every topic subfolder gets its own independent
methodology document(s) — the suffix is checked per file, and
`always_include` expansion (below) only ever pulls in other chunks from
the *same* topic, so `rag_data/astro_transit/foo_methodology.txt` and
`rag_data/astro_synastry/bar_methodology.txt` never activate each other.
When any such chunk is present, `build_prompt` also switches from a plain
"answer using this context" prompt to one that asks the model to reason
step by step — list the relevant facts, think through how they interact
per the methodology, *then* answer — instead of a one-shot lookup-style
response. This is a prompting change on the existing chat model, not a
separate reasoning model; a dedicated reasoning-tuned model (loaded
alongside the main one, the same pattern as the vision model) is a
plausible next step if this isn't enough on its own, but is meaningfully
more infrastructure and hasn't been built — try this first and see
whether the depth of synthesis is actually the bottleneck before adding
it.

## Known gotchas

- **`TypeError: unhashable type: 'dict'` when opening `/`** — the old call
  `templates.TemplateResponse("index.html", {"request": request})` isn't
  compatible with modern Starlette (the signature changed to
  `(request, name, ...)`). `routes/pages.py` already uses the correct call:
  `templates.TemplateResponse(request=request, name="index.html")`.
- **Model fails to load / format error** — the file must be **GGUF**, not the
  older `.bin`/`.ggml` (ggmlv2/ggmlv3) formats, which current llama.cpp
  doesn't read.
- **`ImportError` for a constant from `utils.config`** — if a code update
  breaks importing some constant, an old `utils/config.py` is likely still on
  disk — replace it wholesale rather than patching it by hand.
- **Image requests stuck `pending` forever** — check that ycplt_img is
  reachable at `IMAGE_SERVICE_HOST`/`IMAGE_SERVICE_PORT` (see its own
  firewall notes) and that its job queue isn't stuck; `utils/image_jobs.py`
  logs poll errors to stdout.
- **Replies still feel short even with no `max_tokens` cap** — the real
  ceiling is `N_CTX` (the context window, in tokens, shared between the
  prompt and the reply). Raise it in `.env` if you need longer answers;
  a bigger `N_CTX` costs more RAM and makes each token slightly slower to
  generate, so it's a deliberate trade-off, not a free change.
- **Questions about an attached image resolve to an error** — the vision
  model lives on ycplt_img, not here (see "Understanding an uploaded
  image" above); check that service's own `GET /health` `vision` field
  and make sure the two moondream2 GGUF files are in *its* `models/`
  directory (a common mix-up: they don't belong in this app's `models/`,
  which only holds the chat LLM).
- **First question about an attached image is slow** — expected: ycplt_img
  loads the vision model lazily on first `caption` job, not at its own
  startup. Subsequent questions are fast, same as any other model after
  its one-time load.

## Not done yet (but the architecture allows for it)

- A RAG toggle in the browser UI itself (API-only for now).
- Searching chat history (renaming and downloading a chat archive are done
  — see "Interface" above).
- Pagination of message history for very long conversations.
- Masked inpainting from the browser UI (mask drawing) — the API/ycplt_img
  side already supports a mask (`mode="inpaint"`, see ycplt_img's own
  README), but nothing in the browser UI produces one yet; today an
  attached-image edit is always a whole-image img2img instruction.
- Chat-UI integration for birth profiles — import/export against the
  AstroZet `.zbs` format is implemented (see "Birth profiles" above), but
  there's no browser UI yet for actually *using* a saved profile inside a
  conversation (e.g. referencing one by name instead of retyping full
  birth data each time). Deliberately parked: it's unclear whether that
  should be a picker, a button next to the composer, a slash-command, or
  something else, and this needs more thought before building it.
