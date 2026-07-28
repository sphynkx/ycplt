# ycplt — local chat server (FastAPI + llama-cpp-python)

A local ChatGPT-like app: a FastAPI server running a GGUF model on CPU
(llama-cpp-python), a browser UI (sidebar with parallel chats, persisted
history), optional RAG search over your own documents, code extraction from
model replies as downloadable file attachments, and automatic image
generation/editing requests routed to a companion service
([ycplt_img](https://github.com/sphynkx/ycplt_img)) based on the meaning of
the user's message (no manual toggle).

## Project layout

```
app.py                    — entry point: FastAPI app, routers, startup wiring
routes/
  chat.py                   — POST /chat, GET /health
  conversations.py          — /api/conversations — list/create/history/delete
  files.py                  — /api/files/{id} — download file attachments
  pages.py                  — GET / (browser chat page)
db/
  connection.py              — SQLite connection, schema, init_db(), migrations
  repository.py               — CRUD: conversations, messages, files
utils/
  config.py                  — all settings, loaded from .env (python-dotenv)
  llm.py                      — loads and calls the GGUF model (llama-cpp-python)
  rag.py                      — optional RAG (FAISS + sentence-transformers)
  codeblocks.py                — extracts fenced code blocks from model replies
  intent.py                    — LLM-based classifier: is this an image request?
  image_client.py               — HTTP client for the ycplt_img job queue
  image_jobs.py                 — background poller resolving pending image jobs
  tools.py                      — registry of built-in tools (datetime, calculator, ...)
  tool_router.py                 — LLM-based classifier: does this need a tool?
templates/
  index.html                  — sidebar + chat UI markup (fetch to /chat and /api/*)
static/
  js/app.js                    — browser-side JavaScript, served at /static/js/app.js
build_index.py             — builds the FAISS index from RAG source documents
install/
  requirements.txt            — Python dependencies
  .env.example                 — template for .env (copy and adjust)
  ycplt.service                — systemd unit file
models/model.gguf          — the model file (you provide it, see below)
rag_data/                  — RAG source documents (*.txt, *.pdf) — you provide them
data/                      — generated data:
  faiss_index.bin, meta.pkl  — RAG index (build_index.py)
  chat.sqlite3                — conversations/messages/files (created at startup)
```

## Hardware and why this stack

Reference hardware: i7-5500U (2 physical cores / 4 threads), 12 GB RAM,
GeForce 940M (2 GB) — no usable GPU acceleration for LLM inference. Hence:

- Inference via **llama-cpp-python** — native GGUF support, CPU inference,
  actively maintained.
- A **3B-class model in Q4_K_M quantization**, not 7B: on this CPU a 7B model
  runs at ~1 token/sec, too slow for comfortable chat.
- GPU is not used (`N_GPU_LAYERS=0` by default): the 940M (compute capability
  5.0, 2 GB VRAM) gives no practical speedup.
- Chat history lives in **SQLite** (`data/chat.sqlite3`) — enough for a
  single-user local app, no separate database server needed.

## Installation

```bash
cd /opt
git clone https://github.com/sphynkx/ycplt
cd ycplt
python -m venv .venv
source .venv/bin/activate
pip install -r install/requirements.txt
```

## Configuration (.env)

Copy the template and adjust as needed:

```bash
cp install/.env.example .env
```

All settings are read via `python-dotenv` in `utils/config.py`. Priority
(highest first): a real process environment variable > a value from `.env` >
the hardcoded default.

Path-valued settings (`MODEL_PATH`, `DB_PATH`, `RAG_DATA_DIR`, `INDEX_PATH`,
`META_PATH`) may be relative or absolute. A relative value is resolved
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
| `N_CTX` | `2048` | Context window |
| `N_GPU_LAYERS` | `0` | Layers offloaded to GPU (0 = CPU only) |
| `DB_PATH` | `data/chat.sqlite3` | Chat history database |
| `RAG_DATA_DIR` | `rag_data` | RAG source documents folder |
| `INDEX_PATH` | `data/faiss_index.bin` | Built FAISS index |
| `META_PATH` | `data/meta.pkl` | RAG chunk metadata |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers embedding model |
| `TOP_K` | `3` | Number of RAG chunks retrieved per query |
| `IMAGE_SERVICE_HOST` | `192.168.7.7` | ycplt_img host |
| `IMAGE_SERVICE_PORT` | `4011` | ycplt_img port |
| `IMAGE_POLL_INTERVAL_SEC` | `10` | How often the background poller checks ycplt_img |
| `IMAGE_HTTP_TIMEOUT_SEC` | `10` | Timeout for short status/submit requests (not generation itself) |

## Model

Download a GGUF model and place it at `models/` (or point `MODEL_PATH` at it):

```bash
wget https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf -O models/qwen2.5-3b-instruct-q4_k_m.gguf
```

Recommendation for the reference hardware:

- **Qwen2.5-3B-Instruct-GGUF** (file `qwen2.5-3b-instruct-q4_k_m.gguf`, ~2 GB) —
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF — good speed/quality
  balance, handles Russian well.
- If too slow — **Qwen2.5-1.5B-Instruct-GGUF** (faster, lower quality).
- If you want more capability and don't mind the slowdown — 7B models in
  Q4_K_M (Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3), though not
  comfortable on this CPU.

## Running

```bash
python app.py
# or: uvicorn app:app --host <HOST> --port <PORT>
```

The chat UI is served at `http://<HOST>:<PORT>/` (default
`http://127.0.0.1:4010/`). On first run, `data/chat.sqlite3` is created
automatically with the current schema; on later runs, `init_db()` migrates
an existing database in place if new columns were added (no data loss).

### Running as a systemd service

```bash
sudo cp install/ycplt.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ycplt
```

The unit file uses `EnvironmentFile=/opt/ycplt/.env` and intentionally has no
`User=` line (this deployment runs as root). Adjust `WorkingDirectory` and
`ExecStart` if the app lives somewhere other than `/opt/ycplt`.

## Interface

- **Sidebar** — "+ New chat" button and a list of conversations (ChatGPT-style).
  Clicking a conversation switches to it; the "✕" deletes it (with confirmation).
- **Parallel chats** — each conversation is stored separately in the database;
  any number can be kept simultaneously and switched between via the sidebar.
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

## API

| Method/path | Description |
|---|---|
| `POST /chat` | Send a message. Body: `{query, conversation_id?, use_rag?, max_tokens?, temperature?}`. Without `conversation_id`, creates a new conversation. `max_tokens` defaults to `null` — no artificial cap; the model generates until it stops on its own or fills the context window (`N_CTX`). Pass an explicit value only to deliberately shorten a reply. If the message is classified as an image request, returns a `pending` placeholder immediately instead of chat text; otherwise returns the model's reply with `sent_at`/`responded_at` (ms), `thinking_ms`, and a `files` list. |
| `GET /health` | Model/RAG-index status and the configured `image_service_url`. |
| `GET /api/conversations` | List conversations (id, title, updated_at), sorted by last activity. |
| `POST /api/conversations` | Manually create an empty conversation (usually unnecessary — `/chat` creates one lazily). |
| `GET /api/conversations/{id}/messages` | Message history for a conversation, including files, timestamps, and `status`. |
| `DELETE /api/conversations/{id}` | Delete a conversation (cascades to its messages and files). |
| `GET /api/files/{id}` | Download a file attachment (extracted code, or a generated image). |

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

Built-in tools:

- **`get_current_datetime`** — current date, time, and day of week.
- **`calculate`** — arithmetic expressions (`+ - * / // % **`, parentheses).
  Evaluated via a restricted `ast` walk, not `eval()` — it can only ever
  do arithmetic on numbers, never call functions, import anything, or
  access names, so a malformed or adversarial expression just returns an
  error string instead of executing.

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

1. Install indexing dependencies:

   ```bash
   pip install sentence-transformers faiss-cpu numpy
   pip install pypdf   # only needed if some documents are *.pdf
   ```

2. Put your files (`.txt` and/or `.pdf`) in `rag_data/` (created automatically
   on first run of the script if missing).

3. Build the index:

   ```bash
   python build_index.py
   ```

   This splits documents into chunks, computes embeddings
   (`all-MiniLM-L6-v2`), and saves the FAISS index and metadata to
   `data/faiss_index.bin` and `data/meta.pkl` — the same paths `utils/rag.py`
   reads at app startup.

4. Restart the server so it picks up the index at startup (the log will show
   `RAG index loaded: N chunks`).

5. Add `"use_rag": true` to a `/chat` request (no toggle in the browser UI
   yet — API/curl only):

   ```bash
   curl -X POST http://127.0.0.1:4010/chat -H "Content-Type: application/json" \
     -d '{"query": "What do the documents say about X?", "use_rag": true}'
   ```

If no index has been built, or the RAG dependencies aren't installed, the
server keeps working as a normal chat; `use_rag` simply has no effect (see
`utils/rag.py`).

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

## Not done yet (but the architecture allows for it)

- A RAG toggle in the browser UI itself (API-only for now).
- More sidebar features (renaming a chat, searching history, etc.).
- Pagination of message history for very long conversations.
