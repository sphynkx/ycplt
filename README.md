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
                                  techniques). Not read directly by the app — copy the file(s)
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

Verified end-to-end on a fresh Fedora install. System packages up front cover
everything RAG source ingestion needs (`antiword`/`.doc`, `p7zip`+
`p7zip-plugins`/`unrar-free` for archives, `djvulibre`/`.djvu` scans,
`poppler-utils`/rendering scanned PDF pages, `tesseract`+its Russian
language pack for OCR'ing anything with no embedded text layer (djvu or
PDF), `chmlib`/`.chm` help files, `gcc`/`cmake`/`python-devel` to build the
`ha` archiver from source — see "RAG — search over your own documents"
below for what each package is for):

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

## Configuration (.env)

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
| `N_CTX` | `32768` | Context window (Qwen2.5's native length — lower it if RAM is tight) |
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
| `RAG_ALWAYS_INCLUDE_MAX_CHARS` | `16000` | Cap on methodology-doc auto-inclusion size (see utils/rag.py) |
| `HF_TOKEN` | (unset) | Optional Hugging Face Hub token (rate limit / warning) |
| `HF_HUB_OFFLINE` | (unset) | Set to `1` once models are cached, to skip Hub network checks entirely |
| `IMAGE_SERVICE_HOST` | `192.168.7.7` | ycplt_img host |
| `IMAGE_SERVICE_PORT` | `4011` | ycplt_img port |
| `IMAGE_POLL_INTERVAL_SEC` | `10` | How often the background poller checks ycplt_img |
| `IMAGE_HTTP_TIMEOUT_SEC` | `10` | Timeout for short status/submit requests (not generation itself) |

## Model

Download a GGUF model and place it at `models/` (or point `MODEL_PATH` at it):

```bash
wget https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf -O models/qwen2.5-3b-instruct-q4_k_m.gguf
```

Recommendation for the original reference hardware (i7-5500U, 2c/4t, 12 GB
RAM, no usable GPU):

- **Qwen2.5-3B-Instruct-GGUF** (file `qwen2.5-3b-instruct-q4_k_m.gguf`, ~2 GB) —
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF — good speed/quality
  balance, handles Russian well.
- If too slow — **Qwen2.5-1.5B-Instruct-GGUF** (faster, lower quality).

On more capable CPU-only hardware (tested on an i7-8700, 6c/12t, 16 GB
RAM, no discrete GPU) a bigger quant is comfortably usable and noticeably
better at Russian instruction-following and long structured answers (the
astro interpretation feature in particular benefits — see below):

- **Qwen2.5-7B-Instruct, Q4_K_M** (~4.7 GB) — meaningful step up from 3B
  while still comfortably fast on a modern 6+ core CPU:
  https://huggingface.co/paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF
  (direct file: https://huggingface.co/paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf)
- **Qwen2.5-14B-Instruct, Q4_K_M** (~9 GB) — noticeably slower per token but
  fits in 16 GB RAM alongside the default `N_CTX=32768`; the strongest
  option confirmed to work well for this project's astro-interpretation
  path (correct grammar, good adherence to the sectioned-answer format,
  minimal hallucination) on this hardware tier so far:
  https://huggingface.co/TheRains/Qwen2.5-14B-Instruct-Q4_K_M-GGUF
  (direct file: https://huggingface.co/TheRains/Qwen2.5-14B-Instruct-Q4_K_M-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf)
- If 14B feels tight on RAM together with the full 32768-token context,
  lower `N_CTX` (e.g. to 8192-16384) rather than dropping back to a
  smaller model — a single astro answer's actual prompt rarely needs the
  full window.

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

The unit file uses `EnvironmentFile=/var/www/ycplt/.env` and intentionally
has no `User=` line (this deployment runs as root). Adjust
`WorkingDirectory` and `ExecStart` if the app lives somewhere other than
`/var/www/ycplt`.

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

## API

| Method/path | Description |
|---|---|
| `POST /chat` | Send a message. Body: `{query, conversation_id?, use_rag?, max_tokens?, temperature?, image_data?, image_filename?, image_mime_type?, strength?}`. Without `conversation_id`, creates a new conversation. `max_tokens` defaults to `null` — no artificial cap; the model generates until it stops on its own or fills the context window (`N_CTX`). `image_data` is a base64-encoded image (no `data:...;base64,` prefix); if present, `utils/intent.py` classifies the accompanying text as an edit instruction (submits an img2img job — `strength`, 0..1, default 0.75, controls how much the result may diverge from the input) or a question about the image (submits a caption job) — see "Editing"/"Understanding an uploaded image" below. Any of image-generation, image-edit, or image-caption requests return a `pending` placeholder immediately instead of chat text; otherwise returns the model's reply with `sent_at`/`responded_at` (ms), `thinking_ms`, and a `files` list. |
| `GET /health` | Model/RAG-index status and the configured `image_service_url`. For vision/generation model diagnostics, see ycplt_img's own `GET /health` — this app doesn't hold any of those models. |
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
5. From there, an edit job follows the identical pending →
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
  reliability at transcribing or reasoning about it.

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
  comparison above. `pip install kerykeion timezonefinder geonamescache`
  (kerykeion is AGPL-3.0 — see `utils/astro.py`'s docstring if you plan to
  redistribute this project) if you want this tool available; without
  timezonefinder/geonamescache specifically, their part is simply skipped
  gracefully — kerykeion alone still gets you the explicit-coordinates
  path, just not automatic timezone lookup or city-name resolution
  (best-effort imports inside the tool functions, same graceful-absence
  pattern as everywhere else optional in this project).

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
  | `astro_horar` | *(none yet)* | reserved for horary astrology, not implemented in this app yet — create the folder now, fill it in whenever that technique gets built |

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

1. Install indexing dependencies (`pip install -r install/requirements.txt` covers all of
   the Python packages below in one go):

   ```bash
   pip install sentence-transformers faiss-cpu numpy
   pip install pypdf                 # only needed for *.pdf sources
   pip install charset-normalizer    # auto-detects .txt/.html encoding (cp1251, koi8-r, ...)
   pip install beautifulsoup4        # only needed for *.html/*.htm sources
   pip install striprtf              # only needed for *.rtf sources (pure Python, no system tool)
   pip install patool                # only needed for *.rar/*.arj/*.7z source archives
   ```

   `.zip` archives are read directly (stdlib, no extra package). `.rar`/`.arj`/`.7z`,
   legacy binary `.doc` (MS Word 97-2003, NOT `.docx`), `.ha` and `.chm` archives, `.djvu`/`.djv`
   scans, and scanned (no-text-layer) `.pdf` pages all need matching **system** tools — see the
   "Installation" section above for the package names and the `ha` build steps (Debian/Ubuntu
   and Fedora both covered there).

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
   current 6-subfolder scheme and which `install/methodologies/*_methodology.txt` master copy
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
- More sidebar features (renaming a chat, searching history, etc.).
- Pagination of message history for very long conversations.
- Masked inpainting from the browser UI (mask drawing) — the API/ycplt_img
  side already supports a mask (`mode="inpaint"`, see ycplt_img's own
  README), but nothing in the browser UI produces one yet; today an
  attached-image edit is always a whole-image img2img instruction.
