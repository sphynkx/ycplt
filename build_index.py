"""
Builds the FAISS index(es) for RAG — one per corpus.

What "corpus" means here: rag_data/ (RAG_DATA_DIR) is organized into
topic subfolders (rag_data/<topic>/...), plus optionally some loose files
directly in rag_data/ itself (topic=None, informally called "root" below).
Each of those — every topic subfolder, and the root if it has any loose
files — is its own independent corpus, with its own index+meta pair under
INDEX_DIR (utils/config.py; default data/rag_index/<topic>/). Rebuilding
one corpus never reads, re-OCRs, or re-embeds any other corpus's
documents. This replaced an earlier single-combined-index design once a
real corpus (several topic subfolders, some containing OCR-heavy
.djvu/.pdf scans) made a single from-scratch rebuild take hours with no
visibility into progress and no way to redo just the one corpus that
actually changed.

What it does, per corpus:
  1. Reads every source document belonging to that corpus — *.txt,
     *.html/*.htm, *.pdf (OCR'd automatically for pages with no text
     layer), *.doc, *.djvu/*.djv, and *.zip/*.rar/*.arj/*.7z/*.chm archives
     (whose contents are extracted and read the same way, recursively,
     including archives nested inside archives) — reading multiple
     documents (and, within one document, multiple OCR pages) CONCURRENTLY,
     bounded by INDEX_CONCURRENCY (utils/config.py). This is what actually
     matters for wall-clock time: reading is dominated by external
     processes (antiword, djvutxt, ddjvu, pdftoppm, tesseract,
     extract_chmLib, patool), not by Python itself, and those processes
     were previously run one at a time, strictly sequentially, even though
     nothing stops several of them running at once on however many cores
     the machine has.
  2. Splits text into chunks (by paragraph, with a max-length cap per chunk).
  3. Computes embeddings (sentence-transformers) and builds a FAISS index
     covering that corpus's documents only.
  4. Saves the index and metadata under INDEX_DIR/<topic>/ (or INDEX_DIR/
     _root/ for the no-topic case) — utils/rag.py loads every corpus found
     there and merges their retrieval results at query time, so this is
     transparent to anything reading RAG results (see utils/rag.py's module
     docstring for how the merge works and why it doesn't change retrieval
     behavior compared to one single combined index).

Command-line usage:
    python build_index.py                  # build every corpus found under rag_data/
    python build_index.py <topic>           # build only rag_data/<topic>/
    python build_index.py <path/to/folder>  # build only that folder as its own corpus
    python build_index.py root              # build only the loose files directly in rag_data/
                                             # ("." or "_root" also work)

Building "every corpus" still means each one gets its own index+meta pair
and its own progress output — it's a loop over build_one_corpus() below,
not a return to one combined index. Re-run this for a single topic any
time only that topic's source documents changed, instead of paying to
re-read every other topic's documents too.

Organizing documents by topic:
  Put documents for different subjects in their own subfolder under
  rag_data/ — e.g. rag_data/astrology/planets.txt, rag_data/cooking/pasta.txt.
  Every chunk is tagged with its topic (the subfolder's name) in its
  corpus's index metadata; retrieval still searches across every loaded
  corpus in one pass regardless of topic (there's no topic-selection
  mechanism in the app — see utils/rag.py). Files directly in rag_data/
  itself (no subfolder) get topic=None and form their own "root" corpus.

Methodology documents (always included, not just similarity-matched):
  Ordinary chunk-similarity search is good at finding isolated facts, but
  bad at surfacing a document that describes HOW to combine facts into a
  conclusion (its wording rarely resembles a specific question). Name such
  a document with a "_methodology" suffix before the extension — e.g.
  rag_data/astrology/interpretation_methodology.txt — and utils/rag.py
  will always include its chunks in the prompt whenever any other chunk
  from the same topic was retrieved, regardless of that document's own
  similarity ranking. See utils/rag.py's module docstring for how this
  feeds into a reasoning-oriented prompt instead of a plain lookup one.
  build_index() below warns at build time if a topic's combined
  always-include content exceeds RAG_ALWAYS_INCLUDE_MAX_CHARS. This is not
  a cost/quota limit (there's no billing here, it's a local model) — it's
  a safety margin under N_CTX, the model's context window: a hard
  technical ceiling on how many tokens llama.cpp can process in one call
  at all, imposed by the model's own architecture (Qwen2.5 was trained on
  sequences up to 32768 tokens; feeding it more doesn't just cost more, it
  either errors out outright or produces degraded output past what it was
  trained to attend over). RAG_ALWAYS_INCLUDE_MAX_CHARS exists so a large
  methodology corpus, the real computed chart data, the user's question,
  and the model's own answer don't collectively exceed that ceiling in a
  single request — raise it (and N_CTX, and the RAM for the KV cache that
  scales with N_CTX) as high as your hardware and the model's trained
  context length support, but it can't be made truly unbounded independent
  of both.

Source documents and the indexing output live in separate folders:
  rag_data/   — put your own source files here, in topic subfolders
  data/rag_index/<topic>/  — faiss_index.bin and meta.pkl per corpus (generated by this script)

Supported source formats and their dependencies:
  .txt   — any encoding (UTF-8, cp1251, koi8-r, UTF-16, ...); auto-detected
           via charset-normalizer if installed, else a fixed fallback list.
  .html/.htm — same encoding handling as .txt, then tags/scripts/styles are
           stripped (requires beautifulsoup4).
  .rtf   — text-based format, no system tool needed (requires striprtf).
  .pdf   — requires pypdf for the text layer. Checked PAGE BY PAGE: any
           page pypdf extracts little/nothing from (a scanned page with no
           real text layer, common in old scanned books mixed with a
           text-layer title page) is rendered to an image via `pdftoppm`
           (poppler-utils) and OCR'd via `tesseract` instead — the same
           OCR pipeline as the .djvu fallback below, just page-by-page
           rather than whole-document, and with every thin page's OCR
           dispatched concurrently rather than one page at a time. Only
           the actually-thin pages are OCR'd, so a mostly-text PDF with
           one scanned insert page doesn't pay the OCR cost for pages that
           didn't need it.
  .doc   — legacy binary MS Word (97-2003, NOT .docx — python-docx doesn't
           read this format). Prefers the antiword CLI tool if installed
           for a clean extraction; falls back to a crude best-effort byte
           scrape otherwise, which is rougher but still searchable rather
           than skipping the file outright.
  .djvu/.djv — scanned documents (common for old/rare books and journal
           scans). Two very different cases, handled automatically:
           (a) the file already has an embedded OCR text layer (common for
           scans someone else already OCR'd before distributing) — read
           directly and fast via the `djvutxt` CLI tool; (b) no text layer
           at all (a "raw" photographic scan) — each page is rendered to an
           image via `ddjvu` and OCR'd on the fly via `tesseract`, with
           every page's ddjvu+tesseract pair dispatched concurrently
           (bounded by INDEX_CONCURRENCY) rather than one page at a time.
           Requires djvulibre (`djvutxt`, `ddjvu`, `djvused`) for either
           case, plus `tesseract` with a Russian language pack
           (`tesseract-ocr-rus` / `tesseract-langpack-rus`) for case (b) —
           see README.md for exact package names per distro. Missing tools
           are treated like everywhere else here: the file is skipped with
           an explanatory warning, the rest of the build continues.
  .zip   — read directly (Python's stdlib zipfile, no extra dependency).
  .rar/.arj/.7z — requires the patool package plus a matching system tool
           it shells out to. See README.md for exact package names on
           Debian/Ubuntu and Fedora — they differ, and some of this has
           been in flux (e.g. Fedora is transitioning away from p7zip
           toward a new official "7zip" package as of Fedora 43/44).
  .ha    — the old "HA" DOS-era archiver (Harri Hirvola, ~1995): genuinely
           dead and not packaged by any current Linux distro, and not even
           recognized by patool — but a modern buildable reimplementation
           exists (github.com/val-khokhlov/ha) with real build steps in
           README.md's "Installation" section. Works once a "ha" binary
           built from that (or any other) source is on PATH; otherwise
           skipped with an explanatory warning.
  .chm   — Compiled HTML Help (WinHelp's successor) — really just a
           compressed container of HTML "chapter" pages plus a sidebar
           table-of-contents/index in a proprietary, non-HTML format
           (.hhc/.hhk — not readable text, silently skipped like any other
           unrecognized file found inside a container). Requires the
           `extract_chmLib` CLI tool (package `chmlib` on Fedora,
           `libchm-bin` on Debian/Ubuntu) to unpack it; once unpacked, its
           .html/.htm chapters are read exactly like any other HTML file
           above — no CHM-specific text extraction of its own.
  Archives are extracted to a temporary directory, walked recursively
  (nested archives are extracted too, up to a small depth limit to guard
  against runaway/zip-bomb-style nesting), and every recognized file found
  inside is read the same way as if it sat directly in rag_data/ (members
  are also read concurrently with each other) — its topic is inherited
  from the archive's own location, and its chunk ids are prefixed with the
  archive's path plus its path inside the archive, so two same-named files
  in different archives never collide.

Concurrency (INDEX_CONCURRENCY, utils/config.py, default 4):
  Reading a corpus is I/O- and external-process-bound, not CPU-bound in
  Python itself — most of the wall-clock time is spent waiting on
  antiword/djvutxt/ddjvu/pdftoppm/tesseract/extract_chmLib/patool
  subprocesses, one per document (or, for OCR, one pair per page). Those
  are independent of each other, so they're run concurrently via asyncio,
  each blocking subprocess call wrapped in a worker thread
  (asyncio.to_thread) and bounded by a shared asyncio.Semaphore sized to
  INDEX_CONCURRENCY — this caps how many external processes run AT ONCE
  regardless of whether the concurrency comes from many separate documents
  or many pages of one large scanned document, so raising it is the one
  knob that matters for indexing speed on a real corpus. The embedding
  step (sentence-transformers) is deliberately NOT parallelized this way —
  it's a single already-vectorized batch call over all of a corpus's
  chunks at once, which is the efficient way to use the CPU for that part;
  concurrency only helps the reading/OCR phase.

Install dependencies (if not already installed):
    pip install sentence-transformers faiss-cpu numpy
    pip install -r install/requirements.txt   # includes the optional readers above

After building, RAG becomes available via the "use_rag": true flag in a
/chat request — restart the server (uvicorn) so it picks up the fresh
corpus/corpora on startup. Re-run this script for a single topic any time
just that topic's source documents change, or for everything after
changing EMBED_MODEL (embeddings from a different model aren't compatible
with an existing index even if the vector dimension happens to match).
"""
import asyncio
import glob
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from utils.config import (
    EMBED_MODEL,
    INDEX_CONCURRENCY,
    INDEX_DIR,
    INDEX_PATH,
    META_PATH,
    RAG_ALWAYS_INCLUDE_MAX_CHARS,
    RAG_DATA_DIR,
)

# Rough character-length cap per chunk — to avoid exceeding the embedding
# model's context window.
MAX_CHUNK_CHARS = 800

_METHODOLOGY_SUFFIX = "_methodology"

_TEXT_LIKE_EXTENSIONS = (".txt", ".html", ".htm", ".pdf", ".doc", ".rtf", ".djvu", ".djv")

# A djvu with no embedded OCR text layer still makes `djvutxt` exit 0 with
# empty or near-empty output (just page-break control chars) — not an
# error, just nothing usable to extract this way. Below this many
# characters for the WHOLE document, treat it as "no text layer" and fall
# back to page-by-page OCR rather than silently indexing an almost-empty
# document.
_DJVU_MIN_TEXT_LAYER_CHARS = 40

# Language pack(s) passed to tesseract for every OCR fallback in this file
# (djvu pages with no text layer, scanned PDF pages) — Russian plus English
# covers this project's own corpus; add more codes (e.g. "+ukr") if your
# own rag_data/ documents need them, as long as the matching tesseract
# language pack is installed (see README.md).
_OCR_LANGUAGES = "rus+eng"

# Below this many extracted characters for a single PDF page, treat pypdf's
# text-layer extraction as having found effectively nothing on that page
# (a scanned page with no real text layer) and OCR it instead. Checked
# per-page rather than per-document, since it's common for a scanned PDF to
# mix a handful of real text-layer pages (e.g. a title page exported from a
# word processor) with the rest as raw photographic scans.
_PDF_MIN_PAGE_TEXT_CHARS = 20

# .ha is the old "HA" DOS-era archiver (Harri Hirvola, ~1995) — genuinely
# dead, not packaged by any current Linux distro and not recognized by
# patool at all (unlike .rar/.arj/.7z, where patool at least knows the
# format and just needs a system tool). It's handled as a special case in
# _extract_archive_async() below: if a "ha" binary happens to be on PATH —
# e.g. built from github.com/val-khokhlov/ha per README.md's "Installation"
# section — it's used; otherwise the file is skipped with an explanatory
# warning. .chm (Compiled HTML Help) is handled as a special case there
# too — it's not a general-purpose archive format patool understands, but
# `extract_chmLib` unpacks it into a directory of HTML chapters just the
# same way _read_archive_members_async() below expects.
_ARCHIVE_EXTENSIONS = (".zip", ".rar", ".arj", ".7z", ".ha", ".chm")
_ALL_EXTENSIONS = _TEXT_LIKE_EXTENSIONS + _ARCHIVE_EXTENSIONS

# Guards against archive-in-archive-in-archive recursion (accidental or a
# deliberate zip bomb) — three levels comfortably covers realistic cases
# like "a .zip of .zips of scanned document exports" without ever walking
# an unbounded/hostile nesting chain.
_MAX_ARCHIVE_DEPTH = 3


async def _run_subprocess(sem: asyncio.Semaphore, args: List[str], **kwargs) -> subprocess.CompletedProcess:
    """Runs a blocking subprocess.run() call in a worker thread
    (asyncio.to_thread), bounded by sem so at most INDEX_CONCURRENCY
    external processes (tesseract, antiword, djvutxt, ddjvu, pdftoppm,
    extract_chmLib, ha, ...) run at once regardless of how many documents
    or OCR pages are being processed concurrently above this call — this
    single choke point is what actually turns "many slow external tools,
    one after another" into real overlapping work on however many cores
    the machine has, without needing to convert every external tool's own
    invocation into a separate concurrency-tracking mechanism."""
    async with sem:
        return await asyncio.to_thread(subprocess.run, args, **kwargs)


def _split_long(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Splits a long paragraph into smaller pieces along sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    parts, current = [], ""
    for sentence in text.replace("\n", " ").split(". "):
        if current and len(current) + len(sentence) + 2 > max_chars:
            parts.append(current.strip())
            current = sentence
        else:
            current = f"{current}. {sentence}" if current else sentence
    if current.strip():
        parts.append(current.strip())
    return parts


# Text files show up in a mix of encodings depending on how/where they
# were created — plain UTF-8 exports, Windows Notepad's cp1251 for
# Cyrillic, older koi8-r/cp866 Cyrillic text, an occasional UTF-16
# clipboard paste, and so on. _read_txt used to assume UTF-8
# unconditionally, so any non-UTF-8 file raised UnicodeDecodeError
# ("... invalid continuation byte...") — which load_documents() caught and
# treated as "skip this file entirely", silently, with just a one-line
# print. A rag_data/ folder with several non-UTF-8 files could end up
# mostly unindexed with no obvious sign short of scrolling console output,
# which can plausibly explain thin or irrelevant retrieval results
# independent of any prompt- or methodology-side issue.
#
# Detection order: plain UTF-8 first (the common case, unambiguous when it
# succeeds); charset-normalizer's statistical detection next, if installed
# (pip install charset-normalizer — see requirements.txt); a fixed list of
# encodings common for Russian-language text as a fallback if that library
# isn't available; and, only if every real encoding fails, UTF-8 with
# lossy replacement characters as an absolute last resort, so indexing
# never silently drops a file outright — the result is at least
# searchable text, with a printed warning so a genuinely broken file can
# still be noticed and fixed at the source.
_FALLBACK_ENCODINGS = ("utf-8-sig", "cp1251", "cp866", "koi8-r", "mac_cyrillic", "utf-16", "cp1252")


def _read_txt(path: str) -> str:
    """Pure Python, no subprocess — stays synchronous even in the
    otherwise-async reading pipeline below (there's nothing to overlap
    with anything else here, and it's fast)."""
    raw = open(path, "rb").read()

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass

    try:
        from charset_normalizer import from_bytes
        best = from_bytes(raw).best()
        if best is not None:
            return str(best)
    except ImportError:
        pass

    for encoding in _FALLBACK_ENCODINGS:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue

    print(
        f"Warning: {path} — could not detect a working encoding, decoding "
        "as UTF-8 with lossy replacement characters (some text may be garbled)"
    )
    return raw.decode("utf-8", errors="replace")


def _read_html(path: str) -> str:
    """Strips tags/scripts/styles, keeping just the visible text. Reuses
    _read_txt for the actual byte decoding — an .html file is just as
    likely to be non-UTF-8 as a plain .txt one (older exported pages,
    Windows-authored HTML whose declared <meta charset> doesn't actually
    match its bytes, etc.), so the same multi-encoding detection applies.
    Pure Python, no subprocess — stays synchronous."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError("Reading HTML requires beautifulsoup4: pip install beautifulsoup4")
    raw_text = _read_txt(path)
    soup = BeautifulSoup(raw_text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n\n")


def _read_rtf(path: str) -> str:
    """RTF is a text-based format (control words like \\rtf1, \\par, ...
    wrapping plain text), so — unlike .doc — this needs no external system
    tool: striprtf is a small pure-Python RTF-to-plaintext converter.
    Pure Python, no subprocess — stays synchronous."""
    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError:
        raise RuntimeError("Reading RTF requires the striprtf package: pip install striprtf")
    raw_text = _read_txt(path)  # RTF's own encoding declarations are unreliable; sniff the bytes instead
    return rtf_to_text(raw_text)


async def _read_doc_async(path: str, sem: asyncio.Semaphore) -> str:
    """Legacy binary MS Word format (97-2003, NOT .docx — python-docx and
    pypdf don't read this at all). Prefers the antiword CLI tool, which
    correctly parses the underlying OLE2/binary structure; falls back to a
    crude best-effort scrape (decode as UTF-16LE, keep only runs that look
    like real text) if antiword isn't installed or can't read this
    particular file, rather than skipping it outright — rougher output,
    but still searchable.

    Antiword being installed doesn't guarantee it can read every file with
    a .doc extension: antiword only understands Word 2.0-2003's binary
    format specifically, so a file that's actually something else wearing
    a .doc extension (an older Word format, WordPerfect, a plain-text
    export, a corrupted file, ...) makes antiword exit with a nonzero
    status and an explanatory message on stderr — which is surfaced below
    instead of being discarded, since "why did this specific file fail"
    is only answerable if that message is actually shown."""
    result = None
    if shutil.which("antiword") is None:
        print(
            f"Warning: {path} — antiword is not installed (or not on PATH); "
            "using a crude fallback extraction for this legacy .doc file. "
            "Install antiword (e.g. `apt install antiword` / `dnf install "
            "antiword`) for reliable results."
        )
    else:
        try:
            result = await _run_subprocess(sem, ["antiword", path], capture_output=True, timeout=30)
        except Exception as e:
            print(f"Warning: antiword failed to run on {path}: {e}; using a crude fallback extraction instead.")

    if result is not None:
        if result.returncode == 0 and result.stdout:
            return result.stdout.decode("utf-8", errors="replace")
        stderr_text = (result.stderr or b"").decode("utf-8", errors="replace").strip()
        print(
            f"Warning: antiword ran but produced no usable text for {path} "
            f"(exit code {result.returncode}"
            + (f": {stderr_text}" if stderr_text else ", no error message on stderr")
            + ") — this usually means the file isn't actually a Word "
            "2.0-2003 binary .doc despite its extension (a different old "
            "format, a corrupted file, or an empty one). Falling back to a "
            "crude text scrape."
        )

    raw = open(path, "rb").read()
    text = raw.decode("utf-16-le", errors="ignore")
    # Legacy .doc interleaves real text with binary control structures, so
    # a naive full decode is mostly noise — keep only runs of at least a
    # few consecutive printable-looking characters.
    runs = re.findall(r"[^\x00-\x08\x0b\x0c\x0e-\x1f]{4,}", text)
    return "\n\n".join(runs)


async def _ocr_pdf_pages_async(path: str, page_indices: List[int], sem: asyncio.Semaphore) -> Dict[int, str]:
    """OCR fallback for PDF pages pypdf couldn't extract text from: renders
    each such page to an image via `pdftoppm` (poppler-utils), then runs
    `tesseract` on it — the same two-tool pipeline as the .djvu OCR
    fallback below, just with a different renderer for the page image.
    page_indices are 0-based (matching enumerate(reader.pages) in
    _read_pdf_async); pdftoppm's own -f/-l page numbering is 1-based.
    Every page is OCR'd concurrently (bounded by sem via _run_subprocess),
    not one at a time."""
    if shutil.which("pdftoppm") is None:
        print(
            f"Warning: skipping OCR for {path} — pdftoppm is not installed (or not on "
            "PATH); reading scanned PDF pages requires poppler-utils — see README.md's "
            "\"Installation\" section."
        )
        return {}
    if shutil.which("tesseract") is None:
        print(
            f"Warning: skipping OCR for {path} — tesseract is not installed (or not on "
            "PATH); install tesseract-ocr plus its Russian language pack — see "
            "README.md's \"Installation\" section."
        )
        return {}

    async def _ocr_one_page(tmp: str, i: int) -> Tuple[int, str]:
        page_num = i + 1
        prefix = os.path.join(tmp, f"page_{page_num}")
        try:
            await _run_subprocess(
                sem,
                ["pdftoppm", "-f", str(page_num), "-l", str(page_num), "-r", "300", "-gray", "-png", path, prefix],
                capture_output=True, timeout=120, check=True,
            )
        except Exception as e:
            print(f"Warning: pdftoppm failed on {path} page {page_num}: {e}")
            return i, ""

        rendered = sorted(glob.glob(f"{prefix}*.png"))
        if not rendered:
            print(f"Warning: pdftoppm produced no image for {path} page {page_num}")
            return i, ""

        try:
            ocr_result = await _run_subprocess(
                sem, ["tesseract", rendered[0], "stdout", "-l", _OCR_LANGUAGES], capture_output=True, timeout=120,
            )
        except Exception as e:
            print(f"Warning: tesseract failed on {path} page {page_num}: {e}")
            return i, ""

        if ocr_result.returncode == 0:
            return i, ocr_result.stdout.decode("utf-8", errors="replace")
        stderr_text = (ocr_result.stderr or b"").decode("utf-8", errors="replace").strip()
        print(f"Warning: tesseract produced no text for {path} page {page_num}" + (f": {stderr_text}" if stderr_text else ""))
        return i, ""

    with tempfile.TemporaryDirectory(prefix="ycplt_pdf_ocr_") as tmp:
        pairs = await asyncio.gather(*(_ocr_one_page(tmp, i) for i in page_indices))
    return {i: text for i, text in pairs if text}


async def _read_pdf_async(path: str, sem: asyncio.Semaphore) -> str:
    """Extracts each page's text layer via pypdf; any page that comes back
    with (near-)nothing — a scanned page with no real text layer, common in
    old scanned books that mix a handful of text-layer pages with the rest
    as raw photographic scans — is OCR'd instead via _ocr_pdf_pages_async(),
    the same tesseract-based pipeline used for .djvu scans, just rendered
    with pdftoppm instead of ddjvu. A fully "born-digital" PDF never
    touches the OCR path at all: this only kicks in for pages that
    actually need it. pypdf's own extraction is pure Python (no
    subprocess), so only the OCR fallback needs the shared semaphore."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("Reading PDF requires the pypdf package: pip install pypdf")
    reader = PdfReader(path)
    texts: List[str] = []
    thin_pages: List[int] = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if len(page_text.strip()) < _PDF_MIN_PAGE_TEXT_CHARS:
            thin_pages.append(i)
        texts.append(page_text)

    if thin_pages:
        print(
            f"{path} — {len(thin_pages)} of {len(texts)} page(s) have no usable text layer, "
            "OCR'ing them instead (slower)"
        )
        for i, ocr_text in (await _ocr_pdf_pages_async(path, thin_pages, sem)).items():
            texts[i] = ocr_text

    return "\n\n".join(texts)


async def _ocr_djvu_async(path: str, sem: asyncio.Semaphore) -> str:
    """OCR fallback for a .djvu with no embedded text layer: renders each
    page to a TIFF via `ddjvu`, then runs `tesseract` on it. Both are
    external system tools (djvulibre and tesseract-ocr respectively, plus a
    tesseract language pack — see README.md); a missing tool is a skip with
    an explanatory warning, same pattern as the rest of this file, rather
    than a crash. Every page is OCR'd concurrently (bounded by sem), not
    one at a time."""
    if shutil.which("djvused") is None:
        print(f"Warning: skipping {path} — djvused (part of djvulibre) is not installed (or not on PATH)")
        return ""
    try:
        page_count_result = await _run_subprocess(sem, ["djvused", "-e", "n", path], capture_output=True, timeout=30)
        page_count = int(page_count_result.stdout.decode("utf-8", errors="replace").strip())
    except Exception as e:
        print(f"Warning: could not determine page count for {path}: {e}")
        return ""

    if shutil.which("ddjvu") is None:
        print(f"Warning: skipping {path} — ddjvu (part of djvulibre) is not installed (or not on PATH)")
        return ""
    if shutil.which("tesseract") is None:
        print(
            f"Warning: skipping OCR for {path} — tesseract is not installed (or not on "
            "PATH); install tesseract-ocr plus its Russian language pack — see "
            "README.md's \"Installation\" section."
        )
        return ""

    async def _ocr_one_page(tmp: str, page: int) -> str:
        image_path = os.path.join(tmp, f"page_{page}.tiff")
        try:
            await _run_subprocess(
                sem, ["ddjvu", "-format=tiff", f"-page={page}", path, image_path],
                capture_output=True, timeout=120, check=True,
            )
        except Exception as e:
            print(f"Warning: ddjvu failed on {path} page {page}: {e}")
            return ""

        try:
            ocr_result = await _run_subprocess(
                sem, ["tesseract", image_path, "stdout", "-l", _OCR_LANGUAGES], capture_output=True, timeout=120,
            )
        except Exception as e:
            print(f"Warning: tesseract failed on {path} page {page}: {e}")
            return ""

        if ocr_result.returncode == 0:
            return ocr_result.stdout.decode("utf-8", errors="replace")
        stderr_text = (ocr_result.stderr or b"").decode("utf-8", errors="replace").strip()
        print(f"Warning: tesseract produced no text for {path} page {page}" + (f": {stderr_text}" if stderr_text else ""))
        return ""

    with tempfile.TemporaryDirectory(prefix="ycplt_djvu_ocr_") as tmp:
        pages_text = await asyncio.gather(*(_ocr_one_page(tmp, p) for p in range(1, page_count + 1)))

    return "\n\n".join(t for t in pages_text if t)


async def _read_djvu_async(path: str, sem: asyncio.Semaphore) -> str:
    """Reads a .djvu/.djv scanned document. Tries the fast path first — an
    embedded OCR text layer, extracted directly via the `djvutxt` CLI tool
    (djvulibre) — and only falls back to the slow path (rendering each page
    to an image via `ddjvu` and OCR'ing it with `tesseract`, concurrently)
    if that comes back empty, i.e. this particular scan has no text layer
    at all. Missing `djvutxt` itself is a hard skip (no cruder fallback
    makes sense for a binary image container the way it does for legacy
    .doc) rather than aborting the whole build."""
    if shutil.which("djvutxt") is None:
        print(
            f"Warning: skipping {path} — djvutxt is not installed (or not on PATH). "
            "Reading .djvu/.djv files requires djvulibre (djvutxt/ddjvu/djvused) — "
            "see README.md's \"Installation\" section for the package name on your distro."
        )
        return ""
    try:
        result = await _run_subprocess(sem, ["djvutxt", path], capture_output=True, timeout=60)
    except Exception as e:
        print(f"Warning: djvutxt failed to run on {path}: {e}")
        return ""

    text = result.stdout.decode("utf-8", errors="replace") if result.returncode == 0 else ""
    if len(text.strip()) >= _DJVU_MIN_TEXT_LAYER_CHARS:
        return text

    print(f"{path} — no usable embedded text layer, OCR'ing each page instead (slower)")
    return await _ocr_djvu_async(path, sem)


async def _read_any_async(path: str, sem: asyncio.Semaphore) -> str:
    """Dispatches to the right reader based on extension — the single
    entry point used both for files found directly under rag_data/ and
    for files pulled out of an archive. Formats with no subprocess
    involved (.txt/.html/.rtf) call their plain synchronous reader
    directly; the rest await their async, subprocess-backed counterpart."""
    lower = path.lower()
    if lower.endswith(".pdf"):
        return await _read_pdf_async(path, sem)
    if lower.endswith((".html", ".htm")):
        return _read_html(path)
    if lower.endswith(".doc"):
        return await _read_doc_async(path, sem)
    if lower.endswith(".rtf"):
        return _read_rtf(path)
    if lower.endswith((".djvu", ".djv")):
        return await _read_djvu_async(path, sem)
    return _read_txt(path)


async def _extract_archive_async(path: str, dest_dir: str, sem: asyncio.Semaphore) -> bool:
    """Extracts path into dest_dir. Returns False (with a printed warning)
    if the archive can't be read — a missing system tool for .rar/.arj/.7z
    is treated the same as a bad encoding elsewhere in this file: skip
    that one source with a clear message, don't abort the whole build."""
    lower = path.lower()
    if lower.endswith(".zip"):
        # Pure Python (stdlib zipfile), no external process — no need for
        # the semaphore/thread pool here.
        try:
            with zipfile.ZipFile(path) as zf:
                zf.extractall(dest_dir)
            return True
        except Exception as e:
            print(f"Warning: failed to extract {path}: {e}")
            return False

    if lower.endswith(".ha"):
        # patool doesn't recognize this format at all (confirmed: it
        # raises "unknown archive format" before even looking for a
        # program), and no current Linux distro packages a "ha" binary —
        # the format has been dead since the mid-1990s. github.com/
        # val-khokhlov/ha is a modern buildable reimplementation (build
        # steps in README.md's "Installation" section); this only works
        # once that (or any other) "ha" binary is on PATH, otherwise it's
        # skipped with an explanation rather than silently failing or
        # crashing the whole build.
        if shutil.which("ha") is None:
            print(
                f"Warning: skipping {path} — .ha (the old HA/Hirvola archiver) needs a "
                "`ha` binary on PATH; none was found. See README.md's \"Installation\" "
                "section for build steps (github.com/val-khokhlov/ha) — or extract this "
                "file manually with whatever old tool you have and add the extracted "
                "contents to rag_data/ directly instead."
            )
            return False
        try:
            await _run_subprocess(sem, ["ha", "x", os.path.abspath(path)], cwd=dest_dir, capture_output=True, timeout=60, check=True)
            return True
        except Exception as e:
            print(f"Warning: `ha` failed to extract {path}: {e}")
            return False

    if lower.endswith(".chm"):
        # CHM (Compiled HTML Help) isn't a general-purpose archive format
        # patool understands — it's unpacked with chmlib's own
        # `extract_chmLib` tool instead, into a plain directory of HTML
        # "chapter" pages (plus a proprietary, non-HTML sidebar
        # table-of-contents/index that _read_archive_members_async() below
        # will just skip, same as any other unrecognized file type inside
        # a container). Package name differs by distro — chmlib on
        # Fedora, libchm-bin on Debian/Ubuntu — see README.md's
        # "Installation" section.
        if shutil.which("extract_chmLib") is None:
            print(
                f"Warning: skipping {path} — extract_chmLib is not installed (or not on "
                "PATH); reading .chm files requires chmlib's extract_chmLib tool — see "
                "README.md's \"Installation\" section for the package name on your distro."
            )
            return False
        try:
            await _run_subprocess(sem, ["extract_chmLib", os.path.abspath(path), dest_dir], capture_output=True, timeout=60, check=True)
            return True
        except Exception as e:
            print(f"Warning: extract_chmLib failed to extract {path}: {e}")
            return False

    try:
        import patoolib
    except ImportError:
        print(
            f"Warning: skipping {path} — reading .rar/.arj/.7z requires the "
            "patool package (pip install patool) plus a matching system "
            "tool — see install/requirements.txt and README.md for exact "
            "package names per distro"
        )
        return False
    try:
        async with sem:
            await asyncio.to_thread(patoolib.extract_archive, path, outdir=dest_dir, interactive=False, verbosity=-1)
        return True
    except Exception as e:
        print(f"Warning: failed to extract {path} (missing system tool for this format?): {e}")
        return False


async def _read_archive_members_async(path: str, sem: asyncio.Semaphore, depth: int = 0) -> List[Tuple[str, str]]:
    """Recursively extracts an archive to a temp dir and reads every
    recognized file inside it CONCURRENTLY (descending into further
    archives too, up to _MAX_ARCHIVE_DEPTH) — returns
    (virtual_relative_path, text) pairs. The temp dir is always cleaned up,
    success or failure."""
    if depth >= _MAX_ARCHIVE_DEPTH:
        print(f"Warning: {path} — nested archive depth limit reached, not descending further")
        return []

    with tempfile.TemporaryDirectory(prefix="ycplt_rag_extract_") as tmp:
        if not await _extract_archive_async(path, tmp, sem):
            return []

        # Collecting the file list is just a directory walk (cheap, no
        # subprocess) — the actual concurrency happens below, reading every
        # member at once instead of one at a time.
        member_paths: List[str] = []
        for root, _dirs, filenames in os.walk(tmp):
            for name in sorted(filenames):
                if name.lower().endswith(_ARCHIVE_EXTENSIONS) or name.lower().endswith(_TEXT_LIKE_EXTENSIONS):
                    member_paths.append(os.path.join(root, name))

        async def _read_member(fpath: str) -> List[Tuple[str, str]]:
            rel = os.path.relpath(fpath, tmp).replace(os.sep, "/")
            if fpath.lower().endswith(_ARCHIVE_EXTENSIONS):
                nested = await _read_archive_members_async(fpath, sem, depth=depth + 1)
                return [(f"{rel}/{nested_rel}", nested_text) for nested_rel, nested_text in nested]
            try:
                return [(rel, await _read_any_async(fpath, sem))]
            except Exception as e:
                print(f"Skipping {path}:{rel}: {e}")
                return []

        results = await asyncio.gather(*(_read_member(p) for p in member_paths))

    members: List[Tuple[str, str]] = []
    for r in results:
        members.extend(r)
    return members


def _find_source_files_for_topic(data_dir: str, topic: Optional[str]) -> List[str]:
    """Finds every supported source file belonging to ONE corpus: either
    the loose files directly in data_dir (topic=None — deliberately NOT
    descending into subfolders, since those are separate corpora handled
    by their own call to this function), or everything recursively under
    data_dir/<topic> for a named topic."""
    base = data_dir if topic is None else os.path.join(data_dir, topic)
    if not os.path.isdir(base):
        return []

    paths: List[str] = []
    if topic is None:
        for name in sorted(os.listdir(base)):
            fpath = os.path.join(base, name)
            if os.path.isfile(fpath) and name.lower().endswith(_ALL_EXTENSIONS):
                paths.append(fpath)
    else:
        for root, _dirs, filenames in os.walk(base):
            for name in sorted(filenames):
                if name.lower().endswith(_ALL_EXTENSIONS):
                    paths.append(os.path.join(root, name))
    return sorted(paths)


def _discover_topics(data_dir: str) -> List[Optional[str]]:
    """Every corpus found under data_dir: None for the loose files directly
    in data_dir (only if there are any), plus one entry per immediate
    subfolder that actually contains at least one supported file
    (recursively). This is the list build_index.py's default (no-argument)
    CLI run iterates over, building one index per corpus."""
    topics: List[Optional[str]] = []
    if _find_source_files_for_topic(data_dir, None):
        topics.append(None)
    if os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            if os.path.isdir(os.path.join(data_dir, name)) and _find_source_files_for_topic(data_dir, name):
                topics.append(name)
    return topics


def _is_methodology(path: str) -> bool:
    """A document named "<anything>_methodology.txt" (or .pdf) is always
    included in the prompt whenever its topic is relevant, instead of
    competing for top-k similarity ranking — see this file's module
    docstring and utils/rag.py."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.lower().endswith(_METHODOLOGY_SUFFIX)


def _append_chunks(docs: List[Dict], doc_key: str, text: str, topic: Optional[str], always_include: bool) -> None:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunk_i = 0
    for p in paragraphs:
        for chunk in _split_long(p):
            docs.append(
                {
                    "id": f"{doc_key}_{chunk_i}",
                    "text": chunk,
                    "topic": topic,
                    "always_include": always_include,
                }
            )
            chunk_i += 1


async def load_documents_async(data_dir: str, topic: Optional[str], sem: asyncio.Semaphore) -> List[Dict]:
    """Reads every source document belonging to ONE corpus (data_dir/<topic>,
    or the loose files directly in data_dir if topic is None) and returns
    its chunk dicts — documents are read CONCURRENTLY (bounded by sem),
    not one at a time."""
    paths = _find_source_files_for_topic(data_dir, topic)

    async def _load_one(path: str) -> List[Dict]:
        # Relative path (not just basename) keeps chunk ids unique even
        # when two same-named files exist at different depths in this topic.
        doc_key = os.path.relpath(path, data_dir).replace(os.sep, "/")
        local_docs: List[Dict] = []

        if path.lower().endswith(_ARCHIVE_EXTENSIONS):
            members = await _read_archive_members_async(path, sem)
            for virtual_rel, text in members:
                # Topic is inherited from the archive's own location, not
                # from anything inside it — an archive is just a container,
                # not its own topic. always_include still follows the
                # "_methodology" naming convention on the file *inside* the
                # archive, since that's the actual document.
                _append_chunks(local_docs, f"{doc_key}/{virtual_rel}", text, topic, _is_methodology(virtual_rel))
            return local_docs

        try:
            text = await _read_any_async(path, sem)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            return []

        _append_chunks(local_docs, doc_key, text, topic, _is_methodology(path))
        return local_docs

    results = await asyncio.gather(*(_load_one(p) for p in paths))
    docs: List[Dict] = []
    for r in results:
        docs.extend(r)
    return docs


def _corpus_dir_name(topic: Optional[str]) -> str:
    return "_root" if topic is None else topic


def _paths_for_topic(topic: Optional[str]) -> Tuple[str, str]:
    """Where one corpus's index+meta pair lives: INDEX_DIR/<topic>/ (or
    INDEX_DIR/_root/ for the no-topic case)."""
    corpus_dir = os.path.join(INDEX_DIR, _corpus_dir_name(topic))
    return os.path.join(corpus_dir, "faiss_index.bin"), os.path.join(corpus_dir, "meta.pkl")


def build_index(
    docs: List[Dict],
    embed_model_name: str = EMBED_MODEL,
    index_path: str = INDEX_PATH,
    meta_path: str = META_PATH,
) -> None:
    """Embeds and writes ONE corpus's index+meta pair. index_path/meta_path
    default to the legacy single-file locations for backward compatibility
    if called without them; build_one_corpus() below always passes the
    per-corpus paths from _paths_for_topic()."""
    model = SentenceTransformer(embed_model_name)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)  # inner product = cosine after normalization
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(docs, f)

    topics = sorted({d["topic"] for d in docs if d["topic"]})
    methodology_docs = sorted({d["id"].rsplit("_", 1)[0] for d in docs if d["always_include"]})

    print(f"  Index built: {len(docs)} chunks, dim={dim}, model={embed_model_name}")
    print(f"    topics: {', '.join(topics) if topics else '(none — root corpus)'}")
    if methodology_docs:
        print(f"    always-included (methodology) docs: {', '.join(methodology_docs)}")

    # Warn if a topic's combined always-include (methodology) text exceeds
    # the budget retrieve_context() enforces at query time (utils/rag.py).
    # Past that budget, retrieve_context() silently stops adding chunks —
    # this happened for real with the astrology methodology doc as it grew
    # (the model quietly never saw its worked example or symbol legend, no
    # matter how the wording was tuned), and was only caught by manually
    # simulating the chunking. This check exists so that doesn't have to
    # happen by hand again for this or any other topic.
    always_include_chars_by_topic: Dict[Optional[str], int] = defaultdict(int)
    for d in docs:
        if d["always_include"]:
            always_include_chars_by_topic[d["topic"]] += len(d["text"])
    for topic, total_chars in sorted(always_include_chars_by_topic.items(), key=lambda kv: str(kv[0])):
        if total_chars > RAG_ALWAYS_INCLUDE_MAX_CHARS:
            print(
                f"    WARNING: topic {topic!r} has {total_chars} chars of always-include "
                f"(methodology) content, over RAG_ALWAYS_INCLUDE_MAX_CHARS "
                f"({RAG_ALWAYS_INCLUDE_MAX_CHARS}). retrieve_context() will silently stop "
                "including it partway through — raise RAG_ALWAYS_INCLUDE_MAX_CHARS or "
                "trim/split this topic's methodology doc(s)."
            )

    print(f"    -> {index_path}")
    print(f"    -> {meta_path}")


def build_one_corpus(data_dir: str, topic: Optional[str], concurrency: int) -> int:
    """Builds (or rebuilds) the index+meta pair for exactly ONE corpus —
    the loose files directly in data_dir (topic=None) or one topic
    subfolder — writing it under its own INDEX_DIR/<topic>/ directory so
    re-running this for one corpus never touches any other corpus's index
    or re-reads/re-OCRs documents that haven't changed. Returns the number
    of chunks written (0 if the corpus was empty)."""
    label = topic if topic is not None else "(root)"
    print(f"--- Building corpus {label!r} ---")

    async def _run() -> List[Dict]:
        sem = asyncio.Semaphore(concurrency)
        return await load_documents_async(data_dir, topic, sem)

    docs = asyncio.run(_run())
    if not docs:
        print(f"  No documents found for corpus {label!r} — skipping.")
        return 0

    index_path, meta_path = _paths_for_topic(topic)
    build_index(docs, index_path=index_path, meta_path=meta_path)
    return len(docs)


def _resolve_corpus_arg(arg: str) -> Optional[Tuple[str, Optional[str]]]:
    """Resolves a build_index.py command-line argument to (data_dir, topic)
    for build_one_corpus(). Accepts: "."/"root"/"_root" for the loose files
    directly in RAG_DATA_DIR; a bare topic name matching an existing
    RAG_DATA_DIR subfolder; or a direct filesystem path to any folder (its
    basename becomes the topic, its parent becomes data_dir) — this last
    form also works for a corpus that lives outside RAG_DATA_DIR entirely.
    Returns None (with a printed message) if nothing matches."""
    if arg in (".", "root", "_root"):
        return RAG_DATA_DIR, None
    if os.path.isdir(os.path.join(RAG_DATA_DIR, arg)):
        return RAG_DATA_DIR, arg
    if os.path.isdir(arg):
        normalized = os.path.normpath(arg)
        return (os.path.dirname(normalized) or "."), os.path.basename(normalized)
    print(f"'{arg}' is not a topic under {RAG_DATA_DIR}/ and not an existing folder — nothing to build.")
    return None


if __name__ == "__main__":
    if not os.path.isdir(RAG_DATA_DIR):
        os.makedirs(RAG_DATA_DIR, exist_ok=True)
        print(f"Created {RAG_DATA_DIR}/ — put source files there (optionally in topic "
              f"subfolders, see this file's docstring for supported formats) and run this "
              f"script again.")
    elif len(sys.argv) > 1:
        resolved = _resolve_corpus_arg(sys.argv[1])
        if resolved is not None:
            resolved_data_dir, resolved_topic = resolved
            build_one_corpus(resolved_data_dir, resolved_topic, INDEX_CONCURRENCY)
    else:
        topics = _discover_topics(RAG_DATA_DIR)
        if not topics:
            print(f"No documents found in {RAG_DATA_DIR}/ (recursively — see this file's "
                  f"docstring for supported formats)")
        else:
            print(f"Found {len(topics)} corpus/corpora: {', '.join(t or '(root)' for t in topics)}\n")
            total_chunks = 0
            for t in topics:
                total_chunks += build_one_corpus(RAG_DATA_DIR, t, INDEX_CONCURRENCY)
            print(f"\nAll corpora built: {total_chunks} chunk(s) total across {len(topics)} corpus/corpora.")
