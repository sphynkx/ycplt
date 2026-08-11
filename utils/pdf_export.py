"""Exports a single chat message — its text plus any image file
attachments (astrology wheel charts, etc.) — as one standalone PDF.

Uses weasyprint (HTML+CSS -> PDF) rather than a low-level drawing library
like reportlab, for two concrete reasons specific to this app's content:

1. Cyrillic. reportlab's built-in base14 fonts (Helvetica, Times, ...) have
   NO Cyrillic glyphs at all — every Russian character would come out as
   tofu boxes unless a TTF font is manually registered via pdfmetrics.
   weasyprint instead renders through the system's installed fonts (same
   as a browser would), so specifying font-family: "DejaVu Sans" in the
   stylesheet below "just works" as long as that font is installed
   (it's a near-universal Linux default, pulled in transitively by many
   packages — see install/requirements.txt's own note on this).
2. SVG. The rendered wheel charts (utils/chart_draw.py) are SVG. weasyprint
   embeds an <img src="data:image/svg+xml;base64,..."> natively; reportlab
   has no SVG support at all and would need a separate rasterization step
   (cairosvg) plus manual page-layout math for every image.

The markdown-lite rendering below (bold via **, headings via #, fenced
code blocks) is a deliberate line-for-line port of static/js/app.js's own
renderProseText/renderInlineFormatted/renderMessageBody — the PDF should
look like the same message the person already saw in the chat UI, not a
differently-formatted document. If that JS ever changes, mirror the
change here too (see each function's docstring below for the exact JS
counterpart).
"""
import base64
import html as html_module
import re
from typing import List, Optional, Tuple

from weasyprint import HTML

_BOLD_SPLIT_RE = re.compile(r"(\*\*[^*\n]+\*\*)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_CODE_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```")


def _esc(text: str) -> str:
    return html_module.escape(text, quote=False)


def _render_inline(text: str) -> str:
    """Mirrors static/js/app.js's renderInlineFormatted: only **bold** is
    special-cased, everything else (including literal <, >, &) is treated
    as plain text, and single newlines become <br> (not a new paragraph)."""
    parts = _BOLD_SPLIT_RE.split(text)
    out = []
    for part in parts:
        if len(part) > 4 and part.startswith("**") and part.endswith("**"):
            out.append(f"<strong>{_esc(part[2:-2])}</strong>")
        else:
            out.append(_esc(part).replace("\n", "<br>"))
    return "".join(out)


def _render_prose_block(text: str) -> str:
    """Mirrors static/js/app.js's renderProseText: line-by-line scan,
    flushing a paragraph buffer on blank lines or heading lines (not a
    blank-line-first split — see that function's own comment for why a
    heading glued directly to the next line, with no blank line between,
    still needs to be recognized)."""
    lines = text.split("\n")
    out: List[str] = []
    buffer: List[str] = []

    def flush() -> None:
        if not buffer:
            return
        para = "\n".join(buffer).strip()
        buffer.clear()
        if para:
            out.append(f'<div class="msg-text">{_render_inline(para)}</div>')

    for line in lines:
        if line.strip() == "":
            flush()
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            level = min(6, len(heading.group(1)) + 2)
            out.append(f'<h{level} class="msg-heading">{_render_inline(heading.group(2).strip())}</h{level}>')
        else:
            buffer.append(line)
    flush()
    return "".join(out)


def _render_message_html(text: str) -> str:
    """Mirrors static/js/app.js's renderMessageBody: pulls fenced code
    blocks out first, routes the surrounding text through the prose
    renderer, keeps everything in original order."""
    out: List[str] = []
    pos = 0
    for m in _CODE_FENCE_RE.finditer(text):
        before = text[pos:m.start()]
        if before:
            out.append(_render_prose_block(before))
        code = _esc(m.group(2))
        out.append(f'<pre class="code-block"><code>{code}</code></pre>')
        pos = m.end()
    rest = text[pos:]
    if rest:
        out.append(_render_prose_block(rest))
    return "".join(out)


_PAGE_CSS = """
@page { size: A4; margin: 2cm 1.8cm; }
body {
  font-family: "DejaVu Sans", "Noto Sans", sans-serif;
  font-size: 11pt;
  color: #1a1a1a;
  line-height: 1.45;
}
h2, h3, h4, h5, h6 { margin: 0.7em 0 0.3em; line-height: 1.25; }
.msg-text { margin: 0 0 0.65em; white-space: pre-wrap; }
.code-block {
  background: #f4f4f4;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 8px 10px;
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 9.5pt;
  white-space: pre-wrap;
  margin: 0 0 0.8em;
}
.chart-image { margin: 1em 0; text-align: center; page-break-inside: avoid; }
.chart-image img { max-width: 100%; }
.meta { color: #777; font-size: 9pt; margin-bottom: 1.1em; border-bottom: 1px solid #eee; padding-bottom: 0.6em; }
"""


def build_message_pdf(text: str, images: List[Tuple[bytes, str]], meta_line: Optional[str] = None) -> bytes:
    """Renders one message as a standalone PDF (bytes).

    text: the message's raw content (same markdown-lite syntax the chat
    UI already renders).
    images: list of (content_bytes, mime_type) for every image attachment
    on this message, in the same order they appear in the chat (SVG chart
    files included — media_type "image/svg+xml" embeds natively).
    meta_line: optional small header line (role + timestamp)."""
    body_html = _render_message_html(text) if text else ""

    def _img_tag(content: bytes, mime: str) -> str:
        return f'<div class="chart-image"><img src="data:{mime};base64,{base64.b64encode(content).decode("ascii")}"/></div>'

    # Wheel charts are always image/svg+xml (see routes/chat.py's
    # _attach_chart_if_applicable) — every other attachment is raster.
    # Charts render before the text body, everything else after, to match
    # the chat UI's own ordering (static/js/app.js's addMessageFromRecord).
    chart_html = "".join(_img_tag(content, mime) for content, mime in images if mime == "image/svg+xml")
    other_html = "".join(_img_tag(content, mime) for content, mime in images if mime != "image/svg+xml")
    meta_html = f'<div class="meta">{_esc(meta_line)}</div>' if meta_line else ""
    document = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        f"<style>{_PAGE_CSS}</style></head><body>"
        f"{meta_html}{chart_html}{body_html}{other_html}"
        "</body></html>"
    )
    return HTML(string=document).write_pdf()
