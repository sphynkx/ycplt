"""Extracts fenced code blocks (```lang ... ```) from the model's reply.

Each such block is turned into a separate file attachment (see
db/repository.py add_file and routes/chat.py) — so code from a reply can be
downloaded on its own, not just read inline in the message text.
"""
import re
from typing import Dict, List

_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)

_EXT_BY_LANG = {
    "python": "py", "py": "py",
    "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts",
    "bash": "sh", "sh": "sh", "shell": "sh",
    "powershell": "ps1", "ps1": "ps1",
    "json": "json",
    "html": "html",
    "css": "css",
    "sql": "sql",
    "yaml": "yaml", "yml": "yaml",
    "java": "java",
    "c": "c", "cpp": "cpp", "c++": "cpp",
    "go": "go",
    "rust": "rs",
    "markdown": "md", "md": "md",
    "xml": "xml",
    "toml": "toml",
}


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """Returns [{"filename": ..., "mime_type": ..., "content": ...}, ...]
    for every fenced code block in the text. Never raises — returns an empty
    list if there are none."""
    blocks = []
    for i, m in enumerate(_FENCE_RE.finditer(text or ""), start=1):
        lang = (m.group(1) or "").strip().lower()
        code = m.group(2).rstrip("\n")
        if not code.strip():
            continue
        ext = _EXT_BY_LANG.get(lang, "txt")
        blocks.append(
            {
                "filename": f"snippet_{i}.{ext}",
                "mime_type": "text/plain",
                "content": code,
            }
        )
    return blocks
