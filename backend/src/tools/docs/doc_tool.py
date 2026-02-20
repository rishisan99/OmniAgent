# tools/docs/doc_tool.py
from __future__ import annotations
from typing import Any, Dict
import re

from backend.src.schemas.results import ToolResult
from backend.src.tools.media.assets import save_asset
from backend.src.tools.rag.loaders import load_docs


def doc_extract_text(path: str, max_chars: int = 12000) -> Dict[str, Any]:
    docs = load_docs([path])
    text = "\n\n".join(d.page_content for d in docs)[:max_chars]
    return ToolResult(
        task_id="doc", kind="doc", ok=True,
        data={"text": text, "pages": len(docs), "source": path},
    ).model_dump()


def doc_generate_markdown(session_id: str, content: str) -> Dict[str, Any]:
    safe = (content or "").strip() or "# Document\n\nNo content generated."
    name, url = save_asset(session_id, "md", safe.encode("utf-8"))
    return ToolResult(
        task_id="doc", kind="doc", ok=True,
        data={"text": safe, "url": url, "filename": name, "mime": "text/markdown"},
    ).model_dump()


def _as_plain_text(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return "No content generated."
    t = re.sub(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```", r"\1", t)
    t = re.sub(r"^#+\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
    t = re.sub(r"\*(.*?)\*", r"\1", t)
    t = re.sub(r"`(.*?)`", r"\1", t)
    return t


def _markdown_lines(s: str) -> list[tuple[str, str]]:
    raw = (s or "").replace("\r\n", "\n").strip()
    fenced = re.match(r"^```(?:markdown|md)?\n([\s\S]*?)\n```$", raw, flags=re.IGNORECASE)
    text = fenced.group(1) if fenced else raw
    out: list[tuple[str, str]] = []
    for row in text.split("\n"):
        line = row.strip()
        if not line:
            out.append(("", "blank"))
            continue
        if line.startswith("### "):
            out.append((line[4:].strip(), "h3"))
        elif line.startswith("## "):
            out.append((line[3:].strip(), "h2"))
        elif line.startswith("# "):
            out.append((line[2:].strip(), "h1"))
        elif re.match(r"^\d+\.\s+", line):
            out.append((line, "h3"))
        else:
            out.append((line, "body"))
    cleaned: list[tuple[str, str]] = []
    for line, style in out:
        if not line:
            cleaned.append((line, style))
            continue
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)
        line = re.sub(r"`(.*?)`", r"\1", line)
        cleaned.append((line, style))
    return cleaned


def _escape_pdf_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _simple_pdf_bytes(content: str) -> bytes:
    lines: list[tuple[str, str]] = []
    for row, style in _markdown_lines(content):
        r = row.encode("latin-1", "replace").decode("latin-1")
        width = 90 if style in {"h1", "h2"} else 95
        if not r:
            lines.append(("", "blank"))
            continue
        while len(r) > width:
            lines.append((r[:width], style))
            r = r[width:]
        lines.append((r, style))
    content = "BT\n/F1 11 Tf\n50 800 Td\n14 TL\n"
    for ln, style in lines[:220]:
        if style == "blank":
            content += "T*\n"
            continue
        if style == "h1":
            content += "/F2 16 Tf\n"
        elif style == "h2":
            content += "/F2 14 Tf\n"
        elif style == "h3":
            content += "/F2 12 Tf\n"
        else:
            content += "/F1 11 Tf\n"
        content += f"({_escape_pdf_text(ln)}) Tj\nT*\n"
    content += "ET\n"

    stream = content.encode("latin-1", "replace")
    objs = []
    objs.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objs.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objs.append(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>\nendobj\n"
    )
    objs.append(f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n{stream.decode('latin-1')}endstream\nendobj\n")
    objs.append("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    objs.append("6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n")

    pdf = "%PDF-1.4\n"
    offsets = [0]
    for obj in objs:
        offsets.append(len(pdf.encode("latin-1")))
        pdf += obj
    xref_pos = len(pdf.encode("latin-1"))
    pdf += f"xref\n0 {len(objs)+1}\n"
    pdf += "0000000000 65535 f \n"
    for off in offsets[1:]:
        pdf += f"{off:010d} 00000 n \n"
    pdf += f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    return pdf.encode("latin-1", "replace")


def _simple_doc_bytes(content: str) -> bytes:
    rows = _markdown_lines(content)
    chunks = ["{\\rtf1\\ansi\\deff0\n"]
    for row, style in rows:
        esc = row.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        if style == "blank":
            chunks.append("\\par\n")
            continue
        if style == "h1":
            chunks.append("\\b\\fs34 " + esc + "\\b0\\fs24\\par\n")
        elif style == "h2":
            chunks.append("\\b\\fs30 " + esc + "\\b0\\fs24\\par\n")
        elif style == "h3":
            chunks.append("\\b\\fs26 " + esc + "\\b0\\fs24\\par\n")
        else:
            chunks.append("\\fs24 " + esc + "\\par\n")
    chunks.append("}")
    rtf = "".join(chunks)
    return rtf.encode("utf-8")


def doc_generate_file(session_id: str, content: str, fmt: str = "txt") -> Dict[str, Any]:
    safe = (content or "").strip() or "# Document\n\nNo content generated."
    plain = _as_plain_text(safe)
    fmt = (fmt or "txt").lower()
    if fmt == "pdf":
        ext = "pdf"
        mime = "application/pdf"
        blob = _simple_pdf_bytes(safe)
    elif fmt == "doc":
        ext = "doc"
        mime = "application/msword"
        blob = _simple_doc_bytes(safe)
    elif fmt == "md":
        ext = "md"
        mime = "text/markdown"
        blob = safe.encode("utf-8")
    else:
        ext = "txt"
        mime = "text/plain"
        blob = plain.encode("utf-8")

    name, url = save_asset(session_id, ext, blob)
    return ToolResult(
        task_id="doc",
        kind="doc",
        ok=True,
        data={"url": url, "filename": name, "mime": mime, "text": plain[:12000]},
    ).model_dump()
