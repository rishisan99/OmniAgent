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
    def wrap_line(text: str, width: int) -> list[str]:
        t = (text or "").strip()
        if not t:
            return [""]
        out: list[str] = []
        while len(t) > width:
            cut = t.rfind(" ", 0, width + 1)
            if cut <= 0:
                cut = width
            out.append(t[:cut].rstrip())
            t = t[cut:].lstrip()
        out.append(t)
        return out

    lines: list[tuple[str, str]] = []
    for row, style in _markdown_lines(content):
        r = row.encode("latin-1", "replace").decode("latin-1")
        width = 88 if style in {"h1", "h2"} else 94
        if not r:
            lines.append(("", "blank"))
            continue
        for part in wrap_line(r, width):
            lines.append((part, style))

    lines_per_page = 46
    chunks: list[list[tuple[str, str]]] = []
    for i in range(0, max(1, len(lines)), lines_per_page):
        chunks.append(lines[i : i + lines_per_page])
    if not chunks:
        chunks = [[("", "blank")]]

    page_count = len(chunks)
    catalog_id = 1
    pages_id = 2
    first_page_id = 3
    font1_id = first_page_id + page_count * 2
    font2_id = font1_id + 1

    objs: list[str] = [""] * (font2_id + 1)
    objs[catalog_id] = f"{catalog_id} 0 obj\n<< /Type /Catalog /Pages {pages_id} 0 R >>\nendobj\n"

    kids = []
    for idx in range(page_count):
        page_id = first_page_id + idx * 2
        content_id = page_id + 1
        kids.append(f"{page_id} 0 R")
        chunk = chunks[idx]
        body = "BT\n/F1 11 Tf\n50 800 Td\n14 TL\n"
        for ln, style in chunk:
            if style == "blank":
                body += "T*\n"
                continue
            if style == "h1":
                body += "/F2 16 Tf\n"
            elif style == "h2":
                body += "/F2 14 Tf\n"
            elif style == "h3":
                body += "/F2 12 Tf\n"
            else:
                body += "/F1 11 Tf\n"
            body += f"({_escape_pdf_text(ln)}) Tj\nT*\n"
        body += "ET\n"
        stream = body.encode("latin-1", "replace")
        objs[content_id] = (
            f"{content_id} 0 obj\n<< /Length {len(stream)} >>\nstream\n"
            f"{stream.decode('latin-1')}endstream\nendobj\n"
        )
        objs[page_id] = (
            f"{page_id} 0 obj\n<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 842] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font1_id} 0 R /F2 {font2_id} 0 R >> >> >>\nendobj\n"
        )

    objs[pages_id] = (
        f"{pages_id} 0 obj\n<< /Type /Pages /Kids [{' '.join(kids)}] /Count {page_count} >>\nendobj\n"
    )
    objs[font1_id] = f"{font1_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    objs[font2_id] = f"{font2_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n"

    pdf = "%PDF-1.4\n"
    offsets = [0]
    for obj_id in range(1, len(objs)):
        obj = objs[obj_id]
        offsets.append(len(pdf.encode("latin-1")))
        pdf += obj
    xref_pos = len(pdf.encode("latin-1"))
    pdf += f"xref\n0 {len(objs)}\n"
    pdf += "0000000000 65535 f \n"
    for off in offsets[1:]:
        pdf += f"{off:010d} 00000 n \n"
    pdf += f"trailer\n<< /Size {len(objs)} /Root {catalog_id} 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
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
