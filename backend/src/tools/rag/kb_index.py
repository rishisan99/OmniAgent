from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from backend.src.tools.rag.chunker import chunk_docs
from backend.src.tools.rag.loaders import load_docs


KB_INDEX_DIR = Path("backend/data/knowledge-base-index/faiss")
KB_STAMP_FILE = Path("backend/data/knowledge-base-index/stamp.json")
KB_ALLOWED_EXT = {".pdf", ".txt", ".md", ".docx"}


def kb_root() -> Path:
    env_root = os.getenv("KB_ROOT_PATH", "").strip()
    if env_root:
        return Path(env_root)
    candidates = [
        Path("backend/docs/knowledge-base"),
        Path("backend/docs"),
        Path("backend/data/docs/knowledge-base"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]


def _kb_files() -> List[Path]:
    root = kb_root()
    if not root.exists():
        return []
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in KB_ALLOWED_EXT:
            out.append(p)
    return sorted(out)


def _stamp_for(files: List[Path]) -> Dict[str, Any]:
    mtimes = [int(p.stat().st_mtime_ns) for p in files] if files else [0]
    chunk_size = int(os.getenv("KB_RAG_CHUNK_SIZE", "900"))
    chunk_overlap = int(os.getenv("KB_RAG_CHUNK_OVERLAP", "150"))
    return {
        "count": len(files),
        "latest_mtime_ns": max(mtimes) if mtimes else 0,
        "root": str(kb_root()),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def _read_stamp() -> Dict[str, Any] | None:
    if not KB_STAMP_FILE.exists():
        return None
    try:
        return json.loads(KB_STAMP_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def kb_index_signature() -> str:
    stamp = _read_stamp() or {}
    return json.dumps(stamp, sort_keys=True, ensure_ascii=False)


def _write_stamp(stamp: Dict[str, Any]) -> None:
    KB_STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    KB_STAMP_FILE.write_text(json.dumps(stamp, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_kb_index(embedding_model: str = "text-embedding-3-small", force: bool = False) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing env var: OPENAI_API_KEY (embeddings)")

    files = _kb_files()
    if not files:
        return {"ok": False, "error": f"No KB files found in {kb_root()}"}

    wanted = _stamp_for(files)
    current = _read_stamp()
    if not force and KB_INDEX_DIR.exists() and current == wanted:
        return {"ok": True, "rebuilt": False, "files": len(files)}

    docs = load_docs([str(p) for p in files])
    if not docs:
        return {"ok": False, "error": "No readable KB documents found"}

    chunk_size = int(os.getenv("KB_RAG_CHUNK_SIZE", "900"))
    chunk_overlap = int(os.getenv("KB_RAG_CHUNK_OVERLAP", "150"))
    chunks = chunk_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    emb = OpenAIEmbeddings(model=embedding_model)
    vs = FAISS.from_documents(chunks, emb)
    KB_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(KB_INDEX_DIR))
    _write_stamp(wanted)
    return {
        "ok": True,
        "rebuilt": True,
        "files": len(files),
        "docs": len(docs),
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
