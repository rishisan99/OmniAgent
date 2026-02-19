# tools/rag/auto_index.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

from backend.src.tools.rag.loaders import load_docs
from backend.src.tools.rag.indexer import build_session_index


def ensure_index(session_id: str, attachments: List[Dict[str, Any]]) -> None:
    idx = Path("backend/data/sessions") / session_id / "rag"
    if idx.exists():
        return
    paths = [a.get("path") for a in attachments if a.get("path")]
    if not paths:
        return
    docs = load_docs(paths)
    if docs:
        build_session_index(session_id, docs)
