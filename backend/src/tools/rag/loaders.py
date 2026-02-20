# tools/rag/loaders.py
from __future__ import annotations
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


def load_docs(paths: List[str]) -> List[Document]:
    docs: List[Document] = []
    supported = {".pdf", ".txt", ".md", ".docx"}
    for p in paths:
        fp = Path(p)
        if not fp.exists():
            continue
        ext = fp.suffix.lower()
        if ext not in supported:
            continue
        before = len(docs)
        try:
            if ext == ".pdf":
                docs += PyPDFLoader(str(fp)).load()
            elif ext in (".txt", ".md"):
                docs += TextLoader(str(fp), encoding="utf-8").load()
            elif ext == ".docx":
                docs += Docx2txtLoader(str(fp)).load()
        except Exception:
            # Skip unreadable/unsupported documents instead of failing the full run.
            continue

        for d in docs[before:]:
            d.metadata.setdefault("source", str(fp))
    return docs
