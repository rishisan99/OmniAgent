# tools/rag/loaders.py
from __future__ import annotations
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


def load_docs(paths: List[str]) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        fp = Path(p)
        if not fp.exists():
            continue
        ext = fp.suffix.lower()
        if ext == ".pdf":
            docs += PyPDFLoader(str(fp)).load()
        elif ext in (".txt", ".md"):
            docs += TextLoader(str(fp), encoding="utf-8").load()
        elif ext in (".docx",):
            docs += Docx2txtLoader(str(fp)).load()
        else:
            docs += TextLoader(str(fp), encoding="utf-8").load()

        for d in docs[-50:]:
            d.metadata.setdefault("source", str(fp))
    return docs
