# tools/rag/_smoke_test.py
from __future__ import annotations
import json
from backend.src.core.config import bootstrap_env
bootstrap_env()

from backend.src.tools.rag.loaders import load_docs
from backend.src.tools.rag.indexer import build_session_index
from backend.src.tools.rag.retriever import rag_search

if __name__ == "__main__":
    sid = "demo"
    docs = load_docs(["backend/data/docs/sample.pdf"])
    print(json.dumps(build_session_index(sid, docs), indent=2))
    print(json.dumps(rag_search(sid, "What is this document about?", top_k=3), indent=2))

