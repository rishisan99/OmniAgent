# tools/rag/indexer.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from backend.src.tools.rag.chunker import chunk_docs
from backend.src.schemas.results import ToolResult


def _rag_dir(session_id: str) -> Path:
    p = Path("backend/data/sessions") / session_id / "rag"
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_session_index(session_id: str, docs: List, embedding_model: str = "text-embedding-3-small") -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing env var: OPENAI_API_KEY (embeddings)")
    chunks = chunk_docs(docs)
    emb = OpenAIEmbeddings(model=embedding_model)
    vs = FAISS.from_documents(chunks, emb)
    vs.save_local(str(_rag_dir(session_id)))
    return ToolResult(
        task_id="rag_index", kind="rag", ok=True,
        data={"session_id": session_id, "docs": len(docs), "chunks": len(chunks)}
    ).model_dump()
