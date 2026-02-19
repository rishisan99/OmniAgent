# tools/rag/retriever.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from backend.src.schemas.results import ToolResult, Citation


def _rag_dir(session_id: str) -> Path:
    return Path("backend/data/sessions") / session_id / "rag"


def rag_search(session_id: str, query: str, top_k: int = 5, embedding_model: str = "text-embedding-3-small") -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing env var: OPENAI_API_KEY (embeddings)")
    idx = _rag_dir(session_id)
    if not idx.exists():
        return ToolResult(task_id="rag", kind="rag", ok=False, error="No session index found").model_dump()

    vs = FAISS.load_local(str(idx), OpenAIEmbeddings(model=embedding_model), allow_dangerous_deserialization=True)
    hits = vs.similarity_search(query, k=top_k)

    rows: List[Dict[str, Any]] = []
    cites: List[Citation] = []
    for d in hits:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        rows.append({"text": d.page_content, "source": src, "page": page})
        title = f"{Path(src).name}" + (f" (p.{page+1})" if isinstance(page, int) else "")
        cites.append(Citation(title=title, url=src, snippet=d.page_content[:300]))

    return ToolResult(task_id="rag", kind="rag", ok=True, data={"query": query, "matches": rows}, citations=cites).model_dump()
