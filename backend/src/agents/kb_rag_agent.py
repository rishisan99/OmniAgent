from __future__ import annotations

from backend.src.stream.emitter import Emitter
from typing import Any, Dict

from backend.src.tools.rag.kb_retriever import kb_search


def run(task: Dict[str, Any], state: Dict[str, Any], em: Emitter, provider: str, model: str) -> Dict[str, Any]:
    task_id = str(task.get("id", "kb1"))
    query = str(task.get("query", state.get("user_text", ""))).strip()
    top_k = int(task.get("top_k", 5))
    em.emit("task_start", {"task_id": task_id, "kind": "kb_rag", "query": query})

    out = kb_search(query, top_k=top_k)
    if not out.get("ok"):
        em.emit("task_result", {"task_id": task_id, "kind": "kb_rag", "ok": False})
        return {"task_id": task_id, "kind": "kb_rag", "ok": False, "error": out.get("error", "KB search failed")}
    # Retrieval-only tool output: main LLM lane will stream the final answer from this context.
    result = {
        "task_id": task_id,
        "kind": "kb_rag",
        "ok": True,
        "data": out.get("data", {}),
        "citations": out.get("citations", []),
    }
    em.emit("task_result", {"task_id": task_id, "kind": "kb_rag", "ok": True})
    return result
