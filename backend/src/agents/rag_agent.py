from __future__ import annotations
from typing import Any, Dict

from backend.src.schemas.tasks import RagTask
from backend.src.stream.emitter import Emitter
from backend.src.tools.rag.auto_index import ensure_index
from backend.src.tools.rag.retriever import rag_search


def run(task: Dict[str, Any], state: Dict[str, Any], em: Emitter) -> Dict[str, Any]:
    t = RagTask.model_validate(task)
    em.emit("task_start", {"task_id": t.id, "kind": t.kind, "query": t.query})

    ensure_index(state["session_id"], state.get("attachments", []))
    out = rag_search(state["session_id"], t.query, top_k=t.top_k)

    em.emit("task_result", {"task_id": t.id, "kind": t.kind, "ok": out.get("ok", False)})
    return out
