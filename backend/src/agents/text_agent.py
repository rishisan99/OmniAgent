# agents/text_agent.py
from __future__ import annotations
from typing import Any, Dict

from backend.src.schemas.tasks import TextTask
from backend.src.stream.emitter import Emitter
from backend.src.llm.factory import get_llm


def run(task: Dict[str, Any], state: Dict[str, Any], em: Emitter, provider: str, model: str) -> Dict[str, Any]:
    t = TextTask.model_validate(task)
    em.emit("task_start", {"task_id": t.id, "kind": t.kind})
    llm = get_llm(provider, model, streaming=False)
    msg = llm.invoke(t.prompt)
    em.emit("task_result", {"task_id": t.id, "kind": t.kind, "ok": True})
    return {"task_id": t.id, "kind": "text", "ok": True, "data": {"text": getattr(msg, "content", str(msg))}}
