# agents/doc_agent.py
from __future__ import annotations
from typing import Any, Dict

from backend.src.schemas.tasks import DocTask
from backend.src.stream.emitter import Emitter
from backend.src.llm.factory import get_llm
from backend.src.tools.docs.doc_tool import doc_extract_text, doc_generate_file


def run(task: Dict[str, Any], state: Dict[str, Any], em: Emitter, provider: str, model: str) -> Dict[str, Any]:
    t = DocTask.model_validate(task)
    em.emit("task_start", {"task_id": t.id, "kind": t.kind})
    if t.instruction == "extract":
        att = next((a for a in state.get("attachments", []) if a.get("id") == t.attachment_id), None)
        out = doc_extract_text(att["path"]) if att else {"ok": False, "error": "Attachment not found"}
    else:
        llm = get_llm(provider, model, streaming=False, temperature=0.2)
        prompt = (
            "Write a clean markdown document from the request below.\n"
            "Use a title, short sections, and concise bullets where useful.\n\n"
            f"REQUEST:\n{t.prompt or state.get('user_text', '')}\n"
        )
        msg = llm.invoke(prompt)
        content = (getattr(msg, "content", "") or "").strip()
        out = doc_generate_file(state["session_id"], content, fmt=t.format)
    em.emit("task_result", {"task_id": t.id, "kind": t.kind, "ok": out.get("ok", False)})
    return out
