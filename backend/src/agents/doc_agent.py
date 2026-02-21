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
        req = str(t.prompt or state.get("user_text", "")).lower()
        wants_brief = any(
            k in req for k in ("brief", "short", "concise", "summary", "summarize", "tldr")
        )
        wants_long_form = (not wants_brief) or any(
            k in req
            for k in (
                "1 page",
                "one page",
                "1.5 page",
                "one and a half page",
                "long form",
                "in detail",
                "detailed",
                "deep dive",
                "comprehensive",
            )
        )
        length_rules = (
            "- Use 2-3 short sections with H2 headings.\n"
            "- Use at most 5 bullets total.\n"
            "- Hard cap: 120 words.\n"
            if wants_brief and not wants_long_form
            else
            "- Target ~2 pages (roughly 700-1000 words).\n"
            "- Use 5-8 short sections with H2 headings.\n"
            "- Use bullets where helpful; max 20 bullets total.\n"
        )
        prompt = (
            "Write a clean, concise markdown document from the request below.\n"
            "Formatting rules:\n"
            "- Keep it brief and well-structured.\n"
            "- Include exactly one H1 title.\n"
            f"{length_rules}"
            "- Prefer bullets over long paragraphs.\n"
            "- Keep each bullet to one sentence.\n"
            "- Avoid filler, repetition, and verbose prose.\n\n"
            f"REQUEST:\n{t.prompt or state.get('user_text', '')}\n"
        )
        msg = llm.invoke(prompt)
        content = (getattr(msg, "content", "") or "").strip()
        out = doc_generate_file(state["session_id"], content, fmt=t.format)
    em.emit("task_result", {"task_id": t.id, "kind": t.kind, "ok": out.get("ok", False)})
    return out
