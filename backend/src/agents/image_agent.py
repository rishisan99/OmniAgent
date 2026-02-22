# agents/image_agent.py
from __future__ import annotations
from typing import Any, Dict

from backend.src.core.logging import get_logger
from backend.src.schemas.tasks import ImageGenTask
from backend.src.stream.emitter import Emitter
from backend.src.tools.media.image_tool import image_generate

logger = get_logger("omniagent.agent.image")


def run(task: Dict[str, Any], state: Dict[str, Any], em: Emitter) -> Dict[str, Any]:
    t = ImageGenTask.model_validate(task)
    logger.info(
        "IMAGE_AGENT_START task_id=%s session_id=%s size=%s subject_lock=%s",
        t.id,
        state.get("session_id"),
        t.size,
        bool(t.subject_lock),
    )
    em.emit("task_start", {"task_id": t.id, "kind": t.kind})
    prompt = t.prompt
    if t.subject_lock and t.subject_lock.lower() not in prompt.lower():
        prompt = (
            f"{prompt}\n\n"
            f"CRITICAL CONSTRAINT: Keep main subject as '{t.subject_lock}'. Do not replace it."
        )
    out = image_generate(state["session_id"], prompt, size=t.size)
    data = dict(out.get("data") or {})
    logger.info(
        "IMAGE_AGENT_RESULT task_id=%s ok=%s filename=%s url=%s",
        t.id,
        out.get("ok", False),
        data.get("filename"),
        data.get("url"),
    )
    em.emit("task_result", {"task_id": t.id, "kind": t.kind, "ok": out.get("ok", False)})
    return out
