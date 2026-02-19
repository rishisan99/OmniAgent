# agents/vision_agent.py
from __future__ import annotations
from typing import Any, Dict

from backend.src.schemas.tasks import VisionTask
from backend.src.stream.emitter import Emitter
from backend.src.tools.vision.vision_tool import vision_analyze


def run(task: Dict[str, Any], state: Dict[str, Any], em: Emitter) -> Dict[str, Any]:
    t = VisionTask.model_validate(task)
    em.emit("task_start", {"task_id": t.id, "kind": t.kind})
    att = next((a for a in state.get("attachments", []) if a.get("id") == t.image_attachment_id), None)
    out = vision_analyze(t.prompt, att["path"]) if att else {"ok": False, "error": "Image not found"}
    em.emit("task_result", {"task_id": t.id, "kind": t.kind, "ok": out.get("ok", False)})
    return out
