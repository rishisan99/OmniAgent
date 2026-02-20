# agents/router.py
from __future__ import annotations
from typing import Any, Dict

from backend.src.stream.emitter import Emitter
from backend.src.schemas.tasks import Task
from backend.src.schemas.tasks import task_adapter

from backend.src.agents.web_agent import run as run_web
from backend.src.agents.rag_agent import run as run_rag
from backend.src.agents.kb_rag_agent import run as run_kb_rag
from backend.src.agents.text_agent import run as run_text

from backend.src.agents.image_agent import run as run_image
from backend.src.agents.audio_agent import run as run_audio
from backend.src.agents.doc_agent import run as run_doc
from backend.src.agents.vision_agent import run as run_vision


def run_task(task: Dict[str, Any], state: Dict[str, Any], em: Emitter, provider: str, model: str) -> Dict[str, Any]:
    t = task_adapter.validate_python(task)
    if t.kind == "web":
        return run_web(task, state, em)
    if t.kind == "rag":
        return run_rag(task, state, em)
    if t.kind == "kb_rag":
        return run_kb_rag(task, state, em, provider, model)
    if t.kind == "text":
        return run_text(task, state, em, provider, model)
    if t.kind == "image_gen":
        return run_image(task, state, em)
    if t.kind == "tts":
        return run_audio(task, state, em)
    if t.kind == "doc":
        return run_doc(task, state, em, provider, model)
    if t.kind == "vision":
        return run_vision(task, state, em)

    em.emit("error", {"task_id": t.id, "error": f"Agent not implemented for kind={t.kind}"})
    return {"task_id": t.id, "kind": t.kind, "ok": False, "error": "Agent not implemented"}
