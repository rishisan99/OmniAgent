# agents/audio_agent.py
from __future__ import annotations
from typing import Any, Dict

from backend.src.schemas.tasks import AudioTask
from backend.src.stream.emitter import Emitter
from backend.src.tools.media.tts_tool import tts_generate


def run(task: Dict[str, Any], state: Dict[str, Any], em: Emitter) -> Dict[str, Any]:
    t = AudioTask.model_validate(task)
    em.emit("task_start", {"task_id": t.id, "kind": t.kind})
    out = tts_generate(state["session_id"], t.text, voice=t.voice)
    em.emit("task_result", {"task_id": t.id, "kind": t.kind, "ok": out.get("ok", False)})
    return out
