# tools/media/tts_tool.py
from __future__ import annotations
import os
from typing import Any, Dict

from openai import OpenAI

from backend.src.schemas.results import ToolResult
from backend.src.tools.media.assets import save_asset


def tts_generate(session_id: str, text: str, voice: str = "alloy") -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing env var: OPENAI_API_KEY")
    model = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
    client = OpenAI()
    audio = client.audio.speech.create(model=model, voice=voice, input=text)
    name, url = save_asset(session_id, "mp3", audio.read())
    return ToolResult(
        task_id="tts", kind="tts", ok=True,
        data={"url": url, "filename": name, "mime": "audio/mpeg", "voice": voice, "model": model},
    ).model_dump()
