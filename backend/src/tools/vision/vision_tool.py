# tools/vision/vision_tool.py
from __future__ import annotations
import base64
import os
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from backend.src.schemas.results import ToolResult


def vision_analyze(prompt: str, image_path: str) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing env var: OPENAI_API_KEY")
    model = os.getenv("VISION_MODEL", "gpt-4o-mini")
    b64 = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
    msg = [{"type": "text", "text": prompt},
           {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
    llm = ChatOpenAI(model=model, temperature=0.2)
    out = llm.invoke([{"role": "user", "content": msg}]).content
    return ToolResult(task_id="vision", kind="vision", ok=True, data={"text": out, "model": model}).model_dump()
