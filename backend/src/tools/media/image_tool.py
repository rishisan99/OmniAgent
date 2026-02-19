# tools/media/image_tool.py
from __future__ import annotations
import base64
import os
from typing import Any, Dict

from openai import OpenAI

from backend.src.schemas.results import ToolResult
from backend.src.tools.media.assets import save_asset


def image_generate(session_id: str, prompt: str, size: str = "1024x1024") -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing env var: OPENAI_API_KEY")
    model = os.getenv("IMAGE_MODEL", "gpt-image-1")
    client = OpenAI()
    r = client.images.generate(model=model, prompt=prompt, size=size)
    b64 = r.data[0].b64_json
    name, url = save_asset(session_id, "png", base64.b64decode(b64))
    return ToolResult(
        task_id="image_gen", kind="image_gen", ok=True,
        data={"url": url, "filename": name, "mime": "image/png", "size": size, "model": model, "prompt": prompt},
    ).model_dump()
