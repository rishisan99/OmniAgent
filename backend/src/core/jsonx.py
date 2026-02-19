# core/jsonx.py
from __future__ import annotations
import json
from typing import Any, Dict


def extract_json(text: str) -> Dict[str, Any]:
    """Best-effort: parse full text, else parse the first {...} block."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM output")

    try:
        return json.loads(text)
    except Exception:
        pass

    a = text.find("{")
    b = text.rfind("}")
    if a == -1 or b == -1 or b <= a:
        raise ValueError("No JSON object found in LLM output")

    return json.loads(text[a : b + 1])
