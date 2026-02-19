# llm/base.py
from __future__ import annotations
import os
from typing import Any, Dict, Optional

from backend.src.core.constants import DEFAULT_MODEL, DEFAULT_PROVIDER, PROVIDER_MODELS, SUPPORTED_PROVIDERS


def require_env(name: str) -> None:
    if not os.getenv(name):
        raise RuntimeError(f"Missing env var: {name}")


def normalize(provider: Optional[str], model: Optional[str]) -> tuple[str, str]:
    p = (provider or DEFAULT_PROVIDER).lower()
    if p not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {p}")
    m = (model or DEFAULT_MODEL).strip()
    if not m:
        raise ValueError("Model cannot be empty")
    # Allow custom/updated model ids beyond the UI defaults.
    if m not in PROVIDER_MODELS.get(p, ()):
        if p == "openai" and m.startswith(("gpt-", "o", "chatgpt-")):
            return p, m
        if p == "anthropic" and m.startswith("claude-"):
            return p, m
        if p == "gemini" and m.startswith("gemini-"):
            return p, m
    return p, m


def common_kwargs(streaming: bool, temperature: float) -> Dict[str, Any]:
    return {"streaming": streaming, "temperature": temperature}
