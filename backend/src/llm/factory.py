# llm/factory.py
from __future__ import annotations
from typing import List, Optional

from backend.src.core.constants import PROVIDER_MODELS
from backend.src.llm.base import normalize
from backend.src.llm.openai_llm import build_openai
from backend.src.llm.anthropic_llm import build_anthropic
from backend.src.llm.gemini_llm import build_gemini

PROVIDER_FALLBACK_MODELS = {
    "openai": ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"),
    "anthropic": (
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ),
    "gemini": (
        "gemini-1.5-flash-002",
        "gemini-1.5-pro-002",
        "gemini-2.0-flash",
    ),
}


def is_not_found_error(e: Exception) -> bool:
    s = str(e).lower()
    return "not_found" in s or "not found" in s or "404" in s


def model_candidates(provider: str, selected_model: str) -> List[str]:
    out: List[str] = []
    for m in (selected_model, *PROVIDER_MODELS.get(provider, ()), *PROVIDER_FALLBACK_MODELS.get(provider, ())):
        if m and m not in out:
            out.append(m)
    return out


def get_llm(
    provider: Optional[str],
    model: Optional[str],
    streaming: bool = True,
    temperature: float = 0.2,
):
    p, m = normalize(provider, model)
    try:
        if p == "openai":
            return build_openai(m, streaming, temperature)
        if p == "anthropic":
            return build_anthropic(m, streaming, temperature)
        return build_gemini(m, streaming, temperature)
    except RuntimeError as e:
        # If a selected provider key is missing, gracefully fall back to OpenAI.
        if p != "openai":
            return build_openai("gpt-4o-mini", streaming, temperature)
        raise e
