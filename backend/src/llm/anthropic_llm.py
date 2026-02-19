# llm/anthropic_llm.py
from __future__ import annotations
from langchain_anthropic import ChatAnthropic

from backend.src.llm.base import common_kwargs, require_env


def build_anthropic(model: str, streaming: bool, temperature: float):
    require_env("ANTHROPIC_API_KEY")
    return ChatAnthropic(model=model, **common_kwargs(streaming, temperature))
