# llm/openai_llm.py
from __future__ import annotations
from langchain_openai import ChatOpenAI

from backend.src.llm.base import common_kwargs, require_env


def build_openai(model: str, streaming: bool, temperature: float):
    require_env("OPENAI_API_KEY")
    return ChatOpenAI(model=model, **common_kwargs(streaming, temperature))
