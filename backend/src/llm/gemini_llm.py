# llm/gemini_llm.py
from __future__ import annotations
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.src.llm.base import common_kwargs, require_env


def build_gemini(model: str, streaming: bool, temperature: float):
    # LangChain supports GOOGLE_API_KEY env (recommended)
    require_env("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(model=model, **common_kwargs(streaming, temperature))
