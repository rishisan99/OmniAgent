# graph/streaming.py
from __future__ import annotations
from typing import Any, Optional

from backend.src.llm.factory import get_llm, is_not_found_error, model_candidates
from backend.src.stream.emitter import Emitter


async def stream_tokens(
    prompt: str,
    em: Emitter,
    provider: str,
    model: str,
    temperature: float = 0.2,
) -> str:
    """Provider-agnostic token streaming with model-id fallback retries on 404/not-found."""
    last_err: Optional[Exception] = None
    for idx, candidate in enumerate(model_candidates(provider, model)):
        llm = get_llm(provider, candidate, streaming=True, temperature=temperature)
        acc = ""
        try:
            async for chunk in llm.astream(prompt):
                tok = getattr(chunk, "content", "") or ""
                if tok:
                    acc += tok
                    em.emit("token", {"text": tok})
            return acc
        except Exception as e:
            last_err = e
            if idx < len(model_candidates(provider, model)) - 1 and is_not_found_error(e):
                continue
            raise
    if last_err:
        raise last_err
    return ""
