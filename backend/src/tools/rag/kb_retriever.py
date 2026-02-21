from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from backend.src.schemas.results import Citation, ToolResult
from backend.src.tools.rag.kb_index import KB_INDEX_DIR, ensure_kb_index, kb_index_signature


_VS_CACHE: Dict[str, Any] = {"sig": "", "vs": None}
_QUERY_CACHE: Dict[str, Dict[str, Any]] = {}


def _entity_hint(query: str) -> str:
    q = (query or "").strip()
    mq = re.search(r'"([^"]{2,})"', q) or re.search(r"'([^']{2,})'", q)
    if mq:
        return re.sub(r"\s+", " ", mq.group(1).strip())
    patterns = [
        # Flexible conversational patterns, including polite prefixes.
        r"(?:^|\b)(?:can you|could you|please)\s+(?:tell me about|about|who is|profile of)\s+(?:employee|employees|person)?\s*([a-zA-Z][a-zA-Z .'-]{2,})",
        r"(?:^|\b)(?:tell me about|about|who is|profile of)\s+(?:employee|employees|person)?\s*([a-zA-Z][a-zA-Z .'-]{2,})",
        # Explicit role mention anywhere in query.
        r"\b(?:employee|employees|person)\s+([a-zA-Z][a-zA-Z .'-]{2,})",
    ]
    for p in patterns:
        m = re.search(p, q, flags=re.IGNORECASE)
        if m:
            name = re.sub(r"\s+", " ", m.group(1).strip(" .?!,;:\"'"))
            # Strip leading role words if capture over-includes them.
            name = re.sub(r"^(employee|employees|person)\s+", "", name, flags=re.IGNORECASE)
            if name:
                return name
    return ""


def _source_boost(query: str, source: str, entity_hint: str) -> int:
    src = (source or "").lower()
    score = 0
    if entity_hint:
        hint_tokens = [t for t in re.split(r"\s+", entity_hint.lower()) if len(t) >= 2]
        if hint_tokens and all(t in src for t in hint_tokens):
            score += 100
    q_tokens = [t for t in re.split(r"\s+", re.sub(r"[^a-z0-9 ]+", " ", query.lower())) if len(t) >= 3]
    score += sum(1 for t in q_tokens if t in src)
    return score


def load_kb_vectorstore(embedding_model: str = "text-embedding-3-small") -> FAISS:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing env var: OPENAI_API_KEY (embeddings)")
    idx_state = ensure_kb_index(embedding_model=embedding_model, force=False)
    if not idx_state.get("ok"):
        raise RuntimeError(str(idx_state.get("error", "KB index unavailable")))
    if not KB_INDEX_DIR.exists():
        raise RuntimeError("KB index missing")
    sig = kb_index_signature()
    cached_sig = str(_VS_CACHE.get("sig", ""))
    cached_vs = _VS_CACHE.get("vs")
    if cached_vs is not None and cached_sig == sig:
        return cached_vs
    vs = FAISS.load_local(
        str(KB_INDEX_DIR),
        OpenAIEmbeddings(model=embedding_model),
        allow_dangerous_deserialization=True,
    )
    _VS_CACHE["sig"] = sig
    _VS_CACHE["vs"] = vs
    return vs


def kb_top_chunks(query: str, top_k: int = 5, embedding_model: str = "text-embedding-3-small") -> List[Document]:
    vs = load_kb_vectorstore(embedding_model=embedding_model)
    return vs.similarity_search(query, k=max(1, int(top_k)))


def kb_search(query: str, top_k: int = 6, embedding_model: str = "text-embedding-3-small") -> Dict[str, Any]:
    q_key = re.sub(r"\s+", " ", (query or "").strip().lower())
    sig = kb_index_signature()
    ttl = int(os.getenv("KB_RAG_CACHE_TTL_SEC", "180"))
    key = f"{q_key}|k={int(top_k)}|sig={sig}"
    cached = _QUERY_CACHE.get(key)
    now = time.time()
    if cached and now - float(cached.get("ts", 0)) <= ttl:
        return dict(cached.get("result") or {})

    try:
        vs = load_kb_vectorstore(embedding_model=embedding_model)
        fetch_k = max(8, int(top_k) * 4)
        hits = vs.similarity_search_with_score(query, k=fetch_k)
    except Exception as e:
        return ToolResult(task_id="kb_rag", kind="kb_rag", ok=False, error=str(e)).model_dump()

    entity_hint = _entity_hint(query)
    scored_hits: List[tuple[Any, float]] = []
    for d, dist in hits:
        src = str(d.metadata.get("source", ""))
        # Lower distance is better for FAISS; convert to descending score.
        base_score = -float(dist)
        boost = float(_source_boost(query, src, entity_hint))
        scored_hits.append((d, base_score + boost))
    scored_hits.sort(key=lambda x: x[1], reverse=True)

    filtered_hits = scored_hits
    if entity_hint:
        hint_tokens = [t for t in re.split(r"\s+", entity_hint.lower()) if len(t) >= 2]
        strict = []
        for d, s in scored_hits:
            src = str(d.metadata.get("source", "")).lower()
            if hint_tokens and all(t in src for t in hint_tokens):
                strict.append((d, s))
        if strict:
            filtered_hits = strict
        else:
            result = ToolResult(
                task_id="kb_rag",
                kind="kb_rag",
                ok=True,
                data={"query": query, "matches": [], "entity_not_found": entity_hint},
                citations=[],
            ).model_dump()
            _QUERY_CACHE[key] = {"ts": now, "result": result}
            return result
    top_hits = filtered_hits[: max(1, int(top_k))]

    rows: List[Dict[str, Any]] = []
    cites: List[Citation] = []
    for d, score in top_hits:
        src = str(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page")
        rel_src = src
        try:
            rel_src = str(Path(src).relative_to(Path.cwd()))
        except Exception:
            rel_src = src
        title = Path(src).name if src else "unknown"
        if isinstance(page, int):
            title = f"{title} (p.{page + 1})"
        rows.append({"text": d.page_content, "source": rel_src, "page": page, "score": float(score)})
        cites.append(Citation(title=title, url=rel_src, snippet=d.page_content[:260]))

    result = ToolResult(
        task_id="kb_rag",
        kind="kb_rag",
        ok=True,
        data={"query": query, "matches": rows},
        citations=cites,
    ).model_dump()
    _QUERY_CACHE[key] = {"ts": now, "result": result}
    # Keep cache bounded.
    if len(_QUERY_CACHE) > 512:
        oldest = sorted(_QUERY_CACHE.items(), key=lambda kv: float(kv[1].get("ts", 0)))[:64]
        for k, _ in oldest:
            _QUERY_CACHE.pop(k, None)
    return result
