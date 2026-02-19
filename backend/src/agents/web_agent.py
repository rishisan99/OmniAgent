# agents/web_agent.py
from __future__ import annotations
from typing import Any, Dict, List

from backend.src.schemas.tasks import WebTask
from backend.src.stream.emitter import Emitter
from backend.src.tools.web.tavily_tool import tavily_search
from backend.src.tools.web.wiki_tool import wikipedia_search
from backend.src.tools.web.arxiv_tool import arxiv_search


def run(task: Dict[str, Any], state: Dict[str, Any], em: Emitter) -> Dict[str, Any]:
    t = WebTask.model_validate(task)
    em.emit("task_start", {"task_id": t.id, "kind": t.kind, "query": t.query, "sources": t.sources})

    outs: List[Dict[str, Any]] = []
    errs: List[str] = []

    def _safe_call(name: str, fn):
        try:
            outs.append(fn())
        except Exception as e:
            errs.append(f"{name}: {e}")

    if "tavily" in t.sources:
        _safe_call("tavily", lambda: tavily_search.invoke({"query": t.query, "top_k": t.top_k}))
    if "wikipedia" in t.sources:
        _safe_call("wikipedia", lambda: wikipedia_search(t.query, top_k=min(3, t.top_k)))
    if "arxiv" in t.sources:
        _safe_call("arxiv", lambda: arxiv_search.invoke({"query": t.query, "top_k": t.top_k}))

    ok = any(o.get("ok") for o in outs) if outs else False
    em.emit("task_result", {"task_id": t.id, "kind": t.kind, "ok": ok, "errors": errs})
    return {
        "task_id": t.id,
        "kind": "web",
        "ok": ok,
        "data": {"parts": outs, "errors": errs},
        "citations": sum([o.get("citations", []) for o in outs], []),
        "error": "; ".join(errs) if (errs and not ok) else None,
    }
