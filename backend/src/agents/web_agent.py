# agents/web_agent.py
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

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

    calls: List[tuple[str, Callable[[], Dict[str, Any]]]] = []
    if "tavily" in t.sources:
        calls.append(("tavily", lambda: tavily_search.invoke({"query": t.query, "top_k": t.top_k})))
    if "wikipedia" in t.sources:
        calls.append(("wikipedia", lambda: wikipedia_search(t.query, top_k=min(3, t.top_k))))
    if "arxiv" in t.sources:
        calls.append(("arxiv", lambda: arxiv_search.invoke({"query": t.query, "top_k": t.top_k})))

    if calls:
        with ThreadPoolExecutor(max_workers=len(calls)) as ex:
            fut_to_name = {ex.submit(fn): name for name, fn in calls}
            for fut in as_completed(fut_to_name):
                name = fut_to_name[fut]
                try:
                    outs.append(fut.result())
                except Exception as e:
                    errs.append(f"{name}: {e}")

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
