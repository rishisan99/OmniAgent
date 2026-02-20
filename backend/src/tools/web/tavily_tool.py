# tools/web/tavily_tool.py
from __future__ import annotations
import os
import re
from typing import Any, Dict, List

from langchain.tools import tool
from langchain_tavily import TavilySearch

from backend.src.schemas.results import ToolResult, Citation


@tool("tavily_search")
def tavily_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Web search via Tavily (langchain-tavily). Returns results + citations."""
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Missing env var: TAVILY_API_KEY")

    q = (query or "").strip()
    q_l = q.lower()
    is_news_query = any(k in q_l for k in ("news", "headline", "headlines", "latest", "recent", "today", "update"))
    topic = "news" if is_news_query else "general"

    # Light rewrite to bias freshness when user asks for recent updates.
    effective_query = q
    if is_news_query and not re.search(r"\b(today|this week|past \d+ days?)\b", q_l):
        effective_query = f"{q} today latest updates"

    t = TavilySearch(max_results=top_k, topic=topic)
    out: Dict[str, Any] = t.invoke({"query": effective_query})

    rows: List[Dict[str, Any]] = out.get("results", [])
    if is_news_query:
        # Drop low-signal aggregator/search pages for cleaner news summaries.
        blocked = ("google.com/search", "news.google.com", "/tag/", "/topic/", "/topics/")
        rows = [r for r in rows if not any(b in str(r.get("url", "")).lower() for b in blocked)]
    cites = [
        Citation(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("content"))
        for r in rows
    ]
    return ToolResult(
        task_id="tavily", kind="web", ok=True,
        data={"query": q, "effective_query": effective_query, **{**out, "results": rows}}, citations=cites
    ).model_dump()
