# tools/web/tavily_tool.py
from __future__ import annotations
import os
from typing import Any, Dict, List

from langchain.tools import tool
from langchain_tavily import TavilySearch

from backend.src.schemas.results import ToolResult, Citation


@tool("tavily_search")
def tavily_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Web search via Tavily (langchain-tavily). Returns results + citations."""
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Missing env var: TAVILY_API_KEY")

    t = TavilySearch(max_results=top_k, topic="general")
    out: Dict[str, Any] = t.invoke({"query": query})

    rows: List[Dict[str, Any]] = out.get("results", [])
    cites = [
        Citation(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("content"))
        for r in rows
    ]
    return ToolResult(
        task_id="tavily", kind="web", ok=True,
        data={"query": query, **out}, citations=cites
    ).model_dump()
