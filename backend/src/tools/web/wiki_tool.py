# tools/web/wiki_tool.py
from __future__ import annotations
import os
from typing import Any, Dict, List

import requests

from backend.src.schemas.results import ToolResult


def wikipedia_search(query: str, top_k: int = 3) -> Dict[str, Any]:
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": top_k}
        headers = {"User-Agent": os.getenv("WIKI_UA", "OmniAgent/0.1 (contact: you@example.com)")}
        r = requests.get(url, params=params, headers=headers, timeout=12)
        r.raise_for_status()
        data = r.json()
        hits = (data.get("query", {}).get("search") or [])[:top_k]

        items: List[Dict[str, Any]] = []
        cites: List[Dict[str, Any]] = []
        for h in hits:
            title = h.get("title", "")
            link = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            items.append({"title": title, "url": link, "snippet": h.get("snippet", "")})
            cites.append({"title": title, "url": link})

        return ToolResult(task_id="wiki", kind="web", ok=True, data={"items": items}, citations=cites).model_dump()
    except Exception as e:
        return ToolResult(task_id="wiki", kind="web", ok=False, error=str(e)).model_dump()
