# tools/web/arxiv_tool.py
from __future__ import annotations
import re
from typing import Any, Dict, List

from langchain.tools import tool
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
import arxiv as arxiv_py

from backend.src.schemas.results import ToolResult, Citation


@tool("arxiv_search")
def arxiv_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search arXiv and return short paper summaries."""
    q = (query or "").strip()
    year_match = re.search(r"\b(20\d{2})\b", q)
    year = int(year_match.group(1)) if year_match else None
    topic = re.sub(r"\b(20\d{2})\b", " ", q)
    topic = re.sub(r"\b(in|from|on|about)\b\s*$", "", topic, flags=re.IGNORECASE).strip()
    if not topic:
        topic = q

    api_query = topic
    if year:
        api_query = f"all:{topic} AND submittedDate:[{year}01010000 TO {year}12312359]"

    try:
        search = arxiv_py.Search(
            query=api_query,
            max_results=max(5, int(top_k)),
            sort_by=arxiv_py.SortCriterion.SubmittedDate,
        )
        client = arxiv_py.Client(page_size=max(5, int(top_k)), delay_seconds=0.0, num_retries=2)
        rows: List[Dict[str, Any]] = []
        cites: List[Citation] = []
        for r in client.results(search):
            abs_url = getattr(r, "entry_id", "") or ""
            pdf_url = getattr(r, "pdf_url", "") or ""
            summary = (getattr(r, "summary", "") or "").strip().replace("\n", " ")
            title = (getattr(r, "title", "") or "").strip()
            published = str(getattr(r, "published", "") or "")
            authors = [a.name for a in getattr(r, "authors", [])] if getattr(r, "authors", None) else []
            rows.append(
                {
                    "title": title,
                    "url": abs_url,
                    "pdf_url": pdf_url,
                    "summary": summary,
                    "authors": authors,
                    "published": published,
                }
            )
            if title and abs_url:
                cites.append(Citation(title=title, url=abs_url, snippet=summary[:300]))
        if year:
            rows = [x for x in rows if str(x.get("published", "")).startswith(str(year))]
            cites = [c for c in cites if any(c.url == x.get("url") for x in rows)]
        rows = rows[: max(1, int(top_k))]
        cites = cites[: max(1, int(top_k))]
        if rows:
            return ToolResult(
                task_id="arxiv",
                kind="web",
                ok=True,
                data={"query": q, "effective_query": api_query, "items": rows},
                citations=cites,
            ).model_dump()
    except Exception:
        pass

    # Fallback for environments where direct arxiv client fails.
    api = ArxivAPIWrapper(top_k_results=top_k, doc_content_chars_max=1500)
    runner = ArxivQueryRun(api_wrapper=api)
    text = runner.run(topic or q)
    cites: List[Citation] = [Citation(title=f"arXiv search: {q}", url="https://arxiv.org/search/", snippet=text[:300])]
    return ToolResult(
        task_id="arxiv",
        kind="web",
        ok=True,
        data={"query": q, "effective_query": topic or q, "text": text},
        citations=cites,
    ).model_dump()
