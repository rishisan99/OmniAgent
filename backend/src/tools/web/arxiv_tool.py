# tools/web/arxiv_tool.py
from __future__ import annotations
from typing import Any, Dict, List

from langchain.tools import tool
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

from backend.src.schemas.results import ToolResult, Citation


@tool("arxiv_search")
def arxiv_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search arXiv and return short paper summaries."""
    api = ArxivAPIWrapper(top_k_results=top_k, doc_content_chars_max=1500)
    runner = ArxivQueryRun(api_wrapper=api)
    text = runner.run(query)
    cites: List[Citation] = [Citation(title=f"arXiv search: {query}", url="https://arxiv.org", snippet=text[:300])]
    return ToolResult(task_id="arxiv", kind="web", ok=True, data={"query": query, "text": text}, citations=cites).model_dump()
