# tools/web/_smoke_test.py
from __future__ import annotations
import json

from backend.src.core.config import bootstrap_env
bootstrap_env()

from backend.src.tools.web.tavily_tool import tavily_search
from backend.src.tools.web.wiki_tool import wikipedia_search
from backend.src.tools.web.arxiv_tool import arxiv_search

if __name__ == "__main__":
    print(json.dumps(tavily_search.invoke({"query": "LangGraph SSE streaming", "top_k": 3}), indent=2))
    print(json.dumps(wikipedia_search.invoke({"query": "LangChain", "top_k": 2}), indent=2))
    print(json.dumps(arxiv_search.invoke({"query": "retrieval augmented generation", "top_k": 3}), indent=2))
