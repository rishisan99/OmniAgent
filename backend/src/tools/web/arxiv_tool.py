# tools/web/arxiv_tool.py
from __future__ import annotations
import re
from typing import Any, Dict, List

from langchain.tools import tool
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
import arxiv as arxiv_py

from backend.src.schemas.results import ToolResult, Citation


GENAI_HINT_TERMS = {
    "gen ai",
    "genai",
    "generative ai",
    "foundation model",
    "foundation models",
    "large language model",
    "large language models",
    "llm",
    "llms",
}

GENAI_BOOST_TERMS = [
    "generative ai",
    "generative model",
    "foundation model",
    "large language model",
    "llm",
    "diffusion",
    "text-to-image",
    "text to image",
    "image generation",
    "prompting",
    "instruction tuning",
    "rlhf",
    "rlaif",
    "multimodal",
]


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower())).strip()


def _clean_topic_query(raw: str) -> str:
    t = _normalize_whitespace(raw)
    # Strip common command wrappers so content query remains.
    wrappers = [
        r"^(?:can you|could you|please)\s+",
        r"^(?:find|search|show|get|list)\s+(?:me\s+)?",
        r"^(?:the\s+)?(?:research\s+)?papers?\s*(?:on|about|for|:)\s+",
        r"^(?:from\s+arxiv[,:\s]+)",
        r"^(?:in\s+arxiv[,:\s]+)",
    ]
    out = t
    for p in wrappers:
        out = re.sub(p, "", out, flags=re.IGNORECASE).strip()
    return _normalize_whitespace(out or t)


def _extract_title_hint(topic: str) -> str:
    # Prefer explicitly quoted title when present.
    m = re.search(r'"([^"]{6,})"', topic)
    if m:
        return _normalize_whitespace(m.group(1))
    m = re.search(r"'([^']{6,})'", topic)
    if m:
        return _normalize_whitespace(m.group(1))

    # Handle common explicit patterns.
    p = re.search(r"(?:paper|research paper|title)\s*:\s*(.+)$", topic, flags=re.IGNORECASE)
    if p and p.group(1).strip():
        return _normalize_whitespace(p.group(1))

    # If user likely asked for one specific paper title.
    low = topic.lower()
    if any(k in low for k in ("paper", "research paper", "find me", "can find me")) and len(topic.split()) >= 4:
        return _normalize_whitespace(topic)

    return ""


def _topic_terms(topic: str) -> List[str]:
    s = re.sub(r"[^a-z0-9\s\-]+", " ", topic.lower())
    raw = [t for t in re.split(r"\s+", s) if t]
    stop = {
        "the",
        "a",
        "an",
        "in",
        "on",
        "for",
        "about",
        "of",
        "to",
        "and",
        "paper",
        "papers",
        "research",
        "recent",
        "latest",
        "find",
        "me",
        "can",
        "you",
        "please",
        "show",
        "list",
        "get",
        "search",
        "from",
        "arxiv",
        "is",
        "this",
        "that",
        "with",
        "using",
        "by",
        "at",
        "as",
    }
    return [t for t in raw if t not in stop and len(t) >= 2]


def _is_genai_intent(topic: str) -> bool:
    t = topic.lower()
    return any(h in t for h in GENAI_HINT_TERMS)


def _build_effective_query(topic: str, year: int | None, title_hint: str = "") -> str:
    clean_topic = _normalize_whitespace(topic)
    if title_hint:
        t = _normalize_whitespace(title_hint)
        api_query = f'ti:"{t}" OR all:"{t}"'
    elif _is_genai_intent(clean_topic):
        # Bias toward CS GenAI literature while preserving recency.
        genai_clause = (
            '(all:"generative ai" OR all:"large language model" OR all:llm OR '
            'all:"foundation model" OR all:diffusion OR all:"text-to-image" OR all:multimodal)'
        )
        api_query = f"cat:cs.* AND {genai_clause}"
    else:
        api_query = f"all:{clean_topic}"
    if year:
        api_query = f"{api_query} AND submittedDate:[{year}01010000 TO {year}12312359]"
    return api_query


def _score_row(
    row: Dict[str, Any],
    topic_terms: List[str],
    genai_intent: bool,
    title_hint: str = "",
) -> int:
    title = str(row.get("title", "")).lower()
    summary = str(row.get("summary", "")).lower()
    score = 0
    norm_title = _normalize_for_match(title)
    norm_hint = _normalize_for_match(title_hint)

    if norm_hint:
        if norm_title == norm_hint:
            score += 1000
        elif norm_hint in norm_title:
            score += 450
        hint_tokens = [t for t in norm_hint.split() if len(t) >= 3]
        if hint_tokens:
            overlap = sum(1 for t in hint_tokens if t in norm_title)
            score += int((overlap / len(hint_tokens)) * 250)

    for term in topic_terms:
        if term in title:
            score += 5
        elif term in summary:
            score += 2

    if genai_intent:
        for term in GENAI_BOOST_TERMS:
            if term in title:
                score += 6
            elif term in summary:
                score += 3

    # Small recency bonus from submittedDate sorting order.
    if str(row.get("published", "")):
        score += 1
    return score


def _rank_and_filter(rows: List[Dict[str, Any]], topic: str, top_k: int, title_hint: str = "") -> List[Dict[str, Any]]:
    terms = _topic_terms(topic)
    genai_intent = _is_genai_intent(topic)
    scored = [(r, _score_row(r, terms, genai_intent, title_hint=title_hint)) for r in rows]
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)

    # Keep positive-signal entries first.
    if title_hint:
        # For exact-title requests, require stronger signal.
        filtered = [r for r, s in ranked if s >= 120]
    else:
        filtered = [r for r, s in ranked if s > 0]
    if len(filtered) < max(1, top_k):
        # If query is very broad/noisy, return best-effort ranked recents.
        filtered = [r for r, _ in ranked]

    return filtered[: max(1, int(top_k))]


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

    topic = _clean_topic_query(topic)
    title_hint = _extract_title_hint(topic)
    api_query = _build_effective_query(topic, year, title_hint=title_hint)

    try:
        search = arxiv_py.Search(
            query=api_query,
            max_results=max(15, int(top_k) * 6),
            sort_by=(arxiv_py.SortCriterion.Relevance if title_hint else arxiv_py.SortCriterion.SubmittedDate),
        )
        client = arxiv_py.Client(page_size=max(15, int(top_k) * 6), delay_seconds=0.0, num_retries=2)
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
        rows = _rank_and_filter(rows, topic, top_k, title_hint=title_hint)
        allowed_urls = {x.get("url") for x in rows}
        cites = [c for c in cites if c.url in allowed_urls][: max(1, int(top_k))]
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
