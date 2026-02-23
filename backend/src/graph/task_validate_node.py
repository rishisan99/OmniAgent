from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from backend.src.graph.agent_memory import push_note


def _task_key(t: Dict[str, Any]) -> Tuple[str, str]:
    kind = str(t.get("kind", ""))
    anchor = str(t.get("query") or t.get("prompt") or t.get("text") or t.get("instruction") or "")
    return kind, anchor.strip().lower()


def task_validate_node():
    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        tasks = list(state.get("tasks") or [])
        user_text = str(state.get("user_text") or "")
        user_l = user_text.lower()
        explicit_image_gen = bool(
            re.search(
                r"(?:generate|create|make)\s+(?:an?\s+)?(?:image|picture|photo)\b|(?:image|picture|photo)\s+of\b",
                user_l,
            )
        )
        explicit_web_ask = any(
            k in user_l
            for k in ("web", "internet", "online", "news", "headline", "headlines", "search", "arxiv", "paper", "research")
        )
        explicit_text_ask = any(
            k in user_l
            for k in ("explain", "describe", "summarize", "summary", "tell me", "what is", "why", "how")
        )
        cleaned: List[Dict[str, Any]] = []
        seen = set()
        dropped = 0
        for t in tasks:
            if not isinstance(t, dict) or not t.get("kind"):
                dropped += 1
                continue
            k = _task_key(t)
            if k in seen:
                dropped += 1
                continue
            seen.add(k)
            item = dict(t)
            if item.get("kind") in {"web", "rag", "kb_rag"}:
                try:
                    item["top_k"] = max(1, min(int(item.get("top_k", 5)), 8))
                except Exception:
                    item["top_k"] = 5
            cleaned.append(item)

        # Deterministic guardrail: pure image-generation asks should not pull web/arxiv lanes.
        if explicit_image_gen and not explicit_web_ask and not explicit_text_ask:
            has_image_gen = any(t.get("kind") == "image_gen" for t in cleaned)
            if has_image_gen:
                before = len(cleaned)
                cleaned = [t for t in cleaned if t.get("kind") not in {"web", "rag", "kb_rag"}]
                dropped += max(0, before - len(cleaned))

        return {
            "tasks": cleaned,
            "agent_memory": push_note(
                state,
                node="task_validate",
                summary="Tasks validated",
                extra={"input": len(tasks), "output": len(cleaned), "dropped": dropped},
            ),
        }

    return _run
