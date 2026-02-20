from __future__ import annotations

from typing import Any, Dict, List, Tuple

from backend.src.graph.agent_memory import push_note


def _task_key(t: Dict[str, Any]) -> Tuple[str, str]:
    kind = str(t.get("kind", ""))
    anchor = str(t.get("query") or t.get("prompt") or t.get("text") or t.get("instruction") or "")
    return kind, anchor.strip().lower()


def task_validate_node():
    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        tasks = list(state.get("tasks") or [])
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
