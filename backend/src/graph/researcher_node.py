from __future__ import annotations

import os
from typing import Any, Dict

from backend.src.graph.agent_memory import push_note
from backend.src.llm.factory import get_llm


def researcher_node(provider: str, model: str):
    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        user = str(state.get("user_text", "")).strip()
        tasks = [str(t.get("kind")) for t in state.get("tasks", []) if isinstance(t, dict)]
        intent = state.get("intent") or {}
        contract = dict(state.get("response_contract") or {})
        if not user:
            return {"response_contract": contract}

        prompt = (
            "Role: Researcher Agent.\n"
            "Goal: identify what evidence should be prioritized for this user request.\n"
            "Return 3 concise bullets only. No markdown heading.\n\n"
            f"User request: {user}\n"
            f"Intent: {intent}\n"
            f"Planned task kinds: {tasks}\n"
        )
        try:
            p = os.getenv("INTENT_PROVIDER", provider)
            m = os.getenv("INTENT_MODEL", model)
            llm = get_llm(p, m, streaming=False, temperature=0.0)
            msg = llm.invoke(prompt)
            brief = (getattr(msg, "content", "") or "").strip()
        except Exception:
            brief = "- Prioritize directly relevant retrieved evidence.\n- Resolve entity ambiguity.\n- Keep response concise and grounded."

        contract["researcher_brief"] = brief
        return {
            "response_contract": contract,
            "agent_memory": push_note(
                state,
                node="researcher",
                summary="Research brief prepared",
                extra={"task_kinds": tasks},
            ),
        }

    return _run
