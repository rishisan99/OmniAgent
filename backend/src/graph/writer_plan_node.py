from __future__ import annotations

import os
from typing import Any, Dict

from backend.src.graph.agent_memory import push_note
from backend.src.llm.factory import get_llm


def writer_plan_node(provider: str, model: str):
    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        user = str(state.get("user_text", "")).strip()
        contract = dict(state.get("response_contract") or {})
        researcher = str(contract.get("researcher_brief", "")).strip()
        if not user:
            return {"response_contract": contract}

        prompt = (
            "Role: Writer Planner Agent.\n"
            "Create a compact response plan with:\n"
            "1) answer shape\n"
            "2) key points to include\n"
            "3) brevity/style guidance\n"
            "Return plain text, max 6 lines.\n\n"
            f"User request: {user}\n"
            f"Researcher brief:\n{researcher}\n"
        )
        try:
            p = os.getenv("INTENT_PROVIDER", provider)
            m = os.getenv("INTENT_MODEL", model)
            llm = get_llm(p, m, streaming=False, temperature=0.1)
            msg = llm.invoke(prompt)
            plan_text = (getattr(msg, "content", "") or "").strip()
        except Exception:
            plan_text = "Answer directly using strongest evidence first; keep concise; avoid unsupported claims."

        contract["writer_plan"] = plan_text
        return {
            "response_contract": contract,
            "agent_memory": push_note(
                state,
                node="writer_plan",
                summary="Writer plan prepared",
                extra={},
            ),
        }

    return _run
