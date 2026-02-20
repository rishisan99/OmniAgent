from __future__ import annotations

import os
from typing import Any, Dict

from backend.src.graph.agent_memory import push_note
from backend.src.llm.factory import get_llm


def critic_plan_node(provider: str, model: str):
    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        contract = dict(state.get("response_contract") or {})
        user = str(state.get("user_text", "")).strip()
        writer_plan = str(contract.get("writer_plan", "")).strip()
        if not user:
            return {"response_contract": contract}

        prompt = (
            "Role: Critic Agent.\n"
            "Identify risks of hallucination or irrelevance in the response plan.\n"
            "Return:\n"
            "- risk list (max 3)\n"
            "- one corrective rule\n"
            "Keep concise plain text.\n\n"
            f"User request: {user}\n"
            f"Writer plan:\n{writer_plan}\n"
        )
        try:
            p = os.getenv("INTENT_PROVIDER", provider)
            m = os.getenv("INTENT_MODEL", model)
            llm = get_llm(p, m, streaming=False, temperature=0.0)
            msg = llm.invoke(prompt)
            critic = (getattr(msg, "content", "") or "").strip()
        except Exception:
            critic = "- Risk: unsupported claims\n- Risk: entity mix-up\nRule: only state what context supports."

        contract["critic_checks"] = critic
        return {
            "response_contract": contract,
            "agent_memory": push_note(
                state,
                node="critic_plan",
                summary="Critic checks prepared",
                extra={},
            ),
        }

    return _run
