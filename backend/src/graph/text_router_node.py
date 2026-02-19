# graph/text_router_node.py
from __future__ import annotations
from typing import Any, Dict

from backend.src.schemas.plan import RunPlan


def text_router_node():
    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        plan = RunPlan.model_validate(state["plan"])
        user = (state.get("user_text") or "").lower()

        if not plan.text.enabled:
            return {"plan": plan.model_dump(), "text_instructions": ""}

        if any(k in user for k in ("bullet", "5 points", "points", "bullets")):
            plan.text.style = "bullet"
        elif any(k in user for k in ("detail", "deep", "explain")):
            plan.text.style = "detailed"
        else:
            plan.text.style = "direct"

        plan.text.instruction = f"Answer in style={plan.text.style}."
        return {"plan": plan.model_dump(), "text_instructions": plan.text.instruction}
    return _run
