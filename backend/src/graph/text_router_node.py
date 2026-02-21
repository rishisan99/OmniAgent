# graph/text_router_node.py
from __future__ import annotations
from typing import Any, Dict

from backend.src.graph.agent_memory import push_note
from backend.src.schemas.plan import RunPlan


def text_router_node():
    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        plan = RunPlan.model_validate(state["plan"])

        if not plan.text.enabled:
            return {
                "plan": plan.model_dump(),
                "text_instructions": "",
                "agent_memory": push_note(
                    state,
                    node="text_router",
                    summary="Text lane disabled",
                    extra={"style": None},
                ),
            }

        plan.text.instruction = (
            "Length policy:\n"
            "- For explanation/overview/definition requests, provide about 1 page (roughly 350-500 words).\n"
            "- For greetings, acknowledgements, or very simple asks, keep it concise (1-4 lines).\n"
            "- For mixed asks, allocate length proportionally and avoid unnecessary verbosity.\n"
            "Format with clear headings and concise bullets when useful."
        )
        return {
            "plan": plan.model_dump(),
            "text_instructions": plan.text.instruction,
            "agent_memory": push_note(
                state,
                node="text_router",
                summary="Text lane configured",
                extra={"style": plan.text.style},
            ),
        }
    return _run
