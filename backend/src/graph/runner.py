# graph/runner.py
from __future__ import annotations
from typing import Any, Dict, Callable, Optional

from langgraph.graph import StateGraph, END

from backend.src.schemas.state import AgentState
from backend.src.stream.emitter import Emitter
from backend.src.graph.ack_node import ack_node
from backend.src.graph.context_node import context_node
from backend.src.graph.intent_llm_node import intent_llm_node
from backend.src.graph.planner_node import planner_node
from backend.src.graph.text_router_node import text_router_node
from backend.src.graph.tool_router_node import tool_router_node
from backend.src.graph.task_validate_node import task_validate_node
from backend.src.graph.role_pack_node import role_pack_node
from backend.src.graph.lanes_node import lanes_node
from backend.src.graph.reflect_node import reflect_node
from backend.src.schemas.plan import RunPlan


def _needs_tools(plan: RunPlan) -> bool:
    flags = plan.flags or {}
    return any(
        bool(flags.get(k))
        for k in (
            "needs_web",
            "needs_rag",
            "needs_kb_rag",
            "needs_doc",
            "needs_vision",
            "needs_tts",
            "needs_image_gen",
        )
    )


def _route_after_planner(state: Dict[str, Any]) -> str:
    plan = RunPlan.model_validate(state.get("plan") or {"mode": "text_only", "text": {"enabled": True}})
    if plan.text.enabled:
        return "text_router"
    if _needs_tools(plan):
        return "tool_router"
    return "lanes"


def _route_after_text_router(state: Dict[str, Any]) -> str:
    plan = RunPlan.model_validate(state.get("plan") or {"mode": "text_only", "text": {"enabled": True}})
    return "tool_router" if _needs_tools(plan) else "lanes"


def _route_after_reflect(state: Dict[str, Any]) -> str:
    runtime = dict(state.get("plan_runtime") or {})
    return "tool_router" if bool(runtime.get("replan_requested")) else "end"


def build_graph(provider: str, model: str):
    g = StateGraph(AgentState)
    g.add_node("ack", ack_node())
    g.add_node("context", context_node())
    g.add_node("intent", intent_llm_node(provider, model))
    g.add_node("planner", planner_node())
    g.add_node("text_router", text_router_node())
    g.add_node("tool_router", tool_router_node())
    g.add_node("task_validate", task_validate_node())
    g.add_node("role_pack", role_pack_node(provider, model))
    g.add_node("lanes", lanes_node(provider, model))
    g.add_node("reflect", reflect_node())
    g.set_entry_point("ack")
    g.add_edge("ack", "context")
    g.add_edge("context", "intent")
    g.add_edge("intent", "planner")
    g.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "text_router": "text_router",
            "tool_router": "tool_router",
            "lanes": "lanes",
        },
    )
    g.add_conditional_edges(
        "text_router",
        _route_after_text_router,
        {
            "tool_router": "tool_router",
            "lanes": "lanes",
        },
    )
    g.add_edge("tool_router", "task_validate")
    g.add_edge("task_validate", "role_pack")
    g.add_edge("role_pack", "lanes")
    g.add_edge("lanes", "reflect")
    g.add_conditional_edges(
        "reflect",
        _route_after_reflect,
        {
            "tool_router": "tool_router",
            "end": END,
        },
    )
    return g.compile()


async def run_graph(app, state: Dict[str, Any], send: Callable[[Dict[str, Any]], None], run_id: str, trace_id: Optional[str] = None):
    em = Emitter(run_id=run_id, trace_id=trace_id, send=send)
    em.emit("run_start", {"session_id": state.get("session_id")})
    state["emitter"] = em
    try:
        out = await app.ainvoke(state)
        em.emit("run_end", {"ok": True})
        return out
    except Exception as e:
        em.emit("error", {"error": str(e)})
        em.emit("run_end", {"ok": False})
        return {"final_text": "", "tool_outputs": {}, "error": str(e)}
