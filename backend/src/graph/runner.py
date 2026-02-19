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
from backend.src.graph.lanes_node import lanes_node


def build_graph(provider: str, model: str):
    g = StateGraph(AgentState)
    g.add_node("ack", ack_node())
    g.add_node("context", context_node())
    g.add_node("intent", intent_llm_node(provider, model))
    g.add_node("planner", planner_node())
    g.add_node("text_router", text_router_node())
    g.add_node("tool_router", tool_router_node())
    g.add_node("lanes", lanes_node(provider, model))
    g.set_entry_point("ack")
    g.add_edge("ack", "context")
    g.add_edge("context", "intent")
    g.add_edge("intent", "planner")
    g.add_edge("planner", "text_router")
    g.add_edge("text_router", "tool_router")
    g.add_edge("tool_router", "lanes")
    g.add_edge("lanes", END)
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
