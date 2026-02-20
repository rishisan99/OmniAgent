# graph/ack_node.py
from __future__ import annotations
from typing import Any, Dict

from backend.src.graph.agent_memory import push_note


def ack_node():
    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "agent_memory": push_note(
                state,
                node="ack",
                summary="Run acknowledged",
            )
        }
    return _run
