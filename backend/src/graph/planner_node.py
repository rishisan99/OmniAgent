from __future__ import annotations
import re
from typing import Any, Dict, Optional

from backend.src.graph.agent_memory import push_note


def _extract_subject(prompt: str) -> Optional[str]:
    s = (prompt or "").strip()
    if not s:
        return None
    m = re.search(r"(?:image|photo|picture)\s+of\s+(.+)$", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" .!?")
    m = re.search(r"\bof\s+(.+)$", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" .!?")
    words = [w for w in re.split(r"\s+", s) if w]
    return " ".join(words[-3:]) if words else None


def planner_node():
    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        intent = state.get("intent") or {}
        plan = state.get("plan") or {}
        flags = (plan.get("flags") or {}) if isinstance(plan, dict) else {}
        linked = state.get("linked_artifact") or {}
        intent_type = intent.get("intent_type", "chat")
        target = intent.get("target_modality", "text")
        confidence = float(intent.get("confidence", 0.6))

        subject_lock = None
        if intent_type == "edit" and target == "image" and linked.get("kind") == "image":
            subject_lock = _extract_subject(linked.get("prompt") or "")

        has_tool_lanes = any(
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
        plan_runtime = {
            "intent_type": intent_type,
            "target_modality": target,
            "confidence": confidence,
            "max_replans": 1 if (intent_type == "edit" and target == "image") else 0,
            "subject_lock": subject_lock,
            "iteration": 0,
            "max_iterations": 2 if has_tool_lanes else 1,
            "max_rewrites": 1,
            "replan_requested": False,
            "replan_reason": "",
        }
        return {
            "plan_runtime": plan_runtime,
            "agent_memory": push_note(
                state,
                node="planner",
                summary="Runtime plan prepared",
                extra={"intent_type": intent_type, "target_modality": target, "confidence": confidence},
            ),
        }

    return _run
