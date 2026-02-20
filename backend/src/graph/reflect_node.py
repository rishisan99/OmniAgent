from __future__ import annotations

from typing import Any, Dict, List

from backend.src.graph.agent_memory import push_note
from backend.src.schemas.plan import RunPlan


def reflect_node():
    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        runtime = dict(state.get("plan_runtime") or {})
        iteration = int(runtime.get("iteration", 0)) + 1
        max_iterations = int(runtime.get("max_iterations", 1))
        tasks: List[Dict[str, Any]] = list(state.get("tasks") or [])
        outs: Dict[str, Any] = dict(state.get("tool_outputs") or {})
        final_text = str(state.get("final_text") or "").strip()
        plan = RunPlan.model_validate(state.get("plan") or {"mode": "text_only", "text": {"enabled": True}})

        task_ids = [str(t.get("id")) for t in tasks if isinstance(t, dict)]
        success = 0
        failed = 0
        for tid in task_ids:
            out = outs.get(tid)
            if not isinstance(out, dict):
                continue
            if out.get("ok"):
                success += 1
            else:
                failed += 1

        replan_requested = False
        replan_reason = ""

        # Bounded retry policy: one more iteration if tools were requested and all failed.
        if tasks and success == 0 and failed > 0 and iteration < max_iterations:
            replan_requested = True
            replan_reason = "all_tools_failed_retry_once"
            plan.text.enabled = True
            plan.mode = "text_plus_tools"
            flags = dict(plan.flags or {})
            # If KB retrieval failed, try web fallback once.
            had_kb = any(str(t.get("kind")) == "kb_rag" for t in tasks)
            if had_kb:
                flags["needs_web"] = True
            plan.flags = flags

        runtime.update(
            {
                "iteration": iteration,
                "replan_requested": replan_requested,
                "replan_reason": replan_reason,
                "success_count": success,
                "failed_count": failed,
                "has_final_text": bool(final_text),
            }
        )
        return {
            "plan": plan.model_dump(),
            "plan_runtime": runtime,
            "agent_memory": push_note(
                state,
                node="reflect",
                summary="Reflection complete",
                extra={
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "success": success,
                    "failed": failed,
                    "replan_requested": replan_requested,
                    "reason": replan_reason,
                },
            ),
        }

    return _run
