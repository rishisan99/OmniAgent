from __future__ import annotations

import os
from typing import Any, Dict

from backend.src.core.jsonx import extract_json
from backend.src.graph.agent_memory import push_note
from backend.src.llm.factory import get_llm

PLANNER_SYSTEM_PROMPT = (
    "You are a fast planning assistant for response composition.\n"
    "Return compact, actionable planning signals only.\n"
    "Be concise, grounded, and avoid unnecessary verbosity.\n"
)


def role_pack_node(provider: str, model: str):
    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        user = str(state.get("user_text", "")).strip()
        tasks = [str(t.get("kind")) for t in state.get("tasks", []) if isinstance(t, dict)]
        intent = state.get("intent") or {}
        contract = dict(state.get("response_contract") or {})
        if not user:
            return {"response_contract": contract}
        # Fast path: for media-only turns (doc/image/audio generation), skip planner LLMs.
        # This preserves non-determinism in final generation while reducing pre-stream latency.
        media_kinds = {"doc", "image_gen", "tts"}
        task_set = set(tasks)
        if task_set and task_set.issubset(media_kinds):
            contract["researcher_brief"] = "Prioritize the user's direct explanation request."
            contract["writer_plan"] = "Answer succinctly in markdown. Do not mention tool execution status."
            contract["critic_checks"] = "Avoid unsupported claims; keep response concise."
            return {
                "response_contract": contract,
                "agent_memory": push_note(
                    state,
                    node="role_pack",
                    summary="Role pack fast-path for media-only tasks",
                    extra={"tasks": tasks},
                ),
            }

        role_provider = os.getenv(
            "ROLE_PROVIDER",
            os.getenv("PLANNER_PROVIDER", os.getenv("INTENT_PROVIDER", provider)),
        )
        role_model = os.getenv(
            "ROLE_MODEL",
            os.getenv("PLANNER_MODEL", os.getenv("INTENT_MODEL", model)),
        )
        llm = get_llm(role_provider, role_model, streaming=False, temperature=0.1)
        prompt = (
            "You are producing a compact collaboration contract for a response engine.\n"
            "Return ONLY JSON with keys: researcher_brief, writer_plan, critic_checks.\n"
            "- researcher_brief: max 3 bullets\n"
            "- writer_plan: max 6 lines\n"
            "- critic_checks: max 3 risks + 1 corrective rule\n\n"
            f"User request: {user}\n"
            f"Intent: {intent}\n"
            f"Planned task kinds: {tasks}\n"
        )

        researcher = "- Prioritize directly relevant evidence.\n- Resolve entity ambiguity.\n- Keep concise and grounded."
        writer = "Answer directly with strongest evidence first; keep concise; avoid unsupported claims."
        critic = "- Risk: unsupported claims\n- Risk: entity mix-up\nRule: only state what retrieved evidence supports."
        try:
            msg = llm.invoke(f"{PLANNER_SYSTEM_PROMPT}\n\n{prompt}")
            raw = (getattr(msg, "content", "") or "").strip()
            data = extract_json(raw) if raw else {}
            researcher = str(data.get("researcher_brief", researcher)).strip() or researcher
            writer = str(data.get("writer_plan", writer)).strip() or writer
            critic = str(data.get("critic_checks", critic)).strip() or critic
        except Exception:
            pass

        contract["researcher_brief"] = researcher
        contract["writer_plan"] = writer
        contract["critic_checks"] = critic
        return {
            "response_contract": contract,
            "agent_memory": push_note(
                state,
                node="role_pack",
                summary="Parallel role pack prepared",
                extra={"tasks": tasks, "provider": role_provider, "model": role_model},
            ),
        }

    return _run
