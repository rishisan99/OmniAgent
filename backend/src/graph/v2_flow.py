from __future__ import annotations

import asyncio
import os
import time
import re
from typing import Any, Dict, List, Optional

from backend.src.agents.router import run_task
from backend.src.core.logging import get_logger
from backend.src.graph.agent_memory import push_note
from backend.src.graph.context_node import context_node
from backend.src.graph.intent_llm_node import intent_llm_node
from backend.src.graph.planner_node import planner_node
from backend.src.graph.role_pack_node import role_pack_node
from backend.src.graph.streaming import stream_tokens
from backend.src.graph.task_validate_node import task_validate_node
from backend.src.graph.text_router_node import text_router_node
from backend.src.graph.tool_router_node import tool_router_node
from backend.src.llm.factory import get_llm
from backend.src.schemas.plan import RunPlan
from backend.src.stream.emitter import Emitter

_GREETING_ONLY_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|sup|what'?s up|good\s+morning|good\s+afternoon|good\s+evening)[!. ]*$",
    re.IGNORECASE,
)
_EXPLICIT_TOOL_CUE_RE = re.compile(
    r"\b("
    r"generate|create|make|draw|image|photo|picture|audio|voice|tts|speak|read aloud|"
    r"pdf|document|docx|upload|file|attachment|"
    r"web|internet|news|headline|headlines|search|find|fetch|google|wikipedia|"
    r"arxiv|paper|preprint|latest|recent|current|today"
    r")\b",
    re.IGNORECASE,
)
logger = get_logger("omniagent.graph.v2")


def _task_title(task: Dict[str, Any]) -> str:
    kind = str(task.get("kind", ""))
    if kind == "web":
        sources = task.get("sources") or []
        if sources == ["arxiv"]:
            return "Results from Arxiv"
        return "Results from Web"
    return {
        "rag": "RAG Context",
        "kb_rag": "Knowledge Base",
        "image_gen": "Generated Image",
        "tts": "Generated Audio",
        "doc": "Generated Document",
        "vision": "Vision Analysis",
    }.get(kind, kind)


def _history_text(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    return "\n".join(f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in history[-12:])


def _tool_context_text(outs: Dict[str, Any]) -> str:
    rows: List[str] = []
    for out in outs.values():
        if not isinstance(out, dict) or not out.get("ok"):
            continue
        kind = str(out.get("kind", ""))
        data = dict(out.get("data") or {})
        support = str(data.get("support_summary", "")).strip()
        if support:
            rows.append(f"{kind.upper()} SUMMARY:\n{support}")
        if kind in {"rag", "kb_rag"}:
            matches = data.get("matches", []) if isinstance(data.get("matches"), list) else []
            snippets = [str(m.get("text", "")).strip()[:380] for m in matches[:4] if isinstance(m, dict)]
            if snippets:
                rows.append(f"{kind.upper()} SNIPPETS:\n" + "\n---\n".join(snippets))
        if kind == "web":
            parts = data.get("parts", []) if isinstance(data.get("parts"), list) else []
            urls: List[str] = []
            for p in parts:
                if not isinstance(p, dict):
                    continue
                pd = p.get("data") or {}
                items = pd.get("items") if isinstance(pd.get("items"), list) else []
                for it in items[:5]:
                    if isinstance(it, dict) and it.get("url"):
                        urls.append(str(it["url"]))
            if urls:
                rows.append("WEB URLS:\n" + "\n".join(f"- {u}" for u in urls[:8]))
    return "\n\n".join(rows)


def _can_fast_text_path(state: Dict[str, Any]) -> bool:
    user = str(state.get("user_text") or "").strip()
    if not user:
        return False
    attachments = state.get("attachments") or []
    if attachments:
        return False
    if _EXPLICIT_TOOL_CUE_RE.search(user):
        return False
    return True


async def _supportive_summary(
    kind: str,
    task: Dict[str, Any],
    out: Dict[str, Any],
    provider: str,
    model: str,
) -> str:
    support_provider = os.getenv("SUPPORT_PROVIDER", os.getenv("PLANNER_PROVIDER", provider))
    support_model = os.getenv("SUPPORT_MODEL", os.getenv("PLANNER_MODEL", model))
    lane_model = {
        "web": os.getenv("WEB_SUPPORT_MODEL", support_model),
        "rag": os.getenv("RAG_SUPPORT_MODEL", support_model),
        "kb_rag": os.getenv("RAG_SUPPORT_MODEL", support_model),
        "vision": os.getenv("VISION_SUPPORT_MODEL", support_model),
    }.get(kind, support_model)
    llm = get_llm(support_provider, lane_model, streaming=False, temperature=0.1)
    data = out.get("data") or {}
    prompt = (
        "Summarize this lane output for the main responder.\n"
        "Return concise markdown with only grounded facts (max 6 lines).\n\n"
        f"Lane kind: {kind}\n"
        f"User query: {task.get('query') or task.get('prompt') or ''}\n"
        f"Lane output data:\n{str(data)[:7000]}\n"
    )
    try:
        msg = llm.invoke(prompt)
        return (getattr(msg, "content", "") or "").strip()
    except Exception:
        return ""


def _checker_payload(tasks: List[Dict[str, Any]], outs: Dict[str, Any], final_text: str) -> Dict[str, Any]:
    requested = len(tasks)
    completed = 0
    failed = 0
    for t in tasks:
        out = outs.get(str(t.get("id")))
        if not isinstance(out, dict):
            continue
        if out.get("ok"):
            completed += 1
        else:
            failed += 1
    return {
        "requested_tasks": requested,
        "completed_tasks": completed,
        "failed_tasks": failed,
        "has_main_text": bool((final_text or "").strip()),
    }


async def run_graph_v2(
    state: Dict[str, Any],
    send,
    run_id: str,
    trace_id: Optional[str],
    provider: str,
    model: str,
) -> Dict[str, Any]:
    logger.info(
        "NODE_CALL node=run_graph_v2 run_id=%s trace_id=%s session_id=%s provider=%s model=%s",
        run_id,
        trace_id,
        state.get("session_id"),
        provider,
        model,
    )
    em = Emitter(run_id=run_id, trace_id=trace_id, send=send)
    em.emit("run_start", {"session_id": state.get("session_id"), "graph_version": "v2"})
    state["emitter"] = em
    state["agent_memory"] = push_note(state, node="initial_message", summary="Initial message node entered", extra={})
    try:
        if _can_fast_text_path(state):
            logger.info("NODE_CALL node=fast_text_path run_id=%s", run_id)
            history = list(state.get("chat_history") or [])
            query_text = str(state.get("user_text", ""))
            text_provider = os.getenv("TEXT_PROVIDER", provider)
            text_model = os.getenv("TEXT_MODEL", model)
            prompt = (
                "You are OmniAgent. Reply directly in markdown.\n"
                "Keep it concise, clear, and grounded.\n"
                + (
                    "This turn is a greeting/social opener.\n"
                    "Reply with exactly one short friendly sentence (max 14 words), no headings.\n"
                    if _GREETING_ONLY_RE.match(query_text)
                    else ""
                )
                + f"Conversation so far:\n{_history_text(history[-4:])}\n\n"
                + f"User message:\n{query_text}\n"
            )
            final_text = await stream_tokens(prompt, em, provider=text_provider, model=text_model, temperature=0.2)
            checker = _checker_payload([], {}, final_text)
            em.emit("run_end", {"ok": True, "graph_version": "v2"})
            return {
                "final_text": final_text,
                "tool_outputs": {},
                "last_image_prompt": state.get("last_image_prompt"),
                "artifact_memory": state.get("artifact_memory") or {},
                "checker": checker,
                "agent_memory": push_note(
                    state,
                    node="fast_text_path",
                    summary="Fast text-only path completed",
                    extra={"text_len": len(final_text or "")},
                ),
            }

        for node_name, node in [
            ("context", context_node()),
            ("planner_intent", intent_llm_node(provider, model)),
            ("planner_runtime", planner_node()),
            ("text_router", text_router_node()),
            ("tool_router", tool_router_node()),
            ("task_validate", task_validate_node()),
            ("role_pack", role_pack_node(provider, model)),
        ]:
            logger.info("NODE_CALL node=%s run_id=%s", node_name, run_id)
            updates = node(state) or {}
            if isinstance(updates, dict):
                state.update(updates)
            logger.info("NODE_DONE node=%s run_id=%s update_keys=%s", node_name, run_id, ",".join(sorted((updates or {}).keys())))
            state["agent_memory"] = push_note(state, node=node_name, summary=f"{node_name} completed", extra={})

        plan = RunPlan.model_validate(state.get("plan") or {"mode": "text_only", "text": {"enabled": True}})
        tasks: List[Dict[str, Any]] = list(state.get("tasks") or [])
        history = list(state.get("chat_history") or [])
        response_contract = dict(state.get("response_contract") or {})
        outs: Dict[str, Any] = {}
        artifact_memory = dict(
            state.get("artifact_memory")
            or {"image": None, "audio": None, "doc": None, "lineage": {"image": [], "audio": [], "doc": []}}
        )
        artifact_memory.setdefault("lineage", {"image": [], "audio": [], "doc": []})
        lineage = artifact_memory["lineage"]
        linked_artifact = state.get("linked_artifact") or {}
        last_image_prompt = state.get("last_image_prompt")

        for t in tasks:
            em.emit("block_start", {"block_id": t["id"], "title": _task_title(t), "kind": t.get("kind")})

        knowledge_kinds = {"web", "rag", "kb_rag", "vision"}
        knowledge_tasks = [t for t in tasks if t.get("kind") in knowledge_kinds]
        other_tasks = [t for t in tasks if t.get("kind") not in knowledge_kinds]
        # Always produce streamed final text for knowledge/research tasks, even when planner mode is tools_only.
        should_emit_text = bool(plan.text.enabled or knowledge_tasks)
        logger.info(
            "PLAN_SUMMARY run_id=%s mode=%s text_enabled=%s tasks=%s should_emit_text=%s",
            run_id,
            plan.mode,
            plan.text.enabled,
            [str(t.get("kind")) for t in tasks],
            should_emit_text,
        )

        async def run_one(t: Dict[str, Any]) -> None:
            nonlocal last_image_prompt
            image_timeout_sec = float(os.getenv("IMAGE_TASK_TIMEOUT_SEC", "90"))
            logger.info(
                "TASK_START run_id=%s task_id=%s kind=%s",
                run_id,
                t.get("id"),
                t.get("kind"),
            )
            try:
                if str(t.get("kind", "")) == "image_gen":
                    out = await asyncio.wait_for(
                        asyncio.to_thread(run_task, t, state, em, provider, model),
                        timeout=max(1.0, image_timeout_sec),
                    )
                else:
                    out = await asyncio.to_thread(run_task, t, state, em, provider, model)
            except asyncio.TimeoutError:
                out = {
                    "task_id": t.get("id"),
                    "kind": t.get("kind"),
                    "ok": False,
                    "error": f"Image generation timed out after {int(max(1.0, image_timeout_sec))}s",
                }
            except Exception as e:
                out = {"task_id": t.get("id"), "kind": t.get("kind"), "ok": False, "error": str(e)}
                logger.exception("TASK_ERROR run_id=%s task_id=%s kind=%s error=%s", run_id, t.get("id"), t.get("kind"), e)

            kind = str(t.get("kind", ""))
            if out.get("ok") and kind in {"web", "rag", "kb_rag", "vision"}:
                support = await _supportive_summary(kind, t, out, provider=provider, model=model)
                if support:
                    data = dict(out.get("data") or {})
                    data["support_summary"] = support
                    data["text"] = support
                    data["mime"] = "text/markdown"
                    out["data"] = data

            if kind == "image_gen" and out.get("ok"):
                data = dict(out.get("data") or {})
                prompt = str(data.get("prompt") or t.get("prompt") or "").strip()
                if prompt:
                    last_image_prompt = prompt
                artifact_memory["image"] = {"id": data.get("filename"), "url": data.get("url"), "prompt": prompt}
                parent_id = linked_artifact.get("id") if linked_artifact.get("kind") == "image" else None
                child_id = data.get("filename")
                if parent_id and child_id and parent_id != child_id:
                    lineage["image"].append(
                        {"parent_id": parent_id, "child_id": child_id, "op": "edit", "ts_ms": int(time.time() * 1000)}
                    )
            if kind == "tts" and out.get("ok"):
                data = dict(out.get("data") or {})
                artifact_memory["audio"] = {"id": data.get("filename"), "url": data.get("url"), "text": str(t.get("text", "")).strip()}
            if kind == "doc" and out.get("ok"):
                data = dict(out.get("data") or {})
                artifact_memory["doc"] = {
                    "id": data.get("filename"),
                    "url": data.get("url"),
                    "text": str(data.get("text", ""))[:2000],
                }

            outs[str(t.get("id"))] = out
            logger.info(
                "TASK_DONE run_id=%s task_id=%s kind=%s ok=%s data_keys=%s",
                run_id,
                t.get("id"),
                kind,
                out.get("ok", False),
                ",".join(sorted((out.get("data") or {}).keys())) if isinstance(out.get("data"), dict) else "",
            )
            em.emit("block_end", {"block_id": t.get("id"), "payload": out})

        async def run_group(group: List[Dict[str, Any]]) -> None:
            if group:
                await asyncio.gather(*[run_one(t) for t in group])

        knowledge_job = asyncio.create_task(run_group(knowledge_tasks)) if knowledge_tasks else None
        other_job = asyncio.create_task(run_group(other_tasks)) if other_tasks else None

        final_text = ""
        if should_emit_text:
            # Wait for knowledge lanes first when main text depends on retrieval context.
            if knowledge_job:
                await knowledge_job
                knowledge_job = None
            text_provider = os.getenv("TEXT_PROVIDER", provider)
            text_model = os.getenv("TEXT_MODEL", model)
            context = _tool_context_text(outs)
            query_text = str(state.get("text_query") or state.get("user_text", ""))
            length_rules = (
                "Length policy:\n"
                "- Explanation/overview/definition requests: target about 1 page (roughly 350-500 words).\n"
                "- Greetings, acknowledgements, or very simple asks: keep concise (1-4 lines).\n"
                "- Mixed asks: allocate length proportionally and avoid filler.\n"
            )
            no_tool_tasks = not bool(tasks)
            recent_history = history[-4:] if no_tool_tasks else history
            kb_no_exact_entity = False
            kb_missing_name = ""
            for v in outs.values():
                if not isinstance(v, dict) or v.get("kind") != "kb_rag" or not v.get("ok"):
                    continue
                d = v.get("data") or {}
                matches = d.get("matches") if isinstance(d.get("matches"), list) else []
                if d.get("entity_not_found") and not matches:
                    kb_no_exact_entity = True
                    kb_missing_name = str(d.get("entity_not_found") or "").strip()
                    break
            if kb_no_exact_entity:
                miss = kb_missing_name or "the requested employee"
                final_text = await stream_tokens(
                    (
                        "Return exactly this markdown.\n\n"
                        "## Knowledge Base Result\n\n"
                        f'No exact record was found for "{miss}" in the Insurellm knowledge base.\n\n'
                        "Try the full official name or verify spelling.\n"
                    ),
                    em,
                    provider=text_provider,
                    model=text_model,
                    temperature=0.0,
                )
                if other_job:
                    await other_job
                if knowledge_job:
                    await knowledge_job
                checker = _checker_payload(tasks, outs, final_text)
                state["checker"] = checker
                state["agent_memory"] = push_note(state, node="checker", summary="Checker completed", extra=checker)
                if tasks:
                    em.emit(
                        "block_start",
                        {"block_id": "__meta_conclusion__", "title": "Conclusion", "kind": "meta_conclusion"},
                    )
                    em.emit(
                        "block_end",
                        {
                            "block_id": "__meta_conclusion__",
                            "payload": {
                                "ok": True,
                                "kind": "meta_conclusion",
                                "data": {"text": "Completed. Results are shown above.", "mime": "text/markdown", "checker": checker},
                            },
                        },
                    )
                em.emit("run_end", {"ok": True, "graph_version": "v2"})
                return {
                    "final_text": final_text,
                    "tool_outputs": outs,
                    "last_image_prompt": last_image_prompt,
                    "artifact_memory": artifact_memory,
                    "checker": checker,
                    "agent_memory": state.get("agent_memory"),
                }
            prompt = (
                "You are OmniAgent. Answer directly in markdown.\n"
                + (
                    "Keep response lightweight and direct.\n"
                    if no_tool_tasks
                    else f"{length_rules}"
                )
                + (
                    "Use plain markdown with minimal structure.\n"
                    if no_tool_tasks
                    else "Prefer short headings and concise bullets.\n"
                )
                + (
                    ""
                    if no_tool_tasks
                    else "Avoid long paragraphs (>3 lines each).\n"
                )
                + (
                    "This turn is a greeting/social opener.\n"
                    "Reply with exactly one short friendly sentence (max 14 words), no headings.\n"
                    if _GREETING_ONLY_RE.match(query_text)
                    else ""
                )
                + "If tool outputs are present, treat them as completed and avoid status chatter.\n"
                + "Never claim inability such as 'I can't create images/audio/documents'.\n"
                + "Do not invent URLs. Use only URLs present in context.\n"
                + f"{state.get('text_instructions', '')}\n\n"
                + (
                    "Planner contract:\n"
                    f"Researcher brief:\n{response_contract.get('researcher_brief', '')}\n\n"
                    f"Writer plan:\n{response_contract.get('writer_plan', '')}\n\n"
                    f"Critic checks:\n{response_contract.get('critic_checks', '')}\n\n"
                    if response_contract and not no_tool_tasks
                    else ""
                )
                + f"Conversation so far:\n{_history_text(recent_history)}\n\n"
                + (f"Tool context:\n{context}\n\n" if context else "")
                + (
                    "KB_RAG indicates no exact entity match for the requested name.\n"
                    "State clearly that no exact record was found in the knowledge base and do not speculate.\n\n"
                    if kb_no_exact_entity
                    else ""
                )
                + f"User message:\n{query_text}\n"
            )
            final_text = await stream_tokens(prompt, em, provider=text_provider, model=text_model, temperature=0.2)
            if any(t.get("kind") in {"image_gen", "tts", "doc"} for t in tasks):
                banned = ("can't create", "cannot create", "unable to create", "i can't create", "i cannot create")
                low = (final_text or "").lower()
                if any(b in low for b in banned):
                    final_text = ""

        if other_job:
            await other_job
        if knowledge_job:
            await knowledge_job

        checker = _checker_payload(tasks, outs, final_text)
        state["checker"] = checker
        state["agent_memory"] = push_note(state, node="checker", summary="Checker completed", extra=checker)

        if tasks:
            em.emit(
                "block_start",
                {"block_id": "__meta_conclusion__", "title": "Conclusion", "kind": "meta_conclusion"},
            )
            em.emit(
                "block_end",
                {
                    "block_id": "__meta_conclusion__",
                    "payload": {
                        "ok": True,
                        "kind": "meta_conclusion",
                        "data": {"text": "Completed. Results are shown above.", "mime": "text/markdown", "checker": checker},
                    },
                },
            )
            state["agent_memory"] = push_note(state, node="final_message", summary="Final message emitted", extra={})

        em.emit("run_end", {"ok": True, "graph_version": "v2"})
        return {
            "final_text": final_text,
            "tool_outputs": outs,
            "last_image_prompt": last_image_prompt,
            "artifact_memory": artifact_memory,
            "checker": checker,
            "agent_memory": state.get("agent_memory"),
        }
    except Exception as e:
        em.emit("error", {"error": str(e), "graph_version": "v2"})
        em.emit("run_end", {"ok": False, "graph_version": "v2"})
        return {"final_text": "", "tool_outputs": {}, "error": str(e)}
