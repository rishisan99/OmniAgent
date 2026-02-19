# graph/lanes_node.py
from __future__ import annotations
import asyncio
import re
import time
from typing import Any, Dict, List

from backend.src.schemas.plan import RunPlan
from backend.src.graph.streaming import stream_tokens
from backend.src.agents.router import run_task


def lanes_node(provider: str, model: str):
    async def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        em = state["emitter"]
        plan = RunPlan.model_validate(state["plan"])
        tasks: List[Dict[str, Any]] = list(state.get("tasks", []))
        outs: Dict[str, Any] = dict(state.get("tool_outputs", {}))
        history = state.get("chat_history", [])[-12:]
        last_image_prompt = state.get("last_image_prompt")
        linked_artifact = state.get("linked_artifact") or {}
        runtime = state.get("plan_runtime") or {}
        artifact_memory = dict(
            state.get("artifact_memory")
            or {"image": None, "audio": None, "doc": None, "lineage": {"image": [], "audio": [], "doc": []}}
        )
        lineage = artifact_memory.get("lineage") or {"image": [], "audio": [], "doc": []}
        artifact_memory["lineage"] = lineage

        def _subject_lock_ok(task_prompt: str, subject_lock: str | None) -> bool:
            if not subject_lock:
                return True
            t = (task_prompt or "").lower()
            required = [x for x in re.split(r"\s+", subject_lock.lower()) if x and len(x) >= 3]
            return bool(required) and all(w in t for w in required[:2])

        def task_title(kind: str) -> str:
            return {
                "web": "Web Results",
                "rag": "Document Context",
                "image_gen": "Generated Image",
                "tts": "Generated Audio",
                "doc": "Generated Document",
                "vision": "Vision Analysis",
            }.get(kind, kind)

        def task_phrase() -> str:
            kinds = [t.get("kind") for t in tasks]
            if not kinds:
                return "response"
            labels: List[str] = []
            if "doc" in kinds:
                labels.append("document")
            if "image_gen" in kinds:
                labels.append("image")
            if "tts" in kinds:
                labels.append("audio")
            if "web" in kinds:
                labels.append("web results")
            if "rag" in kinds:
                labels.append("document analysis")
            if "vision" in kinds:
                labels.append("vision analysis")
            if len(labels) == 1:
                return labels[0]
            if len(labels) == 2:
                return f"{labels[0]} and {labels[1]}"
            return ", ".join(labels[:-1]) + f", and {labels[-1]}"

        for t in tasks:
            em.emit("block_start", {"block_id": t["id"], "title": task_title(t["kind"]), "kind": t["kind"]})

        intro_text = ""
        outro_text = ""
        tools_only_turn = bool(tasks) and not plan.text.enabled
        if tools_only_turn:
            intro_text = f'Sure, I will generate your {task_phrase()}.\n\n'
            em.emit("token", {"text": intro_text})

        async def run_tools() -> None:
            async def one(t):
                task_for_run = dict(t)
                subject_lock = task_for_run.get("subject_lock")
                max_replans = int(runtime.get("max_replans", 0))
                attempts = 0

                while True:
                    attempts += 1
                    r = await asyncio.to_thread(run_task, task_for_run, state, em, provider, model)
                    if task_for_run.get("kind") != "image_gen":
                        break
                    if _subject_lock_ok(task_for_run.get("prompt", ""), subject_lock):
                        break
                    if attempts > max_replans:
                        break
                    # One fast replan with stronger constraints for subject stability.
                    task_for_run["prompt"] = (
                        f"{task_for_run.get('prompt', '')}\n\n"
                        f"CRITICAL CONSTRAINT: Keep the main subject as '{subject_lock}'. "
                        "Do not replace it with any other animal or object."
                    )

                outs[t["id"]] = r
                nonlocal last_image_prompt
                if t.get("kind") == "image_gen" and r.get("ok"):
                    p = ((r.get("data") or {}).get("prompt") or t.get("prompt") or "").strip()
                    if p:
                        last_image_prompt = p
                    d = r.get("data") or {}
                    artifact_memory["image"] = {
                        "id": d.get("filename"),
                        "url": d.get("url"),
                        "prompt": p,
                    }
                    parent_id = linked_artifact.get("id") if linked_artifact.get("kind") == "image" else None
                    child_id = d.get("filename")
                    if parent_id and child_id and parent_id != child_id:
                        lineage["image"].append(
                            {"parent_id": parent_id, "child_id": child_id, "op": "edit", "ts_ms": int(time.time() * 1000)}
                        )
                if t.get("kind") == "tts" and r.get("ok"):
                    d = r.get("data") or {}
                    artifact_memory["audio"] = {
                        "id": d.get("filename"),
                        "url": d.get("url"),
                        "text": (t.get("text") or "").strip(),
                    }
                if t.get("kind") == "doc" and r.get("ok"):
                    d = r.get("data") or {}
                    artifact_memory["doc"] = {
                        "id": d.get("filename"),
                        "url": d.get("url"),
                        "text": (d.get("text") or "")[:2000],
                    }
                em.emit("block_end", {"block_id": t["id"], "payload": r})
            if tasks:
                await asyncio.gather(*[one(t) for t in tasks])

        tools_job = asyncio.create_task(run_tools()) if tasks else None
        knowledge_tasks = {"web", "rag", "vision"}
        needs_context_first = any(t.get("kind") in knowledge_tasks for t in tasks)
        media_tasks = {"image_gen", "tts", "doc"}
        media_only = bool(tasks) and all(t.get("kind") in media_tasks for t in tasks)

        def history_text() -> str:
            if not history:
                return ""
            rows = []
            for m in history:
                role = m.get("role", "user").upper()
                rows.append(f"{role}: {m.get('content', '')}")
            return "\n".join(rows)

        def tool_context_text() -> str:
            if not outs:
                return ""
            rows = []
            for v in outs.values():
                kind = v.get("kind", "tool")
                if not v.get("ok"):
                    continue
                if kind == "rag":
                    matches = (v.get("data") or {}).get("matches", [])[:4]
                    snippets = [m.get("text", "")[:500] for m in matches]
                    if snippets:
                        rows.append("RAG:\n" + "\n---\n".join(snippets))
                if kind == "web":
                    parts = (v.get("data") or {}).get("parts", [])
                    rows.append(f"WEB: {parts}")
                if kind == "vision":
                    rows.append(f"VISION: {(v.get('data') or {}).get('text', '')}")
                if kind == "doc":
                    rows.append(f"DOC: {(v.get('data') or {}).get('text', '')[:1200]}")
            return "\n\n".join(rows)

        llm_text = ""
        if plan.text.enabled:
            if media_only and plan.mode == "tools_only":
                if tools_job:
                    await tools_job
                    tools_job = None
                llm_text = ""
            else:
                if tools_job and needs_context_first:
                    await tools_job
                    tools_job = None
                context = tool_context_text()
                has_media_blocks = any(t.get("kind") in {"image_gen", "tts", "doc"} for t in tasks)
                prompt = (
                    "You are OmniAgent. Answer directly in markdown.\n"
                    "If tool outputs are present, treat them as completed results and do not say you cannot perform generation.\n"
                    "Never output internal labels/headers like CHAT_HISTORY or TOOL_CONTEXT.\n"
                )
                if has_media_blocks:
                    prompt += (
                        "If media/doc tool blocks are present, do not output markdown image/audio/doc links or placeholder URLs.\n"
                        "Do not invent URLs (especially example.com). The UI already renders generated media blocks.\n"
                    )
                prompt += (
                    f"{state.get('text_instructions','')}\n\n"
                    f"Conversation so far:\n{history_text()}\n\n"
                    + (f"Useful context from tools:\n{context}\n\n" if context else "")
                    + f"User message:\n{state.get('user_text','')}\n"
                )
                llm_text = await stream_tokens(prompt, em, provider=provider, model=model, temperature=0.2)

        if tools_job:
            await tools_job

        if tools_only_turn:
            outro_text = f"\n\nHere is your {task_phrase()}."
            em.emit("token", {"text": outro_text})

        final_text = f"{intro_text}{llm_text}{outro_text}"

        return {
            "tool_outputs": outs,
            "final_text": final_text,
            "last_image_prompt": last_image_prompt,
            "artifact_memory": artifact_memory,
        }

    return _run
