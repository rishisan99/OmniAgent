# graph/lanes_node.py
from __future__ import annotations
import asyncio
import os
import re
import time
from typing import Any, Dict, List

from backend.src.llm.factory import get_llm
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

        def task_title(task: Dict[str, Any]) -> str:
            kind = task.get("kind")
            if kind == "web":
                sources = task.get("sources") or []
                if sources == ["arxiv"]:
                    return "Results from Arxiv"
                return "Results from Web"
            return {
                "rag": "Document Context",
                "image_gen": "Generated Image",
                "tts": "Generated Audio",
                "doc": "Generated Document",
                "vision": "Vision Analysis",
            }.get(kind, str(kind))

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

        async def _meta_message(stage: str) -> str:
            tasks_summary = task_phrase()
            fallback = (
                f"Sure, I will handle your {tasks_summary} request."
                if stage == "initial"
                else f"All requested outputs are ready: {tasks_summary}."
            )
            try:
                p = os.getenv("INTENT_PROVIDER", "openai")
                m = os.getenv("INTENT_MODEL", "gpt-4o-mini")
                llm = get_llm(p, m, streaming=False, temperature=0.2)
                prompt = (
                    "Write exactly one short sentence for assistant UX.\n"
                    f"Stage: {stage}\n"
                    f"User query: {state.get('user_text','')}\n"
                    f"Requested tools: {tasks_summary}\n"
                    "Rules: no markdown links, no bullets, no headings, no quotes.\n"
                    "If stage=initial, acknowledge what will be generated.\n"
                    "If stage=conclusion, confirm outputs are ready.\n"
                )
                msg = llm.invoke(prompt)
                out = (getattr(msg, "content", "") or "").strip().replace("\n", " ")
                return out or fallback
            except Exception:
                return fallback

        async def _stream_meta_block(block_id: str, title: str, kind: str, text: str) -> None:
            em.emit("block_start", {"block_id": block_id, "title": title, "kind": kind})
            acc = ""
            parts = (text or "").split(" ")
            for i, w in enumerate(parts):
                tok = w + (" " if i < len(parts) - 1 else "")
                acc += tok
                em.emit("block_token", {"block_id": block_id, "text": tok})
                await asyncio.sleep(0.008)
            em.emit(
                "block_end",
                {
                    "block_id": block_id,
                    "payload": {"ok": True, "kind": kind, "data": {"text": acc.strip(), "mime": "text/markdown"}},
                },
            )

        if tasks and not state.get("initial_meta_emitted"):
            initial = await _meta_message("initial")
            await _stream_meta_block("__meta_initial__", "Initial", "meta_initial", initial)

        for t in tasks:
            em.emit("block_start", {"block_id": t["id"], "title": task_title(t), "kind": t["kind"]})

        knowledge_tasks = {"web", "rag", "vision"}
        needs_context_first = any(t.get("kind") in knowledge_tasks for t in tasks)
        knowledge_task_items = [t for t in tasks if t.get("kind") in knowledge_tasks]
        other_task_items = [t for t in tasks if t.get("kind") not in knowledge_tasks]
        # Run knowledge and non-knowledge lanes independently so text can start
        # as soon as retrieval context is ready (without waiting on slower image/doc/audio tasks).
        async def run_selected(selected: List[Dict[str, Any]]) -> None:
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

            if selected:
                await asyncio.gather(*[one(t) for t in selected])

        knowledge_job = asyncio.create_task(run_selected(knowledge_task_items)) if knowledge_task_items else None
        other_job = asyncio.create_task(run_selected(other_task_items)) if other_task_items else None
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
                    lines: List[str] = []
                    for p in parts:
                        pd = (p.get("data") or {}) if isinstance(p, dict) else {}
                        items = pd.get("items") if isinstance(pd, dict) else None
                        if isinstance(items, list) and items:
                            for it in items[:8]:
                                if not isinstance(it, dict):
                                    continue
                                title = str(it.get("title", "")).strip()
                                url = str(it.get("url", "")).strip()
                                summ = str(it.get("summary", "")).strip()
                                pub = str(it.get("published", "")).strip()
                                if title:
                                    lines.append(f"- {title}")
                                if pub:
                                    lines.append(f"  published: {pub}")
                                if url:
                                    lines.append(f"  url: {url}")
                                if summ:
                                    lines.append(f"  summary: {summ[:300]}")
                        elif isinstance(pd.get("results"), list):
                            for it in pd.get("results", [])[:8]:
                                if not isinstance(it, dict):
                                    continue
                                title = str(it.get("title", "")).strip()
                                url = str(it.get("url", "")).strip()
                                content = str(it.get("content", "")).strip()
                                if title:
                                    lines.append(f"- {title}")
                                if url:
                                    lines.append(f"  url: {url}")
                                if content:
                                    lines.append(f"  snippet: {content[:260]}")
                    if lines:
                        rows.append("WEB:\n" + "\n".join(lines))
                if kind == "vision":
                    rows.append(f"VISION: {(v.get('data') or {}).get('text', '')}")
                if kind == "doc":
                    rows.append(f"DOC: {(v.get('data') or {}).get('text', '')[:1200]}")
            return "\n\n".join(rows)

        llm_text = ""
        if plan.text.enabled:
            if media_only and plan.mode == "tools_only":
                if other_job:
                    await other_job
                    other_job = None
                if knowledge_job:
                    await knowledge_job
                    knowledge_job = None
                llm_text = ""
            else:
                if knowledge_job and needs_context_first:
                    await knowledge_job
                    knowledge_job = None
                context = tool_context_text()
                has_media_blocks = any(t.get("kind") in {"image_gen", "tts", "doc"} for t in tasks)
                web_tasks = [t for t in tasks if t.get("kind") == "web"]
                has_arxiv_context = any("arxiv" in (t.get("sources") or []) for t in web_tasks)
                has_web_context = bool(web_tasks)
                prompt = (
                    "You are OmniAgent. Answer directly in markdown.\n"
                    "If tool outputs are present, treat them as completed results and do not say you cannot perform generation.\n"
                    "Never output internal labels/headers like CHAT_HISTORY or TOOL_CONTEXT.\n"
                    "When a turn has both text and tool outputs, answer ONLY the user's text/explanation requests.\n"
                    "Address all distinct textual asks in the same user message; do not skip any requested text output.\n"
                    "If the user asks for both factual retrieval and creative writing, include both in one response with clear sections.\n"
                    "Do NOT include sections like 'Audio Generation', 'Image Generation', 'Document Generation', or status lines about generated tools.\n"
                    "Do NOT mention that files/audio/images were generated; the UI shows tool blocks separately.\n"
                )
                if has_media_blocks:
                    prompt += (
                        "If media/doc tool blocks are present, do not output markdown image/audio/doc links or placeholder URLs.\n"
                        "Do not invent URLs (especially example.com). The UI already renders generated media blocks.\n"
                    )
                if has_arxiv_context:
                    prompt += (
                        "Start your response with this exact heading on the first line: ## Results from Arxiv\n"
                        "Then provide concise answer content.\n"
                        "If WEB context includes paper entries, list those papers with their real URLs.\n"
                        "Do not claim 'no papers found' when paper entries are present in context.\n"
                    )
                elif has_web_context:
                    prompt += (
                        "Start your response with this exact heading on the first line: ## Results from Web\n"
                        "Then provide concise answer content.\n"
                    )
                if has_web_context or has_arxiv_context:
                    prompt += (
                        "For links, only use URLs explicitly present in tool context/citations. Never invent or guess URLs.\n"
                    )
                prompt += (
                    f"{state.get('text_instructions','')}\n\n"
                    f"Conversation so far:\n{history_text()}\n\n"
                    + (f"Useful context from tools:\n{context}\n\n" if context else "")
                    + f"User message:\n{state.get('text_query') or state.get('user_text','')}\n"
                )
                llm_text = await stream_tokens(prompt, em, provider=provider, model=model, temperature=0.2)

        if knowledge_job:
            await knowledge_job
        if other_job:
            await other_job

        if tasks:
            conclusion = await _meta_message("conclusion")
            await _stream_meta_block("__meta_conclusion__", "Conclusion", "meta_conclusion", conclusion)

        final_text = llm_text

        return {
            "tool_outputs": outs,
            "final_text": final_text,
            "last_image_prompt": last_image_prompt,
            "artifact_memory": artifact_memory,
        }

    return _run
