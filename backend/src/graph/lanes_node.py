# graph/lanes_node.py
from __future__ import annotations
import asyncio
import os
import re
import time
from typing import Any, Dict, List

from backend.src.core.logging import get_logger
from backend.src.llm.factory import get_llm
from backend.src.graph.agent_memory import push_note
from backend.src.schemas.plan import RunPlan
from backend.src.graph.streaming import stream_tokens
from backend.src.agents.router import run_task

_GREETING_ONLY_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|sup|what'?s up|good\s+morning|good\s+afternoon|good\s+evening)[!. ]*$",
    re.IGNORECASE,
)
logger = get_logger("omniagent.graph.lanes")


def lanes_node(provider: str, model: str):
    async def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(
            "NODE_CALL node=lanes run_id=%s trace_id=%s session_id=%s provider=%s model=%s",
            state.get("run_id"),
            state.get("trace_id"),
            state.get("session_id"),
            provider,
            model,
        )
        em = state["emitter"]
        text_provider = os.getenv("TEXT_PROVIDER", provider)
        text_model = os.getenv("TEXT_MODEL", model)
        plan = RunPlan.model_validate(state["plan"])
        tasks: List[Dict[str, Any]] = list(state.get("tasks", []))
        outs: Dict[str, Any] = dict(state.get("tool_outputs", {}))
        history = state.get("chat_history", [])[-12:]
        last_image_prompt = state.get("last_image_prompt")
        linked_artifact = state.get("linked_artifact") or {}
        runtime = state.get("plan_runtime") or {}
        response_contract = dict(state.get("response_contract") or {})
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
                "kb_rag": "Knowledge Base",
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
            if "kb_rag" in kinds:
                labels.append("knowledge-base answer")
            if "vision" in kinds:
                labels.append("vision analysis")
            if len(labels) == 1:
                return labels[0]
            if len(labels) == 2:
                return f"{labels[0]} and {labels[1]}"
            return ", ".join(labels[:-1]) + f", and {labels[-1]}"

        async def _meta_message(stage: str) -> str:
            tasks_summary = task_phrase()
            if stage == "conclusion":
                # Keep conclusion deterministic to avoid hallucinated claims.
                return "Completed. Results are shown above."
            fallback = (
                f"Sure, I will handle your {tasks_summary} request."
                if stage == "initial"
                else "Completed. Results are shown above."
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
            delay_ms = int(os.getenv("META_STREAM_TOKEN_DELAY_MS", "0"))
            for i, w in enumerate(parts):
                tok = w + (" " if i < len(parts) - 1 else "")
                acc += tok
                em.emit("block_token", {"block_id": block_id, "text": tok})
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000.0)
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

        knowledge_tasks = {"web", "rag", "kb_rag", "vision"}
        needs_context_first = any(t.get("kind") in knowledge_tasks for t in tasks)
        knowledge_task_items = [t for t in tasks if t.get("kind") in knowledge_tasks]
        other_task_items = [t for t in tasks if t.get("kind") not in knowledge_tasks]
        # Run knowledge and non-knowledge lanes independently so text can start
        # as soon as retrieval context is ready (without waiting on slower image/doc/audio tasks).
        async def run_selected(selected: List[Dict[str, Any]]) -> None:
            async def one(t):
                logger.info(
                    "TASK_START run_id=%s task_id=%s kind=%s",
                    state.get("run_id"),
                    t.get("id"),
                    t.get("kind"),
                )
                task_for_run = dict(t)
                subject_lock = task_for_run.get("subject_lock")
                max_replans = int(runtime.get("max_replans", 0))
                image_timeout_sec = float(os.getenv("IMAGE_TASK_TIMEOUT_SEC", "90"))
                attempts = 0
                r: Dict[str, Any] = {"task_id": t["id"], "kind": t.get("kind"), "ok": False, "error": "Unknown error"}
                try:
                    while True:
                        attempts += 1
                        if task_for_run.get("kind") == "image_gen":
                            r = await asyncio.wait_for(
                                asyncio.to_thread(run_task, task_for_run, state, em, provider, model),
                                timeout=max(1.0, image_timeout_sec),
                            )
                        else:
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
                except asyncio.TimeoutError:
                    r = {
                        "task_id": t["id"],
                        "kind": t.get("kind"),
                        "ok": False,
                        "error": f"Image generation timed out after {int(max(1.0, image_timeout_sec))}s",
                    }
                except Exception as e:
                    r = {"task_id": t["id"], "kind": t.get("kind"), "ok": False, "error": str(e)}
                    logger.exception(
                        "TASK_ERROR run_id=%s task_id=%s kind=%s error=%s",
                        state.get("run_id"),
                        t.get("id"),
                        t.get("kind"),
                        e,
                    )
                finally:
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
                    logger.info(
                        "TASK_DONE run_id=%s task_id=%s kind=%s ok=%s data_keys=%s",
                        state.get("run_id"),
                        t.get("id"),
                        t.get("kind"),
                        r.get("ok", False),
                        ",".join(sorted((r.get("data") or {}).keys())) if isinstance(r.get("data"), dict) else "",
                    )

            if selected:
                await asyncio.gather(*[one(t) for t in selected])

        knowledge_job = asyncio.create_task(run_selected(knowledge_task_items)) if knowledge_task_items else None
        other_job = asyncio.create_task(run_selected(other_task_items)) if other_task_items else None
        media_tasks = {"image_gen", "tts", "doc"}
        media_only = bool(tasks) and all(t.get("kind") in media_tasks for t in tasks)
        should_emit_text = bool(plan.text.enabled or knowledge_task_items)
        logger.info(
            "LANES_PLAN run_id=%s mode=%s text_enabled=%s should_emit_text=%s tasks=%s",
            state.get("run_id"),
            plan.mode,
            plan.text.enabled,
            should_emit_text,
            [str(t.get("kind")) for t in tasks],
        )

        def history_text() -> str:
            if not history:
                return ""
            rows = []
            for m in history:
                role = m.get("role", "user").upper()
                rows.append(f"{role}: {m.get('content', '')}")
            return "\n".join(rows)

        def tool_context_text() -> str:
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
                if kind == "kb_rag":
                    matches = (v.get("data") or {}).get("matches", [])[:5]
                    snippets = [m.get("text", "")[:500] for m in matches]
                    if snippets:
                        rows.append("KB_RAG:\n" + "\n---\n".join(snippets))
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
            # Fallback to persisted generated/extracted doc text when current turn has no doc/rag tool output.
            if not any(r.startswith("DOC:") or r.startswith("RAG:") for r in rows):
                mem_doc = artifact_memory.get("doc") or {}
                mem_text = str(mem_doc.get("text") or "").strip()
                if mem_text:
                    rows.append(f"DOC: {mem_text[:1200]}")
            return "\n\n".join(rows)

        def arxiv_items_from_outs() -> List[Dict[str, str]]:
            items: List[Dict[str, str]] = []
            seen: set[str] = set()
            for v in outs.values():
                if v.get("kind") != "web" or not v.get("ok"):
                    continue
                parts = (v.get("data") or {}).get("parts", [])
                for p in parts:
                    if not isinstance(p, dict):
                        continue
                    pd = p.get("data") or {}
                    for it in pd.get("items", []) if isinstance(pd.get("items"), list) else []:
                        if not isinstance(it, dict):
                            continue
                        title = str(it.get("title", "")).strip()
                        url = str(it.get("url", "")).strip()
                        if not title or not url:
                            continue
                        # Keep only canonical arXiv paper links and dedupe.
                        if "arxiv.org/abs/" not in url:
                            continue
                        if url in seen:
                            continue
                        seen.add(url)
                        items.append(
                            {
                                "title": title,
                                "url": url,
                                "published": str(it.get("published", "")).strip(),
                                "summary": str(it.get("summary", "")).strip(),
                            }
                        )
            return items

        def kb_unique_citations() -> List[Dict[str, str]]:
            seen: set[tuple[str, str]] = set()
            out_rows: List[Dict[str, str]] = []
            for v in outs.values():
                if v.get("kind") != "kb_rag" or not v.get("ok"):
                    continue
                for c in v.get("citations", []) if isinstance(v.get("citations"), list) else []:
                    if not isinstance(c, dict):
                        continue
                    title = str(c.get("title", "")).strip()
                    url = str(c.get("url", "")).strip()
                    if not title or not url:
                        continue
                    key = (title, url)
                    if key in seen:
                        continue
                    seen.add(key)
                    out_rows.append({"title": title, "url": url})
            return out_rows

        def render_arxiv_markdown(items: List[Dict[str, str]]) -> str:
            lines = ["## Results from Arxiv", "", "Here are recent research papers from arXiv:", ""]
            for i, it in enumerate(items, 1):
                lines.append(f"{i}. [{it['title']}]({it['url']})")
                if it.get("published"):
                    lines.append(f"Published: {it['published'][:10]}")
                if it.get("summary"):
                    lines.append(f"Summary: {it['summary'][:280]}")
                lines.append("")
            return "\n".join(lines).strip()

        async def emit_text_tokens(text: str) -> str:
            if not text:
                return ""
            delay_ms = int(os.getenv("ARXIV_STREAM_TOKEN_DELAY_MS", "0"))
            # Preserve markdown formatting while streaming deterministic text.
            chunks = re.findall(r"\S+|\s+", text)
            for tok in chunks:
                em.emit("token", {"text": tok})
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000.0)
            return text

        def _wordset(s: str) -> set[str]:
            return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w) >= 3}

        def ranked_evidence(query: str) -> List[Dict[str, str]]:
            rows: List[Dict[str, str]] = []
            for v in outs.values():
                if not isinstance(v, dict) or not v.get("ok"):
                    continue
                kind = str(v.get("kind", ""))
                if kind in {"rag", "kb_rag"}:
                    for m in (v.get("data") or {}).get("matches", [])[:12]:
                        if isinstance(m, dict):
                            rows.append(
                                {
                                    "kind": kind,
                                    "source": str(m.get("source", "")).strip(),
                                    "text": str(m.get("text", "")).strip(),
                                }
                            )
                elif kind == "web":
                    for c in v.get("citations", []) if isinstance(v.get("citations"), list) else []:
                        if isinstance(c, dict):
                            rows.append(
                                {
                                    "kind": kind,
                                    "source": str(c.get("url", "")).strip(),
                                    "text": str(c.get("snippet", "")).strip(),
                                }
                            )
            if not rows:
                return []
            qset = _wordset(query)
            scored: List[tuple[Dict[str, str], int]] = []
            for r in rows:
                overlap = len(qset.intersection(_wordset((r.get("text", "") + " " + r.get("source", "")))))
                scored.append((r, overlap))
            scored.sort(key=lambda x: x[1], reverse=True)
            top: List[Dict[str, str]] = []
            seen = set()
            for r, _ in scored:
                key = (r.get("source", ""), r.get("text", "")[:120])
                if key in seen:
                    continue
                seen.add(key)
                top.append(r)
                if len(top) >= 5:
                    break
            return top

        def evidence_text(rows: List[Dict[str, str]]) -> str:
            if not rows:
                return ""
            out = []
            for i, r in enumerate(rows, 1):
                out.append(f"{i}. [{r.get('kind','')}] source={r.get('source','unknown')}\n   snippet={r.get('text','')[:320]}")
            return "\n".join(out)

        def conflict_signals(query: str, rows: List[Dict[str, str]]) -> List[str]:
            q = (query or "").lower().strip()
            m = re.match(r"^\s*(?:who is|tell me about|about)\s+([a-z][a-z .'-]{2,})\??\s*$", q)
            if not m:
                return []
            entity = re.sub(r"\s+", " ", m.group(1)).strip()
            tokens = [t for t in entity.split(" ") if len(t) >= 2]
            if len(tokens) < 2:
                return []
            first, last = tokens[0], tokens[-1]
            mismatched = []
            for r in rows:
                src = (r.get("source", "") or "").lower()
                if first in src and last not in src:
                    mismatched.append(r.get("source", ""))
            if mismatched:
                return [f"Possible entity bleed for target '{entity}' in sources: {', '.join(mismatched[:3])}"]
            return []

        llm_text = ""
        if should_emit_text:
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
                kb_tasks = [t for t in tasks if t.get("kind") == "kb_rag"]
                has_arxiv_context = any("arxiv" in (t.get("sources") or []) for t in web_tasks)
                has_web_context = bool(web_tasks)
                has_kb_context = bool(kb_tasks)
                arxiv_only_web = bool(web_tasks) and all((t.get("sources") or []) == ["arxiv"] for t in web_tasks)
                arxiv_items = arxiv_items_from_outs() if has_arxiv_context else []
                kb_citations = kb_unique_citations() if has_kb_context else []
                kb_no_exact_entity = False
                kb_missing_name = ""
                if has_kb_context:
                    for v in outs.values():
                        if v.get("kind") != "kb_rag" or not v.get("ok"):
                            continue
                        d = v.get("data") or {}
                        matches = d.get("matches") if isinstance(d.get("matches"), list) else []
                        if d.get("entity_not_found") and not matches:
                            kb_no_exact_entity = True
                            kb_missing_name = str(d.get("entity_not_found") or "").strip()
                            break
                if arxiv_only_web and arxiv_items and not has_media_blocks:
                    llm_text = await emit_text_tokens(render_arxiv_markdown(arxiv_items))
                elif kb_no_exact_entity:
                    miss = kb_missing_name or "the requested employee"
                    not_found_text = (
                        "## Knowledge Base Result\n\n"
                        f'No exact record was found for "{miss}" in the Insurellm knowledge base.\n\n'
                        "Try the full official name or verify spelling."
                    )
                    llm_text = await emit_text_tokens(not_found_text)
                else:
                    query_text = state.get("text_query") or state.get("user_text", "")
                    length_rules = (
                        "Length policy:\n"
                        "- Explanation/overview/definition requests: target about 1 page (roughly 350-500 words).\n"
                        "- Greetings, acknowledgements, or very simple asks: keep concise (1-4 lines).\n"
                        "- Mixed asks: allocate length proportionally and avoid filler.\n"
                    )
                    ev_rows = ranked_evidence(query_text)
                    ev_text = evidence_text(ev_rows)
                    conflicts = conflict_signals(query_text, ev_rows)
                    prompt = (
                        "You are OmniAgent. Answer directly in markdown.\n"
                        f"{length_rules}"
                        "Use short headings and compact bullets where useful.\n"
                        "Avoid long paragraphs (>3 lines each).\n"
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
                            "For each paper, copy the exact title text from tool context. Do not paraphrase or invent titles.\n"
                            "Each title must be paired with its own URL from the same paper entry.\n"
                        )
                    elif has_web_context:
                        prompt += (
                            "Start your response with this exact heading on the first line: ## Results from Web\n"
                            "Return ONLY a short numbered list (max 5 items), with this exact per-item layout:\n"
                            "1. **Headline:** <title>\n"
                            "   **Description:** <2-3 lines; factual and concise>\n"
                            "   **Source:** <publisher name> - [Read](url)\n"
                            "No intro sentence. No outro sentence. No placeholder '(link)'.\n"
                            "Use only URLs present in tool context and skip vague/generic results.\n"
                        )
                    if has_web_context or has_arxiv_context:
                        prompt += (
                            "For links, only use URLs explicitly present in tool context/citations. Never invent or guess URLs.\n"
                        )
                    if has_kb_context:
                        prompt += (
                            "The question targets the Insurellm knowledge base.\n"
                            "Use KB_RAG context as the source of truth.\n"
                            "If the question names a specific person/entity, answer only for that exact entity.\n"
                            "Ignore context snippets about similarly named but different entities.\n"
                            "Do not copy large raw chunks verbatim; synthesize a concise answer.\n"
                            "Do NOT include a Sources/Citations section in the answer.\n"
                        )
                    if kb_no_exact_entity:
                        prompt += (
                            "KB_RAG indicates no exact entity match for the requested name.\n"
                            "Respond that no record was found for that exact entity in the knowledge base.\n"
                            "Do not infer role, title, or contributions from other people.\n"
                        )
                    if conflicts:
                        prompt += "Conflict alerts:\n" + "\n".join(f"- {c}" for c in conflicts) + "\n"
                    prompt += (
                        f"{state.get('text_instructions','')}\n\n"
                        + (
                            "Agent collaboration contract:\n"
                            f"Researcher brief:\n{response_contract.get('researcher_brief','')}\n\n"
                            f"Writer plan:\n{response_contract.get('writer_plan','')}\n\n"
                            f"Critic checks:\n{response_contract.get('critic_checks','')}\n\n"
                            if response_contract
                            else ""
                        )
                        + f"Conversation so far:\n{history_text()}\n\n"
                        + (f"Useful context from tools:\n{context}\n\n" if context else "")
                        + (f"Ranked evidence (top 5):\n{ev_text}\n\n" if ev_text else "")
                        + f"User message:\n{state.get('text_query') or state.get('user_text','')}\n"
                    )
                    high_conflict = bool(conflicts)
                    rewrite_budget = int((runtime or {}).get("max_rewrites", 0))
                    if high_conflict and rewrite_budget > 0:
                        llm = get_llm(text_provider, text_model, streaming=False, temperature=0.2)
                        draft_msg = llm.invoke(prompt)
                        draft = (getattr(draft_msg, "content", "") or "").strip()
                        review_prompt = (
                            "Review the draft against ranked evidence.\n"
                            "If unsupported or mixed-entity claims exist, rewrite once to be evidence-grounded.\n"
                            "Return only corrected markdown answer.\n\n"
                            f"Draft:\n{draft}\n\n"
                            f"Ranked evidence:\n{ev_text}\n"
                        )
                        final_msg = llm.invoke(review_prompt)
                        reviewed = (getattr(final_msg, "content", "") or "").strip() or draft
                        llm_text = await emit_text_tokens(reviewed)
                    else:
                        llm_text = await stream_tokens(prompt, em, provider=text_provider, model=text_model, temperature=0.2)
                    # Soft guard: if greeting-only prompt still produced a long answer,
                    # rewrite once into a compact single sentence using the same LLM.
                    if _GREETING_ONLY_RE.match(str(query_text or "")):
                        word_count = len((llm_text or "").strip().split())
                        if word_count > 20:
                            llm = get_llm(text_provider, text_model, streaming=False, temperature=0.1)
                            short = ""
                            try:
                                msg = llm.invoke(
                                    "Rewrite the following as exactly one short friendly sentence "
                                    "(max 18 words), no headings, no bullets, no markdown:\n\n"
                                    f"{llm_text}"
                                )
                                short = (getattr(msg, "content", "") or "").strip()
                            except Exception:
                                short = ""
                            if short:
                                llm_text = short
                    if has_kb_context and kb_citations:
                        lines = ["", "", "## Sources", ""]
                        for i, c in enumerate(kb_citations, 1):
                            lines.append(f"{i}. {c['title']} ({c['url']})")
                        suffix = "\n".join(lines)
                        llm_text += await emit_text_tokens(suffix)

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
            "agent_memory": push_note(
                state,
                node="lanes",
                summary="Execution complete",
                extra={"tasks": [t.get("kind") for t in tasks], "tool_outputs": len(outs), "has_final_text": bool(final_text)},
            ),
        }

    return _run
