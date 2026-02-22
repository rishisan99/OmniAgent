# api/routes_chat.py
from __future__ import annotations
import asyncio
import os
import re
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.src.core.constants import MAX_HISTORY_MESSAGES
from backend.src.graph.runner import build_graph, run_graph
from backend.src.graph.v2_flow import run_graph_v2
from backend.src.llm.factory import get_llm
from backend.src.session.store import get_session, cleanup, clear_session, server_boot_id
from backend.src.stream.sse import sse_gen

router = APIRouter()


class ChatIn(BaseModel):
    session_id: str
    provider: str
    model: str
    text: str


class SessionClearIn(BaseModel):
    session_id: str


def _likely_tool_turn(text: str, has_attachments: bool) -> bool:
    t = (text or "").lower()
    if re.match(r"^\s*(hi|hello|hey|yo|sup|good\s+morning|good\s+afternoon|good\s+evening)[!. ]*$", t):
        return False
    has_action = any(k in t for k in ("generate", "create", "make", "search", "find", "upload"))
    questionish = (
        "?" in t
        or t.startswith(("who ", "what ", "when ", "where ", "which ", "how ", "tell me", "did you know", "do you know"))
    )
    kb_retrieval_hint = any(
        k in t
        for k in (
            "employee",
            "employees",
            "company",
            "contract",
            "product",
            "knowledge base",
            "knowledge-base",
            "database",
            "insurellm",
            "carllm",
            "homellm",
            "markellm",
            "rellm",
        )
    )
    has_tool = any(
        k in t
        for k in (
            "image",
            "audio",
            "voice",
            "tts",
            "pdf",
            "document",
            "doc",
            "txt",
            "web",
            "arxiv",
            "rag",
            "paper",
            "research",
            "news",
            "headline",
            "headlines",
        )
    )
    if has_attachments:
        # Only show upload-analysis preamble when user asks to do something with files/tools.
        file_reference = any(
            k in t
            for k in ("uploaded", "upload", "file", "document", "doc", "pdf", "image", "photo", "picture")
        )
        return has_action or has_tool or file_reference
    if questionish and kb_retrieval_hint:
        return True
    return has_action and has_tool


async def _stream_initial_block(send, user_text: str, provider: str, model: str, has_attachments: bool = False) -> None:
    user_l = (user_text or "").lower()
    def _clean_db_subject(s: str) -> str:
        out = re.sub(r"\s+", " ", (s or "").strip())
        out = re.sub(r"^(employee|employees|person)\s+", "", out, flags=re.IGNORECASE)
        out = out.strip(" \t\r\n.,;:!?\"'")
        return out
    def find_clause(text: str, patterns: list[str]) -> str:
        nxt = (
            r"(?=(?:\s*,|\s*[.;:]\s*|\s+and\s+|\s+also\s+|\s+then\s+)\s*"
            r"(?:generate|create|make|explain|tell|write|summarize|what is|find|search|show|get|fetch)\b|$)"
        )
        for p in patterns:
            m = re.search(p + nxt, text, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip(" ,.;:-\"'")
        return ""

    explain_clause = find_clause(
        user_text,
        [
            r"(?:explain|tell me about|what is|summarize|write(?:\s+a)?\s+story about)\s+(.+?)",
        ],
    )
    doc_clause = find_clause(
        user_text,
        [r"(?:generate|create|make)\s+(?:an?\s+)?(?:pdf|document|docx?|txt|text file)(?:\s+on|\s+about|\s+for)?\s+(.+?)"],
    )
    audio_clause = find_clause(
        user_text,
        [r"(?:generate|create|make)\s+audio(?:\s+for|\s+saying|\s+of)?\s+(.+?)"],
    )
    image_clause = find_clause(
        user_text,
        [r"(?:generate|create|make)\s+(?:an?\s+)?(?:image|picture|photo)(?:\s+for|\s+of)?\s+(.+?)"],
    )
    news_clause = find_clause(
        user_text,
        [
            r"(?:tell me|show|give me|find|search|get|fetch)\s+(?:me\s+)?(?:about\s+)?(?:top\s+\d+\s+)?(?:latest|recent|current)\s+(?:news|headlines|updates?)\s+(?:on|about|for)?\s*(.+?)",
            r"(?:tell me|show|give me|find|search|get|fetch)\s+(?:me\s+)?(?:recent|latest|current)\s+(.+?)(?:news|headlines|updates?)",
            r"(?:latest|recent|current)\s+(?:news|headlines|updates?)\s+(?:on|about|for)?\s*(.+?)",
        ],
    )
    arxiv_clause = find_clause(
        user_text,
        [
            r"(?:find|search|show|get|fetch)\s+(?:me\s+)?(?:the\s+)?(?:search\s+)?(?:paper|papers|research(?: papers?)?)\s*(?:on|about|for|:)?\s+(.+?)",
            r"(?:from\s+)?arxiv\s*[,:\s]*(?:find|search|show|get|fetch|list)?\s*(?:the\s+)?(?:paper|papers|research(?: papers?)?)?\s*(?:on|about|for|:)?\s+(.+?)",
            r"(?:arxiv|papers?|research(?: papers?)?)\s+(?:on|about|for)?\s+(.+?)",
        ],
    )
    db_clause = find_clause(
        user_text,
        [
            r"(?:did you know about|do you know about|tell me about|who is|about)\s+(.+?)",
            r"(?:employee|employees|company|contract|product)\s+(.+?)",
        ],
    )

    parts: list[str] = []
    labels: list[str] = []
    if explain_clause:
        parts.append(f'explain "{explain_clause}"')
        labels.append("text explanation")
    if doc_clause:
        parts.append(f'create a document on "{doc_clause}"')
        labels.append("document")
    if audio_clause:
        parts.append(f'generate audio for "{audio_clause}"')
        labels.append("audio")
    if image_clause:
        parts.append(f'generate an image for "{image_clause}"')
        labels.append("image")
    if arxiv_clause:
        parts.append(f'fetch arxiv papers for "{arxiv_clause}"')
        labels.append("arxiv research")
    elif news_clause:
        parts.append(f'fetch recent news on "{news_clause}"')
        labels.append("news summary")

    is_db_scripted = bool(db_clause) and not any((doc_clause, audio_clause, image_clause, arxiv_clause, news_clause))
    has_file_reference = any(
        k in user_l
        for k in ("uploaded", "upload", "file", "document", "doc", "pdf", "image", "photo", "picture", "attachment")
    )
    if is_db_scripted:
        subject = _clean_db_subject(db_clause) or db_clause
        scripted = f'I will search the database for records on "{subject}".'
    elif has_attachments and has_file_reference:
        scripted = "Analyzing your uploaded file now and preparing the answer."
    elif parts:
        # Deduplicate while preserving order.
        seen_p: set[str] = set()
        uniq_parts: list[str] = []
        for p in parts:
            k = p.lower().strip()
            if k and k not in seen_p:
                seen_p.add(k)
                uniq_parts.append(p)
        seen_l: set[str] = set()
        uniq_labels: list[str] = []
        for l in labels:
            k = l.lower().strip()
            if k and k not in seen_l:
                seen_l.add(k)
                uniq_labels.append(l)
        parts = uniq_parts
        labels = uniq_labels

        summary = ", ".join(labels[:-1]) + (f", and {labels[-1]}" if len(labels) > 1 else labels[0])
        actions = [p.capitalize() for p in parts]
        if len(actions) == 1:
            flow = actions[0]
        elif len(actions) == 2:
            flow = f"{actions[0]}, then {actions[1]}"
        else:
            flow = ", then ".join(actions[:-1]) + f", and finally {actions[-1]}"
        scripted = f"Sure, working on this now; I'll produce {summary}: {flow}."
    else:
        scripted = ""

    em_provider = os.getenv("INITIAL_PROVIDER", os.getenv("INTENT_PROVIDER", provider))
    em_model = os.getenv("INITIAL_MODEL", os.getenv("INTENT_MODEL", model))
    start_delay_ms = int(os.getenv("INITIAL_START_DELAY_MS", "200"))
    delay_ms = int(os.getenv("INITIAL_TOKEN_DELAY_MS", "16"))
    if is_db_scripted:
        start_delay_ms = int(os.getenv("INITIAL_DB_START_DELAY_MS", "0"))
        delay_ms = int(os.getenv("INITIAL_DB_TOKEN_DELAY_MS", "6"))
    prompt = (
        "Write one short sentence acknowledging requested tool outputs.\n"
        "No markdown, no bullets, no quotes.\n"
        f"USER:\n{user_text}\n"
    )
    em_text = ""
    send({"type": "block_start", "data": {"block_id": "__meta_initial__", "title": "Initial", "kind": "meta_initial"}})
    try:
        if start_delay_ms > 0:
            await asyncio.sleep(start_delay_ms / 1000.0)
        if scripted:
            words = scripted.split(" ")
            for i, w in enumerate(words):
                tok = w + (" " if i < len(words) - 1 else "")
                em_text += tok
                send({"type": "block_token", "data": {"block_id": "__meta_initial__", "text": tok}})
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000.0)
        else:
            llm = get_llm(em_provider, em_model, streaming=True, temperature=0.2)
            async for chunk in llm.astream(prompt):
                tok = getattr(chunk, "content", "") or ""
                if tok:
                    em_text += tok
                    send({"type": "block_token", "data": {"block_id": "__meta_initial__", "text": tok}})
    except Exception:
        em_text = "Working on your request now."
        send({"type": "block_token", "data": {"block_id": "__meta_initial__", "text": em_text}})
    finally:
        send(
            {
                "type": "block_end",
                "data": {
                    "block_id": "__meta_initial__",
                    "payload": {
                        "ok": True,
                        "kind": "meta_initial",
                        "data": {"text": em_text.strip() or "Working on your request now.", "mime": "text/markdown"},
                    },
                },
            }
        )


@router.post("/chat/stream")
async def chat_stream(inp: ChatIn):
    cleanup()
    sess = get_session(inp.session_id)
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def send(ev):
        loop.call_soon_threadsafe(q.put_nowait, ev)

    run_id, trace_id = str(uuid4())[:8], str(uuid4())[:8]
    artifact_memory = sess.get("artifact_memory", {}) or {}
    artifact_memory.setdefault("lineage", {"image": [], "audio": [], "doc": []})
    likely_tool_turn = _likely_tool_turn(inp.text, bool(sess.get("attachments", [])))
    state = {"session_id": inp.session_id, "run_id": run_id, "trace_id": trace_id,
             "user_text": inp.text, "attachments": sess.get("attachments", []),
             "chat_history": sess.get("chat_history", []),
             "last_image_prompt": sess.get("last_image_prompt"),
             "artifact_memory": artifact_memory,
             "initial_meta_emitted": likely_tool_turn}

    use_graph_v2 = os.getenv("GRAPH_V2_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    if use_graph_v2:
        task = asyncio.create_task(
            run_graph_v2(state, send, run_id=run_id, trace_id=trace_id, provider=inp.provider, model=inp.model)
        )
    else:
        app = build_graph(inp.provider, inp.model)
        task = asyncio.create_task(run_graph(app, state, send, run_id=run_id, trace_id=trace_id))
    initial_task = (
        asyncio.create_task(_stream_initial_block(send, inp.text, inp.provider, inp.model, has_attachments=bool(sess.get("attachments", []))))
        if likely_tool_turn
        else None
    )

    async def done():
        try:
            if initial_task:
                await asyncio.gather(initial_task, task)
                out = task.result()
            else:
                out = await task
            final_text = (out or {}).get("final_text", "")
            if (out or {}).get("last_image_prompt"):
                sess["last_image_prompt"] = out["last_image_prompt"]
            if (out or {}).get("artifact_memory"):
                sess["artifact_memory"] = out["artifact_memory"]
            sess["chat_history"].append({"role": "user", "content": inp.text})
            if final_text:
                sess["chat_history"].append({"role": "assistant", "content": final_text})
            sess["chat_history"] = sess["chat_history"][-MAX_HISTORY_MESSAGES:]
        finally:
            await q.put(None)

    asyncio.create_task(done())
    return StreamingResponse(
        sse_gen(q),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/session/meta")
def session_meta():
    return {"server_boot_id": server_boot_id()}


@router.post("/session/clear")
def session_clear(inp: SessionClearIn):
    existed = clear_session(inp.session_id)
    return {"ok": True, "cleared": existed, "session_id": inp.session_id}
