# api/routes_chat.py
from __future__ import annotations
import asyncio
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.src.core.constants import MAX_HISTORY_MESSAGES
from backend.src.graph.runner import build_graph, run_graph
from backend.src.session.store import get_session, cleanup
from backend.src.stream.sse import sse_gen

router = APIRouter()


class ChatIn(BaseModel):
    session_id: str
    provider: str
    model: str
    text: str


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
    state = {"session_id": inp.session_id, "run_id": run_id, "trace_id": trace_id,
             "user_text": inp.text, "attachments": sess.get("attachments", []),
             "chat_history": sess.get("chat_history", []),
             "last_image_prompt": sess.get("last_image_prompt"),
             "artifact_memory": artifact_memory}

    app = build_graph(inp.provider, inp.model)
    task = asyncio.create_task(run_graph(app, state, send, run_id=run_id, trace_id=trace_id))

    async def done():
        try:
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
    return StreamingResponse(sse_gen(q), media_type="text/event-stream")
