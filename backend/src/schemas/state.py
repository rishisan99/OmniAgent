# schemas/state.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict


class Attachment(TypedDict, total=False):
    id: str               # stored asset id (server-side)
    kind: str             # "image" | "audio" | "doc"
    name: str
    mime: str
    url: str              # optional public url
    meta: Dict[str, Any]  # page_count, width/height, etc.


class AgentState(TypedDict, total=False):
    # identity + tracing
    session_id: str
    run_id: str
    trace_id: str

    # event emitter (runtime only)
    emitter: Any 

    # user I/O
    user_text: str
    text_query: str
    system_prompt: str
    attachments: List[Attachment]

    chat_history: List[Dict[str, str]]
    artifact_memory: Dict[str, Any]
    context_bundle: Dict[str, Any]
    linked_artifact: Dict[str, Any]
    intent: Dict[str, Any]
    plan_runtime: Dict[str, Any]
    last_image_prompt: Optional[str]
    initial_meta_emitted: bool
    plan: Dict[str, Any]
    tasks: List[Dict[str, Any]]
    tool_outputs: Dict[str, Any]
    final_text: Optional[str]
    text_instructions: str
    agent_memory: Dict[str, Any]
    response_contract: Dict[str, Any]
