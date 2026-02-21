# session/store.py
from __future__ import annotations
import time
from uuid import uuid4
from typing import Any, Dict


_sessions: Dict[str, Dict[str, Any]] = {}
TTL_SECS = 60 * 30
SERVER_BOOT_ID = f"boot_{int(time.time())}_{str(uuid4())[:8]}"


def _default_artifact_memory() -> Dict[str, Any]:
    return {"image": None, "audio": None, "doc": None, "lineage": {"image": [], "audio": [], "doc": []}}


def get_session(session_id: str) -> Dict[str, Any]:
    s = _sessions.get(session_id) or {
        "chat_history": [],
        "attachments": [],
        "artifact_memory": _default_artifact_memory(),
        "ts": time.time(),
    }
    s.setdefault("artifact_memory", _default_artifact_memory())
    if not isinstance(s["artifact_memory"], dict):
        s["artifact_memory"] = _default_artifact_memory()
    s["artifact_memory"].setdefault("lineage", {"image": [], "audio": [], "doc": []})
    s["ts"] = time.time()
    _sessions[session_id] = s
    return s


def cleanup() -> None:
    now = time.time()
    for k in list(_sessions.keys()):
        if now - _sessions[k].get("ts", now) > TTL_SECS:
            _sessions.pop(k, None)


def clear_session(session_id: str) -> bool:
    existed = session_id in _sessions
    _sessions.pop(session_id, None)
    return existed


def server_boot_id() -> str:
    return SERVER_BOOT_ID
