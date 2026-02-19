# session/store.py
from __future__ import annotations
import time
from typing import Any, Dict


_sessions: Dict[str, Dict[str, Any]] = {}
TTL_SECS = 60 * 30


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
