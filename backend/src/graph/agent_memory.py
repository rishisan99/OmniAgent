from __future__ import annotations

import time
from typing import Any, Dict


def push_note(
    state: Dict[str, Any],
    node: str,
    summary: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    mem = dict(state.get("agent_memory") or {})
    notes = list(mem.get("notes") or [])
    notes.append(
        {
            "ts_ms": int(time.time() * 1000),
            "node": node,
            "summary": summary,
            "extra": extra or {},
        }
    )
    mem["notes"] = notes[-120:]
    return mem
