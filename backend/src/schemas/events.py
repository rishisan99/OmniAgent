# schemas/events.py
from __future__ import annotations
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel


EventType = Literal[
    "run_start", "plan", "task_start", "task_result",
    "token", "block_start", "block_end",
    "error", "run_end",
]


class SSEEvent(BaseModel):
    type: EventType
    run_id: str
    trace_id: Optional[str] = None
    ts_ms: int
    data: Dict[str, Any] = {}
