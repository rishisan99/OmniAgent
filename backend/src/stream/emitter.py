# stream/emitter.py
from __future__ import annotations
import time
from typing import Any, Callable, Dict, Optional

from backend.src.schemas.events import SSEEvent


class Emitter:
    def __init__(self, run_id: str, trace_id: Optional[str], send: Callable[[Dict[str, Any]], None]):
        self.run_id, self.trace_id, self.send = run_id, trace_id, send

    def emit(self, type_: str, data: Dict[str, Any]) -> None:
        ev = SSEEvent(type=type_, run_id=self.run_id, trace_id=self.trace_id, ts_ms=int(time.time() * 1000), data=data)
        self.send(ev.model_dump())
