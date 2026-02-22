# stream/emitter.py
from __future__ import annotations
import os
import time
from typing import Any, Callable, Dict, Optional

from backend.src.core.logging import get_logger
from backend.src.schemas.events import SSEEvent

logger = get_logger("omniagent.stream.emitter")


class Emitter:
    def __init__(self, run_id: str, trace_id: Optional[str], send: Callable[[Dict[str, Any]], None]):
        self.run_id, self.trace_id, self.send = run_id, trace_id, send

    def emit(self, type_: str, data: Dict[str, Any]) -> None:
        ev = SSEEvent(type=type_, run_id=self.run_id, trace_id=self.trace_id, ts_ms=int(time.time() * 1000), data=data)
        # Keep logs useful without flooding token-level events unless explicitly requested.
        if type_ != "token" or os.getenv("LOG_SSE_TOKENS", "false").strip().lower() in {"1", "true", "yes", "on"}:
            logger.info(
                "SSE_EMIT type=%s run_id=%s trace_id=%s keys=%s",
                type_,
                self.run_id,
                self.trace_id,
                ",".join(sorted((data or {}).keys())),
            )
        self.send(ev.model_dump())
