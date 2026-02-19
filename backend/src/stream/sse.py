# stream/sse.py
from __future__ import annotations
import json
from typing import Any, AsyncGenerator, Dict


def sse_pack(ev: Dict[str, Any]) -> str:
    return "event: message\ndata: " + json.dumps(ev, ensure_ascii=False) + "\n\n"


async def sse_gen(queue) -> AsyncGenerator[str, None]:
    while True:
        ev = await queue.get()
        if ev is None:
            break
        yield sse_pack(ev)
