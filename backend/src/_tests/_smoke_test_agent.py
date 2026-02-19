# agents/_smoke_test.py
from __future__ import annotations
import json
from backend.src.core.config import bootstrap_env
bootstrap_env()

from backend.src.stream.emitter import Emitter
from backend.src.agents.router import run_task

def send(ev): print("EVENT:", json.dumps(ev))

if __name__ == "__main__":
    state = {"session_id": "demo"}
    em = Emitter(run_id="r1", trace_id="t1", send=send)
    print(run_task({"id":"w1","kind":"web","query":"LangGraph","top_k":2,"sources":["wikipedia"]}, state, em, "openai", "gpt-4o-mini"))
