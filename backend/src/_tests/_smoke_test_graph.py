from __future__ import annotations
import asyncio
import json

from backend.src.core.config import bootstrap_env
bootstrap_env()

from backend.src.graph.runner import build_graph, run_graph


def send(ev):
    print("EV:", json.dumps(ev))


async def run_case(title: str, prompt: str):
    print("\n" + "=" * 40)
    print("CASE:", title)
    print("=" * 40)

    app = build_graph("openai", "gpt-4o-mini")
    state = {
        "session_id": "demo",
        "run_id": "r1",
        "trace_id": "t1",
        "user_text": prompt,
        "attachments": [],
    }
    await run_graph(app, state, send, run_id="r1", trace_id="t1")


async def main():
    await run_case("TEXT ONLY", "Explain LangGraph in 5 bullet points.")
    await run_case("TEXT + WEB", "Explain LangGraph and cite sources from Wikipedia.")
    await run_case("TOOLS ONLY", "Search web for latest papers on RAG.")


if __name__ == "__main__":
    asyncio.run(main())
