from __future__ import annotations
import json
from pathlib import Path

from backend.src.core.config import bootstrap_env
bootstrap_env()

from backend.src.stream.emitter import Emitter
from backend.src.agents.router import run_task


def send(ev):
    print("EV:", json.dumps(ev, ensure_ascii=False))


def pick_first(glob_pat: str) -> str | None:
    p = next(iter(Path(".").glob(glob_pat)), None)
    return str(p) if p else None


if __name__ == "__main__":
    em = Emitter(run_id="r1", trace_id="t1", send=send)

    # Adjust these to your actual files if needed
    pdf_path = pick_first("backend/data/docs/*.pdf") or "backend/data/docs/sample.pdf"
    img_path = pick_first("backend/data/docs/*.png")  # you can replace with: backend/data/docs/sample.png

    attachments = []
    if Path(pdf_path).exists():
        attachments.append({"id": "doc1", "kind": "doc", "name": Path(pdf_path).name, "mime": "application/pdf", "path": pdf_path})
    if img_path and Path(img_path).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and Path(img_path).exists():
        attachments.append({"id": "img1", "kind": "image", "name": Path(img_path).name, "mime": "image/*", "path": img_path})

    state = {"session_id": "demo", "attachments": attachments, "chat_history": []}

    tests = [
        ("TEXT", {"id": "t1", "kind": "text", "prompt": "Say hello in one line."}),
        ("WEB", {"id": "w1", "kind": "web", "query": "LangGraph overview", "top_k": 2, "sources": ["wikipedia"]}),
        ("RAG", {"id": "r1", "kind": "rag", "query": "Summarize the uploaded document", "top_k": 3}),
        ("DOC", {"id": "d1", "kind": "doc", "instruction": "extract", "attachment_id": "doc1"}),
        ("VISION", {"id": "v1", "kind": "vision", "prompt": "Describe this image in 3 bullets.", "image_attachment_id": "img1"}),
        ("IMAGE_GEN", {"id": "i1", "kind": "image_gen", "prompt": "A minimal logo of an owl, flat design", "size": "1024x1024"}),
        ("TTS", {"id": "a1", "kind": "tts", "text": "Hello from OmniAgent!", "voice": "alloy"}),
    ]

    for name, task in tests:
        kind = task["kind"]
        if kind == "rag" and not any(a["kind"] == "doc" for a in attachments):
            print(f"\n--- SKIP {name}: no PDF/doc attachment found ---")
            continue
        if kind == "doc" and not any(a["id"] == "doc1" for a in attachments):
            print(f"\n--- SKIP {name}: doc1 missing ---")
            continue
        if kind == "vision" and not any(a["id"] == "img1" for a in attachments):
            print(f"\n--- SKIP {name}: img1 missing ---")
            continue

        print(f"\n==================== {name} ====================")
        out = run_task(task, state, em, provider="openai", model="gpt-4o-mini")
        print("RESULT:", json.dumps(out, indent=2, ensure_ascii=False))
