# graph/intent_llm_node.py
from __future__ import annotations
import os
from typing import Any, Dict

from backend.src.core.jsonx import extract_json
from backend.src.llm.factory import get_llm, is_not_found_error, model_candidates
from backend.src.schemas.plan import RunPlan, TextPlan


def intent_llm_node(provider: str, model: str):
    def _prompt(user: str, has_files: bool, has_last_image: bool) -> str:
        return (
            "You are an intent classifier for a multimodal assistant.\n"
            "Allowed capabilities: text, image, document, audio, web, rag, arxiv.\n"
            "You MUST only use those capabilities and combinations of them.\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            "{\n"
            '  "mode": "text_only" | "text_plus_tools" | "tools_only",\n'
            '  "tasks": ["text"|"image"|"document"|"audio"|"web"|"rag"|"arxiv"],\n'
            '  "confidence": number,\n'
            '  "intent_type": "create"|"edit"|"analyze"|"retrieve"|"chat"\n'
            "}\n"
            "Rules:\n"
            "- If user asks both writing/explaining and generation, include both text and tool tasks.\n"
            "- If user asks for document/pdf/doc/txt only, do not include text unless explicitly requested.\n"
            "- arxiv is a subset of web retrieval; include task 'arxiv' when user asks papers/research.\n"
            "- For follow-up image edits with previous image context, choose image task.\n"
            "- No extra keys, no prose.\n"
            "Examples:\n"
            'USER: "Explain RAG in bullets and generate audio for hello"\n'
            'JSON: {"mode":"text_plus_tools","tasks":["text","audio"],"confidence":0.93,"intent_type":"create"}\n'
            'USER: "Generate a PDF about AI"\n'
            'JSON: {"mode":"tools_only","tasks":["document"],"confidence":0.95,"intent_type":"create"}\n'
            'USER: "latest AI papers from arxiv"\n'
            'JSON: {"mode":"tools_only","tasks":["arxiv"],"confidence":0.92,"intent_type":"retrieve"}\n'
            'USER: "write a phoenix story and also generate an image"\n'
            'JSON: {"mode":"text_plus_tools","tasks":["text","image"],"confidence":0.94,"intent_type":"create"}\n'
            f"has_files={has_files}; has_last_image={has_last_image}\n"
            f"USER:\n{user}\n"
        )

    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        user = state.get("user_text", "")
        user_l = user.lower()
        has_files = bool(state.get("attachments"))
        ctx = state.get("context_bundle") or {}
        has_last_image = bool(ctx.get("has_last_image"))
        prompt = _prompt(user, has_files, has_last_image)
        raw = ""
        last_err: Exception | None = None
        intent_provider = os.getenv("INTENT_PROVIDER", "openai")
        intent_model = os.getenv("INTENT_MODEL", "gpt-4o-mini")

        candidate_pairs: list[tuple[str, str]] = [(intent_provider, intent_model)]
        for c in model_candidates(provider, model):
            pair = (provider, c)
            if pair not in candidate_pairs:
                candidate_pairs.append(pair)

        for i, (p, candidate) in enumerate(candidate_pairs):
            llm = get_llm(p, candidate, streaming=False, temperature=0.0)
            try:
                raw = (llm.invoke(prompt).content or "").strip()
                break
            except Exception as e:
                last_err = e
                if i < len(candidate_pairs) - 1 and is_not_found_error(e):
                    continue
                raise
        if not raw and last_err:
            raise last_err

        data = extract_json(raw)

        task_list = data.get("tasks") or []
        tasks = [str(t).strip().lower() for t in task_list if str(t).strip()]
        # Deterministic retrieval cues to reduce missed web/arxiv intents.
        if any(k in user_l for k in ("latest", "recent", "news", "top ", "headlines", "current")) and "web" not in tasks and "arxiv" not in tasks:
            tasks.append("web")
        if any(k in user_l for k in ("arxiv", "paper", "research paper", "preprint")) and "arxiv" not in tasks:
            tasks.append("arxiv")
        if ("web" in tasks or "arxiv" in tasks or "rag" in tasks) and "text" not in tasks:
            # Retrieval flows should usually produce a textual synthesis.
            tasks.append("text")

        flags = {
            "needs_web": ("web" in tasks) or ("arxiv" in tasks),
            "needs_rag": "rag" in tasks or has_files,
            "needs_doc": "document" in tasks,
            "needs_vision": False,
            "needs_tts": "audio" in tasks,
            "needs_image_gen": "image" in tasks,
        }

        web_source = "arxiv" if "arxiv" in tasks else ("tavily" if "web" in tasks else None)

        mode = data.get("mode", "text_only")
        if "text" in tasks and any(t in tasks for t in ("image", "document", "audio", "web", "rag", "arxiv")):
            mode = "text_plus_tools"
        elif "text" in tasks:
            mode = "text_only"
        elif tasks:
            mode = "tools_only"

        plan = RunPlan(
            mode=mode,
            text=TextPlan(enabled=mode != "tools_only"),
            flags=flags,
            web_source=web_source,
            note="intent_structured_fast",
        )
        return {
            "plan": plan.model_dump(),
            "intent": {
                "intent_type": str(data.get("intent_type", "chat")),
                "target_modality": "+".join(tasks) if tasks else "text",
                "confidence": float(data.get("confidence", 0.7)),
            },
        }
    return _run
