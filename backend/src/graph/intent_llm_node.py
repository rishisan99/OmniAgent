# graph/intent_llm_node.py
from __future__ import annotations
import os
import re
from typing import Any, Dict

from backend.src.core.jsonx import extract_json
from backend.src.graph.agent_memory import push_note
from backend.src.llm.factory import get_llm, is_not_found_error, model_candidates
from backend.src.schemas.plan import RunPlan, TextPlan


PLANNER_SYSTEM_PROMPT = (
    "You are a strict low-latency planner for a multimodal assistant.\n"
    "Priority: speed, correct tool routing, and valid JSON.\n"
    "Never include prose, markdown, comments, or extra keys.\n"
    "If uncertain, choose text_only with task ['text'].\n"
)


def intent_llm_node(provider: str, model: str):
    def _prompt(user: str, has_files: bool, has_last_image: bool) -> str:
        return (
            "You are an intent classifier for a multimodal assistant.\n"
            "Allowed capabilities: text, image, document, audio, web, rag, arxiv, kb_rag.\n"
            "You MUST only use those capabilities and combinations of them.\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            "{\n"
            '  "mode": "text_only" | "text_plus_tools" | "tools_only",\n'
            '  "tasks": ["text"|"image"|"document"|"audio"|"web"|"rag"|"arxiv"|"kb_rag"],\n'
            '  "confidence": number,\n'
            '  "intent_type": "create"|"edit"|"analyze"|"retrieve"|"chat"\n'
            "}\n"
            "Routing policy:\n"
            "- Default to text_only with ['text'] for greetings/chat/simple Q&A.\n"
            "- Add a non-text task ONLY when explicitly requested by the user.\n"
            "- Do NOT infer audio from general explanation requests. Audio requires explicit ask for audio/voice/tts/speak/read aloud.\n"
            "- Do NOT infer image from general explanation requests. Image requires explicit ask to create/generate/make image/photo/picture.\n"
            "- Do NOT infer web/arxiv unless user explicitly asks web/news/internet/search/arxiv/papers/latest/current.\n"
            "- Use 'arxiv' specifically for paper/preprint/arxiv requests.\n"
            "- Use 'rag' only for questions over uploaded files/documents.\n"
            "- Use 'kb_rag' only for organization KB lookup requests (company/employees/products/contracts) when user asks for that data.\n"
            "- If user asks both explanation and a tool action, use text_plus_tools.\n"
            "- For follow-up image edits with previous image context, choose image task.\n"
            "- No extra keys, no prose.\n"
            "Examples:\n"
            'USER: "hi"\n'
            'JSON: {"mode":"text_only","tasks":["text"],"confidence":0.98,"intent_type":"chat"}\n'
            'USER: "Explain RAG in bullets"\n'
            'JSON: {"mode":"text_only","tasks":["text"],"confidence":0.93,"intent_type":"analyze"}\n'
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
        attachments = state.get("attachments") or []
        artifact_memory = state.get("artifact_memory") or {}
        memory_doc = artifact_memory.get("doc") or {}
        has_memory_doc_text = bool((memory_doc.get("text") or "").strip())
        has_files = bool(attachments)
        has_image_attachment = any(str(a.get("kind", "")).lower() == "image" for a in attachments)
        has_doc_attachment = any(str(a.get("kind", "")).lower() == "doc" for a in attachments)
        ctx = state.get("context_bundle") or {}
        has_last_image = bool(ctx.get("has_last_image"))

        prompt = _prompt(user, has_files, has_last_image)
        raw = ""
        last_err: Exception | None = None
        planner_provider = os.getenv("PLANNER_PROVIDER", os.getenv("INTENT_PROVIDER", "openai"))
        planner_model = os.getenv("PLANNER_MODEL", os.getenv("INTENT_MODEL", "gpt-4o-mini"))

        candidate_pairs: list[tuple[str, str]] = [(planner_provider, planner_model)]
        for c in model_candidates(provider, model):
            pair = (provider, c)
            if pair not in candidate_pairs:
                candidate_pairs.append(pair)

        for i, (p, candidate) in enumerate(candidate_pairs):
            llm = get_llm(p, candidate, streaming=False, temperature=0.0)
            try:
                raw = (llm.invoke(f"{PLANNER_SYSTEM_PROMPT}\n\n{prompt}").content or "").strip()
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
        allowed = {"text", "image", "document", "audio", "web", "rag", "arxiv", "kb_rag"}
        tasks = []
        seen = set()
        for item in task_list:
            t = str(item).strip().lower()
            if not t or t not in allowed or t in seen:
                continue
            seen.add(t)
            tasks.append(t)

        # Deterministic cue-based fallback so UI always gets expected text/tool lanes
        # even if classifier output misses an explicit user ask.
        text_cues = (
            "explain",
            "what is",
            "what's",
            "who is",
            "who's",
            "how ",
            "why ",
            "tell me",
            "summarize",
            "summary",
            "describe",
            "analysis",
            "analyze",
            "write",
            "story",
            "?",
        )
        greeting_re = re.compile(
            r"^\s*(hi|hello|hey|yo|sup|good\s+morning|good\s+afternoon|good\s+evening)[!. ]*$",
            re.IGNORECASE,
        )
        explicit_image = any(k in user_l for k in ("generate image", "create image", "make image", "image of", "picture of", "photo of"))
        explicit_audio = any(k in user_l for k in ("generate audio", "create audio", "make audio", "tts", "voice", "read aloud", "narrate", "speak "))
        explicit_doc = (
            any(k in user_l for k in ("pdf", "document", "docx", "text file", "txt", "markdown"))
            and any(k in user_l for k in ("generate", "create", "make", "write", "export"))
        )
        explicit_web = any(k in user_l for k in ("latest", "recent", "news", "headlines", "web", "internet", "search"))
        explicit_arxiv = any(k in user_l for k in ("arxiv", "paper", "papers", "preprint", "research paper"))
        explicit_kb = any(
            k in user_l
            for k in (
                "knowledge base",
                "knowledge-base",
                "employee",
                "employees",
                "company",
                "contract",
                "product",
                "carllm",
                "homellm",
                "markellm",
                "rellm",
            )
        )
        asks_text = bool(greeting_re.match(user)) or any(k in user_l for k in text_cues)
        if explicit_image and "image" not in tasks:
            tasks.append("image")
        if explicit_audio and "audio" not in tasks:
            tasks.append("audio")
        if explicit_doc and "document" not in tasks:
            tasks.append("document")
        if explicit_arxiv and "arxiv" not in tasks:
            tasks.append("arxiv")
        if explicit_web and "web" not in tasks and "arxiv" not in tasks:
            tasks.append("web")
        if explicit_kb and "kb_rag" not in tasks:
            tasks.append("kb_rag")
        if asks_text and "text" not in tasks:
            tasks.append("text")

        # If a document is already uploaded and user is asking a question about it,
        # route to QA (text + retrieval) instead of document generation/extraction.
        has_doc_context = has_doc_attachment or has_memory_doc_text
        asks_doc_question = has_doc_context and (
            any(k in user_l for k in ("document", "doc", "pdf", "file", "uploaded", "upload"))
            and any(
                k in user_l
                for k in (
                    "explain",
                    "summarize",
                    "summary",
                    "what",
                    "describe",
                    "analyze",
                    "analysis",
                    "content",
                    "contents",
                    "tell me",
                    "question",
                )
            )
        )
        asks_doc_generation = any(k in user_l for k in ("generate", "create", "make", "write", "export"))
        if asks_doc_question and not asks_doc_generation:
            tasks = [t for t in tasks if t != "document"]
            if "text" not in tasks:
                tasks.append("text")
            if has_doc_attachment and "rag" not in tasks:
                tasks.append("rag")
        
        # Attachment-aware fallback for image analysis turns if classifier under-selects.
        image_analysis_cues = (
            "image",
            "photo",
            "picture",
            "attached",
            "this image",
            "what is in",
            "describe",
            "analyze",
            "caption",
        )
        if has_image_attachment and any(k in user_l for k in image_analysis_cues):
            if "image" not in tasks:
                tasks.append("image")
            if "text" not in tasks:
                tasks.append("text")
        if ("web" in tasks or "arxiv" in tasks or "rag" in tasks) and "text" not in tasks:
            # Retrieval flows should usually produce a textual synthesis.
            tasks.append("text")
        if "kb_rag" in tasks and "text" not in tasks:
            tasks.append("text")
        if not tasks:
            # Safety fallback: always preserve a text response lane.
            tasks = ["text"]

        flags = {
            "needs_web": ("web" in tasks) or ("arxiv" in tasks),
            "needs_rag": "rag" in tasks,
            "needs_kb_rag": "kb_rag" in tasks,
            "needs_doc": "document" in tasks,
            "needs_vision": ("image" in tasks) and has_image_attachment,
            "needs_tts": "audio" in tasks,
            # Distinguish image generation/edit from image analysis.
            "needs_image_gen": ("image" in tasks) and not has_image_attachment,
        }

        web_source = "arxiv" if "arxiv" in tasks else ("tavily" if "web" in tasks else None)

        mode = data.get("mode", "text_only")
        if "text" in tasks and any(t in tasks for t in ("image", "document", "audio", "web", "rag", "arxiv", "kb_rag")):
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
            "agent_memory": push_note(
                state,
                node="intent",
                summary="Intent classified",
                extra={"mode": mode, "tasks": tasks, "flags": flags},
            ),
        }
    return _run
