# graph/intent_llm_node.py
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any, Dict

from backend.src.core.jsonx import extract_json
from backend.src.graph.agent_memory import push_note
from backend.src.llm.factory import get_llm, is_not_found_error, model_candidates
from backend.src.schemas.plan import RunPlan, TextPlan


_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|sup|good\s+morning|good\s+afternoon|good\s+evening)[!. ]*$",
    flags=re.IGNORECASE,
)
_KB_TERMS = (
    "insurellm",
    "carllm",
    "homellm",
    "markellm",
    "rellm",
    "knowledge base",
    "knowledge-base",
)
_KB_DOMAIN_TERMS = (
    "company",
    "employee",
    "employees",
    "staff",
    "team",
    "worked at",
    "works at",
    "job title",
    "department",
    "manager",
    "salary",
    "compensation",
    "contract",
    "client",
    "product",
    "insurellm",
    "carllm",
    "homellm",
    "markellm",
    "rellm",
)
_EXPLICIT_WEB_TERMS = (
    "web",
    "internet",
    "online",
    "news",
    "headline",
    "headlines",
    "arxiv",
    "search",
    "google",
    "wikipedia",
    "tavily",
    "latest",
    "recent",
    "current",
)

PLANNER_SYSTEM_PROMPT = (
    "You are a strict low-latency planner for a multimodal assistant.\n"
    "Priority: speed, correct tool routing, and valid JSON.\n"
    "Never include prose, markdown, comments, or extra keys.\n"
    "If uncertain, choose the minimal safe task set and keep text enabled for user-facing answers.\n"
)


def intent_llm_node(provider: str, model: str):
    def _kb_exists() -> bool:
        roots = [
            Path(os.getenv("KB_ROOT_PATH", "")).expanduser() if os.getenv("KB_ROOT_PATH") else None,
            Path("backend/docs/knowledge-base"),
            Path("backend/docs"),
            Path("backend/data/docs/knowledge-base"),
        ]
        for r in roots:
            if r and r.exists() and r.is_dir():
                return True
        return False

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
            "- If user asks to generate/create/make an image, do not add web/arxiv tasks unless user explicitly asks for web/news/research.\n"
            "- For pure image generation requests, prefer tools_only with image task.\n"
            "- arxiv is a subset of web retrieval; include task 'arxiv' when user asks papers/research.\n"
            "- Use 'kb_rag' ONLY for company/employee related requests (organization, employees, products, contracts).\n"
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
        attachments = state.get("attachments") or []
        artifact_memory = state.get("artifact_memory") or {}
        memory_doc = artifact_memory.get("doc") or {}
        has_memory_doc_text = bool((memory_doc.get("text") or "").strip())
        has_files = bool(attachments)
        has_image_attachment = any(str(a.get("kind", "")).lower() == "image" for a in attachments)
        has_doc_attachment = any(str(a.get("kind", "")).lower() == "doc" for a in attachments)
        ctx = state.get("context_bundle") or {}
        has_last_image = bool(ctx.get("has_last_image"))

        # Deterministic fast-path for short greetings.
        if _GREETING_RE.match(user or ""):
            plan = RunPlan(
                mode="text_only",
                text=TextPlan(enabled=True),
                flags={
                    "needs_web": False,
                    "needs_rag": False,
                    "needs_doc": False,
                    "needs_vision": False,
                    "needs_tts": False,
                    "needs_image_gen": False,
                },
                web_source=None,
                note="intent_greeting_fastpath",
            )
            return {
                "plan": plan.model_dump(),
                "intent": {
                    "intent_type": "chat",
                    "target_modality": "text",
                    "confidence": 0.99,
                },
                "agent_memory": push_note(
                    state,
                    node="intent",
                    summary="Intent fast-path greeting",
                    extra={"mode": "text_only", "tasks": ["text"]},
                ),
            }

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
        tasks = [str(t).strip().lower() for t in task_list if str(t).strip()]

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
        
        # Deterministic retrieval cues to reduce missed web/arxiv intents.
        if any(k in user_l for k in ("latest", "recent", "news", "top ", "headlines", "current")) and "web" not in tasks and "arxiv" not in tasks:
            tasks.append("web")
        if any(k in user_l for k in ("arxiv", "paper", "research paper", "preprint")) and "arxiv" not in tasks:
            tasks.append("arxiv")
        questionish = (
            "?" in user
            or user_l.startswith(("who ", "what ", "when ", "where ", "which ", "how ", "tell me", "give me"))
        )
        company_employee_related = any(k in user_l for k in _KB_DOMAIN_TERMS)
        other_tool_cues = any(
            k in user_l for k in ("image", "photo", "picture", "audio", "voice", "pdf", "document", "docx", "upload", "web", "arxiv")
        )
        explicit_web_ask = any(k in user_l for k in _EXPLICIT_WEB_TERMS)
        if company_employee_related and any(k in user_l for k in _KB_TERMS):
            if "kb_rag" not in tasks:
                tasks.append("kb_rag")
        elif _kb_exists() and company_employee_related and questionish and not has_files and not other_tool_cues:
            # Default company-QA path: prefer KB retrieval over generic model memory.
            if "kb_rag" not in tasks:
                tasks.append("kb_rag")
        # If this is a KB-style QA and user did not explicitly ask for web retrieval,
        # suppress web/arxiv to avoid irrelevant internet answers.
        if "kb_rag" in tasks and not explicit_web_ask:
            tasks = [t for t in tasks if t not in ("web", "arxiv")]
        # If an image is attached and user asks to analyze/describe it, route to vision.
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
