# graph/tool_router_node.py
from __future__ import annotations
import re
from typing import Any, Dict, List
from uuid import uuid4

from backend.src.graph.agent_memory import push_note
from backend.src.schemas.plan import RunPlan


def _extract_quoted(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    for q in ('"', "'"):
        i = s.find(q)
        j = s.rfind(q)
        if i != -1 and j != -1 and j > i:
            inner = s[i + 1 : j].strip()
            if inner:
                return inner
    return s


def _strip_prefixes(text: str, prefixes: List[str]) -> str:
    s = (text or "").strip()
    low = s.lower()
    for p in prefixes:
        if low.startswith(p):
            return s[len(p) :].strip(" :.-")
    return s


_NEXT_ACTION = r"(?=(?:\s*,|\s+and\s+|\s+also\s+|\s+then\s+)\s*(?:generate|create|make|explain|tell|write|summarize|what is)\b|$)"


def _clean_clause(s: str) -> str:
    out = (s or "").strip().strip(",.;:-")
    out = out.strip().strip('"').strip("'").strip()
    return out


def _find_clause(text: str, patterns: List[str]) -> str:
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            v = _clean_clause(m.group(1))
            if v:
                return v
    return ""


def _remove_tool_clauses(text: str) -> str:
    s = text
    patterns = [
        r"(?:generate|create|make)\s+(?:an?\s+)?(?:image|picture|photo)(?:\s+for|\s+of)?\s+.+?" + _NEXT_ACTION,
        r"(?:generate|create|make)\s+audio(?:\s+for|\s+saying|\s+of)?\s+.+?" + _NEXT_ACTION,
        r"(?:generate|create|make)\s+(?:an?\s+)?(?:pdf|document|docx?|txt|text file)(?:\s+on|\s+about|\s+for)?\s+.+?" + _NEXT_ACTION,
    ]
    for p in patterns:
        s = re.sub(p, " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(and|also|then)\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" ,.;:-")
    return s


def _doc_format_from_text(text: str) -> str:
    s = (text or "").lower()
    if "pdf" in s:
        return "pdf"
    if "docx" in s or "word" in s or " ms doc" in s or " ms-doc" in s or " .doc" in s:
        return "doc"
    if "txt" in s or "text file" in s or "plain text" in s:
        return "txt"
    if "markdown" in s or " md " in f" {s} ":
        return "md"
    return "txt"


def _is_news_query(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ("news", "headline", "headlines", "latest", "recent", "today", "update"))


def tool_router_node():
    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        plan = RunPlan.model_validate(state["plan"])
        flags = plan.flags or {}
        tasks: List[Dict[str, Any]] = []
        user_text = state.get("user_text", "")
        user_l = user_text.lower()
        text_query = _remove_tool_clauses(user_text)
        intent = state.get("intent") or {}
        runtime = state.get("plan_runtime") or {}
        linked = state.get("linked_artifact") or {}
        last_image_prompt = (state.get("last_image_prompt") or "").strip()

        if flags.get("needs_web"):
            src = plan.web_source or "tavily"
            sources = [src]
            if src == "tavily" and not _is_news_query(user_text):
                sources.append("wikipedia")
            tasks.append({
                "id": str(uuid4())[:8], "kind": "web",
                "query": user_text, "top_k": 5,
                "sources": sources,
            })

        if flags.get("needs_rag"):
            tasks.append({"id": str(uuid4())[:8], "kind": "rag", "query": user_text, "top_k": 5})
        if flags.get("needs_kb_rag"):
            tasks.append({"id": str(uuid4())[:8], "kind": "kb_rag", "query": user_text, "top_k": 6})

        image_edit_cues = (
            "add ",
            "replace ",
            "change ",
            "make it ",
            "but it",
            "not ",
            "fix ",
            "update ",
            "background",
        )
        linked_prompt = (linked.get("prompt") or "").strip() if linked.get("kind") == "image" else ""
        wants_image_edit = (
            (intent.get("intent_type") == "edit" and intent.get("target_modality") == "image")
            or (bool(linked_prompt) and any(c in user_l for c in image_edit_cues))
            or (bool(last_image_prompt) and any(c in user_l for c in image_edit_cues))
        )
        if flags.get("needs_image_gen") or wants_image_edit:
            prompt = _find_clause(
                user_text,
                [
                    r"(?:generate|create|make)\s+(?:an?\s+)?(?:image|picture|photo)(?:\s+for|\s+of)?\s+(.+?)" + _NEXT_ACTION,
                    r"(?:image|picture|photo)\s+of\s+(.+?)" + _NEXT_ACTION,
                ],
            ) or _extract_quoted(
                _strip_prefixes(
                    user_text,
                    [
                        "generate image for",
                        "create image for",
                        "make image for",
                        "image for",
                        "generate an image for",
                    ],
                )
            )
            if wants_image_edit:
                base_prompt = linked_prompt or last_image_prompt
                prompt = (
                    f"{base_prompt}\n\n"
                    f"Apply this edit request: {user_text}\n"
                    "Keep the same main subject unless the user explicitly changes it."
                )
            tasks.append(
                {
                    "id": str(uuid4())[:8],
                    "kind": "image_gen",
                    "prompt": prompt,
                    "size": "1024x1024",
                    "subject_lock": runtime.get("subject_lock"),
                }
            )

        if flags.get("needs_tts"):
            explicit_tts_request = any(
                k in user_l for k in ("audio", "voice", "tts", "speak", "read aloud", "narrate")
            )
            text = _find_clause(
                user_text,
                [
                    r"(?:generate|create|make)\s+audio(?:\s+for|\s+saying|\s+of)?\s+(.+?)" + _NEXT_ACTION,
                    r"(?:say|speak)\s+(.+?)" + _NEXT_ACTION,
                ],
            ) or _extract_quoted(
                _strip_prefixes(
                    user_text,
                    [
                        "generate audio for",
                        "create audio for",
                        "make audio for",
                        "audio for",
                        "say",
                        "speak",
                    ],
                )
            )
            if explicit_tts_request and text:
                tasks.append({"id": str(uuid4())[:8], "kind": "tts", "text": text, "voice": "alloy"})

        if flags.get("needs_doc"):
            doc = next((a for a in state.get("attachments", []) if a.get("kind") == "doc"), None)
            if doc:
                tasks.append(
                    {
                        "id": str(uuid4())[:8],
                        "kind": "doc",
                        "instruction": "extract",
                        "attachment_id": doc["id"],
                        "format": _doc_format_from_text(user_text),
                    }
                )
            else:
                prompt = _find_clause(
                    user_text,
                    [
                        r"(?:generate|create|make)\s+(?:an?\s+)?(?:pdf|document|docx?|txt|text file)(?:\s+on|\s+about|\s+for)?\s+(.+?)" + _NEXT_ACTION,
                        r"(?:doc|document)\s+about\s+(.+?)" + _NEXT_ACTION,
                    ],
                ) or _extract_quoted(
                    _strip_prefixes(
                        user_text,
                        ["generate a doc about", "create a doc about", "make a doc about", "doc about"],
                    )
                )
                tasks.append(
                    {
                        "id": str(uuid4())[:8],
                        "kind": "doc",
                        "instruction": "generate",
                        "prompt": prompt,
                        "format": _doc_format_from_text(user_text),
                    }
                )

        # Safety net: only explicit generation/export requests should route to doc generation.
        explicit_doc_request = (
            any(k in user_l for k in ("pdf", "document", "docx", "text file", "txt", "markdown", " md "))
            and any(k in user_l for k in ("generate", "create", "make", "write", "export"))
        )
        has_doc_task = any(t.get("kind") == "doc" for t in tasks)
        if explicit_doc_request and not has_doc_task:
            prompt = _find_clause(
                user_text,
                [r"(?:pdf|document|docx?|txt|text file)(?:\s+on|\s+about|\s+for)?\s+(.+?)" + _NEXT_ACTION],
            ) or _extract_quoted(user_text) or user_text
            tasks.append(
                {
                    "id": str(uuid4())[:8],
                    "kind": "doc",
                    "instruction": "generate",
                    "prompt": prompt,
                    "format": _doc_format_from_text(user_text),
                }
            )

        if flags.get("needs_vision") and state.get("attachments"):
            img = next((a for a in state["attachments"] if a.get("kind") == "image"), None)
            if img:
                tasks.append(
                    {
                        "id": str(uuid4())[:8],
                        "kind": "vision",
                        "prompt": user_text,
                        "image_attachment_id": img["id"],
                    }
                )

        plan.tool_tasks = tasks
        return {
            "plan": plan.model_dump(),
            "tasks": tasks,
            "text_query": text_query,
            "agent_memory": push_note(
                state,
                node="tool_router",
                summary="Tool lanes selected",
                extra={"task_kinds": [t.get("kind") for t in tasks], "count": len(tasks)},
            ),
        }
    return _run
