# graph/intent_llm_node.py
from __future__ import annotations
from typing import Any, Dict

from backend.src.core.jsonx import extract_json
from backend.src.llm.factory import get_llm, is_not_found_error, model_candidates
from backend.src.schemas.plan import RunPlan, TextPlan


def intent_llm_node(provider: str, model: str):
    def _fast_intent(state: Dict[str, Any]) -> Dict[str, Any] | None:
        user = (state.get("user_text") or "").strip()
        user_l = user.lower()
        ctx = state.get("context_bundle") or {}
        linked = state.get("linked_artifact") or {}

        def has_any(parts: tuple[str, ...]) -> bool:
            return any(p in user_l for p in parts)

        wants_doc = has_any(("doc", "document", "pdf", "notes"))
        wants_explain_text = has_any(
            (
                "tell me",
                "explain",
                "what is",
                "why",
                "how",
                "summarize",
                "write",
                "story",
                "poem",
            )
        )
        wants_text = wants_explain_text
        if wants_doc:
            # For document requests, default to doc-only unless user explicitly asks for both.
            wants_text = wants_explain_text and has_any(
                (
                    "also",
                    "and explain",
                    "plus explain",
                    "with explanation",
                    "and summarize",
                    "plus summarize",
                    "along with explanation",
                )
            )

        if ctx.get("is_image_edit") and linked.get("kind") == "image":
            plan = RunPlan(
                mode="tools_only",
                text=TextPlan(enabled=False),
                flags={"needs_image_gen": True},
                web_source=None,
                note="fast_intent:image_edit",
            )
            return {
                "plan": plan.model_dump(),
                "intent": {
                    "intent_type": "edit",
                    "target_modality": "image",
                    "confidence": 0.95,
                },
            }

        if has_any(("generate", "create", "make")) and has_any(("image", "photo", "picture")):
            plan = RunPlan(
                mode="text_plus_tools" if wants_text else "tools_only",
                text=TextPlan(enabled=wants_text),
                flags={"needs_image_gen": True},
                web_source=None,
                note="fast_intent:image_create_with_text" if wants_text else "fast_intent:image_create",
            )
            return {
                "plan": plan.model_dump(),
                "intent": {
                    "intent_type": "create" if not wants_text else "mixed",
                    "target_modality": "image" if not wants_text else "text+image",
                    "confidence": 0.9 if not wants_text else 0.93,
                },
            }

        # Fast visual fallback:
        # If user asks to generate/create/make something without explicitly saying
        # doc/audio/web/text, treat it as image generation for UX consistency.
        if has_any(("generate", "create", "make")) and not has_any(
            (
                "audio",
                "voice",
                "speech",
                "tts",
                "doc",
                "document",
                "pdf",
                "web",
                "search",
                "latest",
                "current",
                "summarize",
                "explain",
                "code",
                "text",
            )
        ):
            plan = RunPlan(
                mode="text_plus_tools" if wants_text else "tools_only",
                text=TextPlan(enabled=wants_text),
                flags={"needs_image_gen": True},
                web_source=None,
                note="fast_intent:image_fallback_with_text" if wants_text else "fast_intent:image_fallback",
            )
            return {
                "plan": plan.model_dump(),
                "intent": {
                    "intent_type": "create" if not wants_text else "mixed",
                    "target_modality": "image" if not wants_text else "text+image",
                    "confidence": 0.78 if not wants_text and ctx.get("has_last_image") else (0.72 if not wants_text else 0.9),
                },
            }

        if has_any(("generate", "create", "make")) and has_any(("audio", "voice", "speech", "tts")):
            plan = RunPlan(
                mode="text_plus_tools" if wants_text else "tools_only",
                text=TextPlan(enabled=wants_text),
                flags={"needs_tts": True},
                web_source=None,
                note="fast_intent:audio_create_with_text" if wants_text else "fast_intent:audio_create",
            )
            return {
                "plan": plan.model_dump(),
                "intent": {
                    "intent_type": "create" if not wants_text else "mixed",
                    "target_modality": "audio" if not wants_text else "text+audio",
                    "confidence": 0.9 if not wants_text else 0.93,
                },
            }

        if has_any(("generate", "create", "make")) and wants_doc:
            plan = RunPlan(
                mode="text_plus_tools" if wants_text else "tools_only",
                text=TextPlan(enabled=wants_text),
                flags={"needs_doc": True},
                web_source=None,
                note="fast_intent:doc_create_with_text" if wants_text else "fast_intent:doc_create",
            )
            return {
                "plan": plan.model_dump(),
                "intent": {
                    "intent_type": "create" if not wants_text else "mixed",
                    "target_modality": "doc" if not wants_text else "text+doc",
                    "confidence": 0.9 if not wants_text else 0.93,
                },
            }
        return None

    def _prompt(user: str, has_files: bool) -> str:
        return (
            "Classify user request for an assistant.\n"
            "Return ONLY JSON with keys:\n"
            "mode: text_only|text_plus_tools|tools_only\n"
            "flags: {needs_web, needs_rag, needs_doc, needs_vision, needs_tts, needs_image_gen}\n"
            "web_source: tavily|wikipedia|arxiv|null\n"
            "Rules: Mentioning the word 'web' is NOT a web-search request.\n"
            "Use web only if user asks search/latest/current/cite/sources/papers.\n"
            f"has_files={has_files}\nUSER:\n{user}\n"
        )

    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        fast = _fast_intent(state)
        if fast:
            return fast

        user = state.get("user_text", "")
        has_files = bool(state.get("attachments"))
        prompt = _prompt(user, has_files)
        raw = ""
        last_err: Exception | None = None
        candidates = model_candidates(provider, model)
        for i, candidate in enumerate(candidates):
            llm = get_llm(provider, candidate, streaming=False, temperature=0.0)
            try:
                raw = (llm.invoke(prompt).content or "").strip()
                break
            except Exception as e:
                last_err = e
                if i < len(candidates) - 1 and is_not_found_error(e):
                    continue
                raise
        if not raw and last_err:
            raise last_err
        data = extract_json(raw)
        web_source = data.get("web_source")
        if isinstance(web_source, str) and web_source.strip().lower() in {"", "null", "none"}:
            web_source = None

        plan = RunPlan(
            mode=data.get("mode", "text_only"),
            text=TextPlan(enabled=data.get("mode", "text_only") != "tools_only"),
            flags=data.get("flags", {}) or {},
            web_source=web_source,
            note="micro_intent_llm",
        )
        return {
            "plan": plan.model_dump(),
            "intent": {
                "intent_type": "chat",
                "target_modality": "text",
                "confidence": 0.6,
            },
        }
    return _run
