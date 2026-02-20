from __future__ import annotations
from typing import Any, Dict, Optional

from backend.src.graph.agent_memory import push_note


def context_node():
    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        user = (state.get("user_text") or "").strip()
        user_l = user.lower()
        memory = state.get("artifact_memory") or {}
        last_image = memory.get("image")

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
            "foreground",
            "remove ",
        )
        is_image_edit = bool(last_image) and any(c in user_l for c in image_edit_cues)

        linked_artifact: Optional[Dict[str, Any]] = None
        if is_image_edit and isinstance(last_image, dict):
            linked_artifact = {
                "kind": "image",
                "id": last_image.get("id"),
                "prompt": last_image.get("prompt"),
                "url": last_image.get("url"),
            }

        return {
            "context_bundle": {
                "has_last_image": bool(last_image),
                "is_image_edit": is_image_edit,
            },
            "linked_artifact": linked_artifact,
            "agent_memory": push_note(
                state,
                node="context",
                summary="Context prepared",
                extra={"has_last_image": bool(last_image), "is_image_edit": is_image_edit},
            ),
        }

    return _run
