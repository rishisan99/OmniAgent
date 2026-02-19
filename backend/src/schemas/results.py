# schemas/results.py
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel


class Citation(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None


class ToolResult(BaseModel):
    task_id: str
    kind: str
    ok: bool = True
    data: Dict[str, Any] = {}
    citations: List[Citation] = []
    error: Optional[str] = None


class FinalAnswer(BaseModel):
    text: str
    blocks: List[Dict[str, Any]] = []  # optional frontend "blocks"
    citations: List[Citation] = []
