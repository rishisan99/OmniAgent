# schemas/plan.py
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

Mode = Literal["text_only", "text_plus_tools", "tools_only"]
WebSource = Literal["tavily", "wikipedia", "arxiv"]


class TextPlan(BaseModel):
    enabled: bool = True
    style: Literal["direct", "bullet", "detailed"] = "direct"
    instruction: str = ""


class RunPlan(BaseModel):
    mode: Mode
    text: TextPlan
    flags: Dict[str, bool] = Field(default_factory=dict)
    web_source: Optional[WebSource] = None
    tool_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    note: Optional[str] = None
