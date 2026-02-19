# schemas/tasks.py
from __future__ import annotations
from typing import Annotated, List, Literal, Optional, Union
from pydantic import BaseModel, Field, TypeAdapter


class BaseTask(BaseModel):
    id: str = Field(..., description="unique task id")
    kind: str = Field(..., description="task discriminator")


class TextTask(BaseTask):
    kind: Literal["text"] = "text"
    prompt: str


class WebTask(BaseTask):
    kind: Literal["web"] = "web"
    query: str
    top_k: int = 5
    sources: List[Literal["tavily", "wikipedia", "arxiv"]] = ["tavily"]


class RagTask(BaseTask):
    kind: Literal["rag"] = "rag"
    query: str
    top_k: int = 5
    scope: Literal["session"] = "session"


class VisionTask(BaseTask):
    kind: Literal["vision"] = "vision"
    prompt: str
    image_attachment_id: str


class ImageGenTask(BaseTask):
    kind: Literal["image_gen"] = "image_gen"
    prompt: str
    size: Literal["512x512", "1024x1024"] = "1024x1024"
    subject_lock: Optional[str] = None


class AudioTask(BaseTask):
    kind: Literal["tts"] = "tts"
    text: str
    voice: str = "alloy"


class DocTask(BaseTask):
    kind: Literal["doc"] = "doc"
    instruction: str
    attachment_id: Optional[str] = None
    prompt: Optional[str] = None
    format: Literal["pdf", "doc", "txt", "md"] = "txt"


Task = Annotated[
    Union[TextTask, WebTask, RagTask, VisionTask, ImageGenTask, AudioTask, DocTask],
    Field(discriminator="kind"),
]

task_adapter = TypeAdapter(Task)
