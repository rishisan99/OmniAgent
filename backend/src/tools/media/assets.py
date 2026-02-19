# tools/media/assets.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple
from uuid import uuid4


BASE = Path("backend/data/uploads")


def save_asset(session_id: str, ext: str, data: bytes) -> Tuple[str, str]:
    d = BASE / session_id
    d.mkdir(parents=True, exist_ok=True)
    name = f"{str(uuid4())[:8]}.{ext.lstrip('.')}"
    (d / name).write_bytes(data)
    return name, f"/api/assets/{session_id}/{name}"
