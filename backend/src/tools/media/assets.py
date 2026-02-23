# tools/media/assets.py
from __future__ import annotations
from pathlib import Path
import re
import time
from typing import Tuple


BASE = Path("backend/data/uploads")


def save_asset(session_id: str, ext: str, data: bytes) -> Tuple[str, str]:
    d = BASE / session_id
    d.mkdir(parents=True, exist_ok=True)
    safe_session = re.sub(r"[^A-Za-z0-9_-]+", "_", str(session_id or "session")).strip("_") or "session"
    ts_ms = int(time.time() * 1000)
    name = f"{safe_session}_{ts_ms}.{ext.lstrip('.')}"
    (d / name).write_bytes(data)
    return name, f"/api/assets/{session_id}/{name}"
