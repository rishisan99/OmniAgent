# api/routes_upload.py
from __future__ import annotations
from pathlib import Path
from uuid import uuid4
import re

from fastapi import APIRouter, UploadFile, File, Form

from backend.src.session.store import get_session

router = APIRouter()
BASE = Path("backend/data/uploads")


@router.post("/upload")
async def upload(session_id: str = Form(...), f: UploadFile = File(...)):
    sid = session_id
    fid = str(uuid4())[:8]
    out_dir = BASE / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", f.filename or "upload.bin")
    path = out_dir / f"{fid}_{safe_name}"

    # Stream upload to disk to keep request responsive for larger files.
    with path.open("wb") as w:
        while True:
            chunk = await f.read(1024 * 1024)
            if not chunk:
                break
            w.write(chunk)

    kind = "image" if (f.content_type or "").startswith("image/") else ("audio" if (f.content_type or "").startswith("audio/") else "doc")
    att = {"id": fid, "kind": kind, "name": f.filename, "mime": f.content_type, "path": str(path)}

    get_session(sid)["attachments"].append(att)
    return att
