# api/routes_upload.py
from __future__ import annotations
from pathlib import Path
from uuid import uuid4
import re

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

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


@router.get("/uploads/{session_id}")
def list_uploads(session_id: str):
    sess = get_session(session_id)
    return {"attachments": list(sess.get("attachments", []))}


@router.delete("/uploads/{session_id}/{attachment_id}")
def remove_upload(session_id: str, attachment_id: str):
    sess = get_session(session_id)
    atts = list(sess.get("attachments", []))
    keep = [a for a in atts if str(a.get("id")) != attachment_id]
    removed = next((a for a in atts if str(a.get("id")) == attachment_id), None)
    if not removed:
        raise HTTPException(status_code=404, detail="Attachment not found")

    sess["attachments"] = keep
    mem = sess.get("artifact_memory") if isinstance(sess.get("artifact_memory"), dict) else {}
    if isinstance(mem, dict):
        kind = str(removed.get("kind") or "").lower()
        if kind == "doc":
            mem["doc"] = None
        elif kind == "image":
            mem["image"] = None
        sess["artifact_memory"] = mem
    path = str(removed.get("path") or "")
    if path:
        try:
            p = Path(path)
            if p.exists() and p.is_file():
                p.unlink()
        except Exception:
            # Best effort file cleanup; context removal already completed.
            pass
    return {"ok": True, "removed_id": attachment_id}
