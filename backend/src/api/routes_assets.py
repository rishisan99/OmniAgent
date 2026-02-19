# api/routes_assets.py
from __future__ import annotations
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()
BASE = Path("backend/data/uploads")


@router.get("/assets/{session_id}/{filename}")
def asset(session_id: str, filename: str):
    path = BASE / session_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)
