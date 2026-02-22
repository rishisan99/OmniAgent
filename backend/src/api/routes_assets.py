# api/routes_assets.py
from __future__ import annotations
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.src.core.logging import get_logger

router = APIRouter()
BASE = Path("backend/data/uploads")
logger = get_logger("omniagent.api.assets")


@router.get("/assets/{session_id}/{filename}")
def asset(session_id: str, filename: str):
    path = BASE / session_id / filename
    logger.info("ASSET_REQUEST session_id=%s filename=%s path=%s", session_id, filename, path)
    if not path.exists():
        logger.warning("ASSET_MISS session_id=%s filename=%s", session_id, filename)
        raise HTTPException(status_code=404, detail="Not found")
    logger.info("ASSET_HIT session_id=%s filename=%s size_bytes=%s", session_id, filename, path.stat().st_size)
    return FileResponse(path)
