# api/app.py
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.src.core.config import bootstrap_env
from backend.src.api.routes_chat import router as chat_router
from backend.src.api.routes_upload import router as upload_router
from backend.src.api.routes_assets import router as assets_router
from backend.src.api.routes_models import router as models_router

bootstrap_env()
app = FastAPI(title="OmniAgent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(upload_router, prefix="/api")
app.include_router(assets_router, prefix="/api")
app.include_router(models_router, prefix="/api")
