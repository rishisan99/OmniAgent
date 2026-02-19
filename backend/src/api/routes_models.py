# api/routes_models.py
from __future__ import annotations
from fastapi import APIRouter

from backend.src.core.constants import PROVIDER_MODELS, SUPPORTED_PROVIDERS, DEFAULT_MODEL, DEFAULT_PROVIDER

router = APIRouter()


@router.get("/models")
def models():
    return {
        "providers": list(SUPPORTED_PROVIDERS),
        "models": PROVIDER_MODELS,
        "default": {"provider": DEFAULT_PROVIDER, "model": DEFAULT_MODEL},
    }
