# core/constants.py
from __future__ import annotations

SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini")

# frontend dropdown uses this (tools remain fixed/hidden)
PROVIDER_MODELS = {
    "openai": ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"),
    "anthropic": (
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-7-sonnet-latest",
    ),
    "gemini": (
        "gemini-1.5-flash-002",
        "gemini-1.5-pro-002",
        "gemini-2.0-flash",
    ),
}

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o-mini"

MAX_USER_CHARS = 15_000
MAX_HISTORY_MESSAGES = 30
MAX_ATTACHMENTS = 8

# SSE
SSE_RETRY_MS = 1500
