# core/config.py
from __future__ import annotations
from dotenv import load_dotenv

def bootstrap_env() -> None:
    """Load .env into environment variables."""
    load_dotenv()
