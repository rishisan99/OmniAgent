# core/logging.py
from __future__ import annotations
import logging
import os
from typing import Optional


def get_logger(name: str = "omniagent", level: Optional[str] = None) -> logging.Logger:
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(lvl)
    h = logging.StreamHandler()
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    h.setFormatter(logging.Formatter(fmt))
    logger.addHandler(h)
    logger.propagate = False
    return logger
