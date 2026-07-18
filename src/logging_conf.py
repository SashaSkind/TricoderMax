"""
Structured logging with the study accession as correlation id.

Usage:
    from src.logging_conf import configure, get_logger
    configure()
    log = get_logger(__name__).bind(accession="CQ500-0123")
    log.info("panel.start", n_modules=3)
"""

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def configure(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None):
    if not _configured:
        configure()
    return structlog.get_logger(name)
