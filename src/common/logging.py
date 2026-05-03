"""Structured logging using structlog.

In production we emit JSON lines so they're trivial to ship to Loki /
Elasticsearch / CloudWatch. In dev we emit colored console output.

All logs include: timestamp, level, logger, deployment_id, environment.
Per-request and per-camera context is bound by callers via
`structlog.contextvars`.
"""
from __future__ import annotations

import logging
import sys

import structlog

from config.settings import settings


def configure_logging() -> None:
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        timestamper,
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.MODULE,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
    ]

    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, settings.log_level.upper(), logging.INFO)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Pipe stdlib logging (FastAPI/uvicorn/sqlalchemy) through structlog too.
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        deployment_id=settings.deployment_id,
        environment=settings.environment,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
