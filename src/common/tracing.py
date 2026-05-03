"""Sentry init for production crash reporting (optional)."""
from __future__ import annotations

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from config.settings import settings


def init_sentry() -> None:
    if not settings.sentry_dsn:
        return
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        release=f"cctv@{settings.deployment_id}",
        traces_sample_rate=0.05 if settings.is_production else 0.0,
        send_default_pii=False,
        integrations=[FastApiIntegration(), SqlalchemyIntegration()],
    )
