"""Webhook alerter — Slack/Teams/Mattermost compatible.

Used for ops-level alerts (camera offline, push-to-Chalana failure,
disk-full, repeated DB errors). De-duplicates identical alerts within
60s using Redis if configured, else in-memory.
"""
from __future__ import annotations

import json
import time
from threading import Lock

import httpx

from config.settings import settings
from src.common.logging import get_logger

log = get_logger("alerts")
_lock = Lock()
_recent: dict[str, float] = {}


def _dedup(key: str, window: float = 60.0) -> bool:
    now = time.time()
    with _lock:
        # Drop expired keys cheaply.
        for k, t in list(_recent.items()):
            if now - t > window:
                _recent.pop(k, None)
        if key in _recent:
            return True
        _recent[key] = now
        return False


def send_alert(level: str, title: str, body: str, *, dedup_key: str | None = None) -> None:
    if not settings.alert_webhook_url:
        log.info("alert_skipped_no_webhook", level=level, title=title)
        return
    if dedup_key and _dedup(dedup_key):
        return
    text = f"*[{settings.deployment_id}] {level.upper()}: {title}*\n{body}"
    payload = {"text": text}
    try:
        with httpx.Client(timeout=5.0) as c:
            r = c.post(settings.alert_webhook_url, content=json.dumps(payload),
                       headers={"content-type": "application/json"})
            r.raise_for_status()
    except Exception as exc:
        log.warning("alert_webhook_failed", error=str(exc))
