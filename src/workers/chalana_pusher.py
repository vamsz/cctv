"""Chalana push worker.

Reads approved violations not yet pushed, posts them to the Chalana
issuance API with retry/backoff, and marks them pushed on success.
The Chalana API contract is a placeholder — adjust the payload shape to
match the actual handoff spec the police give you.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy import select
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from src.common.alerts import send_alert
from src.common.audit import record as audit
from src.common.db import session_scope
from src.common.logging import configure_logging, get_logger
from src.evidence.models import Violation

log = get_logger("chalana")


@retry(
    retry=retry_if_exception_type(httpx.HTTPError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _push(client: httpx.Client, v: Violation) -> Optional[str]:
    payload = {
        "deployment_id": settings.deployment_id,
        "violation_id": v.id,
        "code": v.code,
        "camera_id": v.camera_id,
        "timestamp": v.timestamp.isoformat() + "Z",
        "plate_text": v.plate_text,
        "ocr_confidence": v.plate_ocr_confidence,
        "rule_confidence": v.rule_confidence,
        "evidence_sha256": v.sha256,
        "chain_hash": v.chain_hash,
    }
    headers = {"authorization": f"Bearer {settings.chalana_api_key.get_secret_value()}",
               "content-type": "application/json"}
    r = client.post(settings.chalana_api_url, json=payload, headers=headers, timeout=10)
    r.raise_for_status()
    body = r.json() if r.content else {}
    return body.get("reference") or body.get("id")


def run_once() -> int:
    if not settings.chalana_push_enabled or not settings.chalana_api_url:
        return 0
    with session_scope() as s:
        rows = s.scalars(
            select(Violation)
            .where(Violation.status == "approved")
            .where(Violation.pushed_to_chalana.is_(False))
            .order_by(Violation.reviewed_at.asc())
            .limit(50)
        ).all()
        if not rows:
            return 0
        n = 0
        with httpx.Client() as client:
            for v in rows:
                try:
                    ref = _push(client, v)
                    v.pushed_to_chalana = True
                    v.pushed_at = datetime.utcnow()
                    v.chalana_reference = ref
                    audit("chalana.pushed", actor_label="chalana_pusher",
                          target_type="violation", target_id=str(v.id),
                          detail={"reference": ref})
                    n += 1
                except httpx.HTTPError as exc:
                    log.exception("chalana_push_failed", violation_id=v.id, error=str(exc))
                    send_alert("error", "Chalana push failed",
                               f"violation {v.id}: {exc}",
                               dedup_key=f"chalana:{v.id}")
        return n


def main() -> None:
    configure_logging()
    log.info("chalana_pusher_started",
             enabled=settings.chalana_push_enabled,
             url=settings.chalana_api_url or "(unset)")
    while True:
        try:
            n = run_once()
            if n:
                log.info("chalana_pushed_batch", count=n)
        except Exception as exc:
            log.exception("chalana_loop_error", error=str(exc))
        time.sleep(15)


if __name__ == "__main__":
    main()
