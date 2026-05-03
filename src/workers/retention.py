"""Retention & archival worker.

Runs as a separate long-lived process (systemd unit + cron-like loop).
Every 6 hours it:
  - finds violations older than EVIDENCE_RETENTION_DAYS
  - for each, copies its assets to ARCHIVE_S3_BUCKET if set
  - deletes the live assets (S3 or local)
  - marks the violation with extras.archived=true

We never delete the DB row — the row is the legal record. Only the
binary assets are tiered to cold storage.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

from sqlalchemy import select

from config.settings import settings
from src.common.alerts import send_alert
from src.common.db import session_scope
from src.common.logging import configure_logging, get_logger
from src.common.object_store import LocalObjectStore, S3ObjectStore, get_object_store
from src.evidence.models import Violation

log = get_logger("retention")


def _archive_one(v: Violation, live, archive) -> None:
    for key in (v.frame_path, v.annotated_path, v.plate_crop_path):
        if not key or not live.exists(key):
            continue
        data = live.get_bytes(key)
        archive.put_bytes(key, data, "image/jpeg")
        live.delete(key)


def run_once() -> int:
    cutoff = datetime.utcnow() - timedelta(days=settings.evidence_retention_days)
    archive = None
    if settings.archive_s3_bucket:
        archive = S3ObjectStore(settings.archive_s3_bucket, settings.s3_region, settings.s3_prefix)

    live = get_object_store()
    with session_scope() as s:
        rows = s.scalars(
            select(Violation)
            .where(Violation.timestamp < cutoff)
            .where((Violation.extras.is_(None)) | (Violation.extras["archived"].astext != "true"))  # type: ignore
            .limit(500)
        ).all()
        n = 0
        for v in rows:
            try:
                if archive:
                    _archive_one(v, live, archive)
                else:
                    for key in (v.frame_path, v.annotated_path, v.plate_crop_path):
                        if key and live.exists(key):
                            live.delete(key)
                v.extras = {**(v.extras or {}), "archived": "true",
                            "archived_at": datetime.utcnow().isoformat()}
                n += 1
            except Exception as exc:
                log.exception("archive_failed", id=v.id, error=str(exc))
        log.info("retention_pass", archived=n, cutoff=cutoff.isoformat())
        return n


def main() -> None:
    configure_logging()
    log.info("retention_worker_started",
             retention_days=settings.evidence_retention_days,
             archive_bucket=settings.archive_s3_bucket or "(none)")
    while True:
        try:
            run_once()
        except Exception as exc:
            log.exception("retention_pass_failed", error=str(exc))
            send_alert("error", "Retention pass failed", str(exc), dedup_key="retention_failure")
        time.sleep(6 * 3600)


if __name__ == "__main__":
    main()
