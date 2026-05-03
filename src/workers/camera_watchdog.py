"""Camera watchdog — fires alerts when a camera stops producing frames.

Polls the camera_health table every 30s. If a camera that was up has not
sent a frame in ALERT_CAMERA_OFFLINE_SECONDS, it sends a webhook alert,
flips is_up to false, and records an audit event.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

from sqlalchemy import select

from config.settings import settings
from src.common.alerts import send_alert
from src.common.audit import record as audit
from src.common.db import session_scope
from src.common.logging import configure_logging, get_logger
from src.common.metrics import camera_up
from src.evidence.models import CameraHealth

log = get_logger("watchdog")


def tick() -> None:
    threshold = datetime.utcnow() - timedelta(seconds=settings.alert_camera_offline_seconds)
    with session_scope() as s:
        rows = s.scalars(select(CameraHealth)).all()
        for row in rows:
            if row.last_frame_at is None:
                continue
            offline = row.last_frame_at < threshold
            camera_up.labels(camera_id=row.camera_id).set(0 if offline else 1)
            if offline and row.is_up:
                row.is_up = False
                send_alert(
                    "warn", "Camera offline",
                    f"{row.camera_id} last frame at {row.last_frame_at.isoformat()}",
                    dedup_key=f"offline:{row.camera_id}",
                )
                audit("camera.offline", actor_label="watchdog", target_type="camera", target_id=row.camera_id)
                log.warning("camera_offline", camera_id=row.camera_id, last_frame_at=row.last_frame_at)
            elif not offline and not row.is_up:
                row.is_up = True
                send_alert("info", "Camera back online", row.camera_id, dedup_key=f"online:{row.camera_id}")
                audit("camera.online", actor_label="watchdog", target_type="camera", target_id=row.camera_id)


def main() -> None:
    configure_logging()
    log.info("camera_watchdog_started", threshold_seconds=settings.alert_camera_offline_seconds)
    while True:
        try:
            tick()
        except Exception as exc:
            log.exception("watchdog_tick_failed", error=str(exc))
        time.sleep(30)


if __name__ == "__main__":
    main()
