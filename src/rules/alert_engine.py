"""Alert priority engine — wraps the rules engine with global controls.

Responsibilities:
- Global kill-switch (enabled flag in rules.yaml)
- Per-priority cooldowns (HIGH/MEDIUM/LOW) keyed on plate+code
- Per-subject cooldowns: same plate + same rule never fires twice within cooldown
- All cooldown state lives in-process; fine for single-node deployment.
"""
from __future__ import annotations

import time
import threading
from typing import Optional

from .types import ViolationCode, ViolationEvent, VIOLATION_PRIORITY

_lock = threading.Lock()
# key: (plate_text_or_track_id, ViolationCode) -> last_fired_epoch
_last_fired: dict[tuple, float] = {}


class AlertEngine:
    def __init__(self, params: dict):
        cfg = params.get("alert_engine", {})
        self.enabled: bool = cfg.get("enabled", True)
        self._cooldowns = {
            "HIGH": float(cfg.get("cooldown_HIGH", 30)),
            "MEDIUM": float(cfg.get("cooldown_MEDIUM", 60)),
            "LOW": float(cfg.get("cooldown_LOW", 120)),
        }

    def filter(self, events: list[ViolationEvent]) -> list[ViolationEvent]:
        if not self.enabled:
            return []
        allowed = []
        for ev in events:
            if self._should_fire(ev):
                self._record(ev)
                allowed.append(ev)
        return allowed

    def _subject_key(self, ev: ViolationEvent) -> tuple:
        # prefer plate for cross-track dedup; fallback to camera+track
        subject = ev.plate_text or f"{ev.camera_id}:{ev.track_id}"
        return (subject, ev.code)

    def _should_fire(self, ev: ViolationEvent) -> bool:
        # WATCHLIST_HIT always fires — bypass cooldowns
        if ev.code == ViolationCode.WATCHLIST_HIT:
            return True
        priority = VIOLATION_PRIORITY.get(ev.code.value, "MEDIUM")
        cooldown = self._cooldowns.get(priority, 60)
        key = self._subject_key(ev)
        with _lock:
            last = _last_fired.get(key)
            return last is None or (time.time() - last) >= cooldown

    def _record(self, ev: ViolationEvent) -> None:
        key = self._subject_key(ev)
        with _lock:
            _last_fired[key] = time.time()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
