"""Plate watchlist — loaded from DB at startup, hot-reloaded on change.

Stored in the `watchlist` DB table. Also accepts a seed CSV at
data/watchlist_seed.csv (one plate per line, optionally tab-separated
with a reason column).
"""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Optional

_lock = threading.RLock()
_entries: dict[str, str] = {}   # plate_pattern -> reason
_loaded = False


def _normalize(plate: str) -> str:
    return plate.upper().strip().replace(" ", "")


def load_from_db() -> None:
    global _loaded
    try:
        from sqlalchemy import select, text
        from src.common.db import session_scope
        from src.evidence.models import WatchlistEntry
        with session_scope() as s:
            rows = s.scalars(select(WatchlistEntry).where(WatchlistEntry.is_active == True)).all()
            with _lock:
                _entries.clear()
                for r in rows:
                    _entries[_normalize(r.plate_pattern)] = r.reason or ""
        _loaded = True
    except Exception:
        pass


def load_from_csv(path: Path) -> None:
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        with _lock:
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                plate = _normalize(row[0])
                reason = row[1].strip() if len(row) > 1 else ""
                _entries[plate] = reason


def is_watchlisted(plate_text: str, match_mode: str = "prefix") -> tuple[bool, str]:
    if not plate_text:
        return False, ""
    normalized = _normalize(plate_text)
    with _lock:
        if match_mode == "exact":
            reason = _entries.get(normalized)
            return (True, reason) if reason is not None else (False, "")
        # prefix: any watchlist entry that is a prefix of the plate
        for pattern, reason in _entries.items():
            if normalized.startswith(pattern) or pattern.startswith(normalized):
                return True, reason
        return False, ""


def add_entry(plate_pattern: str, reason: str = "") -> None:
    with _lock:
        _entries[_normalize(plate_pattern)] = reason


def remove_entry(plate_pattern: str) -> bool:
    with _lock:
        return _entries.pop(_normalize(plate_pattern), None) is not None


def all_entries() -> dict[str, str]:
    with _lock:
        return dict(_entries)


def start_background_reload(interval_seconds: int = 300) -> None:
    def _loop():
        while True:
            time.sleep(interval_seconds)
            load_from_db()
    t = threading.Thread(target=_loop, daemon=True, name="watchlist-reload")
    t.start()
