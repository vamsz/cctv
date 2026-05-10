"""Violence incident lifecycle manager.

State machine per camera:
  None  →  "suspected"  (first Stage-1 positive)
        →  "active"     (ESCALATE_FRAMES consecutive Stage-1 positives)
        →  "resolved"   (RESOLVE_FRAMES consecutive Stage-1 negatives)

The caller (pipeline) is responsible for persisting to the DB when
the returned incident is not None (i.e., on state transitions only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .pose_stage1 import Stage1Result

ESCALATE_FRAMES = 10   # consecutive stage-1 hits to escalate suspected → active
RESOLVE_FRAMES  = 30   # consecutive quiet frames to auto-resolve


@dataclass
class ViolenceIncident:
    camera_id: str
    started_at: datetime
    status: str = "suspected"          # suspected | active | resolved
    score: float = 0.0
    flags: list[str] = field(default_factory=list)
    resolved_at: Optional[datetime] = None
    db_id: Optional[int] = None        # filled after first DB persist
    _hits: int = field(default=0, repr=False)
    _quiet: int = field(default=0, repr=False)


class IncidentManager:
    """Per-camera violence incident tracker."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._inc: Optional[ViolenceIncident] = None

    # ------------------------------------------------------------------

    def update(self, result: Stage1Result) -> Optional[ViolenceIncident]:
        """
        Feed one Stage1Result. Returns the incident only on state
        transitions (suspected, active, resolved) so the caller can
        write to DB / push WS exactly once per transition.
        """
        inc = self._inc
        changed: Optional[ViolenceIncident] = None

        if result.active:
            if inc is None:
                inc = ViolenceIncident(
                    camera_id=self.camera_id,
                    started_at=datetime.utcnow(),
                    score=result.score,
                    flags=list(result.flags),
                )
                self._inc = inc
                changed = inc        # "suspected" transition

            inc._hits += 1
            inc._quiet = 0
            inc.score = max(inc.score, result.score)
            inc.flags = list(set(inc.flags) | set(result.flags))

            if inc._hits >= ESCALATE_FRAMES and inc.status == "suspected":
                inc.status = "active"
                changed = inc        # "active" transition

        else:
            if inc is not None:
                inc._quiet += 1
                if inc._quiet >= RESOLVE_FRAMES:
                    inc.status = "resolved"
                    inc.resolved_at = datetime.utcnow()
                    changed = inc    # "resolved" transition
                    self._inc = None

        return changed

    @property
    def current(self) -> Optional[ViolenceIncident]:
        return self._inc

    def is_active(self) -> bool:
        return self._inc is not None and self._inc.status in ("suspected", "active")

    def current_db_id(self) -> Optional[int]:
        return self._inc.db_id if self._inc is not None else None
