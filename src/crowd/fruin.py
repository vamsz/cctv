"""Fruin Level-of-Service crowd density classifier + safety alert engine.

Adapted from the drone-surveillance project's safety pipeline. We use
Fruin's standard A-F bands as the operator-facing severity language so
"Critical" / "Emergency" map to internationally-recognised crowd-control
thresholds rather than ad-hoc magic numbers.

Fixed-camera tuning:
  - We KEEP the India-tuned warning/danger absolute counts in rules.yaml
    for cameras without a homography (no real-world m² area). The Fruin
    level is computed from the same density estimator.
  - Surge / counter-flow / stagnation use SHORT lookback (30 s for fixed
    cameras vs 60 s on drones) — fixed cameras get richer history per
    second so the trend signal is sharper.

References:
  - Fruin, J.J. "Pedestrian Planning and Design" (1971)
  - Kumbh Mela 2025 deployment learnings (drone_surv/research.md)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FruinLevel(str, Enum):
    A = "A"   # < 0.3 p/m²    — free flow
    B = "B"   # 0.3 - 0.7      — minor conflicts
    C = "C"   # 0.7 - 1.2      — restricted flow
    D = "D"   # 1.2 - 2.5      — severely restricted
    E = "E"   # 2.5 - 5.0      — walking difficult, crush risk
    F = "F"   # > 5.0          — DANGEROUS


class AlertSeverity(str, Enum):
    INFO = "info"           # green
    WARNING = "warning"     # yellow
    CRITICAL = "critical"   # orange
    EMERGENCY = "emergency" # red


@dataclass
class SafetyAlert:
    timestamp: float
    severity: AlertSeverity
    alert_type: str               # density_threshold | surge | counter_flow | stagnation
    zone_name: str
    message: str
    density: float = 0.0
    fruin_level: FruinLevel = FruinLevel.A
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------

class FruinClassifier:
    """Pure-function classifier — no state, just thresholds."""

    THRESHOLDS = [
        (FruinLevel.A, 0.0, 0.3),
        (FruinLevel.B, 0.3, 0.7),
        (FruinLevel.C, 0.7, 1.2),
        (FruinLevel.D, 1.2, 2.5),
        (FruinLevel.E, 2.5, 5.0),
        (FruinLevel.F, 5.0, float("inf")),
    ]
    SEVERITY = {
        FruinLevel.A: AlertSeverity.INFO,
        FruinLevel.B: AlertSeverity.INFO,
        FruinLevel.C: AlertSeverity.WARNING,
        FruinLevel.D: AlertSeverity.WARNING,
        FruinLevel.E: AlertSeverity.CRITICAL,
        FruinLevel.F: AlertSeverity.EMERGENCY,
    }
    DESCRIPTIONS = {
        FruinLevel.A: "Free flow",
        FruinLevel.B: "Minor conflicts",
        FruinLevel.C: "Restricted flow",
        FruinLevel.D: "Severely restricted — deploy crowd control",
        FruinLevel.E: "Walking difficult — crush risk",
        FruinLevel.F: "EMERGENCY — dangerous crush conditions",
    }
    HEX_COLORS = {
        FruinLevel.A: "#22c55e",
        FruinLevel.B: "#84cc16",
        FruinLevel.C: "#eab308",
        FruinLevel.D: "#f97316",
        FruinLevel.E: "#ef4444",
        FruinLevel.F: "#991b1b",
    }

    @classmethod
    def classify(cls, density: float) -> FruinLevel:
        for level, lo, hi in cls.THRESHOLDS:
            if lo <= density < hi:
                return level
        return FruinLevel.F

    @classmethod
    def severity_for(cls, level: FruinLevel) -> AlertSeverity:
        return cls.SEVERITY[level]

    @classmethod
    def description_for(cls, level: FruinLevel) -> str:
        return cls.DESCRIPTIONS[level]

    @classmethod
    def hex_color(cls, level: FruinLevel) -> str:
        return cls.HEX_COLORS[level]


# ---------------------------------------------------------------------------

class SafetyMonitor:
    """Stateful safety analyser per camera.

    Tracks short-window history of (timestamp, density) and emits alerts when:
      - Fruin level reaches E or F (density itself is dangerous)
      - Density rate-of-change exceeds surge_threshold per surge_window
      - Counter-flow score exceeds threshold AND density > 2 p/m²
      - Stagnation fraction exceeds threshold AND density > 3 p/m²
    """

    def __init__(
        self,
        surge_threshold: float = 1.0,         # +1 p/m² over surge_window = surge
        surge_window: float = 30.0,           # sec — shorter than drone (60s)
        counter_flow_threshold: float = 0.6,
        stagnation_threshold: float = 0.5,
        alert_cooldown: float = 30.0,
    ):
        self.surge_threshold = surge_threshold
        self.surge_window = surge_window
        self.counter_flow_threshold = counter_flow_threshold
        self.stagnation_threshold = stagnation_threshold
        self.alert_cooldown = alert_cooldown
        # camera_id → deque[(t, density)]
        self._history: dict[str, deque] = {}
        # (camera_id, alert_type) → last_emitted_t
        self._last_alert: dict[tuple, float] = {}

    def update(
        self,
        camera_id: str,
        density: float,
        counter_flow_score: float = 0.0,
        stagnation_score: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> tuple[FruinLevel, AlertSeverity, list[SafetyAlert], float]:
        ts = timestamp if timestamp is not None else time.time()
        hist = self._history.setdefault(camera_id, deque(maxlen=600))
        hist.append((ts, density))

        level = FruinClassifier.classify(density)
        severity = FruinClassifier.severity_for(level)
        trend = self._trend(camera_id, ts)

        alerts: list[SafetyAlert] = []

        # 1) Density itself dangerous
        if level in (FruinLevel.E, FruinLevel.F):
            self._maybe_emit(alerts, camera_id, "density_threshold",
                             FruinClassifier.severity_for(level), ts,
                             f"DENSITY {density:.1f} p/m² — Fruin {level.value} "
                             f"({FruinClassifier.description_for(level)})",
                             density=density, fruin_level=level)

        # 2) Surge
        if trend > self.surge_threshold / self.surge_window:
            per_min = trend * 60.0
            self._maybe_emit(alerts, camera_id, "surge",
                             AlertSeverity.CRITICAL, ts,
                             f"SURGE: density rising {per_min:.1f} p/m²/min "
                             f"(now {density:.1f} p/m²)",
                             density=density, fruin_level=level,
                             metadata={"trend_per_sec": round(trend, 4)})

        # 3) Counter-flow (only meaningful at moderate density and above)
        if counter_flow_score > self.counter_flow_threshold and density > 2.0:
            self._maybe_emit(alerts, camera_id, "counter_flow",
                             AlertSeverity.CRITICAL, ts,
                             f"COUNTER-FLOW: opposing movement "
                             f"score={counter_flow_score:.2f} at {density:.1f} p/m²",
                             density=density, fruin_level=level,
                             metadata={"counter_flow_score": round(counter_flow_score, 3)})

        # 4) Stagnation in dense crowd = pre-crush
        if stagnation_score > self.stagnation_threshold and density > 3.0:
            self._maybe_emit(alerts, camera_id, "stagnation",
                             AlertSeverity.EMERGENCY, ts,
                             f"STAGNATION: {stagnation_score*100:.0f}% of crowd "
                             f"stationary at {density:.1f} p/m² — pre-crush",
                             density=density, fruin_level=level,
                             metadata={"stagnation_score": round(stagnation_score, 3)})

        return level, severity, alerts, trend

    def _trend(self, camera_id: str, now: float) -> float:
        hist = self._history.get(camera_id) or []
        recent = [(t, d) for t, d in hist if t > now - self.surge_window]
        if len(recent) < 2:
            return 0.0
        dt = recent[-1][0] - recent[0][0]
        if dt <= 0:
            return 0.0
        return (recent[-1][1] - recent[0][1]) / dt

    def _maybe_emit(
        self,
        alerts: list[SafetyAlert],
        camera_id: str,
        alert_type: str,
        severity: AlertSeverity,
        ts: float,
        message: str,
        **kw,
    ) -> None:
        key = (camera_id, alert_type)
        last = self._last_alert.get(key, 0.0)
        if ts - last < self.alert_cooldown:
            return
        self._last_alert[key] = ts
        alerts.append(SafetyAlert(
            timestamp=ts, severity=severity, alert_type=alert_type,
            zone_name=camera_id, message=message, **kw,
        ))
