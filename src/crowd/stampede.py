"""Composite stampede risk scorer.

Per plan.md: NEVER fire on a single signal alone. Two or more signals
must be active simultaneously before the score crosses 0.7 (alert threshold).

Signals and weights:
  density_score      max(density/danger_density - 1, 0) * 0.35
  divergence_score   divergence * 0.20      (people fleeing from a point)
  convergence_spike  convergence * 0.15     (sudden crush inward)
  velocity_variance  min(variance/10, 1) * 0.15   (chaotic movement = panic)
  counter_flow       counter_flow * 0.15    (colliding crowd streams)

Total = sum; capped at 1.0.
Alert thresholds: > 0.45 → WARNING, > 0.70 → CRITICAL
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .density import CrowdState
from .flow import FlowState


@dataclass
class StampedeRisk:
    score: float                    # 0-1
    level: str                      # ok / warning / critical
    flags: list[str] = field(default_factory=list)

    @property
    def is_alert(self) -> bool:
        return self.level != "ok"


_WEIGHTS = {
    "density":      0.35,
    "divergence":   0.20,
    "convergence":  0.15,
    "variance":     0.15,
    "counter_flow": 0.15,
}


class StampedeDetector:
    """
    5-signal composite stampede risk scorer.

    Optional learned classifier: when sklearn is installed and
    `load_classifier(path)` is called with a fitted sklearn model,
    the classifier replaces the hand-tuned threshold gating.
    Train on UMN + PETS-2009 + GBA-Stampedes + GSMADC datasets
    (Sciencedirect S0952197624020992 — ~43,000 frames, public).

    Without a fitted model, uses the original heuristic thresholds.
    """

    def __init__(
        self,
        danger_density: float = 7.0,
        warning_threshold: float = 0.45,
        critical_threshold: float = 0.70,
        min_signals_for_alert: int = 2,
        # India-tuned: higher baselines in dense Indian crowds
        divergence_min: float = 0.35,
        convergence_min: float = 0.30,
        variance_min: float = 3.0,
        counter_flow_min: float = 0.30,
    ):
        self.danger_density = danger_density
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.min_signals = min_signals_for_alert
        self.divergence_min = divergence_min
        self.convergence_min = convergence_min
        self.variance_min = variance_min
        self.counter_flow_min = counter_flow_min
        self._classifier = None   # optional sklearn classifier

    def load_classifier(self, path: str) -> None:
        """Load a pre-fitted sklearn classifier from a pickle file.

        The model must accept a 1D feature vector of shape (5,):
          [density_norm, divergence, convergence, velocity_variance, counter_flow]
        and return predict_proba(X) → (P_ok, P_warning, P_critical).

        Train on UMN + PETS-2009 + GBA-Stampedes + GSMADC (public datasets)
        using scripts/train_stampede_classifier.py.
        """
        import pickle
        with open(path, "rb") as f:
            self._classifier = pickle.load(f)

    def _classify_with_model(
        self,
        crowd: "CrowdState",
        flow: "FlowState",
    ) -> "StampedeRisk | None":
        if self._classifier is None:
            return None
        try:
            import numpy as np
            density_norm = crowd.max_density / self.danger_density if crowd.max_density > 0 else 0.0
            features = np.array([[
                min(density_norm, 2.0),
                flow.divergence if flow.valid else 0.0,
                flow.convergence if flow.valid else 0.0,
                min(flow.variance / 15.0, 2.0) if flow.valid else 0.0,
                flow.counter_flow if flow.valid else 0.0,
            ]], dtype=np.float32)
            proba = self._classifier.predict_proba(features)[0]
            classes = list(self._classifier.classes_)
            def _p(label):
                return proba[classes.index(label)] if label in classes else 0.0
            p_crit = _p("critical")
            p_warn = _p("warning")
            score = round(float(p_crit * 0.9 + p_warn * 0.5), 3)
            if p_crit >= 0.5:
                level = "critical"
            elif p_warn >= 0.5:
                level = "warning"
            else:
                level = "ok"
            return StampedeRisk(score=score, level=level, flags=[f"clf:crit={p_crit:.2f}"])
        except Exception:
            return None

    def assess(self, crowd: "CrowdState", flow: "FlowState") -> StampedeRisk:
        # Learned classifier takes precedence when loaded
        clf_result = self._classify_with_model(crowd, flow)
        if clf_result is not None:
            return clf_result
        return self._assess_heuristic(crowd, flow)

    def _assess_heuristic(self, crowd: CrowdState, flow: FlowState) -> StampedeRisk:
        signals: dict[str, float] = {}
        flags: list[str] = []

        # -- density --------------------------------------------------------
        if crowd.max_density > 0:
            d = max(crowd.max_density / self.danger_density - 0.5, 0.0)
            if d > 0.1:
                signals["density"] = min(d, 1.0) * _WEIGHTS["density"]
                flags.append(f"density {crowd.max_density:.1f}p/m²")

        if not flow.valid:
            n_signals = len(signals)
            raw = sum(signals.values())
            return self._make_result(raw, n_signals, flags)

        # -- divergence (people fleeing from a centre point) ----------------
        if flow.divergence > self.divergence_min:
            signals["divergence"] = flow.divergence * _WEIGHTS["divergence"]
            flags.append(f"divergence {flow.divergence:.2f}")

        # -- convergence spike (crush toward a point) ----------------------
        if flow.convergence > self.convergence_min:
            signals["convergence"] = flow.convergence * _WEIGHTS["convergence"]
            flags.append(f"convergence {flow.convergence:.2f}")

        # -- velocity variance (panic indicator) ---------------------------
        if flow.variance > self.variance_min:
            v = min(flow.variance / 15.0, 1.0)
            signals["variance"] = v * _WEIGHTS["variance"]
            flags.append(f"velocity_variance {flow.variance:.1f}")

        # -- counter-flow (crowd collision) --------------------------------
        if flow.counter_flow > self.counter_flow_min:
            signals["counter_flow"] = flow.counter_flow * _WEIGHTS["counter_flow"]
            flags.append(f"counter_flow {flow.counter_flow:.2f}")

        n_signals = len(signals)
        raw = sum(signals.values())
        return self._make_result(raw, n_signals, flags)

    def _make_result(self, raw: float, n_signals: int, flags: list[str]) -> StampedeRisk:
        if n_signals < self.min_signals:
            return StampedeRisk(score=round(raw, 3), level="ok", flags=flags)
        score = min(raw, 1.0)
        if score >= self.critical_threshold:
            level = "critical"
        elif score >= self.warning_threshold:
            level = "warning"
        else:
            level = "ok"
        return StampedeRisk(score=round(score, 3), level=level, flags=flags)
