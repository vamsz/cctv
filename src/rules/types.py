from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ViolationCode(str, Enum):
    NO_HELMET = "no_helmet"
    RED_LIGHT_JUMP = "red_light_jump"
    PLATE_UNREADABLE = "plate_unreadable"
    WRONG_WAY = "wrong_way"
    TRIPLE_RIDING = "triple_riding"
    OVERSPEED = "overspeed"
    WATCHLIST_HIT = "watchlist_hit"


VIOLATION_PRIORITY: dict[str, str] = {
    "watchlist_hit": "HIGH",
    "wrong_way": "HIGH",
    "red_light_jump": "HIGH",
    "no_helmet": "MEDIUM",
    "triple_riding": "MEDIUM",
    "overspeed": "MEDIUM",
    "plate_unreadable": "LOW",
}


@dataclass
class ViolationEvent:
    code: ViolationCode
    camera_id: str
    track_id: int
    timestamp: float
    frame_idx: int
    confidence: float                                # rule-level confidence in [0, 1]
    plate_text: Optional[str] = None                 # normalized; may be None
    plate_ocr_confidence: Optional[float] = None
    bbox: Optional[tuple[float, float, float, float]] = None  # the offender's vehicle box
    extras: dict = field(default_factory=dict)        # speed_kmh, rider_count, etc.
