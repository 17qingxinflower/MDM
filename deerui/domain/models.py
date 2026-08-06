from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskMode(str, Enum):
    REALTIME = "realtime"
    PLAYBACK = "playback"
    BATCH = "batch"


@dataclass(frozen=True)
class BehaviorClass:
    """One YOLO class label, resolved into the keys the UI and state machine need."""

    action_key: str
    display_zh: str
    display_en: str


# Legacy 4-class mapping (matches `main.py` class_mapping / display_class_mapping /
# english_display_mapping in the parent folder). Kept here as the source of truth so
# the domain layer never imports from workers or config.
DEFAULT_BEHAVIOR_CLASSES: dict[int, BehaviorClass] = {
    0: BehaviorClass("egestion", "Egestion", "Egestion"),
    1: BehaviorClass("Standing_feeding", "Feeding", "Feeding"),
    2: BehaviorClass("Stand_or_walk", "Stand/Walk", "Stand/Walk"),
    3: BehaviorClass("Lie_down_and_rest", "Lie Down", "Lie Down"),
}


@dataclass(frozen=True)
class DetectionResult:
    action_key: str | None
    display_name: str | None  # back-compat alias for display_zh
    confidence: float | None
    bbox_xyxy: tuple[int, int, int, int] | None
    center_xy: tuple[int, int] | None
    timestamp_seconds: float
    display_zh: str | None = None
    display_en: str | None = None


@dataclass(frozen=True)
class BehaviorSegment:
    action_key: str
    display_name: str
    started_at_seconds: float
    ended_at_seconds: float
    duration_seconds: float
    confidence: float | None = None


@dataclass(frozen=True)
class AlertSegment:
    alert_type: str
    action_key: str | None
    display_name: str | None
    started_at_seconds: float
    ended_at_seconds: float
    duration_seconds: float | None
    transition_rate: float | None
    recommendation: str
