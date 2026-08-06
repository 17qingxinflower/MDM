from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from deerui.services.profile_service import validate_deer_code


@dataclass(frozen=True)
class ThresholdSettings:
    min_duration_seconds: float = 5.0
    max_transitions_per_minute: float = 10.0
    action_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class AppRuntimeState:
    model_path: Path | None = None
    video_source: Path | int | None = None
    deer_code: str = "2.1.6"
    thresholds: ThresholdSettings = field(default_factory=ThresholdSettings)

    def set_model_path(self, model_path: Path) -> None:
        self.model_path = model_path

    def set_video_source(self, video_source: Path | int) -> None:
        self.video_source = video_source

    def set_deer_code(self, deer_code: str) -> None:
        self.deer_code = validate_deer_code(deer_code)

    def set_thresholds(self, thresholds: ThresholdSettings) -> None:
        if thresholds.min_duration_seconds <= 0:
            raise ValueError("Behavior duration threshold must be greater than 0")
        if thresholds.max_transitions_per_minute <= 0:
            raise ValueError("Transition-rate threshold must be greater than 0")
        self.thresholds = thresholds
