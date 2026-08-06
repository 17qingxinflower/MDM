from __future__ import annotations

from deerui.domain.models import BehaviorSegment, DetectionResult


class BehaviorStateMachine:
    def __init__(self, action_tolerance_seconds: float = 0.5) -> None:
        self.action_tolerance_seconds = action_tolerance_seconds
        self.current_action_key: str | None = None
        self.current_display_name: str | None = None
        self.action_started_at_seconds: float | None = None
        self.current_confidence: float | None = None
        self.pending_action_key: str | None = None
        self.pending_display_name: str | None = None
        self.pending_started_at_seconds: float | None = None
        self.pending_confidence: float | None = None

    def update(self, detection: DetectionResult) -> BehaviorSegment | None:
        action_key = detection.action_key
        timestamp = detection.timestamp_seconds

        if action_key == self.current_action_key:
            self._clear_pending()
            return None

        if action_key != self.pending_action_key or self.pending_started_at_seconds is None:
            self.pending_action_key = action_key
            self.pending_display_name = detection.display_name
            self.pending_started_at_seconds = timestamp
            self.pending_confidence = detection.confidence
            return None

        if timestamp - self.pending_started_at_seconds < self.action_tolerance_seconds:
            return None

        closed = self._close_at(self.pending_started_at_seconds)
        self.current_action_key = self.pending_action_key
        self.current_display_name = self.pending_display_name
        self.action_started_at_seconds = self.pending_started_at_seconds
        self.current_confidence = self.pending_confidence
        self._clear_pending()
        return closed

    def close(self, timestamp_seconds: float) -> BehaviorSegment | None:
        closed = self._close_at(timestamp_seconds)
        self.current_action_key = None
        self.current_display_name = None
        self.action_started_at_seconds = None
        self.current_confidence = None
        self._clear_pending()
        return closed

    def _close_at(self, end_time: float) -> BehaviorSegment | None:
        if self.current_action_key is None or self.action_started_at_seconds is None:
            return None
        duration = end_time - self.action_started_at_seconds
        if duration < 0.1:
            return None
        return BehaviorSegment(
            action_key=self.current_action_key,
            display_name=self.current_display_name or self.current_action_key,
            started_at_seconds=self.action_started_at_seconds,
            ended_at_seconds=end_time,
            duration_seconds=duration,
            confidence=self.current_confidence,
        )

    def _clear_pending(self) -> None:
        self.pending_action_key = None
        self.pending_display_name = None
        self.pending_started_at_seconds = None
        self.pending_confidence = None
