from __future__ import annotations

from dataclasses import dataclass, field

from deerui.domain.behavior_state import BehaviorStateMachine
from deerui.domain.models import BehaviorSegment, DetectionResult


@dataclass
class BehaviorSegmentTotal:
    action_key: str
    display_name: str
    duration_seconds: float = 0.0
    count: int = 0


@dataclass
class DetectionSession:
    camera_id: str
    deer_code: str
    state_machine: BehaviorStateMachine = field(default_factory=BehaviorStateMachine)
    segments: list[BehaviorSegment] = field(default_factory=list)
    committed_segment_totals: dict[str, BehaviorSegmentTotal] = field(default_factory=dict)
    pruned_segment_count: int = 0

    @property
    def segment_count(self) -> int:
        return self.pruned_segment_count + len(self.segments)

    def accept_detection(self, detection: DetectionResult) -> BehaviorSegment | None:
        segment = self.state_machine.update(detection)
        if segment is not None:
            self.segments.append(segment)
        return segment

    def stop(self, timestamp_seconds: float) -> BehaviorSegment | None:
        segment = self.state_machine.close(timestamp_seconds)
        if segment is not None:
            self.segments.append(segment)
        return segment

    def prune_committed_segments(self, count: int) -> None:
        count = max(min(int(count), len(self.segments)), 0)
        if count <= 0:
            return
        self._accumulate_committed_segments(self.segments[:count])
        del self.segments[:count]
        self.pruned_segment_count += count

    def _accumulate_committed_segments(self, segments: list[BehaviorSegment]) -> None:
        for segment in segments:
            total = self.committed_segment_totals.get(segment.action_key)
            if total is None:
                total = BehaviorSegmentTotal(
                    action_key=segment.action_key,
                    display_name=segment.display_name or segment.action_key,
                )
                self.committed_segment_totals[segment.action_key] = total
            if segment.display_name:
                total.display_name = segment.display_name
            total.duration_seconds += segment.duration_seconds
            total.count += 1

    def current_segment_snapshot(self, timestamp_seconds: float | None) -> BehaviorSegment | None:
        if timestamp_seconds is None:
            return None
        action_key = self.state_machine.current_action_key
        started_at = self.state_machine.action_started_at_seconds
        if action_key is None or started_at is None:
            return None

        end_time = max(float(timestamp_seconds), float(started_at))
        duration = end_time - float(started_at)
        if duration < 0.1:
            return None

        return BehaviorSegment(
            action_key=action_key,
            display_name=self.state_machine.current_display_name or action_key,
            started_at_seconds=float(started_at),
            ended_at_seconds=end_time,
            duration_seconds=duration,
            confidence=self.state_machine.current_confidence,
        )
