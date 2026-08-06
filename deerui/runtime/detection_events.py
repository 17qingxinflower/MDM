from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class DetectionRecordEvent:
    stream_id: str
    detection: object


@dataclass(frozen=True)
class DetectionPreviewEvent:
    stream_id: str
    frame_rgb: object
    detection: object


class DetectionEventDispatcher:
    """Separates must-keep detection records from coalesced preview frames."""

    def __init__(self) -> None:
        self._record_events: Deque[DetectionRecordEvent] = deque()
        self._preview_events: dict[str, DetectionPreviewEvent] = {}

    @property
    def record_queue_size(self) -> int:
        return len(self._record_events)

    @property
    def preview_queue_size(self) -> int:
        return len(self._preview_events)

    def submit_record(self, stream_id: str, detection: object) -> None:
        self._record_events.append(DetectionRecordEvent(stream_id, detection))

    def submit_preview(self, stream_id: str, frame_rgb: object, detection: object) -> None:
        self._preview_events[stream_id] = DetectionPreviewEvent(stream_id, frame_rgb, detection)

    def drain_records(self, *, limit: int) -> list[DetectionRecordEvent]:
        if limit <= 0:
            return []
        events: list[DetectionRecordEvent] = []
        for _ in range(min(limit, len(self._record_events))):
            events.append(self._record_events.popleft())
        return events

    def drain_previews(self, *, limit: int) -> list[DetectionPreviewEvent]:
        if limit <= 0:
            return []
        stream_ids = list(self._preview_events)[:limit]
        events: list[DetectionPreviewEvent] = []
        for stream_id in stream_ids:
            event = self._preview_events.pop(stream_id, None)
            if event is not None:
                events.append(event)
        return events

    def clear(self) -> None:
        self._record_events.clear()
        self._preview_events.clear()
