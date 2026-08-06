from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deerui.runtime.stream_quality import StreamQuality, StreamQualitySnapshot


class StreamQualityEventRecorder:
    """Append quality status transitions as JSONL operational evidence."""

    def __init__(
        self,
        output_path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._last_status_by_stream: dict[str, str] = {}

    def record_if_changed(
        self,
        stream_id: str,
        stream_label: str,
        quality: StreamQuality,
        snapshot: StreamQualitySnapshot,
    ) -> dict[str, Any] | None:
        status = quality.status.value
        if self._last_status_by_stream.get(stream_id) == status:
            return None

        event = self._build_event(stream_id, stream_label, quality, snapshot)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            file.write("\n")
        self._last_status_by_stream[stream_id] = status
        return event

    def _build_event(
        self,
        stream_id: str,
        stream_label: str,
        quality: StreamQuality,
        snapshot: StreamQualitySnapshot,
    ) -> dict[str, Any]:
        timestamp = self._clock()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return {
            "schema_version": 1,
            "timestamp": timestamp.isoformat(),
            "stream_id": stream_id,
            "stream_label": stream_label,
            "quality_status": quality.status.value,
            "record_reliable": quality.record_reliable,
            "quality_label_zh": quality.label_zh,
            "quality_reason_zh": quality.reason_zh,
            "capture_fps": round(max(float(snapshot.capture_fps), 0.0), 3),
            "inference_fps": round(max(float(snapshot.inference_fps), 0.0), 3),
            "seconds_since_last_frame": self._rounded_optional(
                snapshot.seconds_since_last_frame
            ),
            "seconds_since_last_inference": self._rounded_optional(
                snapshot.seconds_since_last_inference
            ),
            "overwrite_drop_rate": round(
                min(max(float(snapshot.overwrite_drop_rate), 0.0), 1.0),
                4,
            ),
        }

    @staticmethod
    def _rounded_optional(value: float | None) -> float | None:
        if value is None:
            return None
        return round(max(float(value), 0.0), 3)
