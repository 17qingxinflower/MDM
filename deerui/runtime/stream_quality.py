from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StreamQualityStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNRELIABLE = "unreliable"
    STALE = "stale"


@dataclass(frozen=True)
class StreamQualityThresholds:
    target_capture_fps: float
    min_capture_fps: float
    target_inference_fps: float
    min_inference_fps: float
    degraded_drop_rate: float = 0.05
    max_drop_rate: float = 0.20
    max_frame_age_seconds: float = 3.0
    max_inference_age_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.target_capture_fps <= 0 or self.min_capture_fps <= 0:
            raise ValueError("capture FPS thresholds must be positive")
        if self.target_inference_fps <= 0 or self.min_inference_fps <= 0:
            raise ValueError("inference FPS thresholds must be positive")
        if self.min_capture_fps > self.target_capture_fps:
            raise ValueError("min_capture_fps cannot exceed target_capture_fps")
        if self.min_inference_fps > self.target_inference_fps:
            raise ValueError("min_inference_fps cannot exceed target_inference_fps")
        if not 0 <= self.degraded_drop_rate <= self.max_drop_rate <= 1:
            raise ValueError("drop-rate thresholds must satisfy 0 <= degraded <= max <= 1")
        if self.max_frame_age_seconds <= 0 or self.max_inference_age_seconds <= 0:
            raise ValueError("staleness thresholds must be positive")


@dataclass(frozen=True)
class StreamQualitySnapshot:
    capture_fps: float
    inference_fps: float
    seconds_since_last_frame: float | None
    seconds_since_last_inference: float | None
    overwrite_drop_rate: float = 0.0


@dataclass(frozen=True)
class StreamQuality:
    status: StreamQualityStatus
    record_reliable: bool
    label_zh: str
    reason_zh: str


def evaluate_stream_quality(
    snapshot: StreamQualitySnapshot,
    thresholds: StreamQualityThresholds,
) -> StreamQuality:
    frame_age = snapshot.seconds_since_last_frame
    inference_age = snapshot.seconds_since_last_inference
    if frame_age is None:
        return _quality(StreamQualityStatus.STALE, False, "Recording Stalled", "Waiting for frames")
    if frame_age > thresholds.max_frame_age_seconds:
        return _quality(StreamQualityStatus.STALE, False, "Recording Stalled", f"Frame stalled for {frame_age:.1f}s")
    if inference_age is None:
        return _quality(StreamQualityStatus.STALE, False, "Recording Stalled", "Waiting for inference results")
    if inference_age > thresholds.max_inference_age_seconds:
        return _quality(
            StreamQualityStatus.STALE,
            False,
            "Recording Stalled",
            f"Inference stalled for {inference_age:.1f}s",
        )

    capture_fps = max(float(snapshot.capture_fps), 0.0)
    inference_fps = max(float(snapshot.inference_fps), 0.0)
    drop_rate = min(max(float(snapshot.overwrite_drop_rate), 0.0), 1.0)

    if capture_fps < thresholds.min_capture_fps:
        return _quality(
            StreamQualityStatus.UNRELIABLE,
            False,
            "Recording Unreliable",
            f"Capture FPS {capture_fps:.1f} is below minimum {thresholds.min_capture_fps:.1f}",
        )
    if inference_fps < thresholds.min_inference_fps:
        return _quality(
            StreamQualityStatus.UNRELIABLE,
            False,
            "Recording Unreliable",
            f"Inference FPS {inference_fps:.1f} is below minimum {thresholds.min_inference_fps:.1f}",
        )
    if drop_rate > thresholds.max_drop_rate:
        return _quality(
            StreamQualityStatus.UNRELIABLE,
            False,
            "Recording Unreliable",
            f"Drop rate {drop_rate:.1%} exceeds limit {thresholds.max_drop_rate:.1%}",
        )

    degraded_reasons: list[str] = []
    if capture_fps < thresholds.target_capture_fps:
        degraded_reasons.append(
            f"Capture FPS {capture_fps:.1f} is below target {thresholds.target_capture_fps:.1f}"
        )
    if inference_fps < thresholds.target_inference_fps:
        degraded_reasons.append(
            f"Inference FPS {inference_fps:.1f} is below target {thresholds.target_inference_fps:.1f}"
        )
    if drop_rate > thresholds.degraded_drop_rate:
        degraded_reasons.append(
            f"Drop rate {drop_rate:.1%} exceeds target {thresholds.degraded_drop_rate:.1%}"
        )
    if degraded_reasons:
        return _quality(
            StreamQualityStatus.DEGRADED,
            True,
            "Recording Degraded",
            "; ".join(degraded_reasons),
        )

    return _quality(StreamQualityStatus.HEALTHY, True, "Recording Reliable", "Capture and inference are stable")


def _quality(
    status: StreamQualityStatus,
    record_reliable: bool,
    label_zh: str,
    reason_zh: str,
) -> StreamQuality:
    return StreamQuality(
        status=status,
        record_reliable=record_reliable,
        label_zh=label_zh,
        reason_zh=reason_zh,
    )
