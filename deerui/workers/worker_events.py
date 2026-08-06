from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WorkerEventType(str, Enum):
    FRAME_READY = "frame_ready"
    DETECTION_READY = "detection_ready"
    SEGMENT_CLOSED = "segment_closed"
    STATS_UPDATED = "stats_updated"
    ERROR = "error"
    FINISHED = "finished"


@dataclass(frozen=True)
class WorkerError:
    camera_id: str
    message: str
    recoverable: bool
