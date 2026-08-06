from __future__ import annotations

from pathlib import Path

try:
    from PySide6.QtCore import QObject, Signal, Slot
except ImportError:  # pragma: no cover
    QObject = object

    class Signal:  # type: ignore[no-redef]
        def __init__(self, *args: object) -> None:
            pass

    def Slot(*args: object, **kwargs: object):  # type: ignore[no-redef]
        def decorator(func):
            return func

        return decorator

from deerui.domain.models import AlertSegment, BehaviorSegment
from deerui.services.export_service import ExportService


class ExportWorker(QObject):
    finished = Signal(str, str)
    error = Signal(str)

    def __init__(
        self,
        behavior_segments: list[BehaviorSegment],
        alert_segments: list[AlertSegment],
        raw_path: Path,
        alert_path: Path,
    ) -> None:
        super().__init__()
        self.behavior_segments = behavior_segments
        self.alert_segments = alert_segments
        self.raw_path = raw_path
        self.alert_path = alert_path
        self.export_service = ExportService()

    @Slot()
    def run(self) -> None:
        try:
            raw = self.export_service.export_behavior_segments(self.behavior_segments, self.raw_path)
            alert = ""
            if self.alert_segments:
                alert = str(
                    self.export_service.export_alert_segments(self.alert_segments, self.alert_path)
                )
            self.finished.emit(str(raw), alert)
        except Exception as exc:
            self.error.emit(str(exc))
