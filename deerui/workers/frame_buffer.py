"""Latest-wins frame buffer between camera capture and inference.

`CameraWorker` runs at the source's native cadence (often >30 FPS for video files).
Running YOLO on every frame would bottleneck the capture thread, and emitting each
frame as a Qt signal across threads causes the queued connection to grow without
bound, leaving the UI many seconds behind reality.

`FrameBuffer` breaks that pipeline: capture writes the latest frame into the buffer
under a mutex; the inference thread polls at its own cadence and always pulls the
freshest frame. Older frames are silently dropped, which is fine because each
inference is independent of the previous one.
"""

from __future__ import annotations

try:
    from PySide6.QtCore import QMutex, QObject, Signal, Slot
except ImportError:  # pragma: no cover - allows --smoke-test without PySide6
    class QObject:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class QMutex:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def lock(self) -> None:
            pass

        def unlock(self) -> None:
            pass

    class Signal:  # type: ignore[no-redef]
        def __init__(self, *args: object) -> None:
            pass

    def Slot(*args: object, **kwargs: object):  # type: ignore[no-redef]
        def decorator(func):
            return func

        return decorator


class FrameBuffer(QObject):
    """Holds the most recent BGR frame from `CameraWorker` for `InferenceWorker`."""

    # Optional diagnostic signal; lets the main window surface "no frames arriving".
    submit_failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._mutex = QMutex()
        self._latest_frame: object | None = None  # opaque ndarray
        self._latest_ts: float | None = None
        self._frame_seq: int = 0  # monotonic counter for change detection

    @Slot(object, float)
    def submit(self, frame: object, timestamp_seconds: float) -> None:
        """Slot called from the main thread when a new frame arrives.

        Overwrites any pending frame; we only ever care about the latest.
        """
        if frame is None:
            return
        self._mutex.lock()
        try:
            self._latest_frame = frame
            self._latest_ts = float(timestamp_seconds)
            self._frame_seq += 1
        finally:
            self._mutex.unlock()

    def consume(self) -> tuple[object | None, float | None, int]:
        """Pull the latest frame (called from the inference thread).

        Returns ``(frame, timestamp, sequence)``. The sequence increments by 1 for
        every successful ``submit`` so the caller can detect "no new frame this tick"
        vs "buffer empty since startup".
        """
        self._mutex.lock()
        try:
            return self._latest_frame, self._latest_ts, self._frame_seq
        finally:
            self._mutex.unlock()

    @property
    def has_frame(self) -> bool:
        self._mutex.lock()
        try:
            return self._latest_frame is not None
        finally:
            self._mutex.unlock()


def latest_sequence(previous_seq: int, current_seq: int) -> bool:
    """Helper for inference workers: did the buffer advance since last poll?"""
    return current_seq > previous_seq
