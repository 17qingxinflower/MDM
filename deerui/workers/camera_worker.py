from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

try:
    import cv2
    from PySide6.QtCore import QObject, Signal, Slot
except ImportError:  # pragma: no cover - dependency availability is checked at runtime
    cv2 = None
    QObject = object

    class Signal:  # type: ignore[no-redef]
        def __init__(self, *args: object) -> None:
            pass

    def Slot(*args: object, **kwargs: object):  # type: ignore[no-redef]
        def decorator(func):
            return func

        return decorator


_logger = logging.getLogger(__name__)


class CameraWorker(QObject):
    frame_ready = Signal(object, float)
    status = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        source: str | int,
        target_fps: float = 30.0,
        reconnect_delay_seconds: float = 2.0,
        max_read_failures: int = 5,
        reconnect_on_eof: bool | None = None,
        open_timeout_milliseconds: int = 5000,
        read_timeout_milliseconds: int = 2000,
        max_buffer_drain_grabs_per_frame: int = 2,
    ) -> None:
        super().__init__()
        self.source = source
        self._target_fps_lock = threading.Lock()
        self.target_fps = max(float(target_fps), 1.0)
        self.reconnect_delay_seconds = max(float(reconnect_delay_seconds), 0.1)
        self.max_read_failures = max(int(max_read_failures), 1)
        self.reconnect_on_eof = self._should_reconnect(source) if reconnect_on_eof is None else reconnect_on_eof
        self.open_timeout_milliseconds = max(int(open_timeout_milliseconds), 0)
        self.read_timeout_milliseconds = max(int(read_timeout_milliseconds), 0)
        self.max_buffer_drain_grabs_per_frame = max(
            int(max_buffer_drain_grabs_per_frame),
            1,
        )
        self._running = False

    @Slot()
    def run(self) -> None:
        if cv2 is None:
            self.error.emit("OpenCV is not installed")
            self.finished.emit()
            return

        self._running = True
        try:
            while self._running:
                cap = self._open_capture()
                if cap is None:
                    if not self.reconnect_on_eof:
                        self.error.emit("Unable to connect to the video stream")
                        return
                    self.status.emit("Unable to connect to the video stream; retrying")
                    self._sleep_before_reconnect()
                    continue

                should_retry = self._read_capture(cap)
                if not should_retry:
                    return
                self._sleep_before_reconnect()
        finally:
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self._running = False

    def _open_capture(self):
        try:
            cap = cv2.VideoCapture(self.source)
        except Exception as exc:  # noqa: BLE001
            _logger.exception("VideoCapture construction failed")
            self.error.emit(f"Failed to create video capture: {exc}")
            return None
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                _logger.exception("cap.release() failed after open failure")
            return None
        self._configure_low_latency_capture(cap)
        return cap

    def _read_capture(self, cap) -> bool:
        if self._is_local_file_source():
            return self._read_local_file_capture(cap)
        if self._should_use_latest_frame_decode(cap):
            return self._read_latest_frame_stream(cap)
        return self._read_capture_with_read(cap)

    def _read_local_file_capture(self, cap) -> bool:
        source_fps = self._source_video_fps(cap)
        target_fps = self._target_fps()
        emit_fps = min(source_fps, target_fps) if source_fps > 0 else target_fps
        sleep_seconds = 1.0 / max(float(emit_fps), 1.0)
        source_frames_per_emit = (
            max(source_fps / max(emit_fps, 1.0), 1.0)
            if source_fps > 0
            else 1.0
        )
        skip_remainder = 0.0
        can_skip = callable(getattr(cap, "grab", None))
        try:
            while self._running:
                try:
                    ok, frame = cap.read()
                except Exception as exc:  # noqa: BLE001
                    _logger.exception("cap.read() raised")
                    self.error.emit(f"Failed to read video frame: {exc}")
                    return False
                if not ok:
                    return False

                self.frame_ready.emit(frame, time.time())
                if not self._running:
                    break

                skip_remainder += source_frames_per_emit - 1.0
                whole_skips = int(skip_remainder)
                if can_skip and whole_skips > 0:
                    for _ in range(whole_skips):
                        if not self._running:
                            break
                        try:
                            if not cap.grab():
                                return False
                        except Exception as exc:  # noqa: BLE001
                            _logger.exception("cap.grab() raised")
                            self.error.emit(f"Failed to skip video frame: {exc}")
                            return False
                    skip_remainder -= whole_skips
                time.sleep(sleep_seconds)
        finally:
            self._release_capture(cap)
        return False

    def _read_capture_with_read(self, cap) -> bool:
        failures = 0
        try:
            while self._running:
                try:
                    ok, frame = cap.read()
                except Exception as exc:  # noqa: BLE001
                    _logger.exception("cap.read() raised")
                    if self.reconnect_on_eof:
                        self.status.emit(f"Failed to read video frame; retrying: {exc}")
                    else:
                        self.error.emit(f"Failed to read video frame: {exc}")
                    failures += 1
                    ok = False
                    frame = None
                if not ok:
                    failures += 1
                    if not self.reconnect_on_eof:
                        return False
                    if failures >= self.max_read_failures:
                        self.status.emit("Video stream interrupted; reconnecting")
                        return True
                    time.sleep(0.05)
                    continue
                failures = 0
                self.frame_ready.emit(frame, time.time())
                sleep_seconds = self._frame_interval_seconds(cap)
                time.sleep(sleep_seconds)
        finally:
            self._release_capture(cap)
        return False

    def _read_latest_frame_stream(self, cap) -> bool:
        failures = 0
        next_emit_at = 0.0
        try:
            while self._running:
                now = time.time()
                if now < next_emit_at:
                    sleep_seconds = min(max(next_emit_at - now, 0.0), 0.005)
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    continue
                try:
                    ok = self._drain_capture_buffer(cap)
                except Exception as exc:  # noqa: BLE001
                    _logger.exception("cap.grab() raised")
                    if self.reconnect_on_eof:
                        self.status.emit(f"Failed to read video frame; retrying: {exc}")
                    else:
                        self.error.emit(f"Failed to read video frame: {exc}")
                    ok = False
                if not ok:
                    failures += 1
                    if not self.reconnect_on_eof:
                        return False
                    if failures >= self.max_read_failures:
                        self.status.emit("Video stream interrupted; reconnecting")
                        return True
                    time.sleep(0.05)
                    continue

                failures = 0
                try:
                    ok, frame = cap.retrieve()
                except Exception as exc:  # noqa: BLE001
                    _logger.exception("cap.retrieve() raised")
                    if self.reconnect_on_eof:
                        self.status.emit(f"Failed to decode video frame; retrying: {exc}")
                    else:
                        self.error.emit(f"Failed to decode video frame: {exc}")
                    ok = False
                    frame = None
                if not ok:
                    failures += 1
                    if not self.reconnect_on_eof:
                        return False
                    if failures >= self.max_read_failures:
                        self.status.emit("Video decode stream interrupted; reconnecting")
                        return True
                    continue

                timestamp = time.time()
                self.frame_ready.emit(frame, timestamp)
                next_emit_at = timestamp + self._target_frame_interval_seconds()
        finally:
            self._release_capture(cap)
        return False

    def _drain_capture_buffer(self, cap) -> bool:
        ok = False
        for _ in range(self.max_buffer_drain_grabs_per_frame):
            ok = cap.grab()
            if not ok:
                return False
        return ok

    def _configure_low_latency_capture(self, cap) -> None:
        if cv2 is None:
            return
        setter = getattr(cap, "set", None)
        if not callable(setter):
            return
        self._set_capture_property(setter, "CAP_PROP_BUFFERSIZE", 1)
        self._set_capture_property(
            setter,
            "CAP_PROP_OPEN_TIMEOUT_MSEC",
            self.open_timeout_milliseconds,
        )
        self._set_capture_property(
            setter,
            "CAP_PROP_READ_TIMEOUT_MSEC",
            self.read_timeout_milliseconds,
        )

    def _set_capture_property(self, setter, property_name: str, value: int | float) -> None:
        prop_id = getattr(cv2, property_name, None) if cv2 is not None else None
        if prop_id is None or value <= 0:
            return
        try:
            setter(prop_id, value)
        except Exception:  # noqa: BLE001
            _logger.debug("failed to set %s", property_name, exc_info=True)

    def _should_use_latest_frame_decode(self, cap) -> bool:
        if self._is_local_file_source():
            return False
        return callable(getattr(cap, "grab", None)) and callable(getattr(cap, "retrieve", None))

    def _release_capture(self, cap) -> None:
        try:
            cap.release()
        except Exception:  # noqa: BLE001
            _logger.exception("cap.release() failed")

    def _frame_interval_seconds(self, cap) -> float:
        fps = self._source_video_fps(cap) if self._is_local_file_source() else 0.0
        if fps <= 0:
            fps = self._target_fps()
        else:
            fps = min(fps, self._target_fps())
        return 1.0 / max(float(fps), 1.0)

    @Slot(float)
    def set_target_fps(self, target_fps: float) -> None:
        with self._target_fps_lock:
            self.target_fps = max(float(target_fps), 1.0)

    def _target_fps(self) -> float:
        with self._target_fps_lock:
            return self.target_fps

    def _target_frame_interval_seconds(self) -> float:
        return 1.0 / max(self._target_fps(), 1.0)

    def _source_video_fps(self, cap) -> float:
        if cv2 is None:
            return 0.0
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
        except Exception:  # noqa: BLE001
            _logger.debug("failed to read video fps", exc_info=True)
            return 0.0
        if fps < 1.0 or fps > 240.0:
            return 0.0
        return fps

    def _is_local_file_source(self) -> bool:
        if isinstance(self.source, int):
            return False
        try:
            path = Path(self.source)
        except TypeError:
            return False
        return path.exists() and path.is_file()

    def _sleep_before_reconnect(self) -> None:
        deadline = time.time() + self.reconnect_delay_seconds
        while self._running and time.time() < deadline:
            time.sleep(0.05)

    @staticmethod
    def _should_reconnect(source: str | int) -> bool:
        if isinstance(source, int):
            return True
        try:
            path = Path(source)
        except TypeError:
            return True
        if path.exists() and path.is_file():
            return False
        return True
