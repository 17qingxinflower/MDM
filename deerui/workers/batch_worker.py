from __future__ import annotations

import logging
from collections.abc import Mapping

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

from deerui.domain.models import BehaviorClass
from deerui.services.detection_service import DetectionSession
from deerui.workers.inference_worker import InferenceWorker


_logger = logging.getLogger(__name__)


class BatchWorker(QObject):
    progress = Signal(float)
    error = Signal(str)
    finished = Signal(list)

    def __init__(
        self,
        video_path: str,
        model_path: str,
        deer_code: str,
        behavior_classes: Mapping[int, BehaviorClass],
        confidence_threshold: float,
        sample_every_n_frames: int = 5,
        inference_device: str = "cpu",
        inference_imgsz: int | None = 640,
        inference_half: bool = True,
        batch_size: int = 1,
        preloaded_model: object | None = None,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.deer_code = deer_code
        self.behavior_classes = dict(behavior_classes)
        self.confidence_threshold = float(confidence_threshold)
        self.sample_every_n_frames = max(int(sample_every_n_frames), 1)
        self.inference_device = inference_device
        self.inference_imgsz = inference_imgsz
        self.inference_half = bool(inference_half)
        self.batch_size = max(int(batch_size), 1)
        self.preloaded_model = preloaded_model
        self._running = False
        self._stop_requested = False

    @Slot()
    def run(self) -> None:
        self._running = True
        self._stop_requested = False
        self.progress.emit(0.0)
        failed = False
        try:
            segments = self._run_batch()
        except Exception as exc:  # noqa: BLE001
            _logger.exception("BatchWorker failed")
            failed = True
            self.error.emit(str(exc))
            segments = []
        finally:
            self._running = False
        if not failed and not self._stop_requested:
            self.progress.emit(100.0)
        self.finished.emit(segments)

    @Slot()
    def stop(self) -> None:
        self._stop_requested = True
        self._running = False

    def _run_batch(self) -> list:
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(f"OpenCV is not installed: {exc}") from exc

        cap = cv2.VideoCapture(self.video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError("Failed to open historical video")

            model = self.preloaded_model
            if model is None:
                raise RuntimeError("AI model has not finished loading")

            detector = InferenceWorker(
                model_path=self.model_path,
                behavior_classes=self.behavior_classes,
                confidence_threshold=self.confidence_threshold,
                frame_buffer=None,
                sample_every_n_frames=self.sample_every_n_frames,
                inference_device=self.inference_device,
                inference_imgsz=self.inference_imgsz,
                inference_half=self.inference_half,
            )
            detector.model = model
            detector._model_loaded = True
            session = DetectionSession(camera_id=self.video_path, deer_code=self.deer_code)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0) or 25.0
            frame_index = 0
            last_timestamp = 0.0
            pending_frames: list[tuple[object, float]] = []

            def flush_pending() -> None:
                if not pending_frames:
                    return
                frames = [frame for frame, _timestamp in pending_frames]
                timestamps = [timestamp for _frame, timestamp in pending_frames]
                try:
                    raw_results = detector._predict(frames if len(frames) > 1 else frames[0])
                    if len(frames) == 1:
                        result_groups = [raw_results]
                    else:
                        result_groups = [[result] for result in raw_results]
                        if len(result_groups) != len(frames):
                            raise RuntimeError("Batch inference returned a different number of results than input frames")
                except Exception:
                    if len(frames) == 1:
                        raise
                    _logger.exception("Batch inference failed; falling back to single-frame inference")
                    result_groups = [detector._predict(frame) for frame in frames]

                for results, timestamp in zip(result_groups, timestamps, strict=True):
                    detection = detector._build_detection(results, timestamp)
                    if detection.action_key is not None:
                        session.accept_detection(detection)
                pending_frames.clear()

            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                last_timestamp = frame_index / fps
                if frame_index % self.sample_every_n_frames == 0:
                    pending_frames.append((frame, last_timestamp))
                    if len(pending_frames) >= self.batch_size:
                        flush_pending()
                if total_frames > 0 and frame_index % 10 == 0:
                    self.progress.emit((frame_index / total_frames) * 100)

            flush_pending()
            session.stop(last_timestamp)
            return session.segments
        finally:
            cap.release()
