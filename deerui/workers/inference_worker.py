from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path

try:
    from PySide6.QtCore import QObject, QTimer, Signal, Slot
except ImportError:  # pragma: no cover - allows --smoke-test without PySide6
    QObject = object  # type: ignore[assignment,misc]

    class Signal:  # type: ignore[no-redef]
        def __init__(self, *args: object) -> None:
            pass

    def Slot(*args: object, **kwargs: object):  # type: ignore[no-redef]
        def decorator(func):
            return func

        return decorator

    class QTimer:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        @staticmethod
        def singleShot(interval_ms: int, target) -> None:  # noqa: ARG004
            return None

from deerui.domain.models import BehaviorClass, DetectionResult
from deerui.vision.overlay import draw_detection_overlay_bgr


_logger = logging.getLogger(__name__)
_PYTORCH_MODEL_SUFFIXES = {".pt", ".pth"}


def is_pytorch_model_path(model_path: str | Path) -> bool:
    return Path(str(model_path)).suffix.lower() in _PYTORCH_MODEL_SUFFIXES


class InferenceWorker(QObject):
    """Runs YOLO on frames pulled from a `FrameBuffer`.

    Lifecycle:
        - Construct on the main thread.
        - ``moveToThread(inference_thread)``.
        - Call ``QTimer.singleShot(0, worker.start)`` so polling starts on the
          worker's event loop, not the main thread.
        - ``start()`` schedules model load via ``QTimer.singleShot`` too, so failures
          show up via the ``error`` signal rather than the constructor.
        - ``stop()`` flips ``_running``; the QTimer keeps firing but does no work.
        - On real teardown, the owning thread is ``quit()`` + ``wait()``.

    The worker emits BOTH a frame-with-overlay (for display) and a DetectionResult
    (for the stats panel + state machine) so the UI can subscribe to whichever it
    needs.
    """

    frame_with_overlay = Signal(object, object)  # (RGB ndarray, DetectionResult)
    detection_ready = Signal(object)  # DetectionResult
    model_loaded = Signal()
    warning = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        model_path: str,
        behavior_classes: Mapping[int, BehaviorClass],
        confidence_threshold: float,
        frame_buffer,
        sample_every_n_frames: int = 5,
        poll_interval_ms: int = 33,
        inference_device: str = "cpu",
        inference_imgsz: int | None = 640,
        inference_half: bool = True,
        adaptive_sampling: bool = True,
        adaptive_base_streams_per_worker: int = 12,
        adaptive_max_sample_interval: int = 5,
        preloaded_model: object | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.model_path = model_path
        self._behavior_classes: dict[int, BehaviorClass] = dict(behavior_classes)
        self.confidence_threshold = float(confidence_threshold)
        self._frame_buffer = frame_buffer
        self.sample_every_n_frames = max(int(sample_every_n_frames), 1)
        self._poll_interval_ms = max(int(poll_interval_ms), 1)
        self.inference_device = inference_device
        self.inference_imgsz = max(int(inference_imgsz or 0), 0)
        self.inference_half = bool(inference_half)
        self.adaptive_sampling = bool(adaptive_sampling)
        self.adaptive_base_streams_per_worker = max(
            int(adaptive_base_streams_per_worker),
            1,
        )
        self.adaptive_max_sample_interval = max(
            int(adaptive_max_sample_interval),
            self.sample_every_n_frames,
        )
        self._preloaded_model = preloaded_model

        self.model = None
        self._running = False
        self._model_loaded = False
        self._last_seen_seq: int = 0
        self._frame_counter: int = 0
        self._poll_timer: QTimer | None = None

        # Defensive imports: ultralytics, cv2, numpy may all be missing during smoke tests.
        self._cv2 = None
        self._np = None
        try:
            import cv2  # type: ignore[import-not-found]

            self._cv2 = cv2
        except ImportError:
            pass
        try:
            import numpy as np  # type: ignore[import-not-found]

            self._np = np
        except ImportError:
            pass

    @Slot()
    def start(self) -> None:
        """Begin polling. Safe to call multiple times; no-op after the first.

        We are invoked from `QThread::started`, which fires BEFORE the inference
        thread's event loop is up and running. Anything we do with `QTimer.start()`
        or `QTimer.singleShot()` directly here is lost; those timers need an
        active event loop to fire. So we re-schedule the heavy work via
        `QTimer.singleShot(0, ...)` once the event loop is guaranteed to be
        running. The 0-ms timer created here will fire as soon as exec() begins.
        """
        if self._running:
            return
        self._running = True
        # Two-step: schedule the actual initialization to run after exec() begins.
        QTimer.singleShot(0, self._begin_polling)

    @Slot()
    def _begin_polling(self) -> None:
        if not self._running:
            return
        self.load_model()
        self._schedule_next_poll()

    @Slot()
    def load_model(self) -> None:
        if not self._running:
            _logger.info("load_model: skipping (worker stopped before load)")
            return
        if self._preloaded_model is not None:
            self.model = self._preloaded_model
            self._model_loaded = True
            self.model_loaded.emit()
            return
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:
            _logger.warning("Ultralytics not installed: %s", exc)
            self.error.emit(f"Ultralytics is not installed: {exc}")
            return

        _logger.info("Loading YOLO model from %s", self.model_path)
        try:
            self.model = YOLO(self.model_path)
            if is_pytorch_model_path(self.model_path):
                self.model.fuse()
                self.model.eval()
        except Exception as exc:  # noqa: BLE001 - Ultralytics raises broad exceptions
            _logger.exception("YOLO model load failed")
            self.error.emit(f"Model Load Failed: {exc}")
            return

        self._model_loaded = True
        _logger.info("Model loaded: %d classes configured", len(self._behavior_classes))
        self.model_loaded.emit()

        # Warn (without clobbering) if the model's class names don't match config.
        try:
            model_names = getattr(self.model, "names", None)
        except Exception:  # noqa: BLE001
            model_names = None
        if model_names is not None:
            try:
                model_ids = {int(k) for k in model_names.keys()}
            except Exception:  # noqa: BLE001
                model_ids = set()
            config_ids = set(self._behavior_classes.keys())
            if model_ids and model_ids != config_ids:
                self.warning.emit(
                    f"Model classes {sorted(model_ids)} do not match configured classes {sorted(config_ids)}; "
                    "please update DEERUI_BEHAVIOR_CLASS_MAPPING"
                )

    @Slot()
    def stop(self) -> None:
        self._running = False
        if self._poll_timer is not None:
            try:
                self._poll_timer.stop()
            except Exception:  # noqa: BLE001
                pass
        self.finished.emit()

    # ------------------------------------------------------------------
    # Internal polling
    # ------------------------------------------------------------------

    def _schedule_next_poll(self) -> None:
        if not self._running:
            return
        if self._poll_timer is None:
            self._poll_timer = QTimer(self)
            self._poll_timer.setInterval(self._poll_interval_ms)
            self._poll_timer.timeout.connect(self._poll_and_process)
        self._poll_timer.start()

    @Slot()
    def _poll_and_process(self) -> None:
        if not self._running or self._frame_buffer is None:
            return
        frame, timestamp, seq = self._frame_buffer.consume()
        if frame is None:
            return
        if seq == self._last_seen_seq:
            return
        self._last_seen_seq = seq
        self._frame_counter += 1
        if self._frame_counter % self.sample_every_n_frames != 0:
            return
        self._infer_and_emit(frame, timestamp or 0.0)

    # ------------------------------------------------------------------
    # Inference + drawing
    # ------------------------------------------------------------------

    def _infer_and_emit(self, frame_bgr, timestamp_seconds: float) -> None:
        if not self._model_loaded or self.model is None:
            # No model yet; emit an empty overlay so the display keeps refreshing.
            self._emit_passthrough(frame_bgr, timestamp_seconds)
            return

        # Frame sanity check; guards against empty/invalid buffers that would crash
        # Ultralytics in the C++ layer (which we cannot recover from).
        try:
            if frame_bgr is None or len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
                self._emit_passthrough(frame_bgr, timestamp_seconds)
                return
        except Exception:  # noqa: BLE001
            self._emit_passthrough(frame_bgr, timestamp_seconds)
            return

        try:
            results = self._predict(frame_bgr)
        except Exception as exc:  # noqa: BLE001
            _logger.exception("YOLO inference failed")
            self.error.emit(f"Inference failed: {exc}")
            self._emit_passthrough(frame_bgr, timestamp_seconds)
            return

        detection = self._build_detection(results, timestamp_seconds)

        overlay_bgr = self._draw_overlay(frame_bgr, detection)
        overlay_rgb = self._bgr_to_rgb(overlay_bgr)
        try:
            self.frame_with_overlay.emit(overlay_rgb, detection)
            self.detection_ready.emit(detection)
        except Exception:  # noqa: BLE001
            # Slot raised (likely destroyed widgets during teardown). Log and bail.
            _logger.exception("emitting overlay/detection signals raised")

    def _build_detection(self, results, timestamp_seconds: float) -> DetectionResult:
        if not results or getattr(results[0], "boxes", None) is None:
            return DetectionResult(
                action_key=None,
                display_name=None,
                confidence=None,
                bbox_xyxy=None,
                center_xy=None,
                timestamp_seconds=timestamp_seconds,
            )
        try:
            box = max(results[0].boxes, key=lambda item: float(item.conf[0].item()))
            confidence = float(box.conf[0].item())
        except Exception:  # noqa: BLE001
            return DetectionResult(
                action_key=None,
                display_name=None,
                confidence=None,
                bbox_xyxy=None,
                center_xy=None,
                timestamp_seconds=timestamp_seconds,
            )

        if confidence < self.confidence_threshold:
            return DetectionResult(
                action_key=None,
                display_name=None,
                confidence=confidence,
                bbox_xyxy=None,
                center_xy=None,
                timestamp_seconds=timestamp_seconds,
            )

        try:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        except Exception:  # noqa: BLE001
            return DetectionResult(
                action_key=None,
                display_name=None,
                confidence=confidence,
                bbox_xyxy=None,
                center_xy=None,
                timestamp_seconds=timestamp_seconds,
            )

        behavior = self._behavior_classes.get(class_id)
        if behavior is None:
            action_key = f"class_{class_id}"
            display_zh = f"Class {class_id}"
            display_en = f"Class {class_id}"
        else:
            action_key = behavior.action_key
            display_zh = behavior.display_zh
            display_en = behavior.display_en

        return DetectionResult(
            action_key=action_key,
            display_name=display_zh,
            confidence=confidence,
            bbox_xyxy=(x1, y1, x2, y2),
            center_xy=((x1 + x2) // 2, (y1 + y2) // 2),
            timestamp_seconds=timestamp_seconds,
            display_zh=display_zh,
            display_en=display_en,
        )

    def effective_sample_interval(self, active_stream_count: int = 1) -> int:
        if not self.adaptive_sampling:
            return self.sample_every_n_frames
        active = max(int(active_stream_count), 1)
        overload_steps = max(
            0,
            math.ceil(active / self.adaptive_base_streams_per_worker) - 1,
        )
        return min(
            self.adaptive_max_sample_interval,
            self.sample_every_n_frames + overload_steps,
        )

    def _predict_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "verbose": False,
            "device": self.inference_device,
        }
        if self.inference_imgsz > 0:
            kwargs["imgsz"] = self.inference_imgsz
        if self.inference_half and str(self.inference_device).startswith("cuda"):
            kwargs["half"] = True
        return kwargs

    def _predict(self, frame_bgr):
        kwargs = self._predict_kwargs()
        attempts: list[dict[str, object]] = [kwargs]
        for keys_to_drop in (("half",), ("imgsz",), ("half", "imgsz")):
            fallback = dict(kwargs)
            for key in keys_to_drop:
                fallback.pop(key, None)
            if fallback not in attempts:
                attempts.append(fallback)
        attempts.append({"verbose": False})

        last_exc: TypeError | None = None
        for attempt in attempts:
            try:
                return self.model(frame_bgr, **attempt)
            except TypeError as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        return self.model(frame_bgr, verbose=False)

    def _draw_overlay(self, frame_bgr, detection: DetectionResult):
        if self._cv2 is None or self._np is None:
            return frame_bgr
        try:
            overlay = frame_bgr.copy()
        except Exception:  # noqa: BLE001
            return frame_bgr
        try:
            from deerui.ui.theme import COLOR_PRIMARY_BGR  # local import avoids Qt dep at smoke-test
        except ImportError:
            color = (16, 185, 129)
        else:
            color = COLOR_PRIMARY_BGR
        return draw_detection_overlay_bgr(
            overlay,
            detection,
            color_bgr=color,
            cv2_module=self._cv2,
            text_size=18,
        )

    def _bgr_to_rgb(self, frame_bgr):
        if self._cv2 is None:
            return frame_bgr
        try:
            return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        except Exception:  # noqa: BLE001
            return frame_bgr

    def _emit_passthrough(self, frame_bgr, timestamp_seconds: float) -> None:
        empty = DetectionResult(
            action_key=None,
            display_name=None,
            confidence=None,
            bbox_xyxy=None,
            center_xy=None,
            timestamp_seconds=timestamp_seconds,
        )
        overlay_rgb = self._bgr_to_rgb(frame_bgr)
        self.frame_with_overlay.emit(overlay_rgb, empty)
        self.detection_ready.emit(empty)
