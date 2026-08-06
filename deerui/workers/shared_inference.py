from __future__ import annotations

import threading
import time
import math
import queue
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

try:
    from PySide6.QtCore import QObject, QTimer, Signal, Slot
except ImportError:  # pragma: no cover
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
from deerui.workers.inference_worker import InferenceWorker


_FAIRNESS_LAG_MULTIPLIER = 6


@dataclass(frozen=True)
class LatestFrameStats:
    submitted_frame_count: int = 0
    consumed_sequence: int = 0
    overwritten_frame_count: int = 0

    @property
    def overwrite_drop_rate(self) -> float:
        if self.submitted_frame_count <= 0:
            return 0.0
        return min(
            max(self.overwritten_frame_count / self.submitted_frame_count, 0.0),
            1.0,
        )


class LatestFrameStore(QObject):
    """Thread-safe latest-wins frame store keyed by camera/stream id."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._frames: dict[str, tuple[object, float, int]] = {}
        self._sequences: defaultdict[str, int] = defaultdict(int)
        self._consumed_sequences: defaultdict[str, int] = defaultdict(int)
        self._seen_frame_counts: defaultdict[str, int] = defaultdict(int)
        self._overwritten_frame_counts: defaultdict[str, int] = defaultdict(int)
        self._last_batched_at_seconds: dict[str, float] = {}

    @Slot(str, object, float)
    def submit(self, stream_id: str, frame: object, timestamp_seconds: float) -> None:
        if frame is None:
            return
        with self._lock:
            self._sequences[stream_id] += 1
            self._frames[stream_id] = (
                frame,
                float(timestamp_seconds),
                self._sequences[stream_id],
            )

    def consume(self, stream_id: str) -> tuple[object | None, float | None, int]:
        with self._lock:
            item = self._frames.get(stream_id)
            if item is None:
                return None, None, 0
            return item

    def mark_consumed(self, stream_id: str, sequence: int) -> LatestFrameStats:
        with self._lock:
            self._mark_consumed_unlocked(stream_id, sequence)
            return self._stream_stats_unlocked(stream_id)

    def consume_batch(
        self,
        batch_size: int,
        start_index: int,
        sample_interval: int,
        *,
        now_seconds: float | None = None,
    ) -> tuple[list[tuple[str, object, float]], int, bool]:
        batch: list[tuple[str, object, float]] = []
        batch_size = max(int(batch_size), 1)
        sample_interval = max(int(sample_interval), 1)
        saw_new_frame = False
        now = time.perf_counter() if now_seconds is None else float(now_seconds)
        with self._lock:
            stream_ids = list(self._frames.keys())
            stream_count = len(stream_ids)
            if stream_count <= 0:
                return [], 0, False
            start_index = min(max(int(start_index), 0), stream_count - 1)
            stream_index_by_id = {stream_id: index for index, stream_id in enumerate(stream_ids)}
            round_robin_rank = {
                stream_ids[(start_index + offset) % stream_count]: offset
                for offset in range(stream_count)
            }
            candidates: list[tuple[float, int, int, str, object, float, int]] = []
            debt_candidates: list[tuple[float, int, int, str, object, float, int]] = []
            debt_lag_threshold = max(sample_interval * _FAIRNESS_LAG_MULTIPLIER, 6)
            min_scan_count = min(
                stream_count,
                max(batch_size * sample_interval + batch_size, batch_size),
            )
            sample_ready_count = 0
            for offset in range(stream_count):
                index = (start_index + offset) % stream_count
                stream_id = stream_ids[index]
                frame, timestamp, sequence = self._frames[stream_id]
                consumed_sequence = self._consumed_sequences.get(stream_id, 0)
                if sequence == consumed_sequence:
                    continue
                lag = max(sequence - consumed_sequence, 0)
                next_seen_count = self._seen_frame_counts[stream_id] + 1
                sample_ready = next_seen_count % sample_interval == 0
                if sample_ready:
                    sample_ready_count += 1
                last_batched_at = self._last_batched_at_seconds.get(stream_id)
                candidate = (
                    last_batched_at if last_batched_at is not None else float("-inf"),
                    -lag,
                    round_robin_rank[stream_id],
                    stream_id,
                    frame,
                    timestamp,
                    sequence,
                )
                candidates.append(candidate)
                if sample_ready and (last_batched_at is None or lag >= debt_lag_threshold):
                    debt_candidates.append(candidate)
                if sample_ready_count >= batch_size and offset + 1 >= min_scan_count:
                    break
            if not candidates:
                return [], start_index, False
            if debt_candidates:
                debt_candidates.sort()
                debt_limit = max(1, batch_size // 2)
                selected_debt = debt_candidates[:debt_limit]
                selected_debt_ids = {candidate[3] for candidate in selected_debt}
                ordered_candidates = selected_debt + [
                    candidate
                    for candidate in candidates
                    if candidate[3] not in selected_debt_ids
                ]
            else:
                ordered_candidates = candidates
            last_index = start_index
            for (
                _last_batched,
                _lag_key,
                _rank,
                stream_id,
                frame,
                timestamp,
                sequence,
            ) in ordered_candidates:
                last_index = stream_index_by_id[stream_id]
                saw_new_frame = True
                self._mark_consumed_unlocked(stream_id, sequence)
                self._seen_frame_counts[stream_id] += 1
                if self._seen_frame_counts[stream_id] % sample_interval != 0:
                    continue
                batch.append((stream_id, frame, timestamp or 0.0))
                self._last_batched_at_seconds[stream_id] = now
                if len(batch) >= batch_size:
                    break
            next_index = (last_index + 1) % stream_count
        return batch, next_index, saw_new_frame

    def stream_stats(self, stream_id: str) -> LatestFrameStats:
        with self._lock:
            return self._stream_stats_unlocked(stream_id)

    def remove_stream(self, stream_id: str) -> None:
        with self._lock:
            self._frames.pop(stream_id, None)
            self._sequences.pop(stream_id, None)
            self._consumed_sequences.pop(stream_id, None)
            self._seen_frame_counts.pop(stream_id, None)
            self._overwritten_frame_counts.pop(stream_id, None)
            self._last_batched_at_seconds.pop(stream_id, None)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._sequences.clear()
            self._consumed_sequences.clear()
            self._seen_frame_counts.clear()
            self._overwritten_frame_counts.clear()
            self._last_batched_at_seconds.clear()

    def stream_ids(self) -> list[str]:
        with self._lock:
            return list(self._frames.keys())

    def _stream_stats_unlocked(self, stream_id: str) -> LatestFrameStats:
        return LatestFrameStats(
            submitted_frame_count=self._sequences.get(stream_id, 0),
            consumed_sequence=self._consumed_sequences.get(stream_id, 0),
            overwritten_frame_count=self._overwritten_frame_counts.get(stream_id, 0),
        )

    def _mark_consumed_unlocked(self, stream_id: str, sequence: int) -> None:
        submitted_sequence = self._sequences.get(stream_id, 0)
        consumed_sequence = min(max(int(sequence), 0), submitted_sequence)
        previous_sequence = self._consumed_sequences.get(stream_id, 0)
        if consumed_sequence <= previous_sequence:
            return
        self._overwritten_frame_counts[stream_id] += max(
            consumed_sequence - previous_sequence - 1,
            0,
        )
        self._consumed_sequences[stream_id] = consumed_sequence


class SharedBatchScheduler:
    """Centralized batch collector shared by one or more inference executors."""

    def __init__(
        self,
        frame_store: LatestFrameStore,
        *,
        batch_size: int = 1,
        max_wait_ms: int = 0,
        adaptive_wait: bool = False,
        max_adaptive_wait_ms: int | None = None,
    ) -> None:
        self._frame_store = frame_store
        self.batch_size = max(int(batch_size), 1)
        self.base_wait_seconds = max(float(max_wait_ms), 0.0) / 1000.0
        adaptive_max_wait_ms = (
            max(float(max_adaptive_wait_ms), float(max_wait_ms))
            if max_adaptive_wait_ms is not None
            else max(float(max_wait_ms) * 2.0, float(max_wait_ms))
        )
        self.max_wait_seconds = max(adaptive_max_wait_ms, 0.0) / 1000.0
        self.current_wait_seconds = self.base_wait_seconds
        self.adaptive_wait = bool(adaptive_wait) and self.base_wait_seconds > 0.0
        self._underfilled_timeout_streak = 0
        self._fast_full_streak = 0
        self._lock = threading.Lock()
        self._round_robin_index = 0
        self._pending_batch: list[tuple[str, object, float]] = []
        self._pending_started_at: float | None = None

    @property
    def current_wait_ms(self) -> float:
        return round(self.current_wait_seconds * 1000.0, 3)

    def stream_ids(self) -> list[str]:
        return self._frame_store.stream_ids()

    def consume_ready_batch(
        self,
        *,
        sample_interval: int,
        active_stream_count: int,
        now_seconds: float | None = None,
    ) -> tuple[list[tuple[str, object, float]], bool]:
        now = time.perf_counter() if now_seconds is None else float(now_seconds)
        saw_new_frame = False
        with self._lock:
            remaining_capacity = self.batch_size - len(self._pending_batch)
            if remaining_capacity > 0:
                batch, self._round_robin_index, saw_new_frame = (
                    self._frame_store.consume_batch(
                        remaining_capacity,
                        self._round_robin_index,
                        sample_interval,
                        now_seconds=now,
                    )
                )
                if batch:
                    if self._pending_started_at is None:
                        self._pending_started_at = now
                    self._append_pending_unlocked(batch)

            if not self._pending_batch:
                return [], saw_new_frame

            target_batch_size = min(self.batch_size, max(int(active_stream_count), 1))
            waited_seconds = (
                0.0
                if self._pending_started_at is None
                else max(now - self._pending_started_at, 0.0)
            )
            if (
                len(self._pending_batch) >= target_batch_size
                or waited_seconds >= self.current_wait_seconds
            ):
                ready = self._pop_ready_batch_unlocked()
                self._adapt_wait_unlocked(len(ready), waited_seconds)
                return ready, saw_new_frame
            return [], saw_new_frame

    def clear(self) -> None:
        with self._lock:
            self._pending_batch.clear()
            self._pending_started_at = None
            self._round_robin_index = 0
            self.current_wait_seconds = self.base_wait_seconds
            self._underfilled_timeout_streak = 0
            self._fast_full_streak = 0

    def _append_pending_unlocked(self, batch: list[tuple[str, object, float]]) -> None:
        for stream_id, frame, timestamp in batch:
            for index, (pending_stream_id, _frame, _timestamp) in enumerate(self._pending_batch):
                if pending_stream_id == stream_id:
                    self._pending_batch[index] = (stream_id, frame, timestamp)
                    break
            else:
                self._pending_batch.append((stream_id, frame, timestamp))

    def _pop_ready_batch_unlocked(self) -> list[tuple[str, object, float]]:
        ready = self._pending_batch[: self.batch_size]
        del self._pending_batch[: self.batch_size]
        self._pending_started_at = time.perf_counter() if self._pending_batch else None
        return ready

    def _adapt_wait_unlocked(self, batch_size: int, waited_seconds: float) -> None:
        if not self.adaptive_wait:
            return
        if self.batch_size <= 1 or self.max_wait_seconds <= self.base_wait_seconds:
            return
        fill_ratio = max(min(batch_size / self.batch_size, 1.0), 0.0)
        timed_out = waited_seconds + 0.0005 >= self.current_wait_seconds
        filled_quickly = fill_ratio >= 0.95 and waited_seconds <= self.current_wait_seconds * 0.25
        step_seconds = 0.002
        if fill_ratio < 0.85 and timed_out:
            self._underfilled_timeout_streak += 1
            self._fast_full_streak = 0
            if self._underfilled_timeout_streak >= 3:
                self.current_wait_seconds = min(
                    self.current_wait_seconds + step_seconds,
                    self.max_wait_seconds,
                )
                self._underfilled_timeout_streak = 0
            return
        if filled_quickly:
            self._fast_full_streak += 1
            self._underfilled_timeout_streak = 0
            if self._fast_full_streak >= 3:
                self.current_wait_seconds = max(
                    self.current_wait_seconds - step_seconds,
                    self.base_wait_seconds,
                )
                self._fast_full_streak = 0
            return
        self._underfilled_timeout_streak = 0
        self._fast_full_streak = 0


class CentralBatchDispatcher:
    """Single batch producer feeding one or more model executor workers."""

    def __init__(
        self,
        frame_store: LatestFrameStore,
        *,
        batch_size: int = 1,
        queue_size: int = 1,
        sample_every_n_frames: int = 1,
        adaptive_sampling: bool = True,
        adaptive_base_streams_per_worker: int = 12,
        adaptive_max_sample_interval: int = 5,
        batch_wait_ms: int = 0,
        adaptive_batch_wait: bool = False,
        poll_interval_ms: int = 2,
    ) -> None:
        self._frame_store = frame_store
        self._scheduler = SharedBatchScheduler(
            frame_store,
            batch_size=batch_size,
            max_wait_ms=batch_wait_ms,
            adaptive_wait=adaptive_batch_wait,
        )
        self.batch_size = max(int(batch_size), 1)
        self.sample_every_n_frames = max(int(sample_every_n_frames), 1)
        self.adaptive_sampling = bool(adaptive_sampling)
        self.adaptive_base_streams_per_worker = max(int(adaptive_base_streams_per_worker), 1)
        self.adaptive_max_sample_interval = max(
            int(adaptive_max_sample_interval),
            self.sample_every_n_frames,
        )
        self.poll_interval_seconds = max(float(poll_interval_ms), 1.0) / 1000.0
        self._queue: queue.Queue[list[tuple[str, object, float]]] = queue.Queue(
            maxsize=max(int(queue_size), 1)
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._dropped_batch_count = 0

    @property
    def dropped_batch_count(self) -> int:
        return self._dropped_batch_count

    def stream_ids(self) -> list[str]:
        return self._scheduler.stream_ids()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="deerui-batch-dispatcher",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_seconds: float = 1.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(float(timeout_seconds), 0.0))
        self._thread = None

    def clear(self) -> None:
        self._scheduler.clear()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def dispatch_once(self, now_seconds: float | None = None) -> bool:
        if self._queue.full():
            return False
        stream_count = len(self._scheduler.stream_ids())
        if stream_count <= 0:
            return False
        batch, _saw_new_frame = self._scheduler.consume_ready_batch(
            sample_interval=self._effective_sample_interval(stream_count),
            active_stream_count=stream_count,
            now_seconds=now_seconds,
        )
        if not batch:
            return False
        self._put_latest_batch(batch)
        return True

    def take_batch(self) -> list[tuple[str, object, float]]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return []

    def put_batch_for_test(self, batch: list[tuple[str, object, float]]) -> None:
        self._put_latest_batch(batch)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            dispatched = self.dispatch_once()
            if not dispatched:
                self._stop_event.wait(self.poll_interval_seconds)

    def _put_latest_batch(self, batch: list[tuple[str, object, float]]) -> None:
        if not batch:
            return
        try:
            self._queue.put_nowait(batch)
        except queue.Full:
            self._dropped_batch_count += 1

    def _effective_sample_interval(self, active_stream_count: int) -> int:
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


class PostprocessDispatcher:
    """Runs CPU-side result parsing and overlay work outside model executors."""

    def __init__(
        self,
        handler,
        *,
        queue_size: int = 2,
    ) -> None:
        self._handler = handler
        self._queue: queue.Queue[
            tuple[list[tuple[str, object, float]], list[object]]
        ] = queue.Queue(maxsize=max(int(queue_size), 1))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="deerui-postprocess-dispatcher",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_seconds: float = 1.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(float(timeout_seconds), 0.0))
        self._thread = None

    def enqueue(self, batch: list[tuple[str, object, float]], results: list[object]) -> bool:
        try:
            self._queue.put_nowait((batch, results))
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                batch, results = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._handler(batch, results)
            finally:
                self._queue.task_done()


class SharedInferenceWorker(InferenceWorker):
    """One YOLO worker that round-robins the latest frame from many streams."""

    frame_with_overlay = Signal(str, object, object)  # stream_id, RGB ndarray, DetectionResult
    detection_ready = Signal(str, object)  # stream_id, DetectionResult

    def __init__(
        self,
        model_path: str,
        behavior_classes: Mapping[int, BehaviorClass],
        confidence_threshold: float,
        frame_store: LatestFrameStore,
        sample_every_n_frames: int = 5,
        poll_interval_ms: int = 15,
        inference_device: str = "cpu",
        batch_size: int = 1,
        overlay_interval_seconds: float = 0.0,
        inference_imgsz: int | None = 640,
        inference_half: bool = True,
        adaptive_sampling: bool = True,
        adaptive_base_streams_per_worker: int = 12,
        adaptive_max_sample_interval: int = 5,
        preloaded_model: object | None = None,
        batch_scheduler: SharedBatchScheduler | None = None,
        batch_max_wait_ms: int = 0,
        batch_dispatcher: CentralBatchDispatcher | None = None,
        postprocess_dispatcher: PostprocessDispatcher | None = None,
        postprocess_queue_size: int = 0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            behavior_classes=behavior_classes,
            confidence_threshold=confidence_threshold,
            frame_buffer=None,
            sample_every_n_frames=sample_every_n_frames,
            poll_interval_ms=poll_interval_ms,
            inference_device=inference_device,
            inference_imgsz=inference_imgsz,
            inference_half=inference_half,
            adaptive_sampling=adaptive_sampling,
            adaptive_base_streams_per_worker=adaptive_base_streams_per_worker,
            adaptive_max_sample_interval=adaptive_max_sample_interval,
            preloaded_model=preloaded_model,
            parent=parent,
        )
        self._frame_store = frame_store
        self._batch_dispatcher = batch_dispatcher
        self._postprocess_dispatcher = postprocess_dispatcher or (
            PostprocessDispatcher(
                self._handle_inference_results,
                queue_size=postprocess_queue_size,
            )
            if postprocess_queue_size > 0
            else None
        )
        self._batch_scheduler = batch_scheduler or SharedBatchScheduler(
            frame_store,
            batch_size=batch_size,
            max_wait_ms=batch_max_wait_ms,
        )
        self._last_seen_seq_by_stream: dict[str, int] = {}
        self._frame_counter_by_stream: defaultdict[str, int] = defaultdict(int)
        self._last_overlay_emit_at_by_stream: dict[str, float] = {}
        self._overlay_lock = threading.Lock()
        self.batch_size = max(int(batch_size), 1)
        self.overlay_interval_seconds = max(float(overlay_interval_seconds), 0.0)

    @Slot()
    def _begin_polling(self) -> None:
        if self._postprocess_dispatcher is not None:
            self._postprocess_dispatcher.start()
        super()._begin_polling()

    @Slot()
    def stop(self) -> None:
        if self._postprocess_dispatcher is not None:
            self._postprocess_dispatcher.stop()
        super().stop()

    @Slot()
    def _poll_and_process(self) -> None:
        self._poll_and_process_once()

    def _poll_and_process_once(self) -> bool:
        if not self._running or self._frame_store is None:
            return False

        if self._batch_dispatcher is not None:
            batch = self._batch_dispatcher.take_batch()
            if not batch:
                return False
            self._prune_removed_stream_state(self._batch_dispatcher.stream_ids())
            self._process_dispatched_batch(batch)
            return True

        stream_ids = self._batch_scheduler.stream_ids()
        if not stream_ids:
            return False
        self._prune_removed_stream_state(stream_ids)
        sample_interval = self.effective_sample_interval(len(stream_ids))

        batch, saw_new_frame = self._batch_scheduler.consume_ready_batch(
            sample_interval=sample_interval,
            active_stream_count=len(stream_ids),
        )
        if not batch:
            return saw_new_frame
        self._process_dispatched_batch(batch)
        return True

    def _process_dispatched_batch(self, batch: list[tuple[str, object, float]]) -> None:
        if len(batch) == 1:
            stream_id, frame, timestamp = batch[0]
            self._infer_and_emit_for_stream(stream_id, frame, timestamp)
            return

        self._infer_and_emit_batch(batch)

    def _infer_and_emit_batch(self, batch: list[tuple[str, object, float]]) -> None:
        if not self._model_loaded or self.model is None:
            for stream_id, frame, timestamp in batch:
                self._emit_passthrough_for_stream(stream_id, frame, timestamp)
            return

        valid_batch: list[tuple[str, object, float]] = []
        for stream_id, frame, timestamp in batch:
            try:
                if frame is None or len(frame.shape) != 3 or frame.shape[2] != 3:
                    self._emit_passthrough_for_stream(stream_id, frame, timestamp)
                    continue
            except Exception:  # noqa: BLE001
                self._emit_passthrough_for_stream(stream_id, frame, timestamp)
                continue
            valid_batch.append((stream_id, frame, timestamp))

        if not valid_batch:
            return

        frames = [frame for _, frame, _ in valid_batch]
        try:
            results = self._predict(frames)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"Inference failed: {exc}")
            for stream_id, frame, timestamp in valid_batch:
                self._emit_passthrough_for_stream(stream_id, frame, timestamp)
            return

        result_items = list(results) if results is not None else []
        if self._enqueue_postprocess(valid_batch, result_items):
            return
        self._handle_inference_results(valid_batch, result_items)

    def _handle_inference_results(
        self,
        valid_batch: list[tuple[str, object, float]],
        result_items: list[object],
    ) -> None:
        for index, (stream_id, frame, timestamp) in enumerate(valid_batch):
            result = result_items[index] if index < len(result_items) else None
            detection = self._build_detection([result] if result is not None else [], timestamp)
            if self._should_emit_overlay_for_stream(stream_id):
                overlay_bgr = self._draw_overlay(frame, detection)
                overlay_rgb = self._bgr_to_rgb(overlay_bgr)
                self.frame_with_overlay.emit(stream_id, overlay_rgb, detection)
            self.detection_ready.emit(stream_id, detection)

    def _infer_and_emit_for_stream(
        self,
        stream_id: str,
        frame_bgr: object,
        timestamp_seconds: float,
    ) -> None:
        if not self._model_loaded or self.model is None:
            self._emit_passthrough_for_stream(stream_id, frame_bgr, timestamp_seconds)
            return

        try:
            if frame_bgr is None or len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
                self._emit_passthrough_for_stream(stream_id, frame_bgr, timestamp_seconds)
                return
        except Exception:  # noqa: BLE001
            self._emit_passthrough_for_stream(stream_id, frame_bgr, timestamp_seconds)
            return

        try:
            results = self._predict(frame_bgr)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"Inference failed: {exc}")
            self._emit_passthrough_for_stream(stream_id, frame_bgr, timestamp_seconds)
            return

        result_items = list(results) if results is not None else []
        batch = [(stream_id, frame_bgr, timestamp_seconds)]
        if self._enqueue_postprocess(batch, result_items):
            return
        self._handle_inference_results(batch, result_items)

    def _enqueue_postprocess(
        self,
        batch: list[tuple[str, object, float]],
        result_items: list[object],
    ) -> bool:
        dispatcher = self._postprocess_dispatcher
        if dispatcher is None:
            return False
        is_running = getattr(dispatcher, "is_running", True)
        if is_running is False:
            return False
        return dispatcher.enqueue(batch, result_items)

    def _emit_passthrough_for_stream(
        self,
        stream_id: str,
        frame_bgr: object,
        timestamp_seconds: float,
    ) -> None:
        empty = DetectionResult(
            action_key=None,
            display_name=None,
            confidence=None,
            bbox_xyxy=None,
            center_xy=None,
            timestamp_seconds=timestamp_seconds,
        )
        if self._should_emit_overlay_for_stream(stream_id):
            self.frame_with_overlay.emit(stream_id, self._bgr_to_rgb(frame_bgr), empty)
        self.detection_ready.emit(stream_id, empty)

    def _prune_removed_stream_state(self, stream_ids: list[str]) -> None:
        active = set(stream_ids)
        for mapping in (
            self._last_seen_seq_by_stream,
            self._frame_counter_by_stream,
            self._last_overlay_emit_at_by_stream,
        ):
            for stream_id in tuple(mapping):
                if stream_id not in active:
                    mapping.pop(stream_id, None)

    def _should_emit_overlay_for_stream(self, stream_id: str) -> bool:
        with self._overlay_lock:
            if self.overlay_interval_seconds <= 0:
                return True
            now = time.perf_counter()
            last_emit_at = self._last_overlay_emit_at_by_stream.get(stream_id)
            if last_emit_at is not None and now - last_emit_at < self.overlay_interval_seconds:
                return False
            self._last_overlay_emit_at_by_stream[stream_id] = now
            return True
