from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deerui.config import load_config, resolve_inference_device
from deerui.domain.models import BehaviorSegment, DEFAULT_BEHAVIOR_CLASSES, DetectionResult
from deerui.services.detection_service import DetectionSession
from scripts.benchmark_stream_capacity import (
    RuntimeBenchmarkDispatcher,
    RuntimeBenchmarkFrameStore,
    benchmark_stream_quality,
    build_predict_kwargs,
    capture_target_fps_for_stream,
    classify_status,
    effective_model_worker_count,
    expected_frame_count,
    model_backend_name,
    percentile,
    process_resource_snapshot,
    video_summary,
)


@dataclass
class ReaderStats:
    completed: bool = False
    frames_emitted: int = 0
    read_failures: int = 0
    started_at_monotonic: float = 0.0
    finished_at_monotonic: float = 0.0


@dataclass
class BehaviorRunResult:
    streams: int
    video: str
    model: str
    duration_seconds: float
    target_fps_per_stream: float
    expected_frames: int
    generated_frames: int
    processed_frames: int
    overwritten_frames: int
    generated_fps_per_stream: float
    processed_fps_per_stream: float
    capture_deficit_rate: float
    overwrite_drop_rate: float
    status: str
    quality_status: str
    record_reliable: bool
    quality_label_zh: str
    quality_reason_zh: str
    sample_interval: int
    p50_inference_ms: float
    p95_inference_ms: float
    mean_batch_size: float
    max_batch_size: int
    process_memory_mb: float
    process_thread_count: int
    gpu_memory_mb: float
    completed_streams: int
    per_stream: list[dict[str, Any]] = field(default_factory=list)
    action_differences: dict[str, dict[str, float]] = field(default_factory=dict)


class OneShotVideoReader(threading.Thread):
    def __init__(
        self,
        *,
        stream_index: int,
        video_path: Path,
        store: RuntimeBenchmarkFrameStore,
        target_fps: float,
        stop_event: threading.Event,
        stats: ReaderStats,
    ) -> None:
        super().__init__(daemon=True)
        self.stream_index = stream_index
        self.video_path = video_path
        self.store = store
        self.target_fps = max(float(target_fps), 1.0)
        self.stop_event = stop_event
        self.stats = stats

    def run(self) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        self.stats.started_at_monotonic = time.perf_counter()
        if not cap.isOpened():
            self.stats.read_failures += 1
            self.store.record_read_failure(self.stream_index)
            self.stats.finished_at_monotonic = time.perf_counter()
            return
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        emit_fps = min(source_fps, self.target_fps) if source_fps > 0 else self.target_fps
        sleep_seconds = 1.0 / max(float(emit_fps), 1.0)
        source_frames_per_emit = (
            max(source_fps / emit_fps, 1.0) if source_fps > 0 and emit_fps > 0 else 1.0
        )
        skip_remainder = 0.0
        next_deadline = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                read_started = time.perf_counter()
                ok, frame = cap.read()
                read_elapsed_ms = (time.perf_counter() - read_started) * 1000.0
                if not ok:
                    break
                self.store.submit(
                    self.stream_index,
                    frame,
                    submitted_at_monotonic=time.perf_counter(),
                    read_elapsed_ms=read_elapsed_ms,
                )
                self.stats.frames_emitted += 1
                skip_remainder += source_frames_per_emit - 1.0
                whole_skips = int(skip_remainder)
                if whole_skips > 0:
                    for _ in range(whole_skips):
                        if not cap.grab():
                            break
                    skip_remainder -= whole_skips
                next_deadline += sleep_seconds
                wait_seconds = next_deadline - time.perf_counter()
                if wait_seconds > 0:
                    self.stop_event.wait(wait_seconds)
                else:
                    next_deadline = time.perf_counter()
        finally:
            cap.release()
            self.stats.completed = True
            self.stats.finished_at_monotonic = time.perf_counter()


def build_detection(result: Any, timestamp_seconds: float, confidence_threshold: float) -> DetectionResult:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return _empty_detection(timestamp_seconds)
    try:
        box_items = list(boxes)
    except TypeError:
        box_items = []
    if not box_items:
        return _empty_detection(timestamp_seconds)
    try:
        box = max(box_items, key=lambda item: float(item.conf[0].item()))
        confidence = float(box.conf[0].item())
    except Exception:  # noqa: BLE001
        return _empty_detection(timestamp_seconds)
    if confidence < confidence_threshold:
        return _empty_detection(timestamp_seconds, confidence=confidence)
    try:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
    except Exception:  # noqa: BLE001
        return _empty_detection(timestamp_seconds, confidence=confidence)
    behavior = DEFAULT_BEHAVIOR_CLASSES.get(class_id)
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
        timestamp_seconds=float(timestamp_seconds),
        display_zh=display_zh,
        display_en=display_en,
    )


def _empty_detection(timestamp_seconds: float, confidence: float | None = None) -> DetectionResult:
    return DetectionResult(
        action_key=None,
        display_name=None,
        confidence=confidence,
        bbox_xyxy=None,
        center_xy=None,
        timestamp_seconds=float(timestamp_seconds),
    )


def summarize_session(
    session: DetectionSession,
    *,
    stream_index: int,
    generated_frames: int,
    processed_frames: int,
    overwritten_frames: int,
    last_detection_ts: float | None,
) -> dict[str, Any]:
    if last_detection_ts is not None:
        session.stop(last_detection_ts)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for segment in session.segments:
        totals[segment.action_key] = totals.get(segment.action_key, 0.0) + segment.duration_seconds
        counts[segment.action_key] = counts.get(segment.action_key, 0) + 1
    total_duration = sum(totals.values())
    return {
        "stream_index": stream_index,
        "generated_frames": generated_frames,
        "processed_frames": processed_frames,
        "overwritten_frames": overwritten_frames,
        "segment_count": len(session.segments),
        "total_behavior_seconds": round(total_duration, 3),
        "action_seconds": {key: round(value, 3) for key, value in sorted(totals.items())},
        "action_segment_counts": dict(sorted(counts.items())),
    }


def action_differences(per_stream: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    action_keys = sorted(
        {
            str(action_key)
            for row in per_stream
            for action_key in dict(row.get("action_seconds", {})).keys()
        }
    )
    differences: dict[str, dict[str, float]] = {}
    for action_key in action_keys:
        values = [
            float(dict(row.get("action_seconds", {})).get(action_key, 0.0))
            for row in per_stream
        ]
        if not values:
            continue
        mean = statistics.mean(values)
        differences[action_key] = {
            "min_seconds": round(min(values), 3),
            "max_seconds": round(max(values), 3),
            "mean_seconds": round(mean, 3),
            "spread_seconds": round(max(values) - min(values), 3),
            "relative_spread": round((max(values) - min(values)) / mean, 4) if mean > 0 else 0.0,
        }
    return differences


def run_behavior_consistency(args: argparse.Namespace) -> BehaviorRunResult:
    config = load_config()
    model_backend = model_backend_name(args.model)
    worker_count = effective_model_worker_count(
        requested_worker_count=args.worker_count,
        model_backend=model_backend,
    )
    predict_kwargs = build_predict_kwargs(args.device, args.imgsz, args.half)
    stream_count = max(int(args.streams), 1)
    stop_event = threading.Event()
    store = RuntimeBenchmarkFrameStore(stream_count)
    dispatcher = RuntimeBenchmarkDispatcher(
        store,
        batch_size=args.batch_size,
        queue_size=max(worker_count, 1),
        sample_every_n_frames=args.sample_every_n_frames,
        adaptive_sampling=args.adaptive_sampling,
        adaptive_base_streams_per_worker=args.adaptive_base_streams_per_worker,
        adaptive_max_sample_interval=args.adaptive_max_sample_interval,
        batch_wait_ms=args.batch_wait_ms,
        adaptive_batch_wait=args.adaptive_batch_wait,
        active_streams_per_worker=math.ceil(stream_count / max(worker_count, 1)),
    )
    reader_stats = [ReaderStats() for _ in range(stream_count)]
    readers = [
        OneShotVideoReader(
            stream_index=index,
            video_path=args.video,
            store=store,
            target_fps=capture_target_fps_for_stream(
                index,
                base_target_fps=args.target_fps,
                preview_stream_count=args.preview_streams,
                preview_target_fps=args.preview_target_fps or args.target_fps,
            ),
            stop_event=stop_event,
            stats=reader_stats[index],
        )
        for index in range(stream_count)
    ]
    sessions = [
        DetectionSession(camera_id=f"stream-{index:03d}", deer_code="benchmark")
        for index in range(stream_count)
    ]
    processed_by_stream = [0 for _ in range(stream_count)]
    last_detection_ts_by_stream: list[float | None] = [None for _ in range(stream_count)]
    inference_ms: list[float] = []
    batch_sizes: list[int] = []
    stats_lock = threading.Lock()

    print(
        f"starting streams={stream_count} worker_count={worker_count} "
        f"target_fps={args.target_fps} sample={dispatcher.sample_interval}",
        flush=True,
    )
    models = [YOLO(str(args.model)) for _ in range(worker_count)]
    cap = cv2.VideoCapture(str(args.video))
    ok, warmup_frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read warmup frame: {args.video}")
    for model in models:
        model([warmup_frame], **predict_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    started = time.perf_counter()
    last_progress = started
    for reader in readers:
        reader.start()
    dispatcher.start()
    dispatcher_stopped = threading.Event()

    def readers_done() -> bool:
        return all(not reader.is_alive() for reader in readers)

    def handle_batch(batch: list[tuple[int, int, Any, float]], result_items: list[Any]) -> None:
        for index, (stream_index, _seq, _frame, timestamp) in enumerate(batch):
            result = result_items[index] if index < len(result_items) else None
            detection = build_detection(result, timestamp, args.confidence)
            sessions[stream_index].accept_detection(detection)
            processed_by_stream[stream_index] += 1
            last_detection_ts_by_stream[stream_index] = detection.timestamp_seconds

    def inference_loop(model: YOLO) -> None:
        while not stop_event.is_set():
            batch = dispatcher.take_batch(timeout_seconds=0.05)
            if not batch:
                if dispatcher_stopped.is_set():
                    return
                continue
            frames = [frame for _stream_index, _seq, frame, _timestamp in batch]
            infer_started = time.perf_counter()
            results = model(frames, **predict_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - infer_started) * 1000.0
            result_items = list(results) if results is not None else []
            handle_batch(batch, result_items)
            with stats_lock:
                inference_ms.append(elapsed_ms)
                batch_sizes.append(len(batch))

    inference_threads = [threading.Thread(target=inference_loop, args=(model,), daemon=True) for model in models]
    for thread in inference_threads:
        thread.start()

    try:
        while not readers_done():
            now = time.perf_counter()
            if now - last_progress >= args.progress_interval:
                generated, overwritten, _read_failures = store.totals()
                processed = sum(processed_by_stream)
                completed = sum(1 for stat in reader_stats if stat.completed)
                elapsed = now - started
                print(
                    f"progress elapsed={elapsed:.1f}s completed={completed}/{stream_count} "
                    f"generated={generated} processed={processed} overwritten={overwritten}",
                    flush=True,
                )
                last_progress = now
            time.sleep(0.5)
        time.sleep(max(float(args.batch_wait_ms) / 1000.0 + 0.05, 0.1))
        dispatcher.stop(timeout_seconds=10.0)
        dispatcher_stopped.set()
        for thread in inference_threads:
            thread.join(timeout=10.0)
    finally:
        stop_event.set()
        dispatcher.stop()
        dispatcher_stopped.set()
        for reader in readers:
            reader.join(timeout=2.0)

    elapsed = max(time.perf_counter() - started, 0.001)
    generated_frames, overwritten_frames, _read_failures = store.totals()
    completed_streams = sum(1 for stat in reader_stats if stat.completed)
    processed_frames = sum(processed_by_stream)
    expected_frames = expected_frame_count(
        stream_count=stream_count,
        duration_seconds=float(video_summary(args.video)["duration_seconds"]),
        base_target_fps=args.target_fps,
        preview_stream_count=args.preview_streams,
        preview_target_fps=args.preview_target_fps or args.target_fps,
    )
    capture_deficit_rate = (
        max(0, expected_frames - generated_frames) / expected_frames if expected_frames else 0.0
    )
    overwrite_drop_rate = overwritten_frames / generated_frames if generated_frames else 0.0
    generated_fps_per_stream = generated_frames / elapsed / stream_count
    processed_fps_per_stream = processed_frames / elapsed / stream_count
    quality = benchmark_stream_quality(
        target_fps_per_stream=args.target_fps,
        configured_sample_interval=args.sample_every_n_frames,
        effective_sample_interval=dispatcher.sample_interval,
        generated_fps_per_stream=generated_fps_per_stream,
        processed_fps_per_stream=processed_fps_per_stream,
        overwrite_drop_rate=overwrite_drop_rate,
    )
    resources = process_resource_snapshot()
    gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        if torch.cuda.is_available()
        else 0.0
    )
    per_stream = [
        summarize_session(
            sessions[index],
            stream_index=index,
            generated_frames=reader_stats[index].frames_emitted,
            processed_frames=processed_by_stream[index],
            overwritten_frames=store.overwritten_frames_for_stream(index),
            last_detection_ts=last_detection_ts_by_stream[index],
        )
        for index in range(stream_count)
    ]
    return BehaviorRunResult(
        streams=stream_count,
        video=str(args.video),
        model=str(args.model),
        duration_seconds=round(elapsed, 3),
        target_fps_per_stream=round(args.target_fps, 3),
        expected_frames=expected_frames,
        generated_frames=generated_frames,
        processed_frames=processed_frames,
        overwritten_frames=overwritten_frames,
        generated_fps_per_stream=round(generated_fps_per_stream, 3),
        processed_fps_per_stream=round(processed_fps_per_stream, 3),
        capture_deficit_rate=round(capture_deficit_rate, 4),
        overwrite_drop_rate=round(overwrite_drop_rate, 4),
        status=classify_status(
            capture_deficit_rate=capture_deficit_rate,
            overwrite_drop_rate=overwrite_drop_rate,
            processed_fps_per_stream=processed_fps_per_stream,
            min_infer_fps_per_stream=args.min_infer_fps,
        ),
        quality_status=quality.status.value,
        record_reliable=quality.record_reliable,
        quality_label_zh=quality.label_zh,
        quality_reason_zh=quality.reason_zh,
        sample_interval=dispatcher.sample_interval,
        p50_inference_ms=round(statistics.median(inference_ms), 2) if inference_ms else 0.0,
        p95_inference_ms=round(percentile(inference_ms, 95.0), 2),
        mean_batch_size=round(statistics.mean(batch_sizes), 2) if batch_sizes else 0.0,
        max_batch_size=max(batch_sizes) if batch_sizes else 0,
        process_memory_mb=resources.process_memory_mb,
        process_thread_count=resources.process_thread_count,
        gpu_memory_mb=round(gpu_memory_mb, 1),
        completed_streams=completed_streams,
        per_stream=per_stream,
        action_differences=action_differences(per_stream),
    )


def write_outputs(result: BehaviorRunResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"stream_behavior_consistency_{timestamp}.json"
    csv_path = output_dir / f"stream_behavior_consistency_{timestamp}.csv"
    json_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    action_keys = sorted(result.action_differences.keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "stream_index",
            "generated_frames",
            "processed_frames",
            "overwritten_frames",
            "segment_count",
            "total_behavior_seconds",
            *action_keys,
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.per_stream:
            action_seconds = dict(row.get("action_seconds", {}))
            writer.writerow(
                {
                    "stream_index": row["stream_index"],
                    "generated_frames": row["generated_frames"],
                    "processed_frames": row["processed_frames"],
                    "overwritten_frames": row["overwritten_frames"],
                    "segment_count": row["segment_count"],
                    "total_behavior_seconds": row["total_behavior_seconds"],
                    **{key: action_seconds.get(key, 0.0) for key in action_keys},
                }
            )
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser(description="Compare behavior records across same-video streams.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=config.model_path)
    parser.add_argument("--streams", type=int, default=64)
    parser.add_argument("--target-fps", type=float, default=config.stream_target_fps)
    parser.add_argument("--preview-streams", type=int, default=0)
    parser.add_argument("--preview-target-fps", type=float, default=config.preview_fps)
    parser.add_argument("--batch-size", type=int, default=config.inference_batch_size)
    parser.add_argument("--batch-wait-ms", type=int, default=config.inference_batch_wait_ms)
    parser.add_argument(
        "--adaptive-batch-wait",
        action=argparse.BooleanOptionalAction,
        default=config.adaptive_batch_wait,
    )
    parser.add_argument("--worker-count", type=int, default=config.inference_worker_count)
    parser.add_argument("--sample-every-n-frames", type=int, default=config.sample_every_n_frames)
    parser.add_argument(
        "--adaptive-sampling",
        action=argparse.BooleanOptionalAction,
        default=config.adaptive_sampling,
    )
    parser.add_argument(
        "--adaptive-base-streams-per-worker",
        type=int,
        default=config.adaptive_base_streams_per_worker,
    )
    parser.add_argument(
        "--adaptive-max-sample-interval",
        type=int,
        default=config.adaptive_max_sample_interval,
    )
    parser.add_argument("--imgsz", type=int, default=config.inference_imgsz)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=config.inference_half)
    parser.add_argument("--device", default=resolve_inference_device(config.inference_device))
    parser.add_argument("--confidence", type=float, default=config.confidence_threshold)
    parser.add_argument("--min-infer-fps", type=float, default=4.0)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=config.output_dir / "benchmarks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")
    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")
    result = run_behavior_consistency(args)
    json_path, csv_path = write_outputs(result, args.output_dir)
    print(
        f"done status={result.status} quality={result.quality_label_zh} "
        f"record_reliable={result.record_reliable} "
        f"processed_fps_per_stream={result.processed_fps_per_stream} "
        f"overwrite_drop_rate={result.overwrite_drop_rate:.1%}",
        flush=True,
    )
    print("action_differences=" + json.dumps(result.action_differences, ensure_ascii=False), flush=True)
    print(f"json={json_path}", flush=True)
    print(f"csv={csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
