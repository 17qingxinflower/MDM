from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import random
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
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
from deerui.domain.models import DEFAULT_BEHAVIOR_CLASSES
from deerui.runtime.stream_quality import (
    StreamQuality,
    StreamQualitySnapshot,
    StreamQualityThresholds,
    evaluate_stream_quality,
)
from deerui.workers.shared_inference import (
    CentralBatchDispatcher as RuntimeCentralBatchDispatcher,
)
from deerui.workers.shared_inference import LatestFrameStore as RuntimeLatestFrameStore


_FAIRNESS_LAG_MULTIPLIER = 6


@dataclass
class StreamSlot:
    frame: Any | None = None
    seq: int = 0
    processed_seq: int = 0
    submitted_at_monotonic: float = 0.0
    last_batched_at_monotonic: float = 0.0
    seen_frames: int = 0
    generated_frames: int = 0
    overwritten_frames: int = 0
    read_failures: int = 0
    fault_dropped_frames: int = 0
    fault_frozen_frames: int = 0
    read_calls: int = 0
    read_time_ms_total: float = 0.0
    read_time_ms_max: float = 0.0


@dataclass
class BenchmarkResult:
    streams: int
    duration_seconds: float
    target_fps_per_stream: float
    expected_frames: int
    generated_frames: int
    processed_frames: int
    overwritten_frames: int
    empty_detections: int
    low_confidence_detections: int
    mean_confidence: float
    p50_inference_ms: float
    p95_inference_ms: float
    generated_fps_total: float
    processed_fps_total: float
    generated_fps_per_stream: float
    processed_fps_per_stream: float
    capture_deficit_rate: float
    overwrite_drop_rate: float
    empty_detection_rate: float
    low_confidence_rate: float
    status: str
    quality_status: str
    record_reliable: bool
    quality_label_zh: str
    quality_reason_zh: str
    gpu_memory_mb: float
    sample_interval: int
    process_memory_mb: float = 0.0
    process_thread_count: int = 0
    run_index: int = 1
    fault_mode: str = "none"
    fault_streams: int = 0
    fault_events: int = 0
    fault_dropped_frames: int = 0
    fault_frozen_frames: int = 0
    read_calls: int = 0
    read_time_ms_total: float = 0.0
    read_time_ms_max: float = 0.0
    inference_batches: int = 0
    mean_batch_size: float = 0.0
    max_batch_size: int = 0
    p50_queue_wait_ms: float = 0.0
    p95_queue_wait_ms: float = 0.0
    p50_model_ms: float = 0.0
    p95_model_ms: float = 0.0
    p50_postprocess_ms: float = 0.0
    p95_postprocess_ms: float = 0.0


@dataclass(frozen=True)
class FaultInjectionConfig:
    mode: str = "none"
    stream_fraction: float = 0.0
    start_seconds: float = 0.0
    duration_seconds: float = 0.0
    repeat_seconds: float = 0.0
    seed: int = 20260701

    @property
    def enabled(self) -> bool:
        return (
            self.mode in {"drop", "freeze"}
            and self.stream_fraction > 0.0
            and self.duration_seconds > 0.0
        )

    def normalized(self) -> "FaultInjectionConfig":
        mode = self.mode if self.mode in {"none", "drop", "freeze"} else "none"
        return FaultInjectionConfig(
            mode=mode,
            stream_fraction=min(max(float(self.stream_fraction), 0.0), 1.0),
            start_seconds=max(float(self.start_seconds), 0.0),
            duration_seconds=max(float(self.duration_seconds), 0.0),
            repeat_seconds=max(float(self.repeat_seconds), 0.0),
            seed=int(self.seed),
        )


@dataclass(frozen=True)
class FaultInjectionPlan:
    mode: str = "none"
    affected_streams: tuple[int, ...] = ()
    start_seconds: float = 0.0
    duration_seconds: float = 0.0
    repeat_seconds: float = 0.0

    @property
    def enabled(self) -> bool:
        return (
            self.mode in {"drop", "freeze"}
            and bool(self.affected_streams)
            and self.duration_seconds > 0.0
        )

    def active_mode_for(self, stream_index: int, elapsed_seconds: float) -> str | None:
        if not self.enabled or int(stream_index) not in self.affected_streams:
            return None
        elapsed_seconds = float(elapsed_seconds)
        if elapsed_seconds < self.start_seconds:
            return None
        if self.repeat_seconds > 0.0:
            repeat_seconds = max(self.repeat_seconds, self.duration_seconds)
            active_offset = (elapsed_seconds - self.start_seconds) % repeat_seconds
            return self.mode if active_offset < self.duration_seconds else None
        if elapsed_seconds < self.start_seconds + self.duration_seconds:
            return self.mode
        return None


@dataclass(frozen=True)
class ProcessResourceSnapshot:
    process_memory_mb: float
    process_thread_count: int


def process_resource_snapshot(process: Any | None = None) -> ProcessResourceSnapshot:
    if process is None:
        try:
            import psutil  # type: ignore[import-not-found]

            process = psutil.Process()
        except Exception:  # noqa: BLE001
            return ProcessResourceSnapshot(
                process_memory_mb=0.0,
                process_thread_count=max(threading.active_count(), 0),
            )
    try:
        rss_bytes = float(process.memory_info().rss)
    except Exception:  # noqa: BLE001
        rss_bytes = 0.0
    try:
        thread_count = int(process.num_threads())
    except Exception:  # noqa: BLE001
        thread_count = threading.active_count()
    return ProcessResourceSnapshot(
        process_memory_mb=round(max(rss_bytes, 0.0) / (1024 * 1024), 1),
        process_thread_count=max(thread_count, 0),
    )


def stream_run_plan(stream_counts: list[int], repeat_count: int) -> list[tuple[int, int]]:
    repeat_count = int(repeat_count)
    if repeat_count <= 0:
        raise ValueError("repeat count must be positive")
    return [
        (run_index, stream_count)
        for run_index in range(1, repeat_count + 1)
        for stream_count in stream_counts
    ]


def build_fault_injection_plan(
    config: FaultInjectionConfig | None,
    *,
    stream_count: int,
) -> FaultInjectionPlan:
    if config is None:
        config = FaultInjectionConfig()
    config = config.normalized()
    stream_count = max(int(stream_count), 0)
    if not config.enabled or stream_count <= 0:
        return FaultInjectionPlan()
    affected_count = min(
        stream_count,
        max(int(math.ceil(stream_count * config.stream_fraction)), 1),
    )
    rng = random.Random(config.seed)
    affected_streams = tuple(sorted(rng.sample(range(stream_count), affected_count)))
    return FaultInjectionPlan(
        mode=config.mode,
        affected_streams=affected_streams,
        start_seconds=config.start_seconds,
        duration_seconds=config.duration_seconds,
        repeat_seconds=config.repeat_seconds,
    )


def build_fault_injection_summary(
    results: list[BenchmarkResult],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    config = dict(metadata.get("fault_injection") or {})
    enabled = bool(config.get("enabled", False))
    total_fault_dropped = sum(int(result.fault_dropped_frames) for result in results)
    total_fault_frozen = sum(int(result.fault_frozen_frames) for result in results)
    total_fault_events = sum(int(result.fault_events) for result in results)
    faulted_runs = [result for result in results if int(result.fault_events) > 0]
    return {
        "enabled": enabled,
        "mode": config.get("mode", "none"),
        "stream_fraction": float(config.get("stream_fraction", 0.0) or 0.0),
        "start_seconds": float(config.get("start_seconds", 0.0) or 0.0),
        "duration_seconds": float(config.get("duration_seconds", 0.0) or 0.0),
        "repeat_seconds": float(config.get("repeat_seconds", 0.0) or 0.0),
        "seed": int(config.get("seed", 0) or 0),
        "faulted_run_count": len(faulted_runs),
        "total_fault_events": total_fault_events,
        "total_fault_dropped_frames": total_fault_dropped,
        "total_fault_frozen_frames": total_fault_frozen,
        "record_reliable_under_faults": all(result.record_reliable for result in faulted_runs)
        if faulted_runs
        else True,
        "max_capture_deficit_rate": max(
            (float(result.capture_deficit_rate) for result in faulted_runs),
            default=0.0,
        ),
        "max_overwrite_drop_rate": max(
            (float(result.overwrite_drop_rate) for result in faulted_runs),
            default=0.0,
        ),
    }


def build_profiler_summary(results: list[BenchmarkResult]) -> dict[str, Any]:
    if not results:
        return {
            "run_count": 0,
            "total_inference_batches": 0,
            "mean_batch_size": 0.0,
            "max_batch_size": 0,
            "max_p95_queue_wait_ms": 0.0,
            "max_p95_model_ms": 0.0,
            "max_p95_postprocess_ms": 0.0,
            "total_read_calls": 0,
            "mean_read_ms": 0.0,
            "max_read_ms": 0.0,
        }
    total_batches = sum(int(result.inference_batches) for result in results)
    weighted_batch_size = sum(
        float(result.mean_batch_size) * int(result.inference_batches)
        for result in results
    )
    total_read_calls = sum(int(result.read_calls) for result in results)
    total_read_time_ms = sum(float(result.read_time_ms_total) for result in results)
    return {
        "run_count": len(results),
        "total_inference_batches": total_batches,
        "mean_batch_size": round(
            weighted_batch_size / total_batches if total_batches else 0.0,
            2,
        ),
        "max_batch_size": max((int(result.max_batch_size) for result in results), default=0),
        "max_p95_queue_wait_ms": max(
            (float(result.p95_queue_wait_ms) for result in results),
            default=0.0,
        ),
        "max_p95_model_ms": max(
            (float(result.p95_model_ms) for result in results),
            default=0.0,
        ),
        "max_p95_postprocess_ms": max(
            (float(result.p95_postprocess_ms) for result in results),
            default=0.0,
        ),
        "total_read_calls": total_read_calls,
        "mean_read_ms": round(
            total_read_time_ms / total_read_calls if total_read_calls else 0.0,
            2,
        ),
        "max_read_ms": max((float(result.read_time_ms_max) for result in results), default=0.0),
    }


def capture_target_fps_for_stream(
    stream_index: int,
    *,
    base_target_fps: float,
    preview_stream_count: int,
    preview_target_fps: float,
) -> float:
    base_target_fps = max(float(base_target_fps), 1.0)
    preview_target_fps = max(float(preview_target_fps), base_target_fps)
    preview_stream_count = max(int(preview_stream_count), 0)
    if int(stream_index) < preview_stream_count:
        return preview_target_fps
    return base_target_fps


def expected_frame_count(
    *,
    stream_count: int,
    duration_seconds: float,
    base_target_fps: float,
    preview_stream_count: int,
    preview_target_fps: float,
) -> int:
    stream_count = max(int(stream_count), 0)
    duration_seconds = max(float(duration_seconds), 0.0)
    preview_count = min(max(int(preview_stream_count), 0), stream_count)
    background_count = max(stream_count - preview_count, 0)
    return int(
        round(
            duration_seconds
            * (
                background_count * max(float(base_target_fps), 1.0)
                + preview_count
                * capture_target_fps_for_stream(
                    0,
                    base_target_fps=base_target_fps,
                    preview_stream_count=1,
                    preview_target_fps=preview_target_fps,
                )
            )
        )
    )


def _resource_group_summary(
    results: list[BenchmarkResult],
    *,
    max_process_memory_delta_mb: float = 512.0,
    max_gpu_memory_delta_mb: float = 512.0,
    max_thread_count_delta: int = 2,
) -> dict[str, Any]:
    ordered = sorted(results, key=lambda result: result.run_index)
    process_values = [float(result.process_memory_mb) for result in ordered]
    thread_values = [int(result.process_thread_count) for result in ordered]
    gpu_values = [float(result.gpu_memory_mb) for result in ordered]
    process_memory_delta = round(process_values[-1] - process_values[0], 1)
    process_thread_delta = thread_values[-1] - thread_values[0]
    gpu_memory_delta = round(gpu_values[-1] - gpu_values[0], 1)
    reasons: list[str] = []
    if process_memory_delta > max_process_memory_delta_mb:
        reasons.append(f"RSS increased by {process_memory_delta:.1f} MB")
    if gpu_memory_delta > max_gpu_memory_delta_mb:
        reasons.append(f"GPU memory increased by {gpu_memory_delta:.1f} MB")
    if process_thread_delta > max_thread_count_delta:
        reasons.append(f"Thread count increased by {process_thread_delta}")
    resource_stable = not reasons
    return {
        "run_count": len(ordered),
        "first_process_memory_mb": process_values[0],
        "last_process_memory_mb": process_values[-1],
        "max_process_memory_mb": max(process_values),
        "process_memory_delta_mb": process_memory_delta,
        "max_process_thread_count": max(thread_values),
        "max_gpu_memory_mb": max(gpu_values),
        "process_thread_delta": process_thread_delta,
        "gpu_memory_delta_mb": gpu_memory_delta,
        "resource_stable": resource_stable,
        "resource_reason_zh": "Resources stable" if resource_stable else "; ".join(reasons),
    }


def build_resource_summary(
    results: list[BenchmarkResult],
    *,
    max_process_memory_delta_mb: float = 512.0,
    max_gpu_memory_delta_mb: float = 512.0,
    max_thread_count_delta: int = 2,
) -> dict[str, Any]:
    if not results:
        return {
            "run_count": 0,
            "first_process_memory_mb": 0.0,
            "last_process_memory_mb": 0.0,
            "max_process_memory_mb": 0.0,
            "process_memory_delta_mb": 0.0,
            "max_process_thread_count": 0,
            "max_gpu_memory_mb": 0.0,
            "process_thread_delta": 0,
            "gpu_memory_delta_mb": 0.0,
            "resource_stable": True,
            "resource_reason_zh": "Resources stable",
            "per_stream_count": [],
        }

    summary = _resource_group_summary(
        results,
        max_process_memory_delta_mb=max_process_memory_delta_mb,
        max_gpu_memory_delta_mb=max_gpu_memory_delta_mb,
        max_thread_count_delta=max_thread_count_delta,
    )
    results_by_streams: dict[int, list[BenchmarkResult]] = {}
    for result in results:
        results_by_streams.setdefault(result.streams, []).append(result)
    summary["per_stream_count"] = [
        {
            "streams": stream_count,
            **_resource_group_summary(
                results_by_streams[stream_count],
                max_process_memory_delta_mb=max_process_memory_delta_mb,
                max_gpu_memory_delta_mb=max_gpu_memory_delta_mb,
                max_thread_count_delta=max_thread_count_delta,
            ),
        }
        for stream_count in sorted(results_by_streams)
    ]
    return summary


def _resource_thresholds_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        max_process_memory_delta_mb = float(metadata.get("max_process_memory_delta_mb", 512.0))
    except (TypeError, ValueError):
        max_process_memory_delta_mb = 512.0
    try:
        max_gpu_memory_delta_mb = float(metadata.get("max_gpu_memory_delta_mb", 512.0))
    except (TypeError, ValueError):
        max_gpu_memory_delta_mb = 512.0
    try:
        max_thread_count_delta = int(metadata.get("max_thread_count_delta", 2))
    except (TypeError, ValueError):
        max_thread_count_delta = 2
    return {
        "max_process_memory_delta_mb": max(max_process_memory_delta_mb, 0.0),
        "max_gpu_memory_delta_mb": max(max_gpu_memory_delta_mb, 0.0),
        "max_thread_count_delta": max(max_thread_count_delta, 0),
    }


def _benchmark_statuses(group: list[BenchmarkResult]) -> list[str]:
    return sorted({result.status for result in group})


def _benchmark_status_stable(group: list[BenchmarkResult]) -> bool:
    return bool(group) and all(result.status == "ok" for result in group)


def _fault_injection_enabled(metadata: dict[str, Any]) -> bool:
    fault_config = metadata.get("fault_injection")
    if not isinstance(fault_config, dict):
        return False
    return bool(fault_config.get("enabled", False))


def _benchmark_status_acceptable(
    group: list[BenchmarkResult],
    *,
    fault_injection_enabled: bool,
) -> bool:
    if not group:
        return False
    if not fault_injection_enabled:
        return _benchmark_status_stable(group)
    return all(
        result.status == "ok"
        or (result.status == "decode_bound" and int(result.fault_events) > 0)
        for result in group
    )


def _fault_group_summary(
    group: list[BenchmarkResult],
    *,
    fault_injection_enabled: bool,
) -> dict[str, Any]:
    faulted_runs = [result for result in group if int(result.fault_events) > 0]
    fault_events = sum(int(result.fault_events) for result in group)
    fault_reliable = True
    if fault_injection_enabled:
        fault_reliable = bool(group) and len(faulted_runs) == len(group) and all(
            result.record_reliable for result in faulted_runs
        )
    return {
        "fault_injection_enabled": bool(fault_injection_enabled),
        "fault_reliable": fault_reliable,
        "candidate_faulted_run_count": len(faulted_runs),
        "candidate_fault_events": fault_events,
        "candidate_fault_dropped_frames": sum(
            int(result.fault_dropped_frames) for result in group
        ),
        "candidate_fault_frozen_frames": sum(
            int(result.fault_frozen_frames) for result in group
        ),
    }


def build_capacity_recommendation(
    results: list[BenchmarkResult],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    try:
        worker_count = max(int(metadata.get("worker_count", 1)), 1)
    except (TypeError, ValueError):
        worker_count = 1
    resource_thresholds = _resource_thresholds_from_metadata(metadata)
    fault_enabled = _fault_injection_enabled(metadata)
    results_by_streams: dict[int, list[BenchmarkResult]] = {}
    for result in results:
        results_by_streams.setdefault(result.streams, []).append(result)
    resource_summaries_by_streams = {
        stream_count: _resource_group_summary(group, **resource_thresholds)
        for stream_count, group in results_by_streams.items()
    }
    fault_summaries_by_streams = {
        stream_count: _fault_group_summary(
            group,
            fault_injection_enabled=fault_enabled,
        )
        for stream_count, group in results_by_streams.items()
    }
    stable_groups = {
        stream_count: group
        for stream_count, group in results_by_streams.items()
        if (
            group
            and all(result.record_reliable for result in group)
            and bool(resource_summaries_by_streams[stream_count]["resource_stable"])
            and _benchmark_status_acceptable(
                group,
                fault_injection_enabled=fault_enabled,
            )
            and bool(fault_summaries_by_streams[stream_count]["fault_reliable"])
        )
    }
    reliable_results = [
        max(group, key=lambda result: result.run_index)
        for group in stable_groups.values()
    ]
    reliable_run_count = sum(1 for result in results if result.record_reliable)
    all_fault_summary = _fault_group_summary(
        results,
        fault_injection_enabled=fault_enabled,
    )
    if not reliable_results:
        return {
            "usable": False,
            "recommended_total_streams": 0,
            "recommended_streams_per_worker": 0,
            "worker_count": worker_count,
            "source_streams": None,
            "quality_status": None,
            "candidate_run_count": 0,
            "candidate_reliable_run_count": reliable_run_count,
            "resource_stable": False,
            "resource_reason_zh": "No tested stream count satisfied both recording reliability and resource stability.",
            "benchmark_status_stable": _benchmark_status_acceptable(
                results,
                fault_injection_enabled=fault_enabled,
            ),
            "benchmark_statuses": sorted({result.status for result in results}),
            **all_fault_summary,
            "target_fps_per_stream": metadata.get("target_fps_per_stream"),
            "sample_every_n_frames": metadata.get("sample_every_n_frames"),
            "reason_zh": "No tested stream count was reliable, resource-stable, and not overloaded in every run; keeping the current capacity configuration.",
        }

    source = max(reliable_results, key=lambda result: result.streams)
    source_group = stable_groups.get(source.streams, [source])
    source_resource_summary = resource_summaries_by_streams[source.streams]
    source_fault_summary = fault_summaries_by_streams[source.streams]
    candidate_reliable_run_count = sum(1 for result in source_group if result.record_reliable)
    streams_per_worker = source.streams // worker_count
    if streams_per_worker <= 0:
        return {
            "usable": False,
            "recommended_total_streams": 0,
            "recommended_streams_per_worker": 0,
            "worker_count": worker_count,
            "source_streams": source.streams,
            "quality_status": source.quality_status,
            "candidate_run_count": len(source_group),
            "candidate_reliable_run_count": candidate_reliable_run_count,
            "resource_stable": source_resource_summary["resource_stable"],
            "resource_reason_zh": source_resource_summary["resource_reason_zh"],
            "benchmark_status_stable": _benchmark_status_acceptable(
                source_group,
                fault_injection_enabled=fault_enabled,
            ),
            "benchmark_statuses": _benchmark_statuses(source_group),
            **source_fault_summary,
            "target_fps_per_stream": metadata.get("target_fps_per_stream"),
            "sample_every_n_frames": metadata.get("sample_every_n_frames"),
            "reason_zh": "The consistently reliable stream count is smaller than the worker count; no capacity recommendation is generated.",
        }

    recommended_total = streams_per_worker * worker_count
    return {
        "usable": True,
        "recommended_total_streams": recommended_total,
        "recommended_streams_per_worker": streams_per_worker,
        "worker_count": worker_count,
        "source_streams": source.streams,
        "quality_status": source.quality_status,
        "candidate_run_count": len(source_group),
        "candidate_reliable_run_count": candidate_reliable_run_count,
        "resource_stable": source_resource_summary["resource_stable"],
        "resource_reason_zh": source_resource_summary["resource_reason_zh"],
        "process_memory_delta_mb": source_resource_summary["process_memory_delta_mb"],
        "gpu_memory_delta_mb": source_resource_summary["gpu_memory_delta_mb"],
        "process_thread_delta": source_resource_summary["process_thread_delta"],
        "benchmark_status_stable": _benchmark_status_acceptable(
            source_group,
            fault_injection_enabled=fault_enabled,
        ),
        "benchmark_statuses": _benchmark_statuses(source_group),
        **source_fault_summary,
        "target_fps_per_stream": metadata.get("target_fps_per_stream"),
        "sample_every_n_frames": metadata.get("sample_every_n_frames"),
        "reason_zh": (
            f"Maximum reliable per-run benchmark result: {source.streams} streams; "
            f"rounded conservatively across {worker_count} workers to {recommended_total} streams."
        ),
    }


class LatestFrameStore:
    def __init__(self, stream_count: int) -> None:
        self._lock = threading.Lock()
        self.slots = [StreamSlot() for _ in range(stream_count)]

    def submit(
        self,
        stream_index: int,
        frame: Any,
        *,
        submitted_at_monotonic: float | None = None,
        read_elapsed_ms: float = 0.0,
    ) -> None:
        with self._lock:
            slot = self.slots[stream_index]
            if slot.seq > slot.processed_seq:
                slot.overwritten_frames += 1
            slot.seq += 1
            slot.generated_frames += 1
            slot.submitted_at_monotonic = (
                time.perf_counter()
                if submitted_at_monotonic is None
                else float(submitted_at_monotonic)
            )
            if read_elapsed_ms > 0.0:
                slot.read_calls += 1
                slot.read_time_ms_total += float(read_elapsed_ms)
                slot.read_time_ms_max = max(slot.read_time_ms_max, float(read_elapsed_ms))
            slot.frame = frame

    def record_read_failure(self, stream_index: int) -> None:
        with self._lock:
            self.slots[stream_index].read_failures += 1

    def record_fault_drop(self, stream_index: int) -> None:
        with self._lock:
            slot = self.slots[stream_index]
            slot.read_failures += 1
            slot.fault_dropped_frames += 1

    def submit_fault_freeze(
        self,
        stream_index: int,
        frame: Any,
        *,
        submitted_at_monotonic: float | None = None,
    ) -> None:
        with self._lock:
            slot = self.slots[stream_index]
            if slot.seq > slot.processed_seq:
                slot.overwritten_frames += 1
            slot.seq += 1
            slot.generated_frames += 1
            slot.submitted_at_monotonic = (
                time.perf_counter()
                if submitted_at_monotonic is None
                else float(submitted_at_monotonic)
            )
            slot.fault_frozen_frames += 1
            slot.frame = frame

    def consume_batch(
        self,
        batch_size: int,
        start_index: int,
        sample_interval: int,
        *,
        now_seconds: float | None = None,
    ) -> tuple[list[tuple[int, int, Any, float]], int]:
        batch: list[tuple[int, int, Any, float]] = []
        sample_interval = max(int(sample_interval), 1)
        now = time.perf_counter() if now_seconds is None else float(now_seconds)
        with self._lock:
            stream_count = len(self.slots)
            if stream_count <= 0:
                return [], 0
            start_index = min(max(int(start_index), 0), stream_count - 1)
            candidates: list[tuple[float, int, int, int, StreamSlot]] = []
            debt_candidates: list[tuple[float, int, int, int, StreamSlot]] = []
            debt_lag_threshold = max(sample_interval * _FAIRNESS_LAG_MULTIPLIER, 6)
            min_scan_count = min(
                stream_count,
                max(max(int(batch_size), 1) * sample_interval + max(int(batch_size), 1), 1),
            )
            sample_ready_count = 0
            for offset in range(stream_count):
                index = (start_index + offset) % stream_count
                slot = self.slots[index]
                if slot.frame is None or slot.seq == slot.processed_seq:
                    continue
                lag = max(slot.seq - slot.processed_seq, 0)
                next_seen_count = slot.seen_frames + 1
                sample_ready = next_seen_count % sample_interval == 0
                if sample_ready:
                    sample_ready_count += 1
                last_batched_at = slot.last_batched_at_monotonic or None
                candidate = (
                    last_batched_at if last_batched_at is not None else float("-inf"),
                    -lag,
                    offset,
                    index,
                    slot,
                )
                candidates.append(candidate)
                if sample_ready and (last_batched_at is None or lag >= debt_lag_threshold):
                    debt_candidates.append(candidate)
                if sample_ready_count >= max(int(batch_size), 1) and offset + 1 >= min_scan_count:
                    break
            if not candidates:
                return [], start_index
            if debt_candidates:
                debt_candidates.sort()
                debt_limit = max(1, max(int(batch_size), 1) // 2)
                selected_debt = debt_candidates[:debt_limit]
                selected_debt_indexes = {candidate[3] for candidate in selected_debt}
                ordered_candidates = selected_debt + [
                    candidate
                    for candidate in candidates
                    if candidate[3] not in selected_debt_indexes
                ]
            else:
                ordered_candidates = candidates
            last_index = start_index
            for _last_batched, _lag_key, _rank, index, slot in ordered_candidates:
                last_index = index
                slot.processed_seq = slot.seq
                slot.seen_frames += 1
                if slot.seen_frames % sample_interval != 0:
                    continue
                batch.append((index, slot.seq, slot.frame, slot.submitted_at_monotonic))
                slot.last_batched_at_monotonic = now
                if len(batch) >= batch_size:
                    break
            next_index = (last_index + 1) % stream_count
        return batch, next_index

    def totals(self) -> tuple[int, int, int]:
        with self._lock:
            generated = sum(slot.generated_frames for slot in self.slots)
            overwritten = sum(slot.overwritten_frames for slot in self.slots)
            read_failures = sum(slot.read_failures for slot in self.slots)
        return generated, overwritten, read_failures

    def fault_totals(self) -> tuple[int, int]:
        with self._lock:
            fault_dropped = sum(slot.fault_dropped_frames for slot in self.slots)
            fault_frozen = sum(slot.fault_frozen_frames for slot in self.slots)
        return fault_dropped, fault_frozen

    def read_latency_totals(self) -> tuple[int, float, float]:
        with self._lock:
            read_calls = sum(slot.read_calls for slot in self.slots)
            read_time_ms_total = sum(slot.read_time_ms_total for slot in self.slots)
            read_time_ms_max = max(
                (slot.read_time_ms_max for slot in self.slots),
                default=0.0,
            )
        return read_calls, round(read_time_ms_total, 4), round(read_time_ms_max, 4)


class SharedBatchScheduler:
    def __init__(
        self,
        store: LatestFrameStore,
        *,
        batch_size: int,
        max_wait_ms: int = 0,
        adaptive_wait: bool = False,
        max_adaptive_wait_ms: int | None = None,
    ) -> None:
        self.store = store
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
        self._pending_batch: list[tuple[int, int, Any, float]] = []
        self._pending_started_at: float | None = None

    @property
    def current_wait_ms(self) -> float:
        return round(self.current_wait_seconds * 1000.0, 3)

    def consume_ready_batch(
        self,
        *,
        sample_interval: int,
        active_stream_count: int,
        now_seconds: float | None = None,
    ) -> list[tuple[int, int, Any, float]]:
        now = time.perf_counter() if now_seconds is None else float(now_seconds)
        with self._lock:
            remaining_capacity = self.batch_size - len(self._pending_batch)
            if remaining_capacity > 0:
                batch, self._round_robin_index = self.store.consume_batch(
                    remaining_capacity,
                    self._round_robin_index,
                    sample_interval,
                    now_seconds=now,
                )
                if batch:
                    if self._pending_started_at is None:
                        self._pending_started_at = now
                    self._append_pending_unlocked(batch)

            if not self._pending_batch:
                return []

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
                return ready
            return []

    def _append_pending_unlocked(
        self,
        batch: list[tuple[int, int, Any, float]],
    ) -> None:
        for stream_index, seq, frame, submitted_at in batch:
            for index, (pending_stream_index, _seq, _frame, _submitted_at) in enumerate(
                self._pending_batch
            ):
                if pending_stream_index == stream_index:
                    self._pending_batch[index] = (stream_index, seq, frame, submitted_at)
                    break
            else:
                self._pending_batch.append((stream_index, seq, frame, submitted_at))

    def _pop_ready_batch_unlocked(self) -> list[tuple[int, int, Any, float]]:
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
    def __init__(
        self,
        store: LatestFrameStore,
        *,
        batch_size: int,
        queue_size: int,
        sample_every_n_frames: int,
        adaptive_sampling: bool,
        adaptive_base_streams_per_worker: int = 12,
        adaptive_max_sample_interval: int = 5,
        batch_wait_ms: int = 0,
        adaptive_batch_wait: bool = False,
        active_streams_per_worker: int = 1,
    ) -> None:
        self.store = store
        self.scheduler = SharedBatchScheduler(
            store,
            batch_size=batch_size,
            max_wait_ms=batch_wait_ms,
            adaptive_wait=adaptive_batch_wait,
        )
        self.batch_size = max(int(batch_size), 1)
        self.sample_interval = effective_sample_interval(
            sample_every_n_frames=sample_every_n_frames,
            adaptive_sampling=adaptive_sampling,
            adaptive_base_streams_per_worker=adaptive_base_streams_per_worker,
            adaptive_max_sample_interval=adaptive_max_sample_interval,
            active_streams_per_worker=active_streams_per_worker,
        )
        self._queue: queue.Queue[list[tuple[int, int, Any, float]]] = queue.Queue(
            maxsize=max(int(queue_size), 1)
        )
        self.dropped_batch_count = 0

    def dispatch_once(self, now_seconds: float | None = None) -> bool:
        if self._queue.full():
            return False
        batch = self.scheduler.consume_ready_batch(
            sample_interval=self.sample_interval,
            active_stream_count=len(self.store.slots),
            now_seconds=now_seconds,
        )
        if not batch:
            return False
        self._put_latest_batch(batch)
        return True

    def take_batch(self, timeout_seconds: float = 0.01) -> list[tuple[int, int, Any, float]]:
        try:
            return self._queue.get(timeout=max(float(timeout_seconds), 0.0))
        except queue.Empty:
            return []

    def _put_latest_batch(self, batch: list[tuple[int, int, Any, float]]) -> None:
        try:
            self._queue.put_nowait(batch)
        except queue.Full:
                self.dropped_batch_count += 1


class RuntimeBenchmarkFrameStore(RuntimeLatestFrameStore):
    """Add benchmark counters around the application's live latest-frame store."""

    def __init__(self, stream_count: int) -> None:
        super().__init__()
        self.stream_count = max(int(stream_count), 1)
        self._metrics_lock = threading.Lock()
        self._submitted_frame_counts = [0 for _ in range(self.stream_count)]
        self._read_failure_counts = [0 for _ in range(self.stream_count)]
        self._fault_drop_counts = [0 for _ in range(self.stream_count)]
        self._fault_freeze_counts = [0 for _ in range(self.stream_count)]
        self._read_latencies_ms: list[float] = []

    @staticmethod
    def _stream_id(stream_index: int) -> str:
        return f"benchmark-stream-{int(stream_index):03d}"

    def submit(
        self,
        stream_index: int,
        frame: Any,
        *,
        submitted_at_monotonic: float | None = None,
        read_elapsed_ms: float | None = None,
    ) -> None:
        index = int(stream_index)
        if frame is None or index < 0 or index >= self.stream_count:
            return
        timestamp = (
            time.perf_counter()
            if submitted_at_monotonic is None
            else float(submitted_at_monotonic)
        )
        with self._metrics_lock:
            self._submitted_frame_counts[index] += 1
            if read_elapsed_ms is not None:
                self._read_latencies_ms.append(max(float(read_elapsed_ms), 0.0))
        super().submit(self._stream_id(index), frame, timestamp)

    def submit_fault_freeze(self, stream_index: int, frame: Any) -> None:
        index = int(stream_index)
        if index < 0 or index >= self.stream_count:
            return
        with self._metrics_lock:
            self._fault_freeze_counts[index] += 1
        self.submit(index, frame)

    def record_fault_drop(self, stream_index: int) -> None:
        index = int(stream_index)
        if index < 0 or index >= self.stream_count:
            return
        with self._metrics_lock:
            self._fault_drop_counts[index] += 1

    def record_read_failure(self, stream_index: int) -> None:
        index = int(stream_index)
        if index < 0 or index >= self.stream_count:
            return
        with self._metrics_lock:
            self._read_failure_counts[index] += 1

    def totals(self) -> tuple[int, int, int]:
        with self._metrics_lock:
            submitted = sum(self._submitted_frame_counts)
            read_failures = sum(self._read_failure_counts)
        overwritten = sum(
            self.stream_stats(self._stream_id(index)).overwritten_frame_count
            for index in range(self.stream_count)
        )
        return submitted, overwritten, read_failures

    def fault_totals(self) -> tuple[int, int]:
        with self._metrics_lock:
            return sum(self._fault_drop_counts), sum(self._fault_freeze_counts)

    def read_latency_totals(self) -> tuple[int, float, float]:
        with self._metrics_lock:
            if not self._read_latencies_ms:
                return 0, 0.0, 0.0
            return (
                len(self._read_latencies_ms),
                sum(self._read_latencies_ms),
                max(self._read_latencies_ms),
            )

    def overwritten_frames_for_stream(self, stream_index: int) -> int:
        return self.stream_stats(
            self._stream_id(stream_index)
        ).overwritten_frame_count


class RuntimeBenchmarkDispatcher(RuntimeCentralBatchDispatcher):
    """Use the live dispatcher while adapting batch IDs for benchmark statistics."""

    def __init__(
        self,
        store: RuntimeBenchmarkFrameStore,
        *,
        batch_size: int,
        queue_size: int,
        sample_every_n_frames: int,
        adaptive_sampling: bool,
        adaptive_base_streams_per_worker: int = 12,
        adaptive_max_sample_interval: int = 5,
        batch_wait_ms: int = 0,
        adaptive_batch_wait: bool = False,
        active_streams_per_worker: int = 1,  # Kept for call-site compatibility.
    ) -> None:
        del active_streams_per_worker
        super().__init__(
            store,
            batch_size=batch_size,
            queue_size=queue_size,
            sample_every_n_frames=sample_every_n_frames,
            adaptive_sampling=adaptive_sampling,
            adaptive_base_streams_per_worker=adaptive_base_streams_per_worker,
            adaptive_max_sample_interval=adaptive_max_sample_interval,
            batch_wait_ms=batch_wait_ms,
            adaptive_batch_wait=adaptive_batch_wait,
        )

    @property
    def sample_interval(self) -> int:
        return self._effective_sample_interval(len(self.stream_ids()))

    def take_batch(
        self,
        timeout_seconds: float = 0.01,
    ) -> list[tuple[int, int, Any, float]]:
        deadline = time.perf_counter() + max(float(timeout_seconds), 0.0)
        while True:
            batch = super().take_batch()
            if batch:
                return [
                    (
                        int(stream_id.rsplit("-", 1)[-1]),
                        0,
                        frame,
                        timestamp,
                    )
                    for stream_id, frame, timestamp in batch
                ]
            if time.perf_counter() >= deadline:
                return []
            time.sleep(0.001)


class PostprocessDispatcher:
    def __init__(
        self,
        handler,
        *,
        queue_size: int = 2,
    ) -> None:
        self._handler = handler
        self._queue: queue.Queue[
            tuple[list[tuple[int, int, Any, float]], list[Any]]
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
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(float(timeout_seconds), 0.0))
        self._thread = None

    def enqueue(self, batch: list[tuple[int, int, Any, float]], results: list[Any]) -> bool:
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


class VideoLoopReader(threading.Thread):
    def __init__(
        self,
        *,
        stream_index: int,
        video_path: Path,
        store: LatestFrameStore,
        target_fps: float,
        stop_event: threading.Event,
        fault_plan: FaultInjectionPlan | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.stream_index = stream_index
        self.video_path = video_path
        self.store = store
        self.target_fps = max(float(target_fps), 1.0)
        self.stop_event = stop_event
        self.fault_plan = fault_plan or FaultInjectionPlan()

    def run(self) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            self.store.record_read_failure(self.stream_index)
            return
        frame_interval = 1.0 / self.target_fps
        next_deadline = time.perf_counter()
        started = next_deadline
        last_frame: Any | None = None

        def wait_for_next_frame() -> None:
            nonlocal next_deadline
            next_deadline += frame_interval
            sleep_seconds = next_deadline - time.perf_counter()
            if sleep_seconds > 0:
                self.stop_event.wait(sleep_seconds)
            else:
                next_deadline = time.perf_counter()

        try:
            while not self.stop_event.is_set():
                elapsed = time.perf_counter() - started
                fault_mode = self.fault_plan.active_mode_for(self.stream_index, elapsed)
                if fault_mode == "drop":
                    self.store.record_fault_drop(self.stream_index)
                    wait_for_next_frame()
                    continue
                if fault_mode == "freeze" and last_frame is not None:
                    self.store.submit_fault_freeze(self.stream_index, last_frame)
                    wait_for_next_frame()
                    continue
                read_started = time.perf_counter()
                ok, frame = cap.read()
                read_elapsed_ms = (time.perf_counter() - read_started) * 1000.0
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.store.record_read_failure(self.stream_index)
                    continue
                last_frame = frame
                self.store.submit(
                    self.stream_index,
                    frame,
                    read_elapsed_ms=read_elapsed_ms,
                )
                wait_for_next_frame()
        finally:
            cap.release()


def parse_stream_counts(value: str) -> list[int]:
    counts: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        count = int(part)
        if count <= 0:
            raise ValueError("stream counts must be positive")
        counts.append(count)
    if not counts:
        raise ValueError("at least one stream count is required")
    return counts


def video_fps(video_path: Path, fallback: float) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
    finally:
        cap.release()
    if fps < 1.0 or fps > 240.0:
        return fallback
    return fps


def video_summary(video_path: Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
        height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    finally:
        cap.release()
    return {
        "fps": fps,
        "frames": frames,
        "duration_seconds": frames / fps if fps > 0 else 0.0,
        "width": width,
        "height": height,
    }


def model_backend_name(model_path: Path) -> str:
    suffix = model_path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return "pytorch"
    if suffix in {".engine", ".plan"}:
        return "tensorrt"
    if suffix == ".onnx":
        return "onnx"
    if suffix:
        return suffix.lstrip(".")
    return "unknown"


def effective_model_worker_count(
    *,
    requested_worker_count: int,
    model_backend: str,
) -> int:
    requested_worker_count = max(int(requested_worker_count), 1)
    if str(model_backend).strip().lower() == "tensorrt":
        return min(requested_worker_count, 2)
    return requested_worker_count


def classify_status(
    *,
    capture_deficit_rate: float,
    overwrite_drop_rate: float,
    processed_fps_per_stream: float,
    min_infer_fps_per_stream: float,
) -> str:
    if capture_deficit_rate > 0.15:
        return "decode_bound"
    if processed_fps_per_stream < min_infer_fps_per_stream or overwrite_drop_rate > 0.35:
        return "overloaded"
    if overwrite_drop_rate > 0.15:
        return "marginal"
    return "ok"


def benchmark_stream_quality(
    *,
    target_fps_per_stream: float,
    configured_sample_interval: int,
    effective_sample_interval: int,
    generated_fps_per_stream: float,
    processed_fps_per_stream: float,
    overwrite_drop_rate: float,
) -> StreamQuality:
    target_capture_fps = max(float(target_fps_per_stream), 1.0)
    target_inference_fps = max(
        target_capture_fps / max(int(configured_sample_interval), 1),
        1.0,
    )
    thresholds = StreamQualityThresholds(
        target_capture_fps=target_capture_fps,
        min_capture_fps=max(target_capture_fps * 0.4, 1.0),
        target_inference_fps=target_inference_fps,
        min_inference_fps=max(target_inference_fps * 0.6, 1.0),
    )
    return evaluate_stream_quality(
        StreamQualitySnapshot(
            capture_fps=generated_fps_per_stream,
            inference_fps=processed_fps_per_stream,
            seconds_since_last_frame=0.0,
            seconds_since_last_inference=0.0,
            overwrite_drop_rate=overwrite_drop_rate,
        ),
        thresholds,
    )


def best_detection(result: Any, confidence_threshold: float) -> tuple[str | None, float | None]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None, None
    try:
        box_items = list(boxes)
    except TypeError:
        box_items = []
    if not box_items:
        return None, None
    try:
        box = max(box_items, key=lambda item: float(item.conf[0].item()))
        confidence = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
    except Exception:  # noqa: BLE001
        return None, None
    if confidence < confidence_threshold:
        return None, confidence
    behavior = DEFAULT_BEHAVIOR_CLASSES.get(class_id)
    return (behavior.display_zh if behavior else f"class_{class_id}"), confidence


def build_predict_kwargs(device: str, imgsz: int | None, half: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"verbose": False, "device": device}
    if imgsz is not None and int(imgsz) > 0:
        kwargs["imgsz"] = int(imgsz)
    if half and str(device).startswith("cuda"):
        kwargs["half"] = True
    return kwargs


def effective_sample_interval(
    *,
    sample_every_n_frames: int,
    adaptive_sampling: bool,
    adaptive_base_streams_per_worker: int,
    adaptive_max_sample_interval: int,
    active_streams_per_worker: int,
) -> int:
    sample_every_n_frames = max(int(sample_every_n_frames), 1)
    if not adaptive_sampling:
        return sample_every_n_frames
    active = max(int(active_streams_per_worker), 1)
    base = max(int(adaptive_base_streams_per_worker), 1)
    max_interval = max(int(adaptive_max_sample_interval), sample_every_n_frames)
    overload_steps = max(0, math.ceil(active / base) - 1)
    return min(max_interval, sample_every_n_frames + overload_steps)


def run_once(
    *,
    run_index: int = 1,
    stream_count: int,
    video_path: Path,
    models: list[YOLO],
    device: str,
    duration_seconds: float,
    target_fps: float,
    batch_size: int,
    batch_wait_ms: int,
    adaptive_batch_wait: bool,
    async_postprocess: bool,
    confidence_threshold: float,
    min_infer_fps_per_stream: float,
    predict_kwargs: dict[str, Any],
    sample_every_n_frames: int,
    adaptive_sampling: bool,
    adaptive_base_streams_per_worker: int,
    adaptive_max_sample_interval: int,
    preview_stream_count: int = 0,
    preview_target_fps: float = 0.0,
    fault_config: FaultInjectionConfig | None = None,
) -> BenchmarkResult:
    stop_event = threading.Event()
    store = RuntimeBenchmarkFrameStore(stream_count)
    batch_dispatcher = RuntimeBenchmarkDispatcher(
        store,
        batch_size=batch_size,
        queue_size=max(len(models), 1),
        sample_every_n_frames=sample_every_n_frames,
        adaptive_sampling=adaptive_sampling,
        adaptive_base_streams_per_worker=adaptive_base_streams_per_worker,
        adaptive_max_sample_interval=adaptive_max_sample_interval,
        batch_wait_ms=batch_wait_ms,
        adaptive_batch_wait=adaptive_batch_wait,
        active_streams_per_worker=math.ceil(stream_count / max(len(models), 1)),
    )
    fault_plan = build_fault_injection_plan(fault_config, stream_count=stream_count)
    readers = [
        VideoLoopReader(
            stream_index=index,
            video_path=video_path,
            store=store,
            target_fps=capture_target_fps_for_stream(
                index,
                base_target_fps=target_fps,
                preview_stream_count=preview_stream_count,
                preview_target_fps=preview_target_fps or target_fps,
            ),
            stop_event=stop_event,
            fault_plan=fault_plan,
        )
        for index in range(stream_count)
    ]
    for reader in readers:
        reader.start()
    batch_dispatcher.start()

    processed_frames = 0
    empty_detections = 0
    low_confidence_detections = 0
    confidences: list[float] = []
    inference_ms: list[float] = []
    batch_sizes: list[int] = []
    queue_wait_ms: list[float] = []
    model_ms: list[float] = []
    postprocess_ms: list[float] = []
    stats_lock = threading.Lock()
    started = time.perf_counter()
    deadline = started + duration_seconds
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sample_interval = batch_dispatcher.sample_interval

    def handle_postprocess(batch, result_items) -> None:
        nonlocal processed_frames, empty_detections, low_confidence_detections
        local_confidences: list[float] = []
        local_empty = 0
        local_low_confidence = 0
        postprocess_started = time.perf_counter()
        for index in range(len(batch)):
            result = result_items[index] if index < len(result_items) else None
            action, confidence = best_detection(result, confidence_threshold)
            if confidence is not None:
                local_confidences.append(confidence)
            if action is None:
                local_empty += 1
                if confidence is not None and confidence < confidence_threshold:
                    local_low_confidence += 1
        local_postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
        with stats_lock:
            postprocess_ms.append(local_postprocess_ms)
            processed_frames += len(batch)
            empty_detections += local_empty
            low_confidence_detections += local_low_confidence
            confidences.extend(local_confidences)

    postprocess_dispatcher = (
        PostprocessDispatcher(
            handle_postprocess,
            queue_size=max(len(models), 1) * 2,
        )
        if async_postprocess
        else None
    )
    if postprocess_dispatcher is not None:
        postprocess_dispatcher.start()

    def inference_loop(worker_index: int, model: YOLO) -> None:
        while time.perf_counter() < deadline:
            batch = batch_dispatcher.take_batch(timeout_seconds=0.01)
            if not batch:
                continue
            consume_started = time.perf_counter()
            frames = [frame for _stream_index, _seq, frame, _submitted_at in batch]
            local_queue_wait_ms = [
                max((consume_started - float(submitted_at)) * 1000.0, 0.0)
                for _stream_index, _seq, _frame, submitted_at in batch
            ]
            infer_started = time.perf_counter()
            results = model(frames, **predict_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - infer_started) * 1000.0
            result_items = list(results) if results is not None else []
            with stats_lock:
                inference_ms.append(elapsed_ms)
                model_ms.append(elapsed_ms)
                batch_sizes.append(len(batch))
                queue_wait_ms.extend(local_queue_wait_ms)
            if postprocess_dispatcher is None or not postprocess_dispatcher.enqueue(
                batch,
                result_items,
            ):
                handle_postprocess(batch, result_items)

    inference_threads = [
        threading.Thread(target=inference_loop, args=(index, model), daemon=True)
        for index, model in enumerate(models)
    ]
    try:
        for thread in inference_threads:
            thread.start()
        for thread in inference_threads:
            thread.join(timeout=duration_seconds + 5.0)
    finally:
        stop_event.set()
        batch_dispatcher.stop(timeout_seconds=2.0)
        if postprocess_dispatcher is not None:
            postprocess_dispatcher.stop()
        for reader in readers:
            reader.join(timeout=2.0)

    elapsed = max(time.perf_counter() - started, 0.001)
    generated_frames, overwritten_frames, _read_failures = store.totals()
    fault_dropped_frames, fault_frozen_frames = store.fault_totals()
    fault_events = fault_dropped_frames + fault_frozen_frames
    expected_frames = expected_frame_count(
        stream_count=stream_count,
        duration_seconds=duration_seconds,
        base_target_fps=target_fps,
        preview_stream_count=preview_stream_count,
        preview_target_fps=preview_target_fps or target_fps,
    )
    capture_deficit = max(0, expected_frames - generated_frames)
    capture_deficit_rate = capture_deficit / expected_frames if expected_frames else 0.0
    overwrite_drop_rate = overwritten_frames / generated_frames if generated_frames else 0.0
    empty_detection_rate = empty_detections / processed_frames if processed_frames else 0.0
    low_confidence_rate = low_confidence_detections / processed_frames if processed_frames else 0.0
    generated_fps_total = generated_frames / elapsed
    processed_fps_total = processed_frames / elapsed
    generated_fps_per_stream = generated_fps_total / stream_count
    processed_fps_per_stream = processed_fps_total / stream_count
    p50_ms = statistics.median(inference_ms) if inference_ms else 0.0
    p95_ms = percentile(inference_ms, 95.0) if inference_ms else 0.0
    p50_queue_wait_ms = statistics.median(queue_wait_ms) if queue_wait_ms else 0.0
    p95_queue_wait_ms = percentile(queue_wait_ms, 95.0) if queue_wait_ms else 0.0
    p50_model_ms = statistics.median(model_ms) if model_ms else 0.0
    p95_model_ms = percentile(model_ms, 95.0) if model_ms else 0.0
    p50_postprocess_ms = statistics.median(postprocess_ms) if postprocess_ms else 0.0
    p95_postprocess_ms = percentile(postprocess_ms, 95.0) if postprocess_ms else 0.0
    read_calls, read_time_ms_total, read_time_ms_max = store.read_latency_totals()
    gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        if torch.cuda.is_available()
        else 0.0
    )
    process_resources = process_resource_snapshot()
    quality = benchmark_stream_quality(
        target_fps_per_stream=target_fps,
        configured_sample_interval=sample_every_n_frames,
        effective_sample_interval=sample_interval,
        generated_fps_per_stream=generated_fps_per_stream,
        processed_fps_per_stream=processed_fps_per_stream,
        overwrite_drop_rate=overwrite_drop_rate,
    )

    return BenchmarkResult(
        streams=stream_count,
        duration_seconds=round(elapsed, 3),
        target_fps_per_stream=round(target_fps, 3),
        expected_frames=expected_frames,
        generated_frames=generated_frames,
        processed_frames=processed_frames,
        overwritten_frames=overwritten_frames,
        empty_detections=empty_detections,
        low_confidence_detections=low_confidence_detections,
        mean_confidence=round(statistics.mean(confidences), 4) if confidences else 0.0,
        p50_inference_ms=round(p50_ms, 2),
        p95_inference_ms=round(p95_ms, 2),
        generated_fps_total=round(generated_fps_total, 2),
        processed_fps_total=round(processed_fps_total, 2),
        generated_fps_per_stream=round(generated_fps_per_stream, 2),
        processed_fps_per_stream=round(processed_fps_per_stream, 2),
        capture_deficit_rate=round(capture_deficit_rate, 4),
        overwrite_drop_rate=round(overwrite_drop_rate, 4),
        empty_detection_rate=round(empty_detection_rate, 4),
        low_confidence_rate=round(low_confidence_rate, 4),
        status=classify_status(
            capture_deficit_rate=capture_deficit_rate,
            overwrite_drop_rate=overwrite_drop_rate,
            processed_fps_per_stream=processed_fps_per_stream,
            min_infer_fps_per_stream=min_infer_fps_per_stream,
        ),
        quality_status=quality.status.value,
        record_reliable=quality.record_reliable,
        quality_label_zh=quality.label_zh,
        quality_reason_zh=quality.reason_zh,
        gpu_memory_mb=round(gpu_memory_mb, 1),
        sample_interval=sample_interval,
        process_memory_mb=process_resources.process_memory_mb,
        process_thread_count=process_resources.process_thread_count,
        run_index=run_index,
        fault_mode=fault_plan.mode if fault_plan.enabled else "none",
        fault_streams=len(fault_plan.affected_streams),
        fault_events=fault_events,
        fault_dropped_frames=fault_dropped_frames,
        fault_frozen_frames=fault_frozen_frames,
        read_calls=read_calls,
        read_time_ms_total=round(read_time_ms_total, 2),
        read_time_ms_max=round(read_time_ms_max, 2),
        inference_batches=len(batch_sizes),
        mean_batch_size=round(statistics.mean(batch_sizes), 2) if batch_sizes else 0.0,
        max_batch_size=max(batch_sizes) if batch_sizes else 0,
        p50_queue_wait_ms=round(p50_queue_wait_ms, 2),
        p95_queue_wait_ms=round(p95_queue_wait_ms, 2),
        p50_model_ms=round(p50_model_ms, 2),
        p95_model_ms=round(p95_model_ms, 2),
        p50_postprocess_ms=round(p50_postprocess_ms, 2),
        p95_postprocess_ms=round(p95_postprocess_ms, 2),
    )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def write_outputs(
    *,
    output_dir: Path,
    results: list[BenchmarkResult],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"stream_capacity_{timestamp}.json"
    csv_path = output_dir / f"stream_capacity_{timestamp}.csv"
    payload = {
        "metadata": metadata,
        "recommendation": build_capacity_recommendation(results, metadata),
        "resource_summary": build_resource_summary(
            results,
            **_resource_thresholds_from_metadata(metadata),
        ),
        "fault_injection_summary": build_fault_injection_summary(results, metadata),
        "profiler_summary": build_profiler_summary(results),
        "results": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser(description="Benchmark multi-stream YOLO detection capacity.")
    parser.add_argument("--video", type=Path, required=True, help="Video file used to simulate streams.")
    parser.add_argument("--model", type=Path, default=config.model_path, help="YOLO model path.")
    parser.add_argument("--streams", default="1,2,4,6,8,10,12,16", help="Comma-separated stream counts.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the stream-count sweep this many times.")
    parser.add_argument("--duration", type=float, default=8.0, help="Seconds per stream-count run.")
    parser.add_argument("--target-fps", type=float, default=0.0, help="0 means use source video FPS.")
    parser.add_argument(
        "--preview-streams",
        type=int,
        default=0,
        help="Number of streams to simulate at preview FPS; remaining streams use --target-fps.",
    )
    parser.add_argument(
        "--preview-target-fps",
        type=float,
        default=0.0,
        help="Capture FPS for preview streams; 0 means use config preview FPS.",
    )
    parser.add_argument(
        "--fault-mode",
        choices=["none", "drop", "freeze"],
        default="none",
        help="Inject synthetic stream faults: drop simulates disconnect, freeze repeats the last frame.",
    )
    parser.add_argument(
        "--fault-stream-fraction",
        type=float,
        default=0.0,
        help="Fraction of streams affected by fault injection, from 0 to 1.",
    )
    parser.add_argument(
        "--fault-start-seconds",
        type=float,
        default=0.0,
        help="Seconds after each run starts before fault injection begins.",
    )
    parser.add_argument(
        "--fault-duration-seconds",
        type=float,
        default=0.0,
        help="Length of each injected fault window.",
    )
    parser.add_argument(
        "--fault-repeat-seconds",
        type=float,
        default=0.0,
        help="0 means one fault window; otherwise repeat the window at this interval.",
    )
    parser.add_argument("--fault-seed", type=int, default=20260701)
    parser.add_argument("--batch-size", type=int, default=config.inference_batch_size)
    parser.add_argument(
        "--batch-wait-ms",
        type=int,
        default=getattr(config, "inference_batch_wait_ms", 12),
        help="Maximum time to hold a partial batch before inference.",
    )
    parser.add_argument(
        "--adaptive-batch-wait",
        action=argparse.BooleanOptionalAction,
        default=getattr(config, "adaptive_batch_wait", True),
        help="Adapt partial-batch wait time from recent batch fill behavior.",
    )
    parser.add_argument(
        "--async-postprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run result parsing/postprocess in a separate Python thread.",
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
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=config.inference_half,
        help="Use FP16 on CUDA. Use --no-half to force FP32.",
    )
    parser.add_argument("--device", default=resolve_inference_device(config.inference_device))
    parser.add_argument("--confidence", type=float, default=config.confidence_threshold)
    parser.add_argument("--min-infer-fps", type=float, default=4.0)
    parser.add_argument("--max-process-memory-delta-mb", type=float, default=512.0)
    parser.add_argument("--max-gpu-memory-delta-mb", type=float, default=512.0)
    parser.add_argument("--max-thread-count-delta", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=config.output_dir / "benchmarks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")
    if not args.model or not Path(args.model).exists():
        raise SystemExit(f"model not found: {args.model}")

    stream_counts = parse_stream_counts(args.streams)
    try:
        run_plan = stream_run_plan(stream_counts, args.repeat)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    model_backend = model_backend_name(Path(args.model))
    requested_worker_count = max(int(args.worker_count), 1)
    worker_count = effective_model_worker_count(
        requested_worker_count=requested_worker_count,
        model_backend=model_backend,
    )
    summary = video_summary(args.video)
    target_fps = args.target_fps if args.target_fps > 0 else video_fps(args.video, 12.0)
    preview_target_fps = args.preview_target_fps if args.preview_target_fps > 0 else config.preview_fps
    fault_config = FaultInjectionConfig(
        mode=args.fault_mode,
        stream_fraction=args.fault_stream_fraction,
        start_seconds=args.fault_start_seconds,
        duration_seconds=args.fault_duration_seconds,
        repeat_seconds=args.fault_repeat_seconds,
        seed=args.fault_seed,
    ).normalized()
    predict_kwargs = build_predict_kwargs(args.device, args.imgsz, args.half)

    print(f"video={args.video}")
    print(f"model={args.model}")
    print(f"device={args.device}")
    print(f"imgsz={args.imgsz} half={bool(args.half and str(args.device).startswith('cuda'))}")
    print(
        "video_meta="
        f"{summary['width']:.0f}x{summary['height']:.0f} "
        f"fps={summary['fps']:.2f} duration={summary['duration_seconds']:.1f}s"
    )
    print(
        f"target_fps_per_stream={target_fps:.2f} "
        f"preview_streams={max(int(args.preview_streams), 0)} "
        f"preview_target_fps={preview_target_fps:.2f} "
        f"batch_size={args.batch_size} batch_wait_ms={args.batch_wait_ms} "
        f"adaptive_batch_wait={args.adaptive_batch_wait} "
        f"async_postprocess={args.async_postprocess} "
        f"worker_count={worker_count} "
        f"sample_every={args.sample_every_n_frames} adaptive={args.adaptive_sampling} "
        f"repeat={args.repeat}"
    )
    if worker_count != requested_worker_count:
        print(
            f"requested_worker_count={requested_worker_count} "
            f"effective_worker_count={worker_count} model_backend={model_backend}"
        )
    if fault_config.enabled:
        print(
            f"fault_mode={fault_config.mode} "
            f"fault_stream_fraction={fault_config.stream_fraction:.3f} "
            f"fault_start={fault_config.start_seconds:.1f}s "
            f"fault_duration={fault_config.duration_seconds:.1f}s "
            f"fault_repeat={fault_config.repeat_seconds:.1f}s "
            f"fault_seed={fault_config.seed}"
        )

    models = [YOLO(str(args.model)) for _ in range(worker_count)]
    warmup_frame = cv2.imread(str(args.video))
    if warmup_frame is None:
        cap = cv2.VideoCapture(str(args.video))
        ok, warmup_frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("failed to read warmup frame")
    for model in models:
        model([warmup_frame], **predict_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    results: list[BenchmarkResult] = []
    for run_index, stream_count in run_plan:
        result = run_once(
            run_index=run_index,
            stream_count=stream_count,
            video_path=args.video,
            models=models,
            device=args.device,
            duration_seconds=args.duration,
            target_fps=target_fps,
            batch_size=args.batch_size,
            batch_wait_ms=args.batch_wait_ms,
            adaptive_batch_wait=args.adaptive_batch_wait,
            async_postprocess=args.async_postprocess,
            confidence_threshold=args.confidence,
            min_infer_fps_per_stream=args.min_infer_fps,
            predict_kwargs=predict_kwargs,
            sample_every_n_frames=args.sample_every_n_frames,
            adaptive_sampling=args.adaptive_sampling,
            adaptive_base_streams_per_worker=args.adaptive_base_streams_per_worker,
            adaptive_max_sample_interval=args.adaptive_max_sample_interval,
            preview_stream_count=max(int(args.preview_streams), 0),
            preview_target_fps=preview_target_fps,
            fault_config=fault_config,
        )
        results.append(result)
        run_label = f"run={result.run_index}/{args.repeat} " if args.repeat > 1 else ""
        print(
            f"{run_label}{result.streams:>3} streams | status={result.status:<11} "
            f"quality={result.quality_label_zh} "
            f"capture={result.generated_fps_per_stream:>5.1f}/s/stream "
            f"infer={result.processed_fps_per_stream:>5.1f}/s/stream "
            f"sample={result.sample_interval} "
            f"drop={result.overwrite_drop_rate:>5.1%} "
            f"empty={result.empty_detection_rate:>5.1%} "
            f"rss={result.process_memory_mb:>6.1f}MB "
            f"threads={result.process_thread_count:>3} "
            f"p95={result.p95_inference_ms:>6.1f}ms"
        )

    metadata = {
        "video": str(args.video),
        "model": str(args.model),
        "model_backend": model_backend,
        "device": args.device,
        "video_summary": summary,
        "target_fps_per_stream": target_fps,
        "preview_stream_count": max(int(args.preview_streams), 0),
        "preview_target_fps": preview_target_fps,
        "repeat_count": args.repeat,
        "batch_size": args.batch_size,
        "batch_wait_ms": args.batch_wait_ms,
        "adaptive_batch_wait": args.adaptive_batch_wait,
        "async_postprocess": args.async_postprocess,
        "requested_worker_count": requested_worker_count,
        "worker_count": worker_count,
        "sample_every_n_frames": args.sample_every_n_frames,
        "adaptive_sampling": args.adaptive_sampling,
        "adaptive_base_streams_per_worker": args.adaptive_base_streams_per_worker,
        "adaptive_max_sample_interval": args.adaptive_max_sample_interval,
        "imgsz": args.imgsz,
        "half": bool(args.half and str(args.device).startswith("cuda")),
        "confidence_threshold": args.confidence,
        "min_infer_fps_per_stream": args.min_infer_fps,
        "max_process_memory_delta_mb": args.max_process_memory_delta_mb,
        "max_gpu_memory_delta_mb": args.max_gpu_memory_delta_mb,
        "max_thread_count_delta": args.max_thread_count_delta,
        "fault_injection": {
            "enabled": fault_config.enabled,
            **asdict(fault_config),
        },
    }
    json_path, csv_path = write_outputs(output_dir=args.output_dir, results=results, metadata=metadata)
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
