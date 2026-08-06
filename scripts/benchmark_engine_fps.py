from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_stream_capacity import build_predict_kwargs, model_backend_name


@dataclass(frozen=True)
class EngineFpsResult:
    batch_size: int
    iterations: int
    warmup_iterations: int
    frames_processed: int
    elapsed_seconds: float
    fps: float
    p50_latency_ms: float
    p95_latency_ms: float


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        value = int(text)
        if value <= 0:
            raise ValueError("values must be positive integers")
        values.append(value)
    if not values:
        raise ValueError("at least one value is required")
    return values


def build_strategy_estimates(
    results: list[EngineFpsResult],
    *,
    source_fps: float,
    sample_intervals: list[int],
    safety_factor: float,
) -> list[dict[str, float | int]]:
    source_fps = max(float(source_fps), 1.0)
    safety_factor = min(max(float(safety_factor), 0.05), 1.0)
    estimates: list[dict[str, float | int]] = []
    for result in results:
        usable_fps = result.fps * safety_factor
        for sample_interval in sample_intervals:
            sample_interval = max(int(sample_interval), 1)
            required_fps = source_fps / sample_interval
            estimates.append(
                {
                    "batch_size": result.batch_size,
                    "sample_interval": sample_interval,
                    "required_infer_fps_per_stream": round(required_fps, 2),
                    "estimated_streams": int(usable_fps // required_fps),
                }
            )
    return estimates


def video_summary(video_path: Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        return {
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "frame_count": float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0),
            "width": float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0),
            "height": float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0),
        }
    finally:
        cap.release()


def read_video_frames(video_path: Path, count: int) -> list[Any]:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[Any] = []
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        while len(frames) < count:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"failed to read frames from video: {video_path}")
    return frames


def _synchronize_cuda(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = min(max(math.ceil((percentile / 100.0) * len(values)) - 1, 0), len(values) - 1)
    return sorted(values)[index]


def _batch_from_frames(frames: list[Any], batch_size: int, offset: int) -> list[Any]:
    return [frames[(offset + index) % len(frames)] for index in range(batch_size)]


def benchmark_batch(
    model: YOLO,
    frames: list[Any],
    *,
    batch_size: int,
    iterations: int,
    warmup_iterations: int,
    predict_kwargs: dict[str, Any],
    device: str,
) -> EngineFpsResult:
    batch_size = max(int(batch_size), 1)
    iterations = max(int(iterations), 1)
    warmup_iterations = max(int(warmup_iterations), 0)

    for index in range(warmup_iterations):
        model(_batch_from_frames(frames, batch_size, index), **predict_kwargs)
    _synchronize_cuda(device)

    latencies_ms: list[float] = []
    started = time.perf_counter()
    for index in range(iterations):
        infer_started = time.perf_counter()
        model(_batch_from_frames(frames, batch_size, index), **predict_kwargs)
        _synchronize_cuda(device)
        latencies_ms.append((time.perf_counter() - infer_started) * 1000.0)
    elapsed = max(time.perf_counter() - started, 1e-9)
    frames_processed = iterations * batch_size
    return EngineFpsResult(
        batch_size=batch_size,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        frames_processed=frames_processed,
        elapsed_seconds=round(elapsed, 4),
        fps=round(frames_processed / elapsed, 2),
        p50_latency_ms=round(statistics.median(latencies_ms), 2),
        p95_latency_ms=round(_percentile(latencies_ms, 95), 2),
    )


def write_outputs(
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    results: list[EngineFpsResult],
    strategy_estimates: list[dict[str, float | int]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"engine_fps_{timestamp}.json"
    csv_path = output_dir / f"engine_fps_{timestamp}.csv"
    payload = {
        "metadata": metadata,
        "results": [asdict(result) for result in results],
        "strategy_estimates": strategy_estimates,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one YOLO/TensorRT engine FPS.")
    parser.add_argument("--model", type=Path, default=Path("weights") / "best.engine")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--sample-intervals", default="1,2,3,4,5")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--frame-samples", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task", default="detect")
    parser.add_argument("--source-fps", type=float, default=0.0)
    parser.add_argument("--safety-factor", type=float, default=0.70)
    parser.add_argument("--output-dir", type=Path, default=Path("UI_result") / "benchmarks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batches = parse_int_list(args.batches)
    sample_intervals = parse_int_list(args.sample_intervals)
    summary = video_summary(args.video)
    source_fps = args.source_fps if args.source_fps > 0 else summary["fps"]
    if source_fps <= 0:
        source_fps = 10.0
    frames = read_video_frames(args.video, args.frame_samples)
    predict_kwargs = build_predict_kwargs(args.device, args.imgsz, args.half)

    print(f"video={args.video}")
    print(f"model={args.model}")
    print(f"backend={model_backend_name(args.model)}")
    print(f"source_fps={source_fps:.2f} batches={batches} safety_factor={args.safety_factor:.2f}")

    model = YOLO(str(args.model), task=args.task)
    results: list[EngineFpsResult] = []
    for batch_size in batches:
        result = benchmark_batch(
            model,
            frames,
            batch_size=batch_size,
            iterations=args.iterations,
            warmup_iterations=args.warmup,
            predict_kwargs=predict_kwargs,
            device=args.device,
        )
        results.append(result)
        print(
            f"batch={result.batch_size:<2} fps={result.fps:>7.2f} "
            f"p50={result.p50_latency_ms:>6.2f}ms p95={result.p95_latency_ms:>6.2f}ms"
        )

    strategy_estimates = build_strategy_estimates(
        results,
        source_fps=source_fps,
        sample_intervals=sample_intervals,
        safety_factor=args.safety_factor,
    )
    best_by_interval = {}
    for estimate in strategy_estimates:
        interval = estimate["sample_interval"]
        current = best_by_interval.get(interval)
        if current is None or estimate["estimated_streams"] > current["estimated_streams"]:
            best_by_interval[interval] = estimate
    print("strategy_estimates:")
    for interval in sorted(best_by_interval):
        estimate = best_by_interval[interval]
        print(
            f"  sample={estimate['sample_interval']} "
            f"batch={estimate['batch_size']} "
            f"need={estimate['required_infer_fps_per_stream']:.2f}/s/stream "
            f"streams~{estimate['estimated_streams']}"
        )

    metadata = {
        "video": str(args.video),
        "video_summary": summary,
        "model": str(args.model),
        "model_backend": model_backend_name(args.model),
        "device": args.device,
        "imgsz": args.imgsz,
        "half": bool(args.half and str(args.device).startswith("cuda")),
        "source_fps": source_fps,
        "safety_factor": args.safety_factor,
        "sample_intervals": sample_intervals,
        "frame_samples": len(frames),
    }
    json_path, csv_path = write_outputs(
        output_dir=args.output_dir,
        metadata=metadata,
        results=results,
        strategy_estimates=strategy_estimates,
    )
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
