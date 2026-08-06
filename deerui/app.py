from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from deerui.config import load_config
from deerui.logging_config import configure_logging
from deerui.persistence.database import build_engine, init_database


_logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuskDeer Monitor PySide6 application")
    parser.add_argument("--smoke-test", action="store_true", help="validate imports and config only")
    parser.add_argument("--init-db", action="store_true", help="create database tables")
    parser.add_argument(
        "--capacity-benchmark",
        action="store_true",
        help="run the packaged 60/80/100 stream capacity benchmark",
    )
    parser.add_argument("--benchmark-streams", default="60,80,100")
    parser.add_argument("--benchmark-duration", type=float, default=60.0)
    parser.add_argument("--benchmark-video", type=Path, default=None)
    parser.add_argument("--benchmark-target-fps", type=float, default=0.0)
    return parser


def _run_capacity_benchmark(args: argparse.Namespace, config) -> int:
    video = args.benchmark_video or config.benchmark_video_path
    if video is None:
        print("No benchmark video configured. Set DEERUI_BENCHMARK_VIDEO_PATH.", file=sys.stderr)
        return 3
    video = Path(video)
    if not video.exists():
        print(f"Benchmark video not found: {video}", file=sys.stderr)
        return 3

    output_dir = config.output_dir / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    target_fps = args.benchmark_target_fps or config.stream_target_fps

    from scripts import benchmark_stream_capacity

    previous_argv = sys.argv[:]
    sys.argv = [
        "benchmark_stream_capacity",
        "--video",
        str(video),
        "--model",
        str(config.model_path),
        "--streams",
        args.benchmark_streams,
        "--duration",
        str(args.benchmark_duration),
        "--target-fps",
        str(target_fps),
        "--output-dir",
        str(output_dir),
    ]
    try:
        return benchmark_stream_capacity.main()
    finally:
        sys.argv = previous_argv


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    configure_logging(config.output_dir / "logs")

    if args.smoke_test:
        return 0

    if args.init_db:
        engine = build_engine(config.database_url)
        init_database(engine)
        print(f"Database initialized: {config.database_url}")
        return 0

    if args.capacity_benchmark:
        return _run_capacity_benchmark(args, config)

    try:
        from deerui.ui.main_window import run_app
    except ImportError as exc:
        print(f"PySide6 UI dependencies are not installed: {exc}", file=sys.stderr)
        return 2

    try:
        return run_app(config)
    except Exception:  # noqa: BLE001
        # The Qt event loop has its own crash path; if a Python exception escapes
        # before `app.exec()` we want it in deerui.log, not just stderr.
        _logger.exception("Fatal error in run_app")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
