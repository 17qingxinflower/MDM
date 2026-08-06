from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import json
import os

from deerui.domain.models import DEFAULT_BEHAVIOR_CLASSES, BehaviorClass
from deerui.runtime_paths import (
    app_root,
    bundled_resource,
    default_benchmark_video_path,
    default_output_dir,
    env_file_candidates,
    is_frozen,
    packaged_default_database_url,
)


APP_ROOT = app_root()
_ENV_FILES_LOADED = False


@dataclass(frozen=True)
class AppConfig:
    model_path: Path
    database_url: str
    output_dir: Path
    benchmark_video_path: Path | None = None
    frame_queue_size: int = 2
    display_fps: int = 30
    action_threshold: float = 0.25
    action_tolerance_seconds: float = 0.5
    meters_per_pixel: float = 0.01
    trajectory_points: int = 45
    confidence_threshold: float = 0.25
    sample_every_n_frames: int = 2
    inference_strategy: str = "capacity"
    inference_device: str = "auto"
    inference_batch_size: int = 6
    inference_batch_wait_ms: int = 12
    adaptive_batch_wait: bool = True
    inference_worker_count: int = 4
    inference_imgsz: int = 640
    inference_half: bool = True
    adaptive_sampling: bool = True
    adaptive_base_streams_per_worker: int = 12
    adaptive_max_sample_interval: int = 5
    max_preview_streams: int = 8
    max_managed_streams: int = 100
    capacity_profile: str = "rtx4050"
    capacity_calibration_path: Path | None = None
    worker_stream_capacity: int = 10
    stream_target_fps: float = 12.0
    preview_fps: float = 20.0
    worker_heartbeat_timeout_seconds: float = 15.0
    result_flush_interval_seconds: float = 5.0
    inference_watchdog_seconds: float = 8.0
    stream_stale_seconds: float = 5.0
    camera_reconnect_delay_seconds: float = 2.0
    camera_max_read_failures: int = 5
    camera_open_timeout_milliseconds: int = 5000
    camera_read_timeout_milliseconds: int = 2000
    camera_max_buffer_drain_grabs_per_frame: int = 2
    behavior_classes: Mapping[int, BehaviorClass] = field(
        default_factory=lambda: DEFAULT_BEHAVIOR_CLASSES
    )


def _load_behavior_classes() -> Mapping[int, BehaviorClass]:
    """Parse `DEERUI_BEHAVIOR_CLASS_MAPPING` (JSON) or fall back to the legacy defaults.

    JSON shape: `{"<class_id>": ["action_key", "display_zh", "display_en"], ...}`
    On any parse error, fall back to `DEFAULT_BEHAVIOR_CLASSES` and let `load_config`
    warn via stdout; we deliberately do not raise here so a malformed env var does not
    block the GUI from launching.
    """
    raw = os.getenv("DEERUI_BEHAVIOR_CLASS_MAPPING")
    if not raw:
        return DEFAULT_BEHAVIOR_CLASSES
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return DEFAULT_BEHAVIOR_CLASSES
    if not isinstance(parsed, dict):
        return DEFAULT_BEHAVIOR_CLASSES
    result: dict[int, BehaviorClass] = {}
    for key, value in parsed.items():
        try:
            class_id = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            continue
        action_key, display_zh, display_en = (str(v) for v in value)
        result[class_id] = BehaviorClass(action_key, display_zh, display_en)
    return result or DEFAULT_BEHAVIOR_CLASSES


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_env_file(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _load_env_files_once() -> None:
    global _ENV_FILES_LOADED
    if _ENV_FILES_LOADED:
        return
    for candidate in env_file_candidates():
        _load_env_file(candidate)
    _ENV_FILES_LOADED = True


def _path_from_env(name: str, default: Path | None, *, relative_to: Path) -> Path | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else relative_to / path


def resolve_inference_device(requested_device: str | None) -> str:
    requested = (requested_device or "auto").strip().lower()
    if requested in {"", "auto"}:
        return "cuda:0" if _torch_cuda_available() else "cpu"
    return requested


def default_worker_stream_capacity(capacity_profile: str | None) -> int:
    profile = (capacity_profile or "rtx4050").strip().lower()
    if profile in {"rtx5090", "5090"}:
        # Conservative starting point for RTX 5090-class deployments; confirm with
        # site-specific benchmark runs before unattended farm operation.
        return 24
    if profile in {"rtx4050", "4050", "local", "default"}:
        return 10
    return 10


def _is_tensorrt_engine_path(model_path: Path) -> bool:
    return model_path.suffix.lower() in {".engine", ".plan"}


def _normalize_inference_strategy(raw: str | None) -> str:
    strategy = (raw or "capacity").strip().lower().replace("-", "_")
    if strategy in {"smooth", "preview", "smooth_preview"}:
        return "smooth_preview"
    return "capacity"


def _repeated_calibration_is_stable(payload: dict, recommendation: dict) -> bool:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return True
    try:
        repeat_count = int(metadata.get("repeat_count", 1))
    except (TypeError, ValueError):
        repeat_count = 1
    if repeat_count <= 1:
        return True
    try:
        candidate_run_count = int(recommendation.get("candidate_run_count", 0))
        candidate_reliable_run_count = int(
            recommendation.get("candidate_reliable_run_count", 0)
        )
    except (TypeError, ValueError):
        return False
    return candidate_run_count >= repeat_count and candidate_reliable_run_count >= repeat_count


def calibrated_worker_stream_capacity(calibration_path: Path | None) -> int | None:
    if calibration_path is None:
        return None
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    recommendation = payload.get("recommendation", payload)
    if not isinstance(recommendation, dict):
        return None
    if recommendation.get("usable") is not True:
        return None
    if not _repeated_calibration_is_stable(payload, recommendation):
        return None
    try:
        capacity = int(recommendation.get("recommended_streams_per_worker", 0))
    except (TypeError, ValueError):
        return None
    if capacity <= 0:
        return None
    return capacity


def load_config() -> AppConfig:
    _load_env_files_once()
    default_model = bundled_resource("weights", "best.engine")
    if not default_model.exists():
        default_model = bundled_resource("weights", "best.pt")
    model_path = _path_from_env("DEERUI_MODEL_PATH", default_model, relative_to=APP_ROOT)
    assert model_path is not None
    is_tensorrt_engine = _is_tensorrt_engine_path(model_path)
    inference_strategy = _normalize_inference_strategy(
        os.getenv("DEERUI_INFERENCE_STRATEGY")
    )
    tensorrt_smooth_preview = is_tensorrt_engine and inference_strategy == "smooth_preview"
    default_inference_batch_size = 8 if is_tensorrt_engine else 6
    default_inference_worker_count = 2 if is_tensorrt_engine else 4
    default_adaptive_sampling = False if is_tensorrt_engine else True
    default_sample_every_n_frames = 3 if tensorrt_smooth_preview else 2
    default_stream_target_fps = 12
    if is_tensorrt_engine:
        default_stream_target_fps = 15 if tensorrt_smooth_preview else 10
    output_dir = _path_from_env("DEERUI_OUTPUT_DIR", default_output_dir(), relative_to=APP_ROOT)
    assert output_dir is not None
    if is_frozen() or os.getenv("DEERUI_DEPLOYMENT_MODE", "").strip().lower() == "full":
        default_database_url = packaged_default_database_url()
    else:
        default_database_url = f"sqlite+pysqlite:///{APP_ROOT / 'deerui_dev.db'}"
    benchmark_video_path = _path_from_env(
        "DEERUI_BENCHMARK_VIDEO_PATH",
        default_benchmark_video_path(),
        relative_to=APP_ROOT,
    )
    capacity_profile = os.getenv("DEERUI_CAPACITY_PROFILE", "rtx4050").strip().lower()
    capacity_calibration_raw = os.getenv("DEERUI_CAPACITY_CALIBRATION_PATH", "").strip()
    capacity_calibration_path = Path(capacity_calibration_raw) if capacity_calibration_raw else None
    calibrated_capacity = calibrated_worker_stream_capacity(capacity_calibration_path)
    default_capacity = (
        calibrated_capacity
        if calibrated_capacity is not None
        else default_worker_stream_capacity(capacity_profile)
    )
    if calibrated_capacity is None and is_tensorrt_engine:
        engine_default_capacity = 50 if tensorrt_smooth_preview else 60
        default_capacity = max(default_capacity, engine_default_capacity)
    worker_stream_capacity = int(
        os.getenv(
            "DEERUI_WORKER_STREAM_CAPACITY",
            str(default_capacity),
        )
    )
    return AppConfig(
        model_path=model_path,
        database_url=os.getenv(
            "DEERUI_DATABASE_URL",
            default_database_url,
        ),
        output_dir=output_dir,
        benchmark_video_path=benchmark_video_path,
        frame_queue_size=int(os.getenv("DEERUI_FRAME_QUEUE_SIZE", "2")),
        display_fps=int(os.getenv("DEERUI_DISPLAY_FPS", "30")),
        action_threshold=float(os.getenv("DEERUI_ACTION_THRESHOLD", "0.25")),
        action_tolerance_seconds=float(os.getenv("DEERUI_ACTION_TOLERANCE_SECONDS", "0.5")),
        meters_per_pixel=float(os.getenv("DEERUI_METERS_PER_PIXEL", "0.01")),
        trajectory_points=int(os.getenv("DEERUI_TRAJECTORY_POINTS", "45")),
        confidence_threshold=float(os.getenv("DEERUI_CONFIDENCE_THRESHOLD", "0.25")),
        sample_every_n_frames=int(
            os.getenv("DEERUI_SAMPLE_EVERY_N_FRAMES", str(default_sample_every_n_frames))
        ),
        inference_strategy=inference_strategy,
        inference_device=os.getenv("DEERUI_INFERENCE_DEVICE", "auto"),
        inference_batch_size=int(
            os.getenv("DEERUI_INFERENCE_BATCH_SIZE", str(default_inference_batch_size))
        ),
        inference_batch_wait_ms=int(os.getenv("DEERUI_INFERENCE_BATCH_WAIT_MS", "12")),
        adaptive_batch_wait=_env_bool("DEERUI_ADAPTIVE_BATCH_WAIT", True),
        inference_worker_count=int(
            os.getenv("DEERUI_INFERENCE_WORKER_COUNT", str(default_inference_worker_count))
        ),
        inference_imgsz=int(os.getenv("DEERUI_INFERENCE_IMGSZ", "640")),
        inference_half=_env_bool("DEERUI_INFERENCE_HALF", True),
        adaptive_sampling=_env_bool("DEERUI_ADAPTIVE_SAMPLING", default_adaptive_sampling),
        adaptive_base_streams_per_worker=int(
            os.getenv("DEERUI_ADAPTIVE_BASE_STREAMS_PER_WORKER", "12")
        ),
        adaptive_max_sample_interval=int(os.getenv("DEERUI_ADAPTIVE_MAX_SAMPLE_INTERVAL", "5")),
        max_preview_streams=int(os.getenv("DEERUI_MAX_PREVIEW_STREAMS", "8")),
        max_managed_streams=int(os.getenv("DEERUI_MAX_MANAGED_STREAMS", "100")),
        capacity_profile=capacity_profile,
        capacity_calibration_path=capacity_calibration_path,
        worker_stream_capacity=worker_stream_capacity,
        stream_target_fps=float(os.getenv("DEERUI_STREAM_TARGET_FPS", str(default_stream_target_fps))),
        preview_fps=float(os.getenv("DEERUI_PREVIEW_FPS", "20")),
        worker_heartbeat_timeout_seconds=float(
            os.getenv("DEERUI_WORKER_HEARTBEAT_TIMEOUT_SECONDS", "15")
        ),
        result_flush_interval_seconds=float(
            os.getenv("DEERUI_RESULT_FLUSH_INTERVAL_SECONDS", "5")
        ),
        inference_watchdog_seconds=float(os.getenv("DEERUI_INFERENCE_WATCHDOG_SECONDS", "8")),
        stream_stale_seconds=float(os.getenv("DEERUI_STREAM_STALE_SECONDS", "5")),
        camera_reconnect_delay_seconds=float(os.getenv("DEERUI_CAMERA_RECONNECT_DELAY_SECONDS", "2")),
        camera_max_read_failures=int(os.getenv("DEERUI_CAMERA_MAX_READ_FAILURES", "5")),
        camera_open_timeout_milliseconds=int(
            os.getenv("DEERUI_CAMERA_OPEN_TIMEOUT_MILLISECONDS", "5000")
        ),
        camera_read_timeout_milliseconds=int(
            os.getenv("DEERUI_CAMERA_READ_TIMEOUT_MILLISECONDS", "2000")
        ),
        camera_max_buffer_drain_grabs_per_frame=int(
            os.getenv("DEERUI_CAMERA_MAX_DRAIN_GRABS_PER_FRAME", "2")
        ),
        behavior_classes=_load_behavior_classes(),
    )
