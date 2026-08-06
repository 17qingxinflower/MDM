from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from deerui.config import AppConfig, resolve_inference_device
from deerui.domain.alert_rules import AlertRuleEngine
from deerui.domain.models import DetectionResult, TaskMode
from deerui.persistence.database import (
    build_engine,
    build_session_factory,
    init_database,
    session_scope,
)
from deerui.persistence.models import AnalysisTaskRow
from deerui.persistence.repositories import (
    AnalysisTaskRepository,
    DailySummaryRepository,
    DeerProfileRepository,
    SegmentRepository,
)
from deerui.runtime.stream_capacity import (
    StreamCapacityPolicy,
    evaluate_capacity_gate,
)
from deerui.runtime.detection_events import DetectionEventDispatcher
from deerui.runtime.stream_quality import (
    StreamQualitySnapshot,
    StreamQualityStatus,
    StreamQualityThresholds,
    evaluate_stream_quality,
)
from deerui.runtime.stream_quality_events import StreamQualityEventRecorder
from deerui.services.analysis_service import AnalysisService
from deerui.services.app_state import AppRuntimeState, ThresholdSettings
from deerui.services.detection_service import DetectionSession
from deerui.services.export_service import ExportService
from deerui.services.history_service import HistoryService
from deerui.services.profile_service import ProfileService, validate_deer_code
from deerui.ui import theme
from deerui.ui.pages.batch_page import BatchPage
from deerui.ui.pages.chart_dialog import ChartDialog
from deerui.ui.pages.deer_profiles_dialog import DeerProfilesDialog
from deerui.ui.pages.history_page import HistoryPage
from deerui.ui.pages.multi_camera_page import MultiCameraPage
from deerui.ui.pages.realtime_page import RealtimePage
from deerui.ui.pages.threshold_dialog import ThresholdDialog
from deerui.ui import themed_dialogs
from deerui.ui.widgets.sidebar import Sidebar
from deerui.vision.overlay import draw_detection_overlay_bgr
from deerui.workers.camera_worker import CameraWorker
from deerui.workers.capture_supervisor import CaptureRouteSpec, CaptureSupervisor
from deerui.workers.batch_worker import BatchWorker
from deerui.workers.inference_worker import is_pytorch_model_path
from deerui.workers.shared_inference import (
    CentralBatchDispatcher,
    LatestFrameStats,
    LatestFrameStore,
    SharedBatchScheduler,
    SharedInferenceWorker,
)


_logger = logging.getLogger(__name__)


@dataclass
class MultiCameraRuntime:
    deer_code: str | None = None
    source: Path | int | None = None
    camera_thread: QThread | None = None
    camera_worker: CameraWorker | None = None
    detection_session: DetectionSession | None = None
    started_at: datetime | None = None
    last_paint_at: float | None = None
    last_frame_ts: float | None = None
    last_inference_ts: float | None = None
    last_inference_submit_ts: float | None = None
    last_detection: DetectionResult | None = None
    last_preview_seq: int = 0
    fps: float = 0.0
    inference_fps: float = 0.0
    first_overlay_seen: bool = False
    task_id: int | None = None
    deer_id: int | None = None
    task_code: str | None = None
    segments_committed: int = 0
    last_flush_at: float | None = None
    preview_enabled: bool = True
    preview_suspended_for_quality: bool = False
    capture_target_fps: float = 0.0
    quality_submitted_frame_count: int = 0
    quality_overwritten_frame_count: int = 0


class InferenceFrameStoreRouter:
    def __init__(self, stores: list[LatestFrameStore], indexer) -> None:
        self._stores = stores
        self._indexer = indexer

    def submit(self, stream_id: str, frame: object, timestamp_seconds: float) -> None:
        self._stores[self._indexer(stream_id)].submit(stream_id, frame, timestamp_seconds)

    def consume(self, stream_id: str) -> tuple[object | None, float | None, int]:
        return self._stores[self._indexer(stream_id)].consume(stream_id)

    def mark_consumed(self, stream_id: str, sequence: int) -> LatestFrameStats:
        return self._stores[self._indexer(stream_id)].mark_consumed(stream_id, sequence)

    def stream_stats(self, stream_id: str) -> LatestFrameStats:
        return self._stores[self._indexer(stream_id)].stream_stats(stream_id)

    def remove_stream(self, stream_id: str) -> None:
        for store in self._stores:
            store.remove_stream(stream_id)

    def stream_ids(self) -> list[str]:
        ids: list[str] = []
        for store in self._stores:
            ids.extend(store.stream_ids())
        return ids


def _is_tensorrt_model_path(model_path: Path | None) -> bool:
    if model_path is None:
        return False
    return model_path.suffix.lower() in {".engine", ".plan"}


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.inference_device = resolve_inference_device(config.inference_device)
        self.runtime_state = AppRuntimeState()
        self.stream_quality_event_recorder = StreamQualityEventRecorder(
            config.output_dir / "stream_quality_events.jsonl"
        )

        # Detection lifecycle state.
        self.is_detecting: bool = False
        self.camera_thread: QThread | None = None
        self.camera_worker: CameraWorker | None = None
        self.inference_thread: QThread | None = None
        self.inference_worker: SharedInferenceWorker | None = None
        self.inference_worker_count = self._effective_inference_worker_count(config.model_path)
        self.inference_threads: list[QThread] = []
        self.inference_workers: list[SharedInferenceWorker] = []
        self.frame_stores = [
            LatestFrameStore(self)
            for _ in range(self._inference_frame_store_count(config.model_path))
        ]
        self.inference_batch_schedulers = self._build_inference_batch_schedulers(
            self.frame_stores
        )
        self.inference_batch_dispatchers: list[CentralBatchDispatcher] = []
        self.frame_store = InferenceFrameStoreRouter(self.frame_stores, self._inference_worker_index)
        self.preview_frame_store = LatestFrameStore(self)
        self.realtime_stream_id = "realtime"
        self.detection_session: DetectionSession | None = None
        self._last_realtime_detection: DetectionResult | None = None
        self._realtime_last_preview_seq: int = 0
        self.multi_camera_runtimes: dict[str, MultiCameraRuntime] = {}
        self.multi_preview_order: list[str] = []
        self.capture_supervisor = CaptureSupervisor(
            self,
            worker_factory=lambda source, **kwargs: CameraWorker(source, **kwargs),
        )
        self.batch_thread: QThread | None = None
        self.batch_worker: BatchWorker | None = None
        self.batch_video_path: Path | None = None
        self.batch_deer_code: str = self.runtime_state.deer_code
        self.batch_started_at: datetime | None = None
        self.batch_failed: bool = False
        self.batch_cancelled: bool = False
        self.capacity_benchmark_process: subprocess.Popen | None = None
        self.preloaded_model: object | None = None
        self.preloaded_model_path: Path | None = None
        self.model_loader_thread: threading.Thread | None = None
        self.model_loader_result: tuple[Path, object | None, str | None] | None = None
        self.model_loader_lock = threading.Lock()
        self.model_loader_timer = QTimer(self)
        self.model_loader_timer.setInterval(200)
        self.model_loader_timer.timeout.connect(self._poll_model_loader)
        self.last_frame_activity_at: float | None = None
        self.last_inference_activity_at: float | None = None
        self.detection_event_dispatcher = DetectionEventDispatcher()
        self.detection_event_timer = QTimer(self)
        self.detection_event_timer.setInterval(5)
        self.detection_event_timer.timeout.connect(self._drain_detection_events)
        self.detection_event_timer.start()
        self.inference_restart_count: int = 0
        self.inference_watchdog_timer = QTimer(self)
        self.inference_watchdog_timer.setInterval(1000)
        self.inference_watchdog_timer.timeout.connect(self._check_inference_health)
        self.inference_watchdog_timer.start()
        self.capture_health_timer = QTimer(self)
        self.capture_health_timer.setInterval(1000)
        self.capture_health_timer.timeout.connect(self._check_multi_capture_health)
        self.capture_health_timer.start()

        # FPS calculation state.
        self._capture_last_ts: float | None = None
        self._capture_fps: float = 0.0
        self._inference_last_ts: float | None = None
        self._inference_fps: float = 0.0
        self._first_overlay_seen: bool = False
        self._last_video_paint_at: float | None = None
        self._ui_frame_interval_seconds: float = 1.0 / max(float(config.preview_fps), 1.0)
        self.realtime_preview_timer = QTimer(self)
        self.realtime_preview_timer.setInterval(
            max(int(round(self._ui_frame_interval_seconds * 1000.0)), 1)
        )
        self.realtime_preview_timer.timeout.connect(self._paint_realtime_preview_frame)
        self.realtime_preview_timer.start()
        self.multi_preview_timer = QTimer(self)
        self.multi_preview_timer.setInterval(
            max(int(round(self._ui_frame_interval_seconds * 1000.0)), 1)
        )
        self.multi_preview_timer.timeout.connect(self._paint_multi_preview_frames)
        self.multi_preview_timer.start()

        # Detection session timing (for DB persistence on stop).
        self._session_started_at: datetime | None = None
        self._session_camera_source: Path | int | None = None
        self._session_segments_committed: int = 0
        self._session_task_id: int | None = None
        self._session_deer_id: int | None = None
        self._session_task_code: str | None = None
        self._last_result_flush_at: float | None = None

        # Database wiring.
        self.engine = build_engine(config.database_url)
        self.session_factory = build_session_factory(self.engine)
        self.database_error: str | None = None
        try:
            init_database(self.engine)
        except Exception as exc:  # noqa: BLE001
            self.database_error = str(exc)
        if self.database_error is None:
            self._recover_interrupted_realtime_tasks()

        self.setWindowTitle("MuskDeer Monitor | Intelligent Behavior Monitoring")
        self.resize(1400, 900)
        self.setMinimumSize(1200, 780)
        self.setStyleSheet(theme.APP_STYLESHEET)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.stack = QStackedWidget()
        self.pages = {
            "realtime": RealtimePage(),
            "multi": MultiCameraPage(),
            "batch": BatchPage(),
            "history": HistoryPage(),
        }
        for page in self.pages.values():
            self.stack.addWidget(page)

        self.sidebar.navigate.connect(self.show_page)
        self.sidebar.load_model_requested.connect(self.choose_model_path)
        self.sidebar.load_source_requested.connect(self.choose_video_source)
        self.sidebar.start_stop_requested.connect(self.toggle_detection)
        self.sidebar.thresholds_requested.connect(self.open_threshold_dialog)
        self.sidebar.profiles_requested.connect(self.open_deer_profile_dialog)
        self.pages["multi"].create_slots_requested.connect(self.create_multi_camera_slots)
        self.pages["multi"].stress_test_requested.connect(self.start_capacity_stress_test)
        self.pages["multi"].mount_source_requested.connect(self.mount_multi_camera_source)
        self.pages["multi"].analyze_requested.connect(self.start_multi_camera_analysis)
        self.pages["multi"].preview_requested.connect(self.open_multi_camera_preview)
        self.pages["history"].filter_changed.connect(self.refresh_history_page)
        self.pages["history"].batch_delete_requested.connect(self.delete_selected_history_records)
        self.pages["history"].table.view_dashboard_requested.connect(self.open_history_dashboard)
        self.pages["history"].table.open_raw_requested.connect(self.open_history_raw_records)
        self.pages["history"].table.open_alert_requested.connect(self.open_history_alert_records)
        self.pages["history"].table.delete_requested.connect(self.delete_history_record)
        self.pages["batch"].select_video_requested.connect(self.choose_batch_video)
        self.pages["batch"].select_deer_requested.connect(self.choose_batch_deer)
        self.pages["batch"].start_requested.connect(self.toggle_batch_analysis)
        layout.addWidget(self.sidebar)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(container)
        self.show_page("realtime")
        self._initialize_configured_model_path()

    # ------------------------------------------------------------------
    # Sidebar / page navigation
    # ------------------------------------------------------------------

    def _initialize_configured_model_path(self) -> None:
        model_path = self.config.model_path
        self.runtime_state.set_model_path(model_path)
        self.pages["batch"].set_model_path(model_path)
        if model_path.exists():
            engine_label = "TensorRT configured" if _is_tensorrt_model_path(model_path) else "Configured"
            self.sidebar.set_model_status(f"{model_path.name} ({engine_label})")
        else:
            self.sidebar.set_model_status(f"Missing {model_path.name}")

    def show_page(self, page: str) -> None:
        widget = self.pages[page]
        self.stack.setCurrentWidget(widget)
        self.sidebar.set_active(page)
        if page == "history":
            self.refresh_history_page()

    def start_capacity_stress_test(self) -> None:
        if (
            self.capacity_benchmark_process is not None
            and self.capacity_benchmark_process.poll() is None
        ):
            themed_dialogs.show_info(self, "Benchmark Running", "The 60/80/100 stream benchmark is already running in the background.")
            return
        video_path = self.config.benchmark_video_path
        if video_path is None or not Path(video_path).exists():
            themed_dialogs.show_warning(
                self,
                "Cannot Start Benchmark",
                "The bundled benchmark video was not found. Check benchmark_assets or set DEERUI_BENCHMARK_VIDEO_PATH.",
            )
            return
        benchmark_dir = self.config.output_dir / "benchmarks"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        log_path = benchmark_dir / f"capacity_stress_{datetime.now():%Y%m%d_%H%M%S}.log"
        command = self._capacity_benchmark_command(Path(video_path))
        log_file = log_path.open("w", encoding="utf-8")
        try:
            self.capacity_benchmark_process = subprocess.Popen(
                command,
                cwd=str(
                    Path(sys.executable).resolve().parent
                    if getattr(sys, "frozen", False)
                    else Path.cwd()
                ),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_file.close()
        except Exception as exc:  # noqa: BLE001
            log_file.close()
            themed_dialogs.show_warning(self, "Benchmark Start Failed", str(exc))
            return
        themed_dialogs.show_info(
            self,
            "Benchmark Started",
            f"The 60/80/100 stream benchmark is running in the background. Results will be written to:\n{benchmark_dir}\nLog: {log_path}",
        )

    def _capacity_benchmark_command(self, video_path: Path) -> list[str]:
        common_args = [
            "--capacity-benchmark",
            "--benchmark-streams",
            "60,80,100",
            "--benchmark-duration",
            "60",
            "--benchmark-video",
            str(video_path),
        ]
        if getattr(sys, "frozen", False):
            return [str(Path(sys.executable).resolve()), *common_args]
        return [sys.executable, "-m", "deerui.app", *common_args]

    def apply_model_path(self, model_path: Path) -> None:
        self.runtime_state.set_model_path(model_path)
        if not self._has_running_inference_workers():
            self._configure_inference_frame_stores(model_path)
        self.sidebar.set_model_status(model_path.name)
        self.pages["batch"].set_model_path(model_path)
        self._start_model_preload(model_path)

    def apply_video_source(self, source: Path | int) -> None:
        self.runtime_state.set_video_source(source)
        if isinstance(source, Path):
            self.sidebar.set_source_status(source.name)
        else:
            self.sidebar.set_source_status(f"Camera {source}")
        self.pages["realtime"].set_video_source(source)

    def apply_deer_code(self, deer_code: str) -> None:
        self.runtime_state.set_deer_code(deer_code)
        self.batch_deer_code = self.runtime_state.deer_code
        self.pages["realtime"].set_deer_code(self.runtime_state.deer_code)
        self.pages["batch"].set_deer_code(self.runtime_state.deer_code)

    def apply_thresholds(self, thresholds: ThresholdSettings) -> None:
        self.runtime_state.set_thresholds(thresholds)

    def _start_model_preload(self, model_path: Path) -> None:
        self.preloaded_model = None
        self.preloaded_model_path = None
        with self.model_loader_lock:
            self.model_loader_result = None

        if not model_path.exists():
            return
        if self.model_loader_thread is not None and self.model_loader_thread.is_alive():
            return

        self.sidebar.set_model_status(f"Loading: {model_path.name} - {self.inference_device}")

        def load() -> None:
            model = None
            error: str | None = None
            try:
                from ultralytics import YOLO  # type: ignore[import-not-found]

                model = YOLO(str(model_path))
                if is_pytorch_model_path(model_path):
                    try:
                        if self.inference_device != "cpu":
                            model.to(self.inference_device)
                        model.fuse()
                        model.eval()
                    except Exception:  # noqa: BLE001
                        _logger.info("YOLO fuse/eval skipped during preload", exc_info=True)
                self._warmup_preloaded_model(model)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
            with self.model_loader_lock:
                self.model_loader_result = (model_path, model, error)

        self.model_loader_thread = threading.Thread(
            target=load,
            name="deerui-yolo-loader",
            daemon=True,
        )
        self.model_loader_thread.start()
        self.model_loader_timer.start()

    def _poll_model_loader(self) -> None:
        with self.model_loader_lock:
            result = self.model_loader_result
            self.model_loader_result = None
        if result is None:
            return

        model_path, model, error = result
        if model_path != self.runtime_state.model_path:
            return
        self.model_loader_timer.stop()
        if error is not None or model is None:
            self.preloaded_model = None
            self.preloaded_model_path = None
            self.sidebar.set_model_status(f"Load failed: {model_path.name}")
            self.pages["realtime"].stats_panel.set_warning(f"Model Load Failed: {error}")
            return
        self.preloaded_model = model
        self.preloaded_model_path = model_path
        self.sidebar.set_model_status(f"{model_path.name} - {self.inference_device}")
        self.pages["realtime"].stats_panel.set_warning(None)

    def _warmup_preloaded_model(self, model: object) -> None:
        try:
            import numpy as np

            sample = np.zeros((320, 320, 3), dtype=np.uint8)
            model(sample, verbose=False, device=self.inference_device)
        except Exception:  # noqa: BLE001
            _logger.info("YOLO warmup skipped", exc_info=True)

    # ------------------------------------------------------------------
    # Detection lifecycle
    # ------------------------------------------------------------------

    def toggle_detection(self) -> None:
        if not self.runtime_state.model_path:
            themed_dialogs.show_warning(self, "Cannot Start Detection", "Please load the AI detection model first.")
            return
        if self.runtime_state.video_source is None:
            themed_dialogs.show_warning(self, "Cannot Start Detection", "Please connect a video or camera first.")
            return

        if self.is_detecting:
            self.stop_detection()
        else:
            self.start_detection()

    def start_detection(self) -> None:
        """Start the camera + inference pipeline.

        Re-entrant guard: clicking "Start Detection" while a run is still tearing down
        (e.g. immediately after stop) would leak QThreads, so we no-op.
        """
        if self.camera_thread is not None and self.camera_thread.isRunning():
            return
        source = self.runtime_state.video_source
        if source is None:
            return

        # Reset state.
        self._capture_last_ts = None
        self._capture_fps = 0.0
        self._inference_last_ts = None
        self._inference_fps = 0.0
        self._first_overlay_seen = False
        self._last_video_paint_at = None
        self._last_realtime_detection = None
        self._realtime_last_preview_seq = 0
        self._remove_stream_from_inference(self.realtime_stream_id)
        self._session_started_at = datetime.now(timezone.utc)
        self._session_camera_source = source
        self._session_segments_committed = 0
        self._session_task_id = None
        self._session_deer_id = None
        self._session_task_code = None
        self._last_result_flush_at = time.perf_counter()
        self.detection_session = DetectionSession(
            camera_id=str(source) if isinstance(source, Path) else f"camera:{source}",
            deer_code=self.runtime_state.deer_code,
        )

        if not self._ensure_shared_inference_worker():
            self.detection_session = None
            self._session_started_at = None
            return

        worker_source: str | int = str(source) if isinstance(source, Path) else source

        # ---- Camera thread ----
        self.camera_thread = QThread(self)
        self.camera_worker = CameraWorker(
            worker_source,
            target_fps=self.config.stream_target_fps,
            reconnect_delay_seconds=self.config.camera_reconnect_delay_seconds,
            max_read_failures=self.config.camera_max_read_failures,
            open_timeout_milliseconds=self.config.camera_open_timeout_milliseconds,
            read_timeout_milliseconds=self.config.camera_read_timeout_milliseconds,
            max_buffer_drain_grabs_per_frame=(
                self.config.camera_max_buffer_drain_grabs_per_frame
            ),
        )
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.frame_ready.connect(
            self.submit_realtime_frame,
            Qt.ConnectionType.DirectConnection,
        )
        self.camera_worker.status.connect(self.handle_camera_status)
        self.camera_worker.error.connect(self.handle_camera_error)
        self.camera_worker.finished.connect(self.camera_thread.quit)
        self.camera_worker.finished.connect(self.camera_worker.deleteLater)
        self.camera_thread.finished.connect(self.handle_camera_finished)
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)

        self.is_detecting = True
        self.sidebar.set_detection_running(True)
        stats_panel = self.pages["realtime"].stats_panel
        stats_panel.set_action("Detecting")
        stats_panel.set_warning(None)
        stats_panel.set_stats_summary("Realtime sampling - waiting for frames...")

        self.camera_thread.start()

    def _inference_worker_index(self, stream_id: str) -> int:
        store_count = max(len(self.frame_stores), 1)
        if store_count == 1:
            return 0
        return sum(ord(char) for char in stream_id) % store_count

    def _effective_inference_worker_count(self, model_path: Path | None = None) -> int:
        requested = max(int(self.config.inference_worker_count), 1)
        if _is_tensorrt_model_path(model_path):
            return min(requested, 2)
        return requested

    def _inference_frame_store_count(self, model_path: Path | None = None) -> int:
        if _is_tensorrt_model_path(model_path):
            return 1
        return self._effective_inference_worker_count(model_path)

    def _configure_inference_frame_stores(self, model_path: Path | None = None) -> None:
        worker_count = self._effective_inference_worker_count(model_path)
        store_count = self._inference_frame_store_count(model_path)
        if worker_count == self.inference_worker_count and store_count == len(self.frame_stores):
            self.inference_worker_count = worker_count
            return
        self.inference_worker_count = worker_count
        self.frame_stores = [LatestFrameStore(self) for _ in range(store_count)]
        self.inference_batch_schedulers = self._build_inference_batch_schedulers(
            self.frame_stores
        )
        self.inference_batch_dispatchers = []
        self.frame_store = InferenceFrameStoreRouter(self.frame_stores, self._inference_worker_index)

    def _build_inference_batch_schedulers(
        self,
        frame_stores: list[LatestFrameStore],
    ) -> list[SharedBatchScheduler]:
        return [
            SharedBatchScheduler(
                frame_store,
                batch_size=self.config.inference_batch_size,
                max_wait_ms=self.config.inference_batch_wait_ms,
                adaptive_wait=self.config.adaptive_batch_wait,
            )
            for frame_store in frame_stores
        ]

    def _build_inference_batch_dispatchers(
        self,
        frame_stores: list[LatestFrameStore],
    ) -> list[CentralBatchDispatcher]:
        queue_size = max(self.inference_worker_count, 1)
        return [
            CentralBatchDispatcher(
                frame_store,
                batch_size=self.config.inference_batch_size,
                queue_size=queue_size,
                sample_every_n_frames=self.config.sample_every_n_frames,
                adaptive_sampling=self.config.adaptive_sampling,
                adaptive_base_streams_per_worker=self.config.adaptive_base_streams_per_worker,
                adaptive_max_sample_interval=self.config.adaptive_max_sample_interval,
                batch_wait_ms=self.config.inference_batch_wait_ms,
                adaptive_batch_wait=self.config.adaptive_batch_wait,
                poll_interval_ms=2,
            )
            for frame_store in frame_stores
        ]

    def _submit_frame_to_inference(self, stream_id: str, frame, timestamp_seconds: float) -> None:
        self.frame_store.submit(stream_id, frame, timestamp_seconds)

    def _remove_stream_from_inference(self, stream_id: str) -> None:
        self.frame_store.remove_stream(stream_id)
        self.preview_frame_store.remove_stream(stream_id)

    def _has_running_inference_workers(self) -> bool:
        return any(thread.isRunning() for thread in self.inference_threads)

    def _ensure_shared_inference_worker(self) -> bool:
        if self._has_running_inference_workers():
            return True
        if self.runtime_state.model_path is None:
            themed_dialogs.show_warning(self, "Cannot Start Detection", "Please load the AI detection model first.")
            return False
        if self.preloaded_model_path != self.runtime_state.model_path or self.preloaded_model is None:
            if self.runtime_state.model_path.exists():
                self._start_model_preload(self.runtime_state.model_path)
                themed_dialogs.show_warning(
                    self,
                    "Model Still Loading",
                    "The AI model is still loading in the background. Start detection after loading finishes.",
                )
            else:
                themed_dialogs.show_warning(self, "Cannot Start Detection", "Model file does not exist.")
            return False

        self._configure_inference_frame_stores(self.runtime_state.model_path)
        self.inference_threads = []
        self.inference_workers = []
        self.inference_batch_dispatchers = self._build_inference_batch_dispatchers(
            self.frame_stores
        )
        worker_frame_stores = (
            [self.frame_stores[0] for _ in range(self.inference_worker_count)]
            if _is_tensorrt_model_path(self.runtime_state.model_path)
            else list(self.frame_stores)
        )
        worker_batch_schedulers = (
            [self.inference_batch_schedulers[0] for _ in range(self.inference_worker_count)]
            if _is_tensorrt_model_path(self.runtime_state.model_path)
            else list(self.inference_batch_schedulers)
        )
        worker_batch_dispatchers = (
            [self.inference_batch_dispatchers[0] for _ in range(self.inference_worker_count)]
            if _is_tensorrt_model_path(self.runtime_state.model_path)
            else list(self.inference_batch_dispatchers)
        )
        for batch_dispatcher in self.inference_batch_dispatchers:
            batch_dispatcher.start()
        for index, (frame_store, batch_scheduler, batch_dispatcher) in enumerate(
            zip(
                worker_frame_stores,
                worker_batch_schedulers,
                worker_batch_dispatchers,
                strict=True,
            )
        ):
            thread = QThread(self)
            worker = SharedInferenceWorker(
                model_path=str(self.runtime_state.model_path),
                behavior_classes=self.config.behavior_classes,
                confidence_threshold=self.config.confidence_threshold,
                frame_store=frame_store,
                sample_every_n_frames=self.config.sample_every_n_frames,
                inference_device=self.inference_device,
                batch_size=self.config.inference_batch_size,
                overlay_interval_seconds=self._ui_frame_interval_seconds,
                inference_imgsz=self.config.inference_imgsz,
                inference_half=self.config.inference_half,
                adaptive_sampling=self.config.adaptive_sampling,
                adaptive_base_streams_per_worker=self.config.adaptive_base_streams_per_worker,
                adaptive_max_sample_interval=self.config.adaptive_max_sample_interval,
                preloaded_model=self.preloaded_model if index == 0 else None,
                batch_scheduler=batch_scheduler,
                batch_max_wait_ms=self.config.inference_batch_wait_ms,
                batch_dispatcher=batch_dispatcher,
            )
            worker.moveToThread(thread)
            thread.started.connect(worker.start)
            worker.model_loaded.connect(self.handle_model_loaded)
            worker.warning.connect(self.handle_inference_warning)
            worker.error.connect(self.handle_inference_error)
            worker.frame_with_overlay.connect(self.enqueue_shared_inference_frame)
            worker.detection_ready.connect(self.enqueue_shared_detection)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(self.handle_inference_finished)
            thread.finished.connect(thread.deleteLater)
            self.inference_threads.append(thread)
            self.inference_workers.append(worker)
            thread.start()
        self.inference_thread = self.inference_threads[0] if self.inference_threads else None
        self.inference_worker = self.inference_workers[0] if self.inference_workers else None
        self.last_inference_activity_at = time.perf_counter()
        return True

    def active_detection_thread_count(self) -> int:
        camera_count = 1 if self.camera_thread is not None and self.camera_thread.isRunning() else 0
        return camera_count + sum(1 for thread in self.inference_threads if thread.isRunning())

    def _stop_shared_inference_pool(self, wait_ms: int = 1500) -> None:
        workers = list(self.inference_workers)
        threads = list(self.inference_threads)
        if not workers and self.inference_worker is not None:
            workers = [self.inference_worker]
        if not threads and self.inference_thread is not None:
            threads = [self.inference_thread]

        for batch_dispatcher in self.inference_batch_dispatchers:
            try:
                batch_dispatcher.stop()
            except Exception:  # noqa: BLE001
                _logger.exception("Stopping batch dispatcher failed")
        for worker in workers:
            try:
                worker.stop()
            except Exception:  # noqa: BLE001
                _logger.exception("Stopping inference worker failed")
        for thread in threads:
            if thread is not None and thread.isRunning():
                thread.quit()
                if not thread.wait(wait_ms):
                    thread.terminate()
                    thread.wait(500)
        self.inference_workers = []
        self.inference_threads = []
        self.inference_worker = None
        self.inference_thread = None
        while self.detection_event_dispatcher.record_queue_size:
            self._drain_detection_events(
                record_limit=max(256, self.detection_event_dispatcher.record_queue_size),
                preview_limit=0,
            )
        self.detection_event_dispatcher.clear()
        for frame_store in self.frame_stores:
            frame_store.clear()
        for batch_scheduler in self.inference_batch_schedulers:
            batch_scheduler.clear()
        for batch_dispatcher in self.inference_batch_dispatchers:
            batch_dispatcher.clear()
        self.inference_batch_dispatchers = []
        self.preview_frame_store.clear()

    def stop_detection(self) -> None:
        if not self.is_detecting and self.camera_worker is None and self.inference_worker is None:
            return

        camera_thread = self.camera_thread
        self.is_detecting = False
        self.sidebar.set_detection_running(False)
        stats_panel = self.pages["realtime"].stats_panel
        stats_panel.set_action(None)

        if self.camera_worker is not None:
            self.camera_worker.stop()
        if not self._has_running_multi_cameras():
            self._stop_shared_inference_pool()
        self._remove_stream_from_inference(self.realtime_stream_id)

        # Wait briefly for threads to drain. We deliberately don't block the UI
        # thread for long; fall through and let _finished handlers do final cleanup.
        threads_to_wait = [camera_thread]
        for thread in threads_to_wait:
            if thread is not None and thread.isRunning():
                thread.quit()
                if not thread.wait(1500):
                    thread.terminate()
                    thread.wait(1000)

        # Flush whatever we accumulated to the DB now (after threads are quiesced).
        self._flush_session_to_db()

    def _cleanup_threads(self) -> None:
        """Best-effort teardown used by closeEvent. Idempotent."""
        camera_thread = self.camera_thread
        self.capture_health_timer.stop()
        self.stop_batch_analysis()
        self.stop_all_multi_camera_analysis()
        if self.camera_worker is not None:
            try:
                self.camera_worker.stop()
            except Exception:  # noqa: BLE001
                pass
        self._stop_shared_inference_pool()
        for thread in (camera_thread,):
            if thread is not None and thread.isRunning():
                thread.quit()
                if not thread.wait(1500):
                    thread.terminate()
                    thread.wait(1000)

    def _has_running_multi_cameras(self) -> bool:
        return any(
            runtime.camera_thread is not None and runtime.camera_thread.isRunning()
            for runtime in self.multi_camera_runtimes.values()
        )

    def _active_inference_stream_count(self) -> int:
        count = 0
        if self.is_detecting or self.camera_worker is not None:
            count += 1
        for runtime in self.multi_camera_runtimes.values():
            if runtime.camera_worker is not None:
                count += 1
                continue
            thread = runtime.camera_thread
            if thread is not None and thread.isRunning():
                count += 1
        return count

    def _stream_capacity_policy(self) -> StreamCapacityPolicy:
        return StreamCapacityPolicy(
            worker_count=max(int(self.inference_worker_count), 1),
            reliable_streams_per_worker=max(int(self.config.worker_stream_capacity), 1),
        )

    def _check_stream_capacity_before_start(self, requested_stream_count: int = 1) -> bool:
        decision = evaluate_capacity_gate(
            active_stream_count=self._active_inference_stream_count(),
            requested_stream_count=requested_stream_count,
            policy=self._stream_capacity_policy(),
        )
        if decision.allowed:
            return True
        themed_dialogs.show_warning(
            self,
            "Insufficient Reliable Capacity",
            (
                f"{decision.reason_zh}.\n"
                "To preserve behavior-recording accuracy for each enclosure, reduce the number of simultaneous streams, lower the input FPS, "
                "or switch to a higher-capacity hardware profile and rerun the benchmark."
            ),
        )
        return False

    def _check_inference_health(self) -> None:
        has_active_stream = self.is_detecting or self._has_running_multi_cameras()
        if not has_active_stream or not self._has_running_inference_workers():
            return
        now = time.perf_counter()
        if self.last_frame_activity_at is None:
            return
        if now - self.last_frame_activity_at > self.config.stream_stale_seconds:
            if self.is_detecting:
                self.pages["realtime"].stats_panel.set_warning("The video stream has not produced frames for a long time")
            return
        if self.last_inference_activity_at is None:
            self.last_inference_activity_at = now
            return
        if now - self.last_inference_activity_at <= self.config.inference_watchdog_seconds:
            return
        self.restart_shared_inference_worker("Inference heartbeat timed out; worker restarted automatically")

    def _capture_restart_wait_milliseconds(self) -> int:
        return min(max(int(self.config.camera_read_timeout_milliseconds), 500), 1500)

    def _capture_restart_message(self, snapshot) -> str:
        restart_count = int(getattr(snapshot, "restart_count", 0))
        status = getattr(snapshot, "status", "unknown")
        message = f"Capture restarted #{restart_count}\nstatus={status}"
        last_message = (
            getattr(snapshot, "last_error_message", None)
            or getattr(snapshot, "last_status_message", None)
        )
        if last_message:
            message += f"\nlast={last_message}"
        return message

    def _report_multi_capture_restart(self, slot_id: str) -> None:
        snapshot = self.capture_supervisor.snapshot(slot_id)
        if snapshot is None:
            return
        runtime = self.multi_camera_runtimes.get(slot_id)
        if runtime is not None and hasattr(snapshot, "target_fps"):
            runtime.capture_target_fps = float(snapshot.target_fps)
        self.pages["multi"].set_slot_stats(slot_id, self._capture_restart_message(snapshot))

    def _check_multi_capture_health(self) -> None:
        restart_failed = getattr(self.capture_supervisor, "restart_failed_routes", None)
        restart_stale = getattr(self.capture_supervisor, "restart_stale_routes", None)
        if not callable(restart_failed) or not callable(restart_stale):
            return
        wait_ms = self._capture_restart_wait_milliseconds()
        restarted: list[str] = []
        restarted.extend(
            restart_failed(
                wait_milliseconds=wait_ms,
                max_routes=1,
            )
        )
        restarted.extend(
            restart_stale(
                stale_after_seconds=max(float(self.config.stream_stale_seconds), 1.0),
                wait_milliseconds=wait_ms,
                max_routes=1,
            )
        )
        for slot_id in dict.fromkeys(restarted):
            self._report_multi_capture_restart(slot_id)

    def restart_shared_inference_worker(self, reason: str) -> None:
        if self.preloaded_model is None or self.runtime_state.model_path is None:
            return
        self.inference_restart_count += 1
        _logger.warning("%s (restart_count=%s)", reason, self.inference_restart_count)
        if self.is_detecting:
            self.pages["realtime"].stats_panel.set_warning(f"{reason} #{self.inference_restart_count}")

        self._stop_shared_inference_pool(wait_ms=800)
        self.last_inference_activity_at = time.perf_counter()
        self._ensure_shared_inference_worker()

    # ------------------------------------------------------------------
    # Camera / inference slots
    # ------------------------------------------------------------------

    def submit_realtime_frame(self, frame, timestamp_seconds: float) -> None:
        self.preview_frame_store.submit(self.realtime_stream_id, frame, timestamp_seconds)
        self._submit_frame_to_inference(self.realtime_stream_id, frame, timestamp_seconds)
        self.last_frame_activity_at = time.perf_counter()

    def _paint_realtime_preview_frame(self) -> None:
        if not self.is_detecting and self.camera_worker is None:
            return
        frame, timestamp_seconds, seq = self.preview_frame_store.consume(
            self.realtime_stream_id
        )
        if frame is None or seq == 0 or seq == self._realtime_last_preview_seq:
            return
        self._realtime_last_preview_seq = seq
        display_frame = frame
        if self._first_overlay_seen:
            display_frame = self._draw_detection_overlay_on_bgr(
                frame,
                self._last_realtime_detection,
            )
        if not self._paint_video_frame(display_frame):
            return

        video_panel = self.pages["realtime"].video_panel
        if self._capture_last_ts is not None and timestamp_seconds is not None:
            elapsed = timestamp_seconds - self._capture_last_ts
            if elapsed > 0:
                self._capture_fps = 0.9 * self._capture_fps + 0.1 * (1.0 / elapsed)
                video_panel.set_fps(self._capture_fps)
        if timestamp_seconds is not None:
            self._capture_last_ts = timestamp_seconds
        self._update_stats_summary()

    def handle_camera_frame(self, frame, timestamp_seconds: float) -> None:
        """Cold-start / passthrough path.

        Paints the bare BGR frame so the user sees something immediately, even
        before `InferenceWorker` produces its first overlay. Once detections arrive,
        capture frames keep the preview moving while the latest result supplies
        the overlay.
        """
        try:
            video_panel = self.pages["realtime"].video_panel
            display_frame = frame
            if self._first_overlay_seen:
                display_frame = self._draw_detection_overlay_on_bgr(
                    frame,
                    self._last_realtime_detection,
                )
            self._paint_video_frame(display_frame)
            if self._capture_last_ts is not None:
                elapsed = timestamp_seconds - self._capture_last_ts
                if elapsed > 0:
                    self._capture_fps = 0.9 * self._capture_fps + 0.1 * (1.0 / elapsed)
                    video_panel.set_fps(self._capture_fps)
            self._capture_last_ts = timestamp_seconds
            self._update_stats_summary()
        except Exception:  # noqa: BLE001
            _logger.exception("handle_camera_frame raised")

    def handle_camera_status(self, message: str) -> None:
        _logger.info("Camera status: %s", message)
        self.pages["realtime"].stats_panel.set_warning(message)
        self._update_stats_summary()

    def handle_camera_error(self, message: str) -> None:
        _logger.warning("Camera error: %s", message)
        themed_dialogs.show_warning(self, "Video Source Error", message)
        self.stop_detection()

    def handle_camera_finished(self) -> None:
        self.camera_thread = None
        self.camera_worker = None
        if self.is_detecting:
            # Camera died unexpectedly (file ended, device unplugged); flag it
            # but don't tear down the rest of the state machine automatically.
            self.pages["realtime"].stats_panel.set_warning("Video stream interrupted")
            self._update_stats_summary()
            self.stop_detection()

    def handle_inference_frame(self, frame_rgb, detection) -> None:
        try:
            first_overlay = not self._first_overlay_seen
            video_panel = self.pages["realtime"].video_panel
            if frame_rgb is not None:
                if self._paint_video_frame(frame_rgb, is_rgb=True, force=first_overlay):
                    self._first_overlay_seen = True
                frame_rgb = None
            if frame_rgb is not None:
                video_panel.set_frame(frame_rgb)
            if self._inference_last_ts is not None and detection.timestamp_seconds:
                elapsed = detection.timestamp_seconds - self._inference_last_ts
                if elapsed > 0:
                    self._inference_fps = 0.9 * self._inference_fps + 0.1 * (1.0 / elapsed)
            self._inference_last_ts = detection.timestamp_seconds
            self._update_stats_summary()
        except Exception:  # noqa: BLE001
            _logger.exception("handle_inference_frame raised")

    def handle_realtime_inference_preview(self, detection: DetectionResult) -> None:
        try:
            self._last_realtime_detection = detection
            self._first_overlay_seen = True
            if self._inference_last_ts is not None and detection.timestamp_seconds:
                elapsed = detection.timestamp_seconds - self._inference_last_ts
                if elapsed > 0:
                    self._inference_fps = 0.9 * self._inference_fps + 0.1 * (1.0 / elapsed)
            self._inference_last_ts = detection.timestamp_seconds
            self._update_stats_summary()
        except Exception:  # noqa: BLE001
            _logger.exception("handle_realtime_inference_preview raised")

    def handle_shared_inference_frame(self, stream_id: str, frame_rgb, detection) -> None:
        self.last_inference_activity_at = time.perf_counter()
        if stream_id == self.realtime_stream_id:
            self.handle_realtime_inference_preview(detection)
            return
        slot_id = self._slot_id_from_stream_id(stream_id)
        if slot_id is not None:
            self.handle_multi_inference_frame(slot_id, frame_rgb, detection)

    def handle_shared_detection(self, stream_id: str, detection: DetectionResult) -> None:
        self.last_inference_activity_at = time.perf_counter()
        if stream_id == self.realtime_stream_id:
            self.handle_detection(detection)
            return
        slot_id = self._slot_id_from_stream_id(stream_id)
        if slot_id is not None:
            self.handle_multi_detection(slot_id, detection)

    def enqueue_shared_inference_frame(
        self,
        stream_id: str,
        frame_rgb,
        detection: DetectionResult,
    ) -> None:
        self.last_inference_activity_at = time.perf_counter()
        self.detection_event_dispatcher.submit_preview(stream_id, frame_rgb, detection)

    def enqueue_shared_detection(self, stream_id: str, detection: DetectionResult) -> None:
        self.last_inference_activity_at = time.perf_counter()
        self.detection_event_dispatcher.submit_record(stream_id, detection)

    def _drain_detection_events(
        self,
        *,
        record_limit: int = 256,
        preview_limit: int = 64,
    ) -> None:
        for event in self.detection_event_dispatcher.drain_records(limit=record_limit):
            self.handle_shared_detection(event.stream_id, event.detection)
        for event in self.detection_event_dispatcher.drain_previews(limit=preview_limit):
            self.handle_shared_inference_frame(event.stream_id, event.frame_rgb, event.detection)

    def handle_detection(self, detection: DetectionResult) -> None:
        """Update the state machine and the current-action label."""
        try:
            self._last_realtime_detection = detection
            if self.detection_session is not None:
                self.detection_session.accept_detection(detection)
                self._flush_session_segments_to_db()
            stats_panel = self.pages["realtime"].stats_panel
            stats_panel.set_action(detection.display_zh or detection.display_name)
            self._update_stats_summary()
        except Exception:  # noqa: BLE001
            _logger.exception("handle_detection raised")

    def submit_multi_frame(self, slot_id: str, frame, timestamp_seconds: float) -> None:
        stream_id = self._multi_stream_id(slot_id)
        self.preview_frame_store.submit(stream_id, frame, timestamp_seconds)
        self.last_frame_activity_at = time.perf_counter()
        runtime = self.multi_camera_runtimes.get(slot_id)
        if runtime is None:
            return
        if self._should_submit_multi_frame_to_inference(runtime, timestamp_seconds):
            self._submit_frame_to_inference(stream_id, frame, timestamp_seconds)
        if runtime.last_frame_ts is not None:
            elapsed = timestamp_seconds - runtime.last_frame_ts
            if elapsed > 0:
                runtime.fps = 0.9 * runtime.fps + 0.1 * (1.0 / elapsed)
        runtime.last_frame_ts = timestamp_seconds

    def _should_submit_multi_frame_to_inference(
        self,
        runtime: MultiCameraRuntime,
        timestamp_seconds: float,
    ) -> bool:
        target_fps = max(float(self.config.stream_target_fps), 1.0)
        interval_seconds = 1.0 / target_fps
        last_submitted_at = runtime.last_inference_submit_ts
        if (
            last_submitted_at is not None
            and timestamp_seconds - last_submitted_at < interval_seconds * 0.95
        ):
            return False
        runtime.last_inference_submit_ts = timestamp_seconds
        return True

    def handle_multi_camera_frame(self, slot_id: str, frame, timestamp_seconds: float) -> None:
        runtime = self.multi_camera_runtimes.get(slot_id)
        if runtime is None:
            return
        try:
            if not runtime.first_overlay_seen:
                self._paint_multi_frame(slot_id, frame, runtime=runtime)
            if runtime.last_frame_ts is not None:
                elapsed = timestamp_seconds - runtime.last_frame_ts
                if elapsed > 0:
                    runtime.fps = 0.9 * runtime.fps + 0.1 * (1.0 / elapsed)
            runtime.last_frame_ts = timestamp_seconds
            self._update_multi_stats(slot_id)
        except Exception:  # noqa: BLE001
            _logger.exception("handle_multi_camera_frame raised for %s", slot_id)

    def handle_multi_inference_frame(
        self,
        slot_id: str,
        frame_rgb,
        detection: DetectionResult,
    ) -> None:
        runtime = self.multi_camera_runtimes.get(slot_id)
        if runtime is None:
            return
        if self._paint_multi_frame(
            slot_id,
            frame_rgb,
            runtime=runtime,
            is_rgb=True,
            force=not runtime.first_overlay_seen,
        ):
            runtime.first_overlay_seen = True

    def _paint_multi_preview_frames(self) -> None:
        for slot_id, runtime in list(self.multi_camera_runtimes.items()):
            self._apply_multi_camera_capture_fps(slot_id, runtime)
            dialog_open = self.pages["multi"].is_slot_preview_dialog_open(slot_id)
            if runtime.preview_suspended_for_quality and not dialog_open:
                continue
            if (
                not self._is_multi_preview_enabled(slot_id)
                and not dialog_open
            ):
                continue
            frame, _timestamp, seq = self.preview_frame_store.consume(
                self._multi_stream_id(slot_id)
            )
            if frame is None or seq == 0 or seq == runtime.last_preview_seq:
                continue
            runtime.last_preview_seq = seq
            try:
                display_frame = self._draw_detection_overlay_on_bgr(frame, runtime.last_detection)
                self._paint_multi_frame(slot_id, display_frame, runtime=runtime, force=True)
            except Exception:  # noqa: BLE001
                _logger.exception("multi preview paint failed for %s", slot_id)

    def _draw_detection_overlay_on_bgr(
        self,
        frame,
        detection: DetectionResult | None,
    ):
        if frame is None or detection is None or detection.bbox_xyxy is None:
            return frame
        try:
            return draw_detection_overlay_bgr(
                frame,
                detection,
                color_bgr=theme.COLOR_PRIMARY_BGR,
                text_size=16,
            )
        except Exception:  # noqa: BLE001
            _logger.exception("drawing multi preview overlay failed")
            return frame

    def handle_multi_detection(self, slot_id: str, detection: DetectionResult) -> None:
        runtime = self.multi_camera_runtimes.get(slot_id)
        if runtime is None:
            return
        if runtime.last_inference_ts is not None and detection.timestamp_seconds:
            elapsed = detection.timestamp_seconds - runtime.last_inference_ts
            if elapsed > 0:
                runtime.inference_fps = 0.9 * runtime.inference_fps + 0.1 * (1.0 / elapsed)
        runtime.last_inference_ts = detection.timestamp_seconds
        runtime.last_detection = detection
        try:
            if runtime.detection_session is not None:
                runtime.detection_session.accept_detection(detection)
                self._flush_multi_segments_to_db(slot_id)
            self.pages["multi"].set_slot_action(slot_id, detection.display_zh or detection.display_name)
            self._update_multi_stats(slot_id)
        except Exception:  # noqa: BLE001
            _logger.exception("handle_multi_detection raised for %s", slot_id)

    def handle_multi_camera_status(self, slot_id: str, message: str) -> None:
        _logger.info("Multi camera %s status: %s", slot_id, message)
        self.pages["multi"].set_slot_stats(slot_id, message)

    def handle_multi_camera_error(self, slot_id: str, message: str) -> None:
        _logger.warning("Multi camera %s error: %s", slot_id, message)
        snapshot = self.capture_supervisor.snapshot(slot_id)
        if snapshot is not None and snapshot.status not in {"stopped", "finished"}:
            self.pages["multi"].set_slot_stats(
                slot_id,
                f"Capture error: {message}\nwaiting for automatic restart",
            )
            return
        self.pages["multi"].set_slot_stats(slot_id, f"Video stream error: {message}")
        self.stop_multi_camera_analysis(slot_id)

    def handle_multi_capture_restarted(self, slot_id: str, handle) -> None:
        runtime = self.multi_camera_runtimes.get(slot_id)
        if runtime is None:
            return
        runtime.camera_thread = handle.thread
        runtime.camera_worker = handle.worker
        self.pages["multi"].set_slot_running(slot_id, True)
        self._report_multi_capture_restart(slot_id)

    def handle_multi_camera_finished(self, slot_id: str) -> None:
        runtime = self.multi_camera_runtimes.get(slot_id)
        if runtime is None:
            return
        runtime.camera_thread = None
        runtime.camera_worker = None
        self.pages["multi"].set_slot_running(slot_id, False)
        self._remove_stream_from_inference(self._multi_stream_id(slot_id))
        self._flush_multi_session_to_db(slot_id)

    def _paint_multi_frame(
        self,
        slot_id: str,
        frame,
        *,
        runtime: MultiCameraRuntime,
        is_rgb: bool = False,
        force: bool = False,
    ) -> bool:
        preview_enabled = self._is_multi_preview_enabled(slot_id)
        dialog_open = self.pages["multi"].is_slot_preview_dialog_open(slot_id)
        if runtime.preview_suspended_for_quality and not dialog_open:
            return False
        if not preview_enabled and not dialog_open:
            return False
        if frame is None:
            return False
        now = time.perf_counter()
        if (
            not force
            and runtime.last_paint_at is not None
            and now - runtime.last_paint_at < self._ui_frame_interval_seconds
        ):
            return False
        if not preview_enabled and dialog_open:
            if is_rgb:
                self.pages["multi"].set_slot_dialog_frame_rgb(slot_id, frame)
            else:
                self.pages["multi"].set_slot_dialog_frame_bgr(slot_id, frame)
        elif is_rgb:
            self.pages["multi"].set_slot_frame_rgb(slot_id, frame)
        else:
            self.pages["multi"].set_slot_frame_bgr(slot_id, frame)
        runtime.last_paint_at = now
        return True

    def _update_multi_stats(self, slot_id: str) -> None:
        runtime = self.multi_camera_runtimes.get(slot_id)
        if runtime is None:
            return
        segment_count = (
            runtime.detection_session.segment_count
            if runtime.detection_session is not None
            else 0
        )
        snapshot = self._multi_stream_quality_snapshot(slot_id, runtime)
        quality = evaluate_stream_quality(
            snapshot,
            self._multi_stream_quality_thresholds(),
        )
        self._apply_record_first_preview_guard(slot_id, runtime, quality.status)
        self._record_multi_quality_event(slot_id, quality, snapshot)
        self.pages["multi"].set_slot_stats(
            slot_id,
            (
                f"{quality.label_zh} - {quality.reason_zh}\n"
                f"Capture FPS {runtime.fps:.1f}\n"
                f"Inference FPS {runtime.inference_fps:.1f}\n"
                f"Detected {segment_count} segments"
            ),
        )
        self.pages["multi"].set_slot_behavior_stats(
            slot_id,
            self._build_behavior_stats(runtime.detection_session, runtime.last_inference_ts),
        )

    def _apply_record_first_preview_guard(
        self,
        slot_id: str,
        runtime: MultiCameraRuntime,
        quality_status: StreamQualityStatus,
    ) -> None:
        if quality_status in {
            StreamQualityStatus.DEGRADED,
            StreamQualityStatus.UNRELIABLE,
        }:
            runtime.preview_suspended_for_quality = True
            if runtime.preview_enabled:
                self._set_multi_preview_enabled(slot_id, False)
            else:
                self._apply_multi_camera_capture_fps(slot_id, runtime)
            return
        if quality_status is StreamQualityStatus.HEALTHY and runtime.preview_suspended_for_quality:
            runtime.preview_suspended_for_quality = False
            self._apply_multi_camera_capture_fps(slot_id, runtime)

    def _record_multi_quality_event(
        self,
        slot_id: str,
        quality,
        snapshot: StreamQualitySnapshot,
    ) -> None:
        try:
            self.stream_quality_event_recorder.record_if_changed(
                stream_id=self._multi_stream_id(slot_id),
                stream_label=slot_id,
                quality=quality,
                snapshot=snapshot,
            )
        except Exception:  # noqa: BLE001
            _logger.exception("failed to record stream quality event for %s", slot_id)

    def _multi_stream_quality_snapshot(
        self,
        slot_id: str,
        runtime: MultiCameraRuntime,
    ) -> StreamQualitySnapshot:
        now = time.time()
        frame_stats = self.frame_store.stream_stats(self._multi_stream_id(slot_id))
        overwrite_drop_rate = self._recent_overwrite_drop_rate(runtime, frame_stats)
        return StreamQualitySnapshot(
            capture_fps=runtime.fps,
            inference_fps=runtime.inference_fps,
            seconds_since_last_frame=(
                None if runtime.last_frame_ts is None else max(now - runtime.last_frame_ts, 0.0)
            ),
            seconds_since_last_inference=(
                None
                if runtime.last_inference_ts is None
                else max(now - runtime.last_inference_ts, 0.0)
            ),
            overwrite_drop_rate=overwrite_drop_rate,
        )

    def _recent_overwrite_drop_rate(
        self,
        runtime: MultiCameraRuntime,
        frame_stats: LatestFrameStats,
    ) -> float:
        submitted_delta = (
            frame_stats.submitted_frame_count
            - runtime.quality_submitted_frame_count
        )
        overwritten_delta = (
            frame_stats.overwritten_frame_count
            - runtime.quality_overwritten_frame_count
        )
        runtime.quality_submitted_frame_count = frame_stats.submitted_frame_count
        runtime.quality_overwritten_frame_count = frame_stats.overwritten_frame_count
        if submitted_delta <= 0 or overwritten_delta < 0:
            return 0.0
        return min(max(overwritten_delta / submitted_delta, 0.0), 1.0)

    def _multi_stream_quality_thresholds(self) -> StreamQualityThresholds:
        target_capture_fps = max(float(self.config.stream_target_fps), 1.0)
        sample_interval = max(int(self.config.sample_every_n_frames), 1)
        target_inference_fps = max(target_capture_fps / sample_interval, 1.0)
        return StreamQualityThresholds(
            target_capture_fps=target_capture_fps,
            min_capture_fps=max(target_capture_fps * 0.4, 1.0),
            target_inference_fps=target_inference_fps,
            min_inference_fps=max(target_inference_fps * 0.6, 1.0),
            max_frame_age_seconds=max(float(self.config.stream_stale_seconds), 1.0),
            max_inference_age_seconds=max(float(self.config.inference_watchdog_seconds), 1.0),
        )

    def handle_model_loaded(self) -> None:
        _logger.info("InferenceWorker model loaded")
        # Quiet success; could become "Model ready" if desired.
        pass

    def handle_inference_warning(self, message: str) -> None:
        self.pages["realtime"].stats_panel.set_warning(message)

    def handle_inference_error(self, message: str) -> None:
        # Non-blocking: keep detection running so the camera keeps painting.
        _logger.warning("InferenceWorker error: %s", message)
        self.pages["realtime"].stats_panel.set_warning(f"Model/inference error: {message}")
        self._update_stats_summary()

    def handle_inference_finished(self) -> None:
        if self._has_running_inference_workers():
            return
        self.inference_threads = []
        self.inference_workers = []
        self.inference_thread = None
        self.inference_worker = None

    def _paint_video_frame(self, frame, *, is_rgb: bool = False, force: bool = False) -> bool:
        if frame is None:
            return False
        now = time.perf_counter()
        if (
            not force
            and self._last_video_paint_at is not None
            and now - self._last_video_paint_at < self._ui_frame_interval_seconds
        ):
            return False

        video_panel = self.pages["realtime"].video_panel
        if is_rgb:
            video_panel.set_rgb_frame(frame)
        else:
            video_panel.set_frame(frame)
        self._last_video_paint_at = now
        return True

    # ------------------------------------------------------------------
    # Persistence (stop_detection)
    # ------------------------------------------------------------------

    def _recover_interrupted_realtime_tasks(self) -> None:
        try:
            with session_scope(self.session_factory) as db_session:
                task_repository = AnalysisTaskRepository(db_session)
                recovered = task_repository.complete_interrupted_realtime_tasks(
                    fallback_ended_at=datetime.now(timezone.utc)
                )
                summary_repository = DailySummaryRepository(db_session)
                for task_id, deer_id in recovered:
                    task = db_session.get(AnalysisTaskRow, task_id)
                    if task is None:
                        continue
                    summary_repository.rebuild_for_task_date(task.started_at.date(), deer_id)
        except Exception as exc:  # noqa: BLE001
            self.database_error = str(exc)

    def _segment_flush_due(self, last_flush_at: float | None, *, force: bool = False) -> bool:
        if force:
            return True
        interval = float(self.config.result_flush_interval_seconds)
        if interval <= 0:
            return True
        if last_flush_at is None:
            return True
        return time.perf_counter() - last_flush_at >= interval

    def _ensure_realtime_task_row(self, db_session, session: DetectionSession) -> tuple[int, int]:
        if self._session_task_id is not None:
            deer_id = self._session_deer_id
            if deer_id is None:
                deer_id = DeerProfileRepository(db_session).get_or_create(session.deer_code).id
                self._session_deer_id = deer_id
            return self._session_task_id, deer_id

        deer_id = DeerProfileRepository(db_session).get_or_create(session.deer_code).id
        task_code = self._session_task_code or f"#JOB-{uuid.uuid4().hex[:8].upper()}"
        task = AnalysisTaskRepository(db_session).create(
            task_code=task_code,
            deer_profile_id=deer_id,
            mode=TaskMode.REALTIME,
            started_at=self._session_started_at or datetime.now(timezone.utc),
            camera_source_id=None,
        )
        self._session_task_id = task.id
        self._session_deer_id = deer_id
        self._session_task_code = task_code
        return task.id, deer_id

    def _flush_session_segments_to_db(self, *, force: bool = False) -> bool:
        session = self.detection_session
        if session is None or self._session_started_at is None:
            return True
        if self.database_error:
            return False

        pending_segments = list(session.segments[self._session_segments_committed :])
        current_segment = session.current_segment_snapshot(self._inference_last_ts)
        if not pending_segments and current_segment is None:
            return True
        if not self._segment_flush_due(self._last_result_flush_at, force=force):
            return True

        try:
            with session_scope(self.session_factory) as db_session:
                task_id, _deer_id = self._ensure_realtime_task_row(db_session, session)
                repository = SegmentRepository(db_session)
                repository.add_behavior_segments(task_id, pending_segments)
                if current_segment is not None:
                    repository.upsert_behavior_segment(task_id, current_segment)
            committed_count = self._session_segments_committed + len(pending_segments)
            session.prune_committed_segments(committed_count)
            self._session_segments_committed = 0
            self._last_result_flush_at = time.perf_counter()
            return True
        except Exception as exc:  # noqa: BLE001
            self.database_error = str(exc)
            self.pages["realtime"].stats_panel.set_warning(
                f"Warning: this session was not saved ({len(pending_segments)} segments): {exc}"
            )
            return False

    def _close_session_current_segment(
        self,
        session: DetectionSession,
        latest_timestamp_seconds: float | None,
    ) -> None:
        started_at = session.state_machine.action_started_at_seconds
        if session.state_machine.current_action_key is None or started_at is None:
            return
        if latest_timestamp_seconds is None:
            return
        end_ts = max(float(latest_timestamp_seconds), float(started_at))
        if end_ts - float(started_at) < 0.1:
            return
        session.stop(end_ts)

    def _flush_session_to_db(self) -> None:
        """Finish and persist the realtime detection task."""
        session = self.detection_session
        if session is None or self._session_started_at is None:
            return
        if self.database_error:
            self.pages["realtime"].stats_panel.set_warning(
                f"Warning: this session was not saved: database initialization failed ({self.database_error})"
            )
            return

        started_at = self._session_started_at
        ended_at = datetime.now(timezone.utc)
        self._close_session_current_segment(session, self._inference_last_ts)
        saved_segment_count = session.segment_count

        try:
            if not self._flush_session_segments_to_db(force=True):
                return
            with session_scope(self.session_factory) as db_session:
                task_id, deer_id = self._ensure_realtime_task_row(db_session, session)
                AnalysisTaskRepository(db_session).complete(
                    task_id=task_id,
                    ended_at=ended_at,
                    has_alert=False,
                    raw_excel_path=None,
                    alert_excel_path=None,
                )
                DailySummaryRepository(db_session).rebuild_for_task_date(
                    started_at.date(),
                    deer_id,
                )
            self._session_segments_committed = saved_segment_count
        except Exception as exc:  # noqa: BLE001
            self.database_error = str(exc)
            self.pages["realtime"].stats_panel.set_warning(
                f"Warning: this session was not saved ({saved_segment_count} segments): {exc}"
            )
        finally:
            self.detection_session = None
            self._session_started_at = None
            self._session_task_id = None
            self._session_deer_id = None
            self._session_task_code = None
            self._last_result_flush_at = None

    def _update_stats_summary(self) -> None:
        panel = self.pages["realtime"].stats_panel
        segment_count = self.detection_session.segment_count if self.detection_session else 0
        if self.is_detecting:
            text = (
                f"Realtime sampling - capture FPS {self._capture_fps:.1f} - "
                f"inference FPS {self._inference_fps:.1f} - detected {segment_count} segments"
            )
        elif self._session_segments_committed:
            text = f"Detection finished - saved {self._session_segments_committed} segments"
        elif segment_count:
            text = f"Detected {segment_count} segments"
        else:
            text = "No behavior data yet"
        panel.set_stats_summary(text)
        panel.set_behavior_stats(self._build_realtime_behavior_stats())

    def _build_realtime_behavior_stats(self) -> list[dict[str, object]]:
        return self._build_behavior_stats(self.detection_session, self._inference_last_ts)

    def _build_behavior_stats(
        self,
        session: DetectionSession | None,
        latest_timestamp_seconds: float | None,
    ) -> list[dict[str, object]]:
        stats_by_key: dict[str, dict[str, object]] = {}
        name_to_key: dict[str, str] = {}
        for behavior in self.config.behavior_classes.values():
            stats_by_key[behavior.action_key] = {
                "key": behavior.action_key,
                "name": behavior.display_zh,
                "color": theme.behavior_color(behavior.action_key, behavior.display_zh),
                "duration_seconds": 0.0,
                "count": 0,
                "active": False,
            }
            name_to_key[behavior.display_zh] = behavior.action_key

        if session is not None:
            for total in session.committed_segment_totals.values():
                key = total.action_key
                if key not in stats_by_key:
                    key = name_to_key.get(total.display_name, total.action_key)
                if key not in stats_by_key:
                    stats_by_key[key] = {
                        "key": key,
                        "name": total.display_name or key,
                        "color": theme.behavior_color(key, total.display_name),
                        "duration_seconds": 0.0,
                        "count": 0,
                        "active": False,
                    }
                stats_by_key[key]["duration_seconds"] = (
                    float(stats_by_key[key]["duration_seconds"]) + total.duration_seconds
                )
                stats_by_key[key]["count"] = int(stats_by_key[key]["count"]) + total.count

            for segment in session.segments:
                key = segment.action_key
                if key not in stats_by_key:
                    key = name_to_key.get(segment.display_name, segment.action_key)
                if key not in stats_by_key:
                    stats_by_key[key] = {
                        "key": key,
                        "name": segment.display_name or key,
                        "color": theme.behavior_color(key, segment.display_name),
                        "duration_seconds": 0.0,
                        "count": 0,
                        "active": False,
                    }
                stats_by_key[key]["duration_seconds"] = (
                    float(stats_by_key[key]["duration_seconds"]) + segment.duration_seconds
                )
                stats_by_key[key]["count"] = int(stats_by_key[key]["count"]) + 1

            current_key = session.state_machine.current_action_key
            current_started_at = session.state_machine.action_started_at_seconds
            if current_key is not None:
                if current_key not in stats_by_key:
                    stats_by_key[current_key] = {
                        "key": current_key,
                        "name": session.state_machine.current_display_name or current_key,
                        "color": theme.behavior_color(
                            current_key,
                            session.state_machine.current_display_name,
                        ),
                        "duration_seconds": 0.0,
                        "count": 0,
                        "active": False,
                    }
                stats_by_key[current_key]["active"] = True
                if current_started_at is not None and latest_timestamp_seconds is not None:
                    live_duration = max(0.0, latest_timestamp_seconds - current_started_at)
                    stats_by_key[current_key]["duration_seconds"] = (
                        float(stats_by_key[current_key]["duration_seconds"]) + live_duration
                    )

        total_duration = sum(float(item["duration_seconds"]) for item in stats_by_key.values())
        for item in stats_by_key.values():
            if total_duration > 0:
                item["percent"] = float(item["duration_seconds"]) / total_duration * 100
            else:
                item["percent"] = 0.0
        return list(stats_by_key.values())

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._cleanup_threads()
        self._flush_session_to_db()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

    def choose_model_path(self) -> None:
        path = themed_dialogs.open_file_name(
            self,
            "Select AI Weights",
            "YOLO / TensorRT Model (*.pt *.engine *.plan)",
        )
        if path:
            self.apply_model_path(Path(path))

    def _active_deer_codes(self) -> list[str]:
        if self.database_error:
            return []
        try:
            with session_scope(self.session_factory) as session:
                return ProfileService(session).list_active_deer_codes()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Failed to load deer profile choices: %s", exc)
            return []

    def _ask_deer_code(self, title: str, label: str, *, text: str = "") -> tuple[str, bool]:
        return themed_dialogs.ask_choice_text(
            self,
            title,
            label,
            choices=self._active_deer_codes(),
            text=text,
            placeholder="e.g. 2.1.6",
        )

    def choose_video_source(self) -> None:
        deer_code, ok = self._ask_deer_code(
            "Select Current Deer",
            "Deer ID for the current video",
            text=self.runtime_state.deer_code,
        )
        if not ok:
            return
        try:
            self.apply_deer_code(validate_deer_code(deer_code))
        except ValueError as exc:
            themed_dialogs.show_warning(self, "Invalid Deer ID Format", str(exc))
            return

        path = themed_dialogs.open_file_name(self, "Select Video Source", "Video (*.mp4 *.avi *.mkv)")
        if path:
            self.apply_video_source(Path(path))
            return
        camera_index, ok = themed_dialogs.ask_int(
            self,
            "Camera",
            "Camera index",
            value=0,
            minimum=0,
            maximum=64,
        )
        if ok:
            self.apply_video_source(camera_index)

    def open_threshold_dialog(self) -> None:
        dialog = ThresholdDialog(
            self.runtime_state.thresholds,
            action_names=[behavior.display_zh for behavior in self.config.behavior_classes.values()],
        )
        if dialog.exec():
            self.apply_thresholds(dialog.settings())
            themed_dialogs.show_info(self, "Success", "Alert thresholds were updated.")

    def choose_batch_video(self) -> None:
        path = themed_dialogs.open_file_name(self, "Select Batch Video", "Video (*.mp4 *.avi *.mkv)")
        if not path:
            return
        self.batch_video_path = Path(path)
        self.pages["batch"].set_video_source(self.batch_video_path)
        self.pages["batch"].set_result("Video selected; waiting to start batch analysis.")

    def choose_batch_deer(self) -> None:
        deer_code, ok = self._ask_deer_code(
            "Batch Identity Assignment",
            "Deer ID for the selected video",
            text=self.batch_deer_code,
        )
        if not ok:
            return
        try:
            self.batch_deer_code = validate_deer_code(deer_code)
        except ValueError as exc:
            themed_dialogs.show_warning(self, "Invalid Deer ID Format", str(exc))
            return
        self.pages["batch"].set_deer_code(self.batch_deer_code)

    def toggle_batch_analysis(self) -> None:
        if self.batch_thread is not None and self.batch_thread.isRunning():
            self.stop_batch_analysis()
            return
        if self.runtime_state.model_path is None:
            themed_dialogs.show_warning(self, "Cannot Start Batch", "Please load the AI detection model first.")
            return
        if self.preloaded_model_path != self.runtime_state.model_path or self.preloaded_model is None:
            if self.runtime_state.model_path.exists():
                self._start_model_preload(self.runtime_state.model_path)
                themed_dialogs.show_warning(
                    self,
                    "Model Still Loading",
                    "The AI model is still loading in the background. Start batch analysis after loading finishes.",
                )
            else:
                themed_dialogs.show_warning(self, "Cannot Start Batch", "Model file does not exist.")
            return
        if self.batch_video_path is None:
            themed_dialogs.show_warning(self, "Cannot Start Batch", "Please select a historical video first.")
            return
        try:
            deer_code = validate_deer_code(self.batch_deer_code)
        except ValueError as exc:
            themed_dialogs.show_warning(self, "Invalid Deer ID Format", str(exc))
            return

        self.batch_started_at = datetime.now(timezone.utc)
        self.batch_failed = False
        self.batch_cancelled = False
        self.batch_thread = QThread(self)
        self.batch_worker = BatchWorker(
            video_path=str(self.batch_video_path),
            model_path=str(self.runtime_state.model_path),
            deer_code=deer_code,
            behavior_classes=self.config.behavior_classes,
            confidence_threshold=self.config.confidence_threshold,
            sample_every_n_frames=self.config.sample_every_n_frames,
            inference_device=self.inference_device,
            inference_imgsz=self.config.inference_imgsz,
            inference_half=self.config.inference_half,
            batch_size=self.config.inference_batch_size,
            preloaded_model=self.preloaded_model,
        )
        self.batch_worker.moveToThread(self.batch_thread)
        self.batch_thread.started.connect(self.batch_worker.run)
        self.batch_worker.progress.connect(self.pages["batch"].set_progress)
        self.batch_worker.error.connect(self.handle_batch_error)
        self.batch_worker.finished.connect(self.handle_batch_finished)
        self.batch_worker.finished.connect(self.batch_thread.quit)
        self.batch_worker.finished.connect(self.batch_worker.deleteLater)
        self.batch_thread.finished.connect(self.handle_batch_thread_finished)
        self.batch_thread.finished.connect(self.batch_thread.deleteLater)
        self.pages["batch"].set_running(True)
        self.pages["batch"].set_progress(0)
        self.pages["batch"].set_result("Batch running - offline recognition in progress")
        self.batch_thread.start()

    def stop_batch_analysis(self) -> None:
        if self.batch_thread is not None and self.batch_thread.isRunning():
            self.batch_cancelled = True
        if self.batch_worker is not None:
            self.batch_worker.stop()
        if self.batch_thread is not None and self.batch_thread.isRunning():
            self.batch_thread.quit()
            if not self.batch_thread.wait(1500):
                self.batch_thread.terminate()
                self.batch_thread.wait(1000)
        self.pages["batch"].set_running(False)

    def handle_batch_error(self, message: str) -> None:
        self.batch_failed = True
        self.pages["batch"].set_result(f"Batch failed: {message}")

    def handle_batch_finished(self, segments: list) -> None:
        self.pages["batch"].set_running(False)
        if self.batch_failed:
            return
        if self.batch_cancelled:
            self.pages["batch"].set_result("Batch stopped; unfinished results were not saved.")
            return
        if self.batch_started_at is None or self.batch_video_path is None:
            self.pages["batch"].set_result("Batch finished, but task context is incomplete.")
            return

        ended_at = datetime.now(timezone.utc)
        thresholds = self.runtime_state.thresholds
        alerts = AnalysisService(
            AlertRuleEngine(
                min_duration_seconds=thresholds.min_duration_seconds,
                max_transitions_per_minute=thresholds.max_transitions_per_minute,
                action_thresholds=thresholds.action_thresholds,
            )
        ).analyze_alerts(segments)

        task_code = f"#JOB-{uuid.uuid4().hex[:8].upper()}"
        export_dir = self.config.output_dir / "exports" / task_code.replace("#", "")

        try:
            raw_path = ExportService().export_behavior_segments(
                segments,
                export_dir / "behavior_records.xlsx",
            )
            alert_path = None
            if alerts:
                alert_path = ExportService().export_alert_segments(
                    alerts,
                    export_dir / "alert_records.xlsx",
                )

            with session_scope(self.session_factory) as db_session:
                deer = DeerProfileRepository(db_session).get_or_create(validate_deer_code(self.batch_deer_code))
                task = AnalysisTaskRepository(db_session).create(
                    task_code=task_code,
                    deer_profile_id=deer.id,
                    mode=TaskMode.BATCH,
                    started_at=self.batch_started_at,
                    camera_source_id=None,
                )
                SegmentRepository(db_session).add_behavior_segments(task.id, segments)
                if alerts:
                    SegmentRepository(db_session).add_alert_segments(task.id, alerts)
                AnalysisTaskRepository(db_session).complete(
                    task_id=task.id,
                    ended_at=ended_at,
                    has_alert=bool(alerts),
                    raw_excel_path=str(raw_path),
                    alert_excel_path=str(alert_path) if alert_path else None,
                )
                DailySummaryRepository(db_session).rebuild_for_task_date(
                    self.batch_started_at.date(),
                    deer.id,
                )
            self.pages["batch"].set_result(
                f"Batch complete - detected {len(segments)} segments - {len(alerts)} alerts"
            )
        except Exception as exc:  # noqa: BLE001
            self.pages["batch"].set_result(f"Failed to save batch results: {exc}")

    def handle_batch_thread_finished(self) -> None:
        self.batch_thread = None
        self.batch_worker = None

    def _multi_preview_limit(self) -> int:
        return max(int(getattr(self.config, "max_preview_streams", 8)), 1)

    def _is_multi_preview_enabled(self, slot_id: str) -> bool:
        runtime = self.multi_camera_runtimes.get(slot_id)
        return True if runtime is None else runtime.preview_enabled

    def _set_multi_preview_enabled(self, slot_id: str, enabled: bool) -> None:
        runtime = self.multi_camera_runtimes.setdefault(slot_id, MultiCameraRuntime())
        runtime.preview_enabled = enabled
        self.pages["multi"].set_slot_preview_enabled(slot_id, enabled)
        if enabled:
            if slot_id in self.multi_preview_order:
                self.multi_preview_order.remove(slot_id)
            self.multi_preview_order.append(slot_id)
        else:
            self.multi_preview_order = [item for item in self.multi_preview_order if item != slot_id]
        self._apply_multi_camera_capture_fps(slot_id, runtime)

    def _multi_camera_capture_target_fps(self, slot_id: str) -> float:
        base_fps = max(float(self.config.stream_target_fps), 1.0)
        runtime = self.multi_camera_runtimes.get(slot_id)
        if runtime is not None and runtime.preview_suspended_for_quality:
            return base_fps
        if (
            self._is_multi_preview_enabled(slot_id)
            or self.pages["multi"].is_slot_preview_dialog_open(slot_id)
        ):
            return max(base_fps, float(self.config.preview_fps))
        return base_fps

    def _apply_multi_camera_capture_fps(
        self,
        slot_id: str,
        runtime: MultiCameraRuntime,
    ) -> None:
        target_fps = self._multi_camera_capture_target_fps(slot_id)
        if runtime.capture_target_fps == target_fps:
            return
        runtime.capture_target_fps = target_fps
        if self.capture_supervisor.set_target_fps(slot_id, target_fps):
            return
        worker = runtime.camera_worker
        if worker is None:
            return
        set_target_fps = getattr(worker, "set_target_fps", None)
        if callable(set_target_fps):
            set_target_fps(target_fps)
        else:
            setattr(worker, "target_fps", target_fps)

    def _initialize_multi_preview_slots(self) -> None:
        slots = self.pages["multi"].slots
        limit = self._multi_preview_limit()
        controls_visible = len(slots) > limit
        self.pages["multi"].set_preview_controls_visible(controls_visible)
        self.multi_preview_order = []
        for index, slot in enumerate(slots):
            enabled = index < limit or not controls_visible
            self._set_multi_preview_enabled(slot.slot_id, enabled)
        self.pages["multi"].set_preview_controls_visible(controls_visible)

    def open_multi_camera_preview(self, slot_id: str) -> None:
        runtime = self.multi_camera_runtimes.get(slot_id)
        if self._is_multi_preview_enabled(slot_id):
            return
        self.pages["multi"].open_slot_preview_dialog(slot_id)
        if runtime is None:
            return
        self._apply_multi_camera_capture_fps(slot_id, runtime)
        frame, _timestamp, seq = self.preview_frame_store.consume(self._multi_stream_id(slot_id))
        if frame is not None and seq != 0:
            runtime.last_preview_seq = seq
            display_frame = self._draw_detection_overlay_on_bgr(frame, runtime.last_detection)
            self._paint_multi_frame(slot_id, display_frame, runtime=runtime, force=True)

    def create_multi_camera_slots(self) -> None:
        count, ok = themed_dialogs.ask_int(
            self,
            "Multi-View Matrix",
            "How many monitoring panes should be created?",
            value=4,
            minimum=1,
            maximum=max(int(self.config.max_managed_streams), 1),
        )
        if ok:
            self.stop_all_multi_camera_analysis()
            self.multi_camera_runtimes.clear()
            self.multi_preview_order.clear()
            self.pages["multi"].create_slots(count)
            self._initialize_multi_preview_slots()

    def mount_multi_camera_source(self, slot_id: str) -> None:
        runtime = self.multi_camera_runtimes.setdefault(slot_id, MultiCameraRuntime())
        deer_code, ok = self._ask_deer_code(
            "Pane Identity Assignment",
            f"Select target deer ID for {slot_id}",
            text=self.runtime_state.deer_code,
        )
        if not ok:
            return
        try:
            deer_code = validate_deer_code(deer_code)
        except ValueError as exc:
            themed_dialogs.show_warning(self, "Invalid Deer ID Format", str(exc))
            return
        path = themed_dialogs.open_file_name(self, f"{slot_id} Select Video Source", "Video (*.mp4 *.avi *.mkv)")
        source: Path | int
        if path:
            source = Path(path)
        else:
            camera_index, ok = themed_dialogs.ask_int(
                self,
                f"{slot_id} Camera",
                "Camera index",
                value=0,
                minimum=0,
                maximum=64,
            )
            if not ok:
                return
            source = camera_index
        runtime.deer_code = deer_code
        runtime.source = source
        self.pages["multi"].set_slot_binding(slot_id, deer_code, source)

    def start_multi_camera_analysis(self, slot_id: str) -> None:
        runtime = self.multi_camera_runtimes.setdefault(slot_id, MultiCameraRuntime())
        if runtime.camera_thread is not None and runtime.camera_thread.isRunning():
            self.stop_multi_camera_analysis(slot_id)
            return
        if self.runtime_state.model_path is None:
            themed_dialogs.show_warning(self, "Cannot Start Analysis", "Please load the AI detection model first.")
            return
        if runtime.deer_code is None or runtime.source is None:
            self.mount_multi_camera_source(slot_id)
            runtime = self.multi_camera_runtimes.setdefault(slot_id, runtime)
            if runtime.deer_code is None or runtime.source is None:
                return
        if not self._check_stream_capacity_before_start():
            return
        if not self._ensure_shared_inference_worker():
            return

        worker_source: str | int = (
            str(runtime.source) if isinstance(runtime.source, Path) else runtime.source
        )
        capture_target_fps = self._multi_camera_capture_target_fps(slot_id)
        capture_handle = self.capture_supervisor.start_route(
            CaptureRouteSpec(
                route_id=slot_id,
                source=worker_source,
                target_fps=capture_target_fps,
                reconnect_delay_seconds=self.config.camera_reconnect_delay_seconds,
                max_read_failures=self.config.camera_max_read_failures,
                open_timeout_milliseconds=self.config.camera_open_timeout_milliseconds,
                read_timeout_milliseconds=self.config.camera_read_timeout_milliseconds,
                max_buffer_drain_grabs_per_frame=(
                    self.config.camera_max_buffer_drain_grabs_per_frame
                ),
            ),
            on_frame=lambda _route_id, frame, ts, sid=slot_id: self.submit_multi_frame(
                sid,
                frame,
                ts,
            ),
            on_status=lambda _route_id, message, sid=slot_id: self.handle_multi_camera_status(
                sid,
                message,
            ),
            on_error=lambda _route_id, message, sid=slot_id: self.handle_multi_camera_error(
                sid,
                message,
            ),
            on_finished=lambda _route_id, sid=slot_id: self.handle_multi_camera_finished(sid),
            on_restarted=lambda _route_id, handle, sid=slot_id: self.handle_multi_capture_restarted(
                sid,
                handle,
            ),
        )
        camera_thread = capture_handle.thread
        camera_worker = capture_handle.worker

        runtime.camera_thread = camera_thread
        runtime.camera_worker = camera_worker
        runtime.capture_target_fps = capture_target_fps
        runtime.detection_session = DetectionSession(
            camera_id=str(runtime.source),
            deer_code=runtime.deer_code,
        )
        runtime.started_at = datetime.now(timezone.utc)
        runtime.last_paint_at = None
        runtime.last_frame_ts = None
        runtime.last_inference_ts = None
        runtime.last_inference_submit_ts = None
        runtime.last_detection = None
        runtime.last_preview_seq = 0
        runtime.fps = 0.0
        runtime.inference_fps = 0.0
        runtime.first_overlay_seen = False
        runtime.preview_suspended_for_quality = False
        runtime.task_id = None
        runtime.deer_id = None
        runtime.task_code = None
        runtime.segments_committed = 0
        runtime.last_flush_at = time.perf_counter()
        runtime.quality_submitted_frame_count = 0
        runtime.quality_overwritten_frame_count = 0
        self.pages["multi"].set_slot_running(slot_id, True)
        self.pages["multi"].set_slot_stats(slot_id, "Realtime sampling - waiting for frames")

    def stop_multi_camera_analysis(self, slot_id: str) -> None:
        runtime = self.multi_camera_runtimes.get(slot_id)
        if runtime is None:
            return
        thread = runtime.camera_thread
        stopped_by_supervisor = self.capture_supervisor.stop_route(slot_id)
        if not stopped_by_supervisor and runtime.camera_worker is not None:
            runtime.camera_worker.stop()
        self._remove_stream_from_inference(self._multi_stream_id(slot_id))
        if (
            not stopped_by_supervisor
            and thread is not None
            and thread.isRunning()
        ):
            thread.quit()
            if not thread.wait(1500):
                thread.terminate()
                thread.wait(1000)
        self.pages["multi"].set_slot_running(slot_id, False)
        self._flush_multi_session_to_db(slot_id)
        if not self.is_detecting and not self._has_running_multi_cameras():
            self._stop_shared_inference_pool()

    def stop_all_multi_camera_analysis(self) -> None:
        for slot_id in list(self.multi_camera_runtimes):
            self.stop_multi_camera_analysis(slot_id)

    def _ensure_multi_task_row(
        self,
        db_session,
        runtime: MultiCameraRuntime,
    ) -> tuple[int, int]:
        session = runtime.detection_session
        if session is None:
            raise ValueError("multi camera detection session is not active")
        if runtime.task_id is not None:
            deer_id = runtime.deer_id
            if deer_id is None:
                deer_id = DeerProfileRepository(db_session).get_or_create(session.deer_code).id
                runtime.deer_id = deer_id
            return runtime.task_id, deer_id

        deer_id = DeerProfileRepository(db_session).get_or_create(session.deer_code).id
        task_code = runtime.task_code or f"#JOB-{uuid.uuid4().hex[:8].upper()}"
        task = AnalysisTaskRepository(db_session).create(
            task_code=task_code,
            deer_profile_id=deer_id,
            mode=TaskMode.REALTIME,
            started_at=runtime.started_at or datetime.now(timezone.utc),
            camera_source_id=None,
        )
        runtime.task_id = task.id
        runtime.deer_id = deer_id
        runtime.task_code = task_code
        return task.id, deer_id

    def _flush_multi_segments_to_db(self, slot_id: str, *, force: bool = False) -> bool:
        runtime = self.multi_camera_runtimes.get(slot_id)
        if runtime is None or runtime.detection_session is None or runtime.started_at is None:
            return True
        if self.database_error:
            return False

        pending_segments = list(runtime.detection_session.segments[runtime.segments_committed :])
        current_segment = runtime.detection_session.current_segment_snapshot(runtime.last_inference_ts)
        if not pending_segments and current_segment is None:
            return True
        if not self._segment_flush_due(runtime.last_flush_at, force=force):
            return True

        try:
            with session_scope(self.session_factory) as db_session:
                task_id, _deer_id = self._ensure_multi_task_row(db_session, runtime)
                repository = SegmentRepository(db_session)
                repository.add_behavior_segments(task_id, pending_segments)
                if current_segment is not None:
                    repository.upsert_behavior_segment(task_id, current_segment)
            committed_count = runtime.segments_committed + len(pending_segments)
            runtime.detection_session.prune_committed_segments(committed_count)
            runtime.segments_committed = 0
            runtime.last_flush_at = time.perf_counter()
            return True
        except Exception as exc:  # noqa: BLE001
            self.pages["multi"].set_slot_stats(slot_id, f"Data not saved: {exc}")
            return False

    def _flush_multi_session_to_db(self, slot_id: str) -> None:
        runtime = self.multi_camera_runtimes.get(slot_id)
        if runtime is None or runtime.detection_session is None or runtime.started_at is None:
            return
        if self.database_error:
            self.pages["multi"].set_slot_stats(slot_id, f"Data not saved: {self.database_error}")
            return

        session = runtime.detection_session
        ended_at = datetime.now(timezone.utc)
        self._close_session_current_segment(session, runtime.last_inference_ts)
        saved_segment_count = session.segment_count
        try:
            if not self._flush_multi_segments_to_db(slot_id, force=True):
                return
            with session_scope(self.session_factory) as db_session:
                task_id, deer_id = self._ensure_multi_task_row(db_session, runtime)
                AnalysisTaskRepository(db_session).complete(
                    task_id=task_id,
                    ended_at=ended_at,
                    has_alert=False,
                    raw_excel_path=None,
                    alert_excel_path=None,
                )
                DailySummaryRepository(db_session).rebuild_for_task_date(
                    runtime.started_at.date(),
                    deer_id,
                )
            self.pages["multi"].set_slot_stats(slot_id, f"Detection finished - saved {saved_segment_count} segments")
        except Exception as exc:  # noqa: BLE001
            self.pages["multi"].set_slot_stats(slot_id, f"Data not saved: {exc}")
        finally:
            runtime.detection_session = None
            runtime.started_at = None
            runtime.task_id = None
            runtime.deer_id = None
            runtime.task_code = None
            runtime.segments_committed = 0
            runtime.last_flush_at = None

    def _multi_stream_id(self, slot_id: str) -> str:
        return f"multi:{slot_id}"

    def _slot_id_from_stream_id(self, stream_id: str) -> str | None:
        prefix = "multi:"
        if not stream_id.startswith(prefix):
            return None
        return stream_id[len(prefix) :]

    def open_deer_profile_dialog(self) -> None:
        existing_codes: list[str] = []
        if not self.database_error:
            try:
                with session_scope(self.session_factory) as session:
                    existing_codes = ProfileService(session).list_active_deer_codes()
            except Exception as exc:  # noqa: BLE001
                self.database_error = str(exc)

        dialog = DeerProfilesDialog(existing_codes)
        if dialog.exec():
            try:
                deer_code = dialog.deer_code()
                if not self.database_error:
                    with session_scope(self.session_factory) as session:
                        ProfileService(session).add_profile(deer_code)
                self.apply_deer_code(deer_code)
            except ValueError as exc:
                themed_dialogs.show_warning(self, "Invalid Format", str(exc))

    def refresh_history_page(self) -> None:
        history_page = self.pages["history"]
        if self.database_error:
            history_page.table.set_empty_message()
            return
        try:
            with session_scope(self.session_factory) as session:
                records = HistoryService(session).list_recent_records(
                    query=history_page.query_text(),
                    has_alert=history_page.has_alert_filter(),
                )
            history_page.set_records(records)
        except Exception as exc:  # noqa: BLE001
            self.database_error = str(exc)
            history_page.table.set_empty_message()

    def open_history_dashboard(self, task_id: int | None) -> None:
        if task_id is None:
            themed_dialogs.show_warning(self, "History", "No matching task was found.")
            return
        try:
            with session_scope(self.session_factory) as session:
                service = HistoryService(session)
                task = service.get_task(task_id)
                if task is None:
                    themed_dialogs.show_warning(self, "History", "This task no longer exists.")
                    return
                segments = service.list_behavior_segments(task_id)
                alerts = service.list_alert_segments(task_id)
                task_code = task.task_code
            ChartDialog(task_code, segments, alerts, parent=self).exec()
        except Exception as exc:  # noqa: BLE001
            themed_dialogs.show_warning(self, "Dashboard Open Failed", str(exc))

    def open_history_raw_records(self, task_id: int | None) -> None:
        self._open_history_export(task_id, export_kind="raw")

    def open_history_alert_records(self, task_id: int | None) -> None:
        self._open_history_export(task_id, export_kind="alert")

    def delete_history_record(self, task_id: int | None) -> None:
        if task_id is None:
            themed_dialogs.show_warning(self, "Delete Failed", "No matching task was found.")
            return
        if not themed_dialogs.ask_confirm(
            self,
            "Delete History Record",
            "Delete this analysis record? Related export files will also be deleted.",
            danger=True,
        ):
            return
        try:
            with session_scope(self.session_factory) as session:
                HistoryService(session).delete_task(task_id, delete_files=True)
            self.refresh_history_page()
        except Exception as exc:  # noqa: BLE001
            themed_dialogs.show_warning(self, "Delete Failed", str(exc))

    def delete_selected_history_records(self) -> None:
        task_ids = self.pages["history"].selected_task_ids()
        if not task_ids:
            themed_dialogs.show_warning(self, "Batch Delete", "Please select records to delete first.")
            return
        if not themed_dialogs.ask_confirm(
            self,
            "Batch DeleteHistory",
            f"Delete the selected {len(task_ids)} analysis records? Related export files will also be deleted.",
            danger=True,
        ):
            return
        try:
            with session_scope(self.session_factory) as session:
                HistoryService(session).delete_tasks(task_ids, delete_files=True)
            self.refresh_history_page()
        except Exception as exc:  # noqa: BLE001
            themed_dialogs.show_warning(self, "Batch Delete Failed", str(exc))

    def _open_history_export(self, task_id: int | None, *, export_kind: str) -> None:
        if task_id is None:
            themed_dialogs.show_warning(self, "History", "No matching task was found.")
            return
        try:
            with session_scope(self.session_factory) as session:
                task = HistoryService(session).get_task(task_id)
                if task is None:
                    themed_dialogs.show_warning(self, "History", "This task no longer exists.")
                    return
                path_text = task.raw_excel_path if export_kind == "raw" else task.alert_excel_path
            if not path_text:
                themed_dialogs.show_warning(self, "File Missing", "This task has not generated the requested export file yet.")
                return
            path = Path(path_text)
            if not path.exists():
                themed_dialogs.show_warning(self, "File Missing", f"File not found: {path}")
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        except Exception as exc:  # noqa: BLE001
            themed_dialogs.show_warning(self, "File Open Failed", str(exc))


def run_app(config: AppConfig) -> int:
    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    try:
        return app.exec()
    except Exception:  # noqa: BLE001
        _logger.exception("QApplication.exec() raised; logging traceback before exit")
        return 3
