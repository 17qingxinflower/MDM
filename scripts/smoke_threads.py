"""Diagnostic: simulate main_window's threaded detection without GUI.

Usage:
    D:/PrommgremApps/Anaconda/setup/envs/yolov8n/python.exe scripts/smoke_threads.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer, QThread
from PySide6.QtWidgets import QApplication

from deerui.config import load_config
from deerui.domain.models import DEFAULT_BEHAVIOR_CLASSES
from deerui.workers.camera_worker import CameraWorker
from deerui.workers.frame_buffer import FrameBuffer
from deerui.workers.inference_worker import InferenceWorker

VIDEO = str(
    Path(__file__).resolve().parents[1]
    / "benchmark_assets"
    / "053358f40cf7b5cfe5d8620a94d621b9.mp4"
)


def main() -> int:
    print(f"[smoke] starting at {time.time():.1f}")
    app = QApplication(sys.argv)
    cfg = load_config()
    print(f"[smoke] config model={cfg.model_path} db={cfg.database_url}")

    buf = FrameBuffer()
    worker = InferenceWorker(
        str(cfg.model_path),
        DEFAULT_BEHAVIOR_CLASSES,
        cfg.confidence_threshold,
        buf,
        cfg.sample_every_n_frames,
    )
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.start)

    overlays = [0]
    detections = [0]

    def on_overlay(*_args):
        overlays[0] += 1
        print(f"[smoke] overlay #{overlays[0]}")

    def on_detection(_det):
        detections[0] += 1
        print(f"[smoke] detection #{detections[0]}")

    def on_err(msg):
        print(f"[smoke] WORKER ERROR: {msg}")

    def on_warn(msg):
        print(f"[smoke] WORKER WARN: {msg}")

    def on_loaded():
        print("[smoke] model loaded")

    worker.frame_with_overlay.connect(on_overlay)
    worker.detection_ready.connect(on_detection)
    worker.error.connect(on_err)
    worker.warning.connect(on_warn)
    worker.model_loaded.connect(on_loaded)
    worker.finished.connect(thread.quit)

    cam_thread = QThread()
    cam = CameraWorker(VIDEO, target_fps=15)
    cam.moveToThread(cam_thread)
    cam_thread.started.connect(cam.run)
    cam.frame_ready.connect(buf.submit)
    cam.error.connect(lambda m: print(f"[smoke] CAM ERROR: {m}"))
    cam.finished.connect(cam_thread.quit)

    thread.start()
    cam_thread.start()
    print("[smoke] threads started, will run 8 seconds")

    QTimer.singleShot(8000, app.quit)
    rc = app.exec()
    print(f"[smoke] exec returned {rc}, overlays={overlays[0]}, detections={detections[0]}")

    cam.stop()
    worker.stop()
    cam_thread.wait(2000)
    thread.wait(2000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
