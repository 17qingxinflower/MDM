from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QStackedLayout

from deerui.ui import theme


class VideoPanel(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setProperty("panel", True)
        self.setMinimumSize(760, 520)

        layout = QStackedLayout(self)
        self.video_label = QLabel("Waiting for a video source ...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            f"background: #000000; color: {theme.COLOR_TEXT_SUB}; font-size: 28px; font-weight: 700;"
        )
        layout.addWidget(self.video_label)

        self.fps_label = QLabel("FPS: 0", self.video_label)
        self.fps_label.setStyleSheet(
            f"background: {theme.COLOR_BG_SIDEBAR}; color: white; padding: 4px 8px; border-radius: 4px;"
        )
        self.position_fps_label()

    def set_fps(self, fps: float) -> None:
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.position_fps_label()

    def position_fps_label(self) -> None:
        margin = 18
        self.fps_label.adjustSize()
        x = max(margin, self.video_label.width() - self.fps_label.width() - margin)
        self.fps_label.move(x, margin)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.position_fps_label()

    def set_placeholder(self, text: str) -> None:
        self.video_label.setText(text)
        self.video_label.setPixmap(QPixmap())
        self.position_fps_label()

    def set_frame(self, frame) -> None:
        """Display an OpenCV BGR HxWx3 frame."""
        if frame is None:
            return
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError("Video frame must be an HxWx3 array")
        self.set_rgb_frame(frame[:, :, ::-1].copy())

    def set_rgb_frame(self, frame) -> None:
        """Display an RGB HxWx3 frame."""
        if frame is None:
            return
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError("Video frame must be an HxWx3 array")

        rgb_frame = frame.copy()
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )
        self.video_label.setText("")
        self.position_fps_label()
