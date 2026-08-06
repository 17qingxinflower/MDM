from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from deerui.ui import theme


class BatchPage(QWidget):
    select_video_requested = Signal()
    select_deer_requested = Signal()
    start_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(18)

        title = QLabel("Video Batch Analysis")
        title.setStyleSheet("color: white; font-size: 22px; font-weight: 800;")
        subtitle = QLabel("Run offline behavior recognition on historical videos and save behavior streams, alerts, and task history")
        subtitle.setProperty("muted", True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        panel = QFrame()
        panel.setProperty("panel", True)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(22, 22, 22, 22)
        panel_layout.setSpacing(18)

        status_grid = QGridLayout()
        status_grid.setHorizontalSpacing(16)
        status_grid.setVerticalSpacing(12)
        self.model_label = self._status_label("AI model: not loaded")
        self.video_label = self._status_label("Video: not selected")
        self.deer_label = self._status_label("Target deer: not assigned")
        status_grid.addWidget(self.model_label, 0, 0)
        status_grid.addWidget(self.video_label, 1, 0)
        status_grid.addWidget(self.deer_label, 2, 0)
        panel_layout.addLayout(status_grid)

        actions = QHBoxLayout()
        actions.setSpacing(12)
        self.video_button = QPushButton("Select Video")
        self.deer_button = QPushButton("Assign Deer ID")
        self.start_button = QPushButton("Start Batch")
        self.start_button.setObjectName("primaryButton")
        self.video_button.clicked.connect(self.select_video_requested.emit)
        self.deer_button.clicked.connect(self.select_deer_requested.emit)
        self.start_button.clicked.connect(self.start_requested.emit)
        for button in [self.video_button, self.deer_button]:
            button.setStyleSheet("text-align: center;")
            actions.addWidget(button)
        self.start_button.setStyleSheet(
            f"background: {theme.COLOR_PRIMARY}; color: #04130d; font-weight: 800; text-align: center;"
        )
        actions.addWidget(self.start_button)
        panel_layout.addLayout(actions)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet(
            f"""
            QProgressBar {{
                background: #0D1411;
                color: {theme.COLOR_TEXT_MAIN};
                border: 1px solid {theme.COLOR_BORDER};
                border-radius: 7px;
                height: 22px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: {theme.COLOR_PRIMARY};
                border-radius: 6px;
            }}
            """
        )
        panel_layout.addWidget(self.progress)

        self.result_label = QLabel("Waiting for batch task")
        self.result_label.setProperty("muted", True)
        self.result_label.setWordWrap(True)
        panel_layout.addWidget(self.result_label)

        layout.addWidget(panel)
        layout.addStretch(1)

    def set_model_path(self, model_path: Path) -> None:
        self.model_label.setText(f"AI model: {model_path.name}")

    def set_video_source(self, video_path: Path) -> None:
        self.video_label.setText(f"Video: {video_path.name}")

    def set_deer_code(self, deer_code: str) -> None:
        self.deer_label.setText(f"Target deer: {deer_code}")

    def set_running(self, running: bool) -> None:
        self.start_button.setText("Stop Batch" if running else "Start Batch")
        self.video_button.setEnabled(not running)
        self.deer_button.setEnabled(not running)

    def set_progress(self, value: float) -> None:
        self.progress.setValue(max(0, min(int(value), 100)))

    def set_result(self, text: str) -> None:
        self.result_label.setText(text)

    def _status_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"background: #0F1713; color: {theme.COLOR_TEXT_MAIN}; border: 1px solid {theme.COLOR_BORDER};"
            "border-radius: 8px; padding: 12px 14px; font-size: 13px;"
        )
        return label
