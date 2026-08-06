from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from deerui.ui import theme


class Sidebar(QWidget):
    navigate = Signal(str)
    load_model_requested = Signal()
    load_source_requested = Signal()
    start_stop_requested = Signal()
    thresholds_requested = Signal()
    profiles_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setFixedWidth(260)
        self.setStyleSheet(f"background: {theme.COLOR_BG_SIDEBAR};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 36, 24, 28)
        layout.setSpacing(8)

        title = QLabel("MuskDeer\nMonitor")
        title.setStyleSheet(f"color: {theme.COLOR_PRIMARY}; font-size: 26px; font-weight: 900;")
        subtitle = QLabel("Intelligent Husbandry Assistant")
        subtitle.setProperty("muted", True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(24)

        self.nav_buttons: dict[str, QPushButton] = {}
        for key, text in [
            ("realtime", "Realtime Console"),
            ("multi", "Multi-View Matrix"),
            ("batch", "Video Batch Analysis"),
            ("history", "History Library"),
        ]:
            button = QPushButton(text)
            button.clicked.connect(lambda checked=False, page=key: self.navigate.emit(page))
            layout.addWidget(button)
            self.nav_buttons[key] = button

        profile_button = QPushButton("Deer Profile Registry")
        profile_button.clicked.connect(self.profiles_requested.emit)
        layout.addWidget(profile_button)
        layout.addSpacing(22)

        for text, signal in [
            ("Load AI Model    >", self.load_model_requested),
            ("Connect Video Source >", self.load_source_requested),
            ("Behavior Alert Settings", self.thresholds_requested),
        ]:
            button = QPushButton(text)
            button.clicked.connect(signal.emit)
            layout.addWidget(button)

        layout.addStretch(1)

        self.model_status_label = QLabel("AI model: not loaded")
        self.source_status_label = QLabel("Video source: not connected")
        for label in [self.model_status_label, self.source_status_label]:
            label.setProperty("muted", True)
            label.setWordWrap(True)
            layout.addWidget(label)

        self.start_button = QPushButton("Start Detection")
        self.start_button.setObjectName("startDetectionButton")
        self.start_button.clicked.connect(self.start_stop_requested.emit)
        self.start_button.setStyleSheet(
            f"background: {theme.COLOR_PRIMARY}; color: #04130d; font-weight: 800;"
            " padding: 12px 14px; border-radius: 8px;"
        )
        layout.addWidget(self.start_button)
        self.set_active("realtime")

    def set_active(self, page: str) -> None:
        for key, button in self.nav_buttons.items():
            button.setProperty("active", key == page)
            button.style().unpolish(button)
            button.style().polish(button)

    def set_model_status(self, model_name: str | None) -> None:
        self.model_status_label.setText(f"AI model: {model_name}" if model_name else "AI model: not loaded")

    def set_source_status(self, source_name: str | None) -> None:
        self.source_status_label.setText(f"Video source: {source_name}" if source_name else "Video source: not connected")

    def set_detection_running(self, is_running: bool) -> None:
        self.start_button.setText("Stop Detection" if is_running else "Start Detection")
