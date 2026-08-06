from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from deerui.ui.widgets.behavior_stats_panel import BehaviorStatsPanel
from deerui.ui.widgets.metric_card import MetricCard
from deerui.ui.widgets.video_panel import VideoPanel


class RealtimePage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        cards = QGridLayout()
        cards.setSpacing(16)
        self.deer_card = MetricCard("Target Deer ID", "2.1.6")
        cards.addWidget(self.deer_card, 0, 0)
        cards.addWidget(MetricCard("Ambient Temperature", "18.5 C"), 0, 1)
        cards.addWidget(MetricCard("Relative Humidity", "62 %"), 0, 2)
        cards.addWidget(MetricCard("AI Health Assessment", "Good", highlighted=True), 0, 3)
        layout.addLayout(cards)

        self.model_status_label = QLabel("AI model: not loaded")
        self.source_status_label = QLabel("Video source: not connected")
        self.model_status_label.setProperty("muted", True)
        self.source_status_label.setProperty("muted", True)
        content = QHBoxLayout()
        content.setSpacing(16)
        self.video_panel = VideoPanel()
        self.stats_panel = BehaviorStatsPanel()
        self.stats_panel.setMinimumWidth(330)
        content.addWidget(self.video_panel, 7)
        content.addWidget(self.stats_panel, 3)
        layout.addLayout(content, 1)

    def set_deer_code(self, deer_code: str) -> None:
        self.deer_card.set_value(deer_code)

    def set_model_path(self, model_path: Path) -> None:
        self.model_status_label.setText(f"AI model: {model_path.name}")

    def set_video_source(self, source: Path | int) -> None:
        if isinstance(source, Path):
            self.source_status_label.setText(f"Video source: {source.name}")
            self.video_panel.set_placeholder(f"Selected video: {source.name}")
        else:
            self.source_status_label.setText(f"Video source: camera {source}")
            self.video_panel.set_placeholder(f"Connected to camera {source}")
