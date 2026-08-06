from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from deerui.ui import theme


class MetricCard(QFrame):
    def __init__(self, title: str, value: str, highlighted: bool = False) -> None:
        super().__init__()
        self.setProperty("panel", True)
        self.setMinimumHeight(82)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setProperty("muted", True)
        value_label = QLabel(value)
        value_label.setStyleSheet(
            f"font-size: 24px; font-weight: 700; color: "
            f"{theme.COLOR_PRIMARY if highlighted else theme.COLOR_TEXT_MAIN};"
        )
        layout.addWidget(title_label)
        layout.addWidget(value_label)

        self.value_label = value_label

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)
