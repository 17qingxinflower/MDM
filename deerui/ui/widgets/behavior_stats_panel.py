from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout

from deerui.ui import theme


class BehaviorStatCard(QFrame):
    def __init__(self, name: str, color: str) -> None:
        super().__init__()
        self.color = color
        self.setObjectName("behaviorStatCard")
        self._apply_style(active=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        self.dot_label = QLabel("*")
        self.dot_label.setStyleSheet(f"color: {self.color}; font-size: 14px;")
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet(
            f"color: {theme.COLOR_TEXT_MAIN}; font-size: 14px; font-weight: 800;"
        )
        self.percent_label = QLabel("0%")
        self.percent_label.setStyleSheet(f"color: {theme.COLOR_TEXT_SUB}; font-size: 12px;")
        header.addWidget(self.dot_label)
        header.addWidget(self.name_label, 1)
        header.addWidget(self.percent_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        self.duration_label = QLabel("0.0s")
        self.duration_label.setStyleSheet(
            f"color: {self.color}; font-size: 17px; font-weight: 800;"
        )
        self.count_label = QLabel("0 segments")
        self.count_label.setStyleSheet(f"color: {theme.COLOR_TEXT_SUB}; font-size: 12px;")
        footer.addWidget(self.duration_label)
        footer.addStretch(1)
        footer.addWidget(self.count_label)

        layout.addLayout(header)
        layout.addWidget(self.progress)
        layout.addLayout(footer)

    def update_stat(
        self,
        *,
        duration_seconds: float,
        count: int,
        percent: float,
        active: bool,
    ) -> None:
        self.duration_label.setText(self._format_duration(duration_seconds))
        self.count_label.setText(f"{count} segments")
        self.percent_label.setText(f"{percent:.0f}%")
        self.progress.setValue(max(0, min(100, round(percent))))
        self._apply_style(active=active)
        self.dot_label.setStyleSheet(f"color: {self.color}; font-size: 14px;")
        self.duration_label.setStyleSheet(
            f"color: {self.color}; font-size: 17px; font-weight: 800;"
        )
        self.name_label.setStyleSheet(
            f"color: {theme.COLOR_TEXT_MAIN}; font-size: 14px; "
            f"font-weight: {'900' if active else '800'};"
        )

    def _apply_style(self, *, active: bool) -> None:
        border_color = self.color if active else theme.COLOR_BORDER
        self.setStyleSheet(
            f"""
            QFrame#behaviorStatCard {{
                background: #0D1612;
                border: 1px solid {border_color};
                border-radius: 8px;
            }}
            QLabel {{
                border: none;
                background: transparent;
            }}
            QProgressBar {{
                height: 6px;
                border: none;
                border-radius: 3px;
                background: #17221C;
                text-align: center;
                color: transparent;
            }}
            QProgressBar::chunk {{
                border-radius: 3px;
                background: {self.color};
            }}
            """
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        remain = int(seconds % 60)
        return f"{minutes}m {remain}s"


class BehaviorStatsPanel(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setProperty("panel", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet(
            f"color: {theme.COLOR_WARNING}; font-weight: 700; font-size: 13px;"
        )
        self.warning_label.setWordWrap(True)
        self.warning_label.hide()

        title = QLabel("Current Focus Behavior")
        title.setProperty("muted", True)
        self.action_label = QLabel("Waiting")
        self.action_label.setStyleSheet(
            f"font-size: 30px; font-weight: 800; color: {theme.COLOR_TEXT_SUB};"
        )

        stats_title = QLabel("Realtime Behavior Summary")
        stats_title.setStyleSheet(f"color: {theme.COLOR_PRIMARY}; font-weight: 700;")
        self.stats_label = QLabel("No behavior data yet")
        self.stats_label.setProperty("muted", True)
        self.stats_label.setWordWrap(True)

        self.cards_layout = QVBoxLayout()
        self.cards_layout.setSpacing(10)
        self.behavior_cards: dict[str, BehaviorStatCard] = {}

        layout.addWidget(self.warning_label)
        layout.addWidget(title)
        layout.addWidget(self.action_label)
        layout.addSpacing(12)
        layout.addWidget(stats_title)
        layout.addWidget(self.stats_label)
        layout.addLayout(self.cards_layout)
        layout.addStretch(1)

    def set_action(self, display_name: str | None) -> None:
        if display_name:
            self.action_label.setText(f"* {display_name}")
            self.action_label.setStyleSheet(
                f"font-size: 30px; font-weight: 800; color: {theme.COLOR_PRIMARY};"
            )
        else:
            self.action_label.setText("Waiting")
            self.action_label.setStyleSheet(
                f"font-size: 30px; font-weight: 800; color: {theme.COLOR_TEXT_SUB};"
            )

    def set_stats_summary(self, text: str | None) -> None:
        if text:
            self.stats_label.setText(text)
            self.stats_label.setProperty("muted", False)
            self.stats_label.setStyleSheet(f"color: {theme.COLOR_TEXT_MAIN};")
        else:
            self.stats_label.setText("No behavior data yet")
            self.stats_label.setProperty("muted", True)
            self.stats_label.setStyleSheet("")

    def set_behavior_stats(self, stats: list[dict[str, object]]) -> None:
        seen: set[str] = set()
        for item in stats:
            key = str(item["key"])
            seen.add(key)
            card = self.behavior_cards.get(key)
            if card is None:
                card = BehaviorStatCard(str(item["name"]), str(item.get("color", theme.COLOR_PRIMARY)))
                self.behavior_cards[key] = card
                self.cards_layout.addWidget(card)
            card.update_stat(
                duration_seconds=float(item.get("duration_seconds", 0.0)),
                count=int(item.get("count", 0)),
                percent=float(item.get("percent", 0.0)),
                active=bool(item.get("active", False)),
            )
            card.show()

        for key, card in self.behavior_cards.items():
            if key not in seen:
                card.hide()

    def set_warning(self, message: str | None) -> None:
        if message:
            self.warning_label.setText(f"Warning: {message}")
            self.warning_label.show()
        else:
            self.warning_label.setText("")
            self.warning_label.hide()
