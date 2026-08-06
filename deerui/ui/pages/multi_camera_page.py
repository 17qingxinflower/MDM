from __future__ import annotations

from dataclasses import dataclass, field

from pathlib import Path

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from deerui.ui import theme
from deerui.ui.themed_dialogs import ThemedDialog


@dataclass
class MultiCameraSlot:
    slot_id: str
    panel: QFrame
    video_label: QLabel
    side_panel: QFrame
    bind_label: QLabel
    mount_button: QPushButton
    analyze_button: QPushButton
    preview_button: QPushButton
    action_label: QLabel
    stats_label: QLabel
    behavior_scroll: QScrollArea
    behavior_cards: dict[str, "CompactBehaviorStatCard"]
    preview_controls_visible: bool = False
    preview_enabled: bool = True
    preview_dialog: ThemedDialog | None = None
    dialog_video_label: QLabel | None = None
    dialog_action_label: QLabel | None = None
    dialog_behavior_cards: dict[str, "CompactBehaviorStatCard"] = field(default_factory=dict)
    deer_code: str | None = None
    source: Path | int | None = None


class CompactBehaviorStatCard(QFrame):
    def __init__(self, name: str, color: str) -> None:
        super().__init__()
        self.color = color
        self.setObjectName("multiBehaviorCard")
        self.setMinimumHeight(54)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setSpacing(4)
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet(
            f"color: {theme.COLOR_TEXT_MAIN}; font-size: 10px; font-weight: 800;"
        )
        self.percent_label = QLabel("0%")
        self.percent_label.setStyleSheet(f"color: {theme.COLOR_TEXT_SUB}; font-size: 10px;")
        header.addWidget(self.name_label, 1)
        header.addWidget(self.percent_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)

        footer = QHBoxLayout()
        footer.setSpacing(4)
        self.duration_label = QLabel("0.0s")
        self.duration_label.setStyleSheet(f"color: {self.color}; font-size: 11px; font-weight: 800;")
        self.count_label = QLabel("0 segments")
        self.count_label.setStyleSheet(f"color: {theme.COLOR_TEXT_SUB}; font-size: 10px;")
        footer.addWidget(self.duration_label)
        footer.addWidget(self.count_label)

        layout.addLayout(header)
        layout.addWidget(self.progress)
        layout.addLayout(footer)
        self._apply_style(active=False)

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

    def _apply_style(self, *, active: bool) -> None:
        border_color = self.color if active else theme.COLOR_BORDER
        self.setStyleSheet(
            f"""
            QFrame#multiBehaviorCard {{
                background: #0C1511;
                border: 1px solid {border_color};
                border-radius: 6px;
            }}
            QLabel {{
                border: none;
                background: transparent;
            }}
            QProgressBar {{
                height: 3px;
                border: none;
                border-radius: 2px;
                background: #17221C;
            }}
            QProgressBar::chunk {{
                border-radius: 2px;
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
        return f"{minutes}m{remain}s"


class MultiCameraPage(QWidget):
    create_slots_requested = Signal()
    stress_test_requested = Signal()
    mount_source_requested = Signal(str)
    analyze_requested = Signal(str)
    preview_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.slots: list[MultiCameraSlot] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(30, 30, 30, 30)
        root.setSpacing(18)

        header = QHBoxLayout()
        title_group = QVBoxLayout()
        title = QLabel("Multi-View Monitoring Matrix")
        title.setStyleSheet("color: white; font-size: 22px; font-weight: 800;")
        subtitle = QLabel("Multiple sources share one YOLO inference queue to avoid loading one model per stream")
        subtitle.setProperty("muted", True)
        title_group.addWidget(title)
        title_group.addWidget(subtitle)
        header.addLayout(title_group, 1)

        self.create_button = QPushButton("Create Panes")
        self.create_button.setFixedWidth(140)
        self.create_button.setStyleSheet(
            f"background: {theme.COLOR_PRIMARY}; color: #04130d; font-weight: 800; text-align: center;"
        )
        self.create_button.clicked.connect(self.create_slots_requested.emit)
        header.addWidget(self.create_button)
        self.stress_test_button = QPushButton("60-100 Stream Benchmark")
        self.stress_test_button.setFixedWidth(136)
        self.stress_test_button.setStyleSheet(
            f"background: #1E2923; border-color: {theme.COLOR_BORDER}; color: {theme.COLOR_TEXT_MAIN};"
            "font-weight: 800; text-align: center;"
        )
        self.stress_test_button.clicked.connect(self.stress_test_requested.emit)
        header.addWidget(self.stress_test_button)
        root.addLayout(header)

        search_row = QHBoxLayout()
        search_row.setSpacing(10)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search camera slot, deer ID, or video source; press Enter to open preview")
        self.search_input.setMinimumHeight(38)
        self.search_input.setStyleSheet(
            f"""
            QLineEdit {{
                background: #101915;
                color: {theme.COLOR_TEXT_MAIN};
                border: 1px solid {theme.COLOR_BORDER};
                border-radius: 8px;
                padding: 8px 12px;
            }}
            QLineEdit:focus {{
                border-color: {theme.COLOR_PRIMARY};
            }}
            QLineEdit::placeholder {{
                color: {theme.COLOR_TEXT_SUB};
            }}
            """
        )
        self.search_input.textChanged.connect(self.apply_slot_filter)
        self.search_input.returnPressed.connect(self._open_first_search_match_preview)
        search_row.addWidget(self.search_input, 1)
        root.addLayout(search_row)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet(
            f"""
            QScrollArea {{
                background: {theme.COLOR_BG_MAIN};
                border: none;
            }}
            QScrollArea > QWidget {{
                background: {theme.COLOR_BG_MAIN};
            }}
            QScrollBar:vertical {{
                background: {theme.COLOR_BG_MAIN};
                width: 8px;
            }}
            QScrollBar::handle:vertical {{
                background: {theme.COLOR_BORDER};
                border-radius: 4px;
                min-height: 28px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            """
        )
        self.grid_host = QWidget()
        self.grid_host.setStyleSheet(f"background: {theme.COLOR_BG_MAIN};")
        self.scroll.viewport().setStyleSheet(f"background: {theme.COLOR_BG_MAIN};")
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(8)
        self.scroll.setWidget(self.grid_host)
        root.addWidget(self.scroll, 1)

        self.empty_label = QLabel("Click Create Panes to create 1-8 monitoring slots")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setProperty("muted", True)
        self.grid.addWidget(self.empty_label, 0, 0)

    def create_slots(self, count: int) -> None:
        count = max(1, int(count))
        self._clear_grid()
        self.slots = []
        columns = self._column_count_for_slots(count)
        for index in range(count):
            row, column = divmod(index, columns)
            slot = self._build_slot(index + 1)
            self.slots.append(slot)
            self.grid.addWidget(slot.panel, row, column)
        for column in range(4):
            self.grid.setColumnStretch(column, 0)
        for column in range(columns):
            self.grid.setColumnStretch(column, 1)
        self.apply_slot_filter(self.search_input.text())

    def set_slot_binding(
        self,
        slot_id: str,
        deer_code: str,
        source: Path | int | None = None,
    ) -> None:
        slot = self._slot_by_id(slot_id)
        if slot is None:
            return
        slot.deer_code = deer_code
        if source is not None:
            slot.source = source
        source_text = self._source_text(slot.source)
        slot.bind_label.setText(f"Assigned {deer_code}" + (f"\n{source_text}" if source_text else ""))
        slot.bind_label.setStyleSheet(f"color: {theme.COLOR_PRIMARY}; font-weight: 700;")
        self.apply_slot_filter(self.search_input.text())

    def set_slot_running(self, slot_id: str, running: bool) -> None:
        slot = self._slot_by_id(slot_id)
        if slot is None:
            return
        slot.analyze_button.setText("Stop Analysis" if running else "Start Analysis")
        slot.action_label.setText("Analyzing" if running else "Waiting")
        slot.action_label.setStyleSheet(
            f"color: {theme.COLOR_PRIMARY if running else theme.COLOR_TEXT_SUB};"
            "font-size: 15px; font-weight: 800;"
        )

    def _legacy_set_slot_action(self, slot_id: str, action_text: str | None) -> None:
        slot = self._slot_by_id(slot_id)
        if slot is None:
            return
        slot.action_label.setText(f"* {action_text or 'Waiting'}")

    def set_slot_stats(self, slot_id: str, text: str) -> None:
        slot = self._slot_by_id(slot_id)
        if slot is not None:
            slot.stats_label.setText(text)

    def apply_slot_filter(self, text: str) -> None:
        query = text.strip().lower()
        for slot in self.slots:
            slot.panel.setVisible(not query or query in self._slot_search_text(slot))

    def _open_first_search_match_preview(self) -> None:
        matches = self._matching_slots_for_search()
        if matches:
            self.preview_requested.emit(matches[0].slot_id)

    def _matching_slots_for_search(self) -> list[MultiCameraSlot]:
        query = self.search_input.text().strip().lower()
        if not query:
            return []
        return [slot for slot in self.slots if query in self._slot_search_text(slot)]

    def _slot_search_text(self, slot: MultiCameraSlot) -> str:
        values = (
            slot.slot_id,
            slot.deer_code or "",
            self._source_text(slot.source),
            slot.bind_label.text(),
        )
        return " ".join(values).lower()

    def set_preview_controls_visible(self, visible: bool) -> None:
        for slot in self.slots:
            slot.preview_controls_visible = visible
            slot.preview_button.setVisible(visible and not slot.preview_enabled)

    def _legacy_set_slot_preview_enabled(self, slot_id: str, enabled: bool) -> None:
        slot = self._slot_by_id(slot_id)
        if slot is None:
            return
        slot.preview_button.setText("Previewing" if enabled else "Open Preview")
        slot.preview_button.setEnabled(not enabled)
        if not enabled:
            slot.video_label.clear()
            slot.video_label.setText("Background detection\nClick to open preview")
        elif slot.video_label.text():
            slot.video_label.setText(f"{slot_id} idle")

    def _legacy_set_slot_behavior_stats(self, slot_id: str, stats: list[dict[str, object]]) -> None:
        slot = self._slot_by_id(slot_id)
        if slot is None:
            return
        seen: set[str] = set()
        for item in stats:
            key = str(item["key"])
            seen.add(key)
            card = slot.behavior_cards.get(key)
            if card is None:
                continue
            card.update_stat(
                duration_seconds=float(item.get("duration_seconds", 0.0)),
                count=int(item.get("count", 0)),
                percent=float(item.get("percent", 0.0)),
                active=bool(item.get("active", False)),
            )
            card.show()
        for key, card in slot.behavior_cards.items():
            if key not in seen:
                card.hide()

    def set_slot_frame_bgr(self, slot_id: str, frame) -> None:
        if frame is None:
            return
        self.set_slot_frame_rgb(slot_id, frame[:, :, ::-1].copy())

    def set_slot_frame_rgb(self, slot_id: str, frame) -> None:
        slot = self._slot_by_id(slot_id)
        if slot is None or frame is None:
            return
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return
        rgb_frame = frame.copy()
        height, width, channels = rgb_frame.shape
        image = QImage(
            rgb_frame.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image)
        slot.video_label.setPixmap(
            pixmap.scaled(
                slot.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        slot.video_label.setText("")

    def set_slot_action(self, slot_id: str, action_text: str | None) -> None:
        slot = self._slot_by_id(slot_id)
        if slot is None:
            return
        text = f"* {action_text or 'Waiting'}"
        slot.action_label.setText(text)
        if slot.dialog_action_label is not None:
            slot.dialog_action_label.setText(text)

    def set_slot_preview_enabled(self, slot_id: str, enabled: bool) -> None:
        slot = self._slot_by_id(slot_id)
        if slot is None:
            return
        slot.preview_enabled = enabled
        slot.preview_button.setText("Open Preview")
        slot.preview_button.setEnabled(not enabled)
        slot.preview_button.setVisible(slot.preview_controls_visible and not enabled)
        if enabled:
            slot.video_label.show()
            slot.behavior_scroll.show()
            slot.panel.setMinimumHeight(300)
            slot.side_panel.setMinimumWidth(156)
            slot.side_panel.setMaximumWidth(164)
            slot.side_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
            if slot.video_label.text():
                slot.video_label.setText(f"{slot_id} idle")
            return

        slot.video_label.clear()
        slot.video_label.setText("")
        slot.video_label.hide()
        slot.behavior_scroll.hide()
        slot.panel.setMinimumHeight(96)
        slot.side_panel.setMinimumWidth(0)
        slot.side_panel.setMaximumWidth(16777215)
        slot.side_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_slot_behavior_stats(self, slot_id: str, stats: list[dict[str, object]]) -> None:
        slot = self._slot_by_id(slot_id)
        if slot is None:
            return
        self._update_behavior_cards(slot.behavior_cards, stats)
        if slot.dialog_behavior_cards:
            self._update_behavior_cards(slot.dialog_behavior_cards, stats)

    def _update_behavior_cards(
        self,
        cards: dict[str, CompactBehaviorStatCard],
        stats: list[dict[str, object]],
    ) -> None:
        seen: set[str] = set()
        for item in stats:
            key = str(item["key"])
            seen.add(key)
            card = cards.get(key)
            if card is None:
                continue
            card.update_stat(
                duration_seconds=float(item.get("duration_seconds", 0.0)),
                count=int(item.get("count", 0)),
                percent=float(item.get("percent", 0.0)),
                active=bool(item.get("active", False)),
            )
            card.show()
        for key, card in cards.items():
            if key not in seen:
                card.hide()

    def open_slot_preview_dialog(self, slot_id: str) -> None:
        slot = self._slot_by_id(slot_id)
        if slot is None:
            return
        if slot.preview_dialog is not None and slot.preview_dialog.isVisible():
            slot.preview_dialog.raise_()
            slot.preview_dialog.activateWindow()
            return

        dialog = ThemedDialog(f"{slot_id} Live Preview", self, width=980)
        dialog.setModal(False)
        content = QHBoxLayout()
        content.setSpacing(12)

        video_label = QLabel(f"{slot_id} Waiting for frames")
        video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_label.setMinimumSize(640, 360)
        video_label.setStyleSheet(
            f"background: #000000; color: {theme.COLOR_TEXT_SUB}; font-size: 16px; font-weight: 800;"
        )
        content.addWidget(video_label, 1)

        side = QVBoxLayout()
        side.setSpacing(8)
        action_label = QLabel(slot.action_label.text())
        action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        action_label.setStyleSheet(f"color: {theme.COLOR_PRIMARY}; font-size: 18px; font-weight: 800;")
        side.addWidget(action_label)

        dialog_behavior_cards: dict[str, CompactBehaviorStatCard] = {}
        for key, name in self._behavior_specs():
            card = CompactBehaviorStatCard(name, theme.behavior_color(key, name))
            dialog_behavior_cards[key] = card
            side.addWidget(card)
        side.addStretch(1)
        content.addLayout(side, 0)
        dialog.body_layout.addLayout(content)
        close = dialog.add_button("Close", primary=True)
        close.clicked.connect(dialog.accept)

        slot.preview_dialog = dialog
        slot.dialog_video_label = video_label
        slot.dialog_action_label = action_label
        slot.dialog_behavior_cards = dialog_behavior_cards
        dialog.finished.connect(lambda _code=0, sid=slot_id: self._clear_slot_preview_dialog(sid))
        dialog.show()

    def _clear_slot_preview_dialog(self, slot_id: str) -> None:
        slot = self._slot_by_id(slot_id)
        if slot is None:
            return
        slot.preview_dialog = None
        slot.dialog_video_label = None
        slot.dialog_action_label = None
        slot.dialog_behavior_cards = {}

    def is_slot_preview_dialog_open(self, slot_id: str) -> bool:
        slot = self._slot_by_id(slot_id)
        return bool(slot and slot.preview_dialog is not None and slot.preview_dialog.isVisible())

    def set_slot_dialog_frame_bgr(self, slot_id: str, frame) -> None:
        if frame is None:
            return
        self.set_slot_dialog_frame_rgb(slot_id, frame[:, :, ::-1].copy())

    def set_slot_dialog_frame_rgb(self, slot_id: str, frame) -> None:
        slot = self._slot_by_id(slot_id)
        if slot is None or slot.dialog_video_label is None or frame is None:
            return
        self._set_frame_on_label(slot.dialog_video_label, frame)

    def _set_frame_on_label(self, label: QLabel, frame) -> None:
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return
        rgb_frame = frame.copy()
        height, width, channels = rgb_frame.shape
        image = QImage(
            rgb_frame.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image)
        label.setPixmap(
            pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        label.setText("")

    def _behavior_specs(self) -> tuple[tuple[str, str], ...]:
        return (
            ("Stand_or_walk", "Stand/Walk"),
            ("Standing_feeding", "Feeding"),
            ("egestion", "Egestion"),
            ("Lie_down_and_rest", "Lie Down"),
        )

    def _build_slot(self, number: int) -> MultiCameraSlot:
        slot_id = f"Slot_{number}"
        panel = QFrame()
        panel.setProperty("panel", True)
        panel.setMinimumHeight(300)
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        video_label = QLabel(f"{slot_id} idle")
        video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_label.setMinimumWidth(360)
        video_label.setStyleSheet(
            f"background: #000000; color: {theme.COLOR_TEXT_SUB}; font-size: 15px; font-weight: 700;"
        )
        layout.addWidget(video_label, 1)

        side_panel = QFrame()
        side_panel.setObjectName("multiSidePanel")
        side_panel.setMinimumWidth(156)
        side_panel.setMaximumWidth(164)
        side_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        side_panel.setStyleSheet(
            """
            QFrame#multiSidePanel {
                background: transparent;
                border: none;
            }
            """
        )
        side = QVBoxLayout(side_panel)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(6)
        title = QLabel(f"{slot_id} Controls")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: white; font-size: 13px; font-weight: 800;")
        bind_label = QLabel("No deer assigned")
        bind_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bind_label.setWordWrap(True)
        bind_label.setProperty("muted", True)
        side.addWidget(title)
        side.addWidget(bind_label)

        mount_button = QPushButton("Mount Source")
        mount_button.setMinimumHeight(30)
        mount_button.setStyleSheet(
            f"background: #1E2923; border-color: {theme.COLOR_BORDER}; text-align: center; font-size: 11px;"
        )
        mount_button.clicked.connect(lambda checked=False, sid=slot_id: self.mount_source_requested.emit(sid))
        side.addWidget(mount_button)

        analyze_button = QPushButton("Start Analysis")
        analyze_button.setMinimumHeight(30)
        analyze_button.setStyleSheet(
            f"background: {theme.COLOR_PRIMARY}; color: #04130d; font-weight: 800; text-align: center; font-size: 11px;"
        )
        analyze_button.clicked.connect(lambda checked=False, sid=slot_id: self.analyze_requested.emit(sid))
        side.addWidget(analyze_button)

        preview_button = QPushButton("Open Preview")
        preview_button.setMinimumHeight(28)
        preview_button.setStyleSheet(
            f"background: #121D17; border-color: {theme.COLOR_BORDER}; text-align: center; font-size: 11px;"
        )
        preview_button.clicked.connect(lambda checked=False, sid=slot_id: self.preview_requested.emit(sid))
        preview_button.hide()
        side.addWidget(preview_button)

        action_label = QLabel("Waiting")
        action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        action_label.setStyleSheet(f"color: {theme.COLOR_TEXT_SUB}; font-size: 15px; font-weight: 800;")
        side.addWidget(action_label)

        stats_label = QLabel("No behavior data yet")
        stats_label.setProperty("muted", True)
        stats_label.setWordWrap(True)
        stats_label.hide()

        behavior_cards: dict[str, CompactBehaviorStatCard] = {}
        behavior_host = QWidget()
        behavior_host.setStyleSheet("background: transparent;")
        behavior_grid = QGridLayout(behavior_host)
        behavior_grid.setContentsMargins(0, 0, 0, 0)
        behavior_grid.setHorizontalSpacing(4)
        behavior_grid.setVerticalSpacing(5)
        behavior_specs = (
            ("Stand_or_walk", "Stand/Walk"),
            ("Standing_feeding", "Feeding"),
            ("egestion", "Egestion"),
            ("Lie_down_and_rest", "Lie Down"),
        )
        for index, (key, name) in enumerate(behavior_specs):
            card = CompactBehaviorStatCard(name, theme.behavior_color(key, name))
            behavior_cards[key] = card
            behavior_grid.addWidget(card, index, 0)
        behavior_scroll = QScrollArea()
        behavior_scroll.setWidgetResizable(True)
        behavior_scroll.setFrameShape(QFrame.Shape.NoFrame)
        behavior_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        behavior_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        behavior_scroll.setMinimumHeight(96)
        behavior_scroll.setMaximumHeight(228)
        behavior_scroll.setWidget(behavior_host)
        behavior_scroll.setStyleSheet(
            f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollArea > QWidget {{
                background: transparent;
            }}
            QScrollBar:vertical {{
                background: #08110D;
                width: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {theme.COLOR_BORDER};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            """
        )
        side.addWidget(behavior_scroll, 1)
        layout.addWidget(side_panel, 0)

        return MultiCameraSlot(
            slot_id=slot_id,
            panel=panel,
            video_label=video_label,
            side_panel=side_panel,
            bind_label=bind_label,
            mount_button=mount_button,
            analyze_button=analyze_button,
            preview_button=preview_button,
            action_label=action_label,
            stats_label=stats_label,
            behavior_scroll=behavior_scroll,
            behavior_cards=behavior_cards,
        )

    def _clear_grid(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _column_count_for_slots(self, count: int) -> int:
        if count <= 1:
            return 1
        return 2

    def _slot_by_id(self, slot_id: str) -> MultiCameraSlot | None:
        return next((slot for slot in self.slots if slot.slot_id == slot_id), None)

    def _source_text(self, source: Path | int | None) -> str:
        if source is None:
            return ""
        if isinstance(source, Path):
            return source.name
        return f"Camera {source}"
