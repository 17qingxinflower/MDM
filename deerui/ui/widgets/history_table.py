from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from deerui.services.history_records import HistoryRecord
from deerui.ui import theme


ROW_EVEN = "#0F1713"
ROW_ODD = "#131D18"
ROW_BORDER = "#203029"
DEER_TAG_BG = "#0B3D31"
DEER_TAG_TEXT = "#55E0B2"


@dataclass(frozen=True)
class _TextItem:
    value: str

    def text(self) -> str:
        return self.value


class HistoryTable(QFrame):
    view_dashboard_requested = Signal(object)
    open_raw_requested = Signal(object)
    open_alert_requested = Signal(object)
    delete_requested = Signal(object)

    headers = ["Select", "Job ID", "Deer ID", "Date", "Monitoring Period", "Alert Status", "Actions"]

    def __init__(self) -> None:
        super().__init__()
        self.setProperty("panel", True)
        self.setStyleSheet(
            f"QFrame[panel='true'] {{ background: {theme.COLOR_PANEL}; border: 1px solid {theme.COLOR_BORDER};"
            "border-radius: 12px; }}"
        )
        self._records: list[HistoryRecord] = []
        self._cell_values: list[list[str]] = []
        self._row_checks: list[QCheckBox] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(0)

        header = QFrame()
        header.setFixedHeight(44)
        header.setStyleSheet(f"background: {theme.COLOR_PANEL}; border: none;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 10, 0)
        header_layout.setSpacing(10)
        for index, text in enumerate(self.headers):
            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f"color: #8CA39A; font-size: 12px; font-weight: 700;")
            header_layout.addWidget(label, self._column_weight(index))
        layout.addWidget(header)

        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background: {ROW_BORDER};")
        layout.addWidget(line)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet(
            f"""
            QScrollArea {{
                background: {theme.COLOR_PANEL};
                border: none;
            }}
            QScrollArea > QWidget > QWidget {{
                background: {theme.COLOR_PANEL};
            }}
            QScrollBar:vertical {{
                background: #0D1411;
                width: 10px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical {{
                background: #26362E;
                border-radius: 4px;
                min-height: 24px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            """
        )
        self.content = QWidget()
        self.content.setStyleSheet(f"background: {theme.COLOR_PANEL};")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 5, 0, 0)
        self.content_layout.setSpacing(0)
        self.content_layout.addStretch(1)
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll, 1)

    def rowCount(self) -> int:
        return len(self._cell_values)

    def item(self, row: int, column: int) -> _TextItem:
        return _TextItem(self._cell_values[row][column])

    def set_all_checked(self, checked: bool) -> None:
        for checkbox in self._row_checks:
            checkbox.setChecked(checked)

    def selected_task_ids(self) -> list[int]:
        ids: list[int] = []
        for record, checkbox in zip(self._records, self._row_checks, strict=False):
            if checkbox.isChecked() and record.task_id is not None:
                ids.append(record.task_id)
        return ids

    def set_empty_message(self) -> None:
        self._records = []
        self._cell_values = []
        self._row_checks = []
        self._clear_rows()
        empty = QLabel("No matching records found")
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty.setMinimumHeight(160)
        empty.setStyleSheet(f"background: {theme.COLOR_PANEL}; color: {theme.COLOR_TEXT_SUB}; font-size: 14px;")
        self.content_layout.insertWidget(0, empty)

    def set_records(self, records: list[HistoryRecord]) -> None:
        self._records = records
        self._cell_values = []
        self._row_checks = []
        self._clear_rows()
        if not records:
            self.set_empty_message()
            return

        for index, record in enumerate(records):
            self._cell_values.append(self._record_values(record))
            self.content_layout.insertWidget(index, self._build_row(index, record))

    def _clear_rows(self) -> None:
        while self.content_layout.count() > 1:
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _record_values(self, record: HistoryRecord) -> list[str]:
        return [
            record.task_code,
            record.deer_code,
            record.mode_text,
            record.source_text,
            record.date_text,
            record.period_text,
            "Alert" if record.has_alert else "Normal",
            "Raw / Alerts" if record.raw_excel_path or record.alert_excel_path else "-",
        ]

    def _build_row(self, index: int, record: HistoryRecord) -> QFrame:
        row = QFrame()
        row.setFixedHeight(46)
        row.setStyleSheet(
            f"QFrame {{ background: {ROW_EVEN if index % 2 == 0 else ROW_ODD};"
            f"border-bottom: 1px solid {ROW_BORDER}; }}"
        )
        layout = QHBoxLayout(row)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(10)

        checkbox = QCheckBox()
        checkbox.setStyleSheet(
            f"""
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #2A3A32;
                background: #0B120F;
            }}
            QCheckBox::indicator:checked {{
                background: {theme.COLOR_PRIMARY};
                border-color: {theme.COLOR_PRIMARY};
            }}
            """
        )
        self._row_checks.append(checkbox)
        layout.addWidget(checkbox, self._column_weight(0), Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._label(record.task_code, bold=True), self._column_weight(1))
        layout.addWidget(self._deer_tag(record.deer_code), self._column_weight(2))
        layout.addWidget(self._label(record.date_text, bold=True), self._column_weight(3))
        layout.addWidget(self._label(record.period_text, muted=True), self._column_weight(4))

        status = "* Alert" if record.has_alert else "* Normal"
        status_color = theme.COLOR_DANGER if record.has_alert else theme.COLOR_PRIMARY
        layout.addWidget(self._label(status, color=status_color, bold=True), self._column_weight(5))
        layout.addWidget(self._actions(record), self._column_weight(6))

        return row

    def _label(
        self,
        text: str,
        *,
        muted: bool = False,
        bold: bool = False,
        color: str | None = None,
    ) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        label.setStyleSheet(
            f"background: transparent; color: {color or (theme.COLOR_TEXT_SUB if muted else theme.COLOR_TEXT_MAIN)};"
            f"font-size: 12px; font-weight: {'700' if bold else '400'};"
        )
        return label

    def _deer_tag(self, deer_code: str) -> QFrame:
        tag = QFrame()
        tag.setFixedHeight(24)
        tag.setStyleSheet(f"background: {DEER_TAG_BG}; border: 1px solid #155B49; border-radius: 4px;")
        layout = QHBoxLayout(tag)
        layout.setContentsMargins(8, 0, 8, 0)
        label = QLabel(deer_code)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"background: transparent; color: {DEER_TAG_TEXT}; font-size: 11px; font-weight: 800;")
        layout.addWidget(label)
        return tag

    def _actions(self, record: HistoryRecord) -> QWidget:
        actions = QWidget()
        actions.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(actions)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for text, danger, signal in [
            ("Dashboard", False, self.view_dashboard_requested),
            ("Raw Log", False, self.open_raw_requested),
            ("Alerts", record.has_alert, self.open_alert_requested),
            ("Delete", True, self.delete_requested),
        ]:
            if text == "Alerts" and not record.has_alert:
                continue
            button = QPushButton(text)
            button.setObjectName(f"history{text}Button")
            button.setFixedHeight(28)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(
                lambda checked=False, task_id=record.task_id, requested=signal: requested.emit(task_id)
            )
            if text == "Dashboard":
                button.setStyleSheet(
                    f"background: #1A2922; color: {theme.COLOR_TEXT_MAIN}; border: 1px solid #2A3A32;"
                    "border-radius: 6px; padding: 4px 8px; text-align: center; font-weight: 700;"
                )
            else:
                color = theme.COLOR_DANGER if danger else "#BFD0C8"
                border = theme.COLOR_DANGER if danger else "#2A3A32"
                button.setStyleSheet(
                    f"background: #101915; color: {color}; border: 1px solid {border};"
                    "border-radius: 6px; padding: 4px 8px; text-align: center;"
                )
            layout.addWidget(button)
        return actions

    def _column_weight(self, index: int) -> int:
        return [4, 15, 10, 12, 16, 10, 23][index]
