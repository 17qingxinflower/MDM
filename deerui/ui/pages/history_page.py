from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from deerui.services.history_records import HistoryRecord
from deerui.ui import theme
from deerui.ui.widgets.history_table import HistoryTable


CONTROL_STYLE = f"""
QLineEdit, QComboBox {{
    background: #121A16;
    color: {theme.COLOR_TEXT_MAIN};
    border: 1px solid #26362E;
    border-radius: 7px;
    padding: 8px 10px;
}}
QLineEdit:focus, QComboBox:focus {{
    border-color: {theme.COLOR_PRIMARY};
}}
QLineEdit::placeholder {{
    color: #7B8D84;
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background: #111814;
    color: {theme.COLOR_TEXT_MAIN};
    border: 1px solid {theme.COLOR_BORDER};
    selection-background-color: #173A2F;
}}
QCheckBox {{
    color: {theme.COLOR_TEXT_MAIN};
    spacing: 8px;
}}
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


class HistoryPage(QWidget):
    filter_changed = Signal()
    batch_delete_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet(CONTROL_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(14)

        title_group = QVBoxLayout()
        title = QLabel("History Library")
        title.setStyleSheet("color: white; font-size: 22px; font-weight: 800;")
        subtitle = QLabel("Search and manage recorded musk deer behavior analysis tasks")
        subtitle.setProperty("muted", True)
        title_group.addWidget(title)
        title_group.addWidget(subtitle)
        top_bar.addLayout(title_group, 1)

        self.select_all = QCheckBox("Select All")
        self.select_all.toggled.connect(self._toggle_select_all)

        self.batch_delete_button = QPushButton("Batch Delete")
        self.batch_delete_button.setObjectName("historyBatchDeleteButton")
        self.batch_delete_button.setFixedWidth(92)
        self.batch_delete_button.setStyleSheet(
            f"background: transparent; color: {theme.COLOR_DANGER}; border: 1px solid {theme.COLOR_DANGER};"
            "border-radius: 8px; padding: 8px 10px; text-align: center;"
        )
        self.batch_delete_button.clicked.connect(self.batch_delete_requested.emit)

        self.status_filter = QComboBox()
        self.status_filter.addItems(["Alert status (all)", "Normal", "Alert"])
        self.status_filter.setFixedWidth(150)
        self.status_filter.currentIndexChanged.connect(self.filter_changed.emit)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search deer ID or job ID...")
        self.search.setFixedWidth(250)
        self.search.returnPressed.connect(self.filter_changed.emit)

        self.search_button = QPushButton("Search")
        self.search_button.setFixedWidth(72)
        self.search_button.setStyleSheet(
            f"background: #101915; color: {theme.COLOR_TEXT_MAIN}; border: 1px solid #24342C;"
            "border-radius: 8px; padding: 8px 10px; text-align: center; font-weight: 700;"
        )
        self.search_button.clicked.connect(self.filter_changed.emit)

        for widget in [
            self.select_all,
            self.batch_delete_button,
            self.status_filter,
            self.search,
            self.search_button,
        ]:
            top_bar.addWidget(widget)

        layout.addLayout(top_bar)

        self.table = HistoryTable()
        self.table.set_empty_message()
        layout.addWidget(self.table, 1)

    def set_records(self, records: list[HistoryRecord]) -> None:
        self.table.set_records(records)

    def query_text(self) -> str:
        return self.search.text().strip()

    def has_alert_filter(self) -> bool | None:
        text = self.status_filter.currentText()
        if text == "Alert":
            return True
        if text == "Normal":
            return False
        return None

    def selected_task_ids(self) -> list[int]:
        return self.table.selected_task_ids()

    def _toggle_select_all(self, checked: bool) -> None:
        self.table.set_all_checked(checked)
