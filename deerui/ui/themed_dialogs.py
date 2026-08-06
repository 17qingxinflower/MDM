from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from deerui.ui import theme


THEMED_DIALOG_STYLESHEET = f"""
QDialog {{
    background: transparent;
}}
QFrame#dialogPanel {{
    background: {theme.COLOR_PANEL};
    border: 1px solid {theme.COLOR_BORDER};
    border-radius: 12px;
}}
QFrame#dialogTitleBar {{
    background: #0E1713;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
    border-bottom: 1px solid {theme.COLOR_BORDER};
}}
QLabel#dialogTitle {{
    color: {theme.COLOR_TEXT_MAIN};
    font-size: 15px;
    font-weight: 800;
}}
QLabel#dialogText {{
    color: {theme.COLOR_TEXT_MAIN};
    font-size: 13px;
}}
QLabel[muted="true"] {{
    color: {theme.COLOR_TEXT_SUB};
}}
QLineEdit, QSpinBox, QComboBox {{
    background: #121A16;
    color: {theme.COLOR_TEXT_MAIN};
    border: 1px solid #26362E;
    border-radius: 7px;
    padding: 8px 10px;
    min-height: 24px;
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
    border-color: {theme.COLOR_PRIMARY};
}}
QLineEdit::placeholder {{
    color: #7B8D84;
}}
QComboBox QAbstractItemView {{
    background: #0F1713;
    color: {theme.COLOR_TEXT_MAIN};
    border: 1px solid #26362E;
    selection-background-color: #0B3D31;
    selection-color: {theme.COLOR_PRIMARY};
}}
QPushButton {{
    background: #101915;
    color: {theme.COLOR_TEXT_MAIN};
    border: 1px solid #24342C;
    border-radius: 8px;
    padding: 8px 14px;
    text-align: center;
    min-width: 78px;
}}
QPushButton:hover {{
    background: #17231D;
}}
QPushButton#primaryButton {{
    background: {theme.COLOR_PRIMARY};
    color: #04130D;
    border-color: {theme.COLOR_PRIMARY};
    font-weight: 800;
}}
QPushButton#dangerButton {{
    background: transparent;
    color: {theme.COLOR_DANGER};
    border-color: {theme.COLOR_DANGER};
}}
QPushButton#closeButton {{
    background: transparent;
    color: {theme.COLOR_TEXT_SUB};
    border: none;
    border-radius: 0;
    min-width: 30px;
    padding: 4px;
    font-size: 18px;
}}
QPushButton#closeButton:hover {{
    background: {theme.COLOR_DANGER};
    color: white;
}}
QListWidget {{
    background: #0F1713;
    color: {theme.COLOR_TEXT_MAIN};
    border: 1px solid #26362E;
    border-radius: 7px;
    padding: 6px;
}}
QListWidget::item {{
    padding: 8px;
    border-radius: 5px;
}}
QListWidget::item:selected {{
    background: #0B3D31;
    color: {theme.COLOR_PRIMARY};
}}
QDoubleSpinBox {{
    background: #121A16;
    color: {theme.COLOR_TEXT_MAIN};
    border: 1px solid #26362E;
    border-radius: 7px;
    padding: 8px 10px;
    min-height: 24px;
}}
QDoubleSpinBox:focus {{
    border-color: {theme.COLOR_PRIMARY};
}}
"""


class ThemedDialog(QDialog):
    def __init__(self, title: str, parent: QWidget | None = None, width: int = 360) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumWidth(width)
        self.setStyleSheet(THEMED_DIALOG_STYLESHEET)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        panel = QFrame()
        panel.setObjectName("dialogPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 16)
        panel_layout.setSpacing(0)
        root.addWidget(panel)

        title_bar = QFrame()
        title_bar.setObjectName("dialogTitleBar")
        title_bar.setFixedHeight(42)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(16, 0, 8, 0)
        title_label = QLabel(title)
        title_label.setObjectName("dialogTitle")
        close_button = QPushButton("x")
        close_button.setObjectName("closeButton")
        close_button.clicked.connect(self.reject)
        title_layout.addWidget(title_label, 1)
        title_layout.addWidget(close_button)
        panel_layout.addWidget(title_bar)

        self.body_layout = QVBoxLayout()
        self.body_layout.setContentsMargins(18, 16, 18, 8)
        self.body_layout.setSpacing(10)
        panel_layout.addLayout(self.body_layout)

        self.button_layout = QHBoxLayout()
        self.button_layout.setContentsMargins(18, 4, 18, 0)
        self.button_layout.setSpacing(10)
        self.button_layout.addStretch(1)
        panel_layout.addLayout(self.button_layout)

    def add_button(self, text: str, *, primary: bool = False, danger: bool = False) -> QPushButton:
        button = QPushButton(text)
        if primary:
            button.setObjectName("primaryButton")
        elif danger:
            button.setObjectName("dangerButton")
        self.button_layout.addWidget(button)
        return button


def ask_text(
    parent: QWidget | None,
    title: str,
    label: str,
    *,
    text: str = "",
    placeholder: str = "",
) -> tuple[str, bool]:
    dialog = ThemedDialog(title, parent)
    prompt = QLabel(label)
    prompt.setObjectName("dialogText")
    edit = QLineEdit()
    edit.setText(text)
    edit.setPlaceholderText(placeholder)
    dialog.body_layout.addWidget(prompt)
    dialog.body_layout.addWidget(edit)
    cancel = dialog.add_button("Cancel")
    ok = dialog.add_button("OK", primary=True)
    cancel.clicked.connect(dialog.reject)
    ok.clicked.connect(dialog.accept)
    edit.returnPressed.connect(dialog.accept)
    accepted = dialog.exec() == QDialog.DialogCode.Accepted
    return edit.text(), accepted


def ask_choice_text(
    parent: QWidget | None,
    title: str,
    label: str,
    *,
    choices: list[str] | tuple[str, ...] = (),
    text: str = "",
    placeholder: str = "",
) -> tuple[str, bool]:
    dialog = ThemedDialog(title, parent)
    prompt = QLabel(label)
    prompt.setObjectName("dialogText")
    combo = QComboBox()
    combo.setEditable(True)
    combo.addItems(list(dict.fromkeys(str(choice) for choice in choices if str(choice).strip())))
    if text:
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.setEditText(text)
    combo.lineEdit().setPlaceholderText(placeholder)
    dialog.body_layout.addWidget(prompt)
    dialog.body_layout.addWidget(combo)
    cancel = dialog.add_button("Cancel")
    ok = dialog.add_button("OK", primary=True)
    cancel.setText("Cancel")
    ok.setText("OK")
    cancel.clicked.connect(dialog.reject)
    ok.clicked.connect(dialog.accept)
    combo.lineEdit().returnPressed.connect(dialog.accept)
    accepted = dialog.exec() == QDialog.DialogCode.Accepted
    return combo.currentText(), accepted


def ask_int(
    parent: QWidget | None,
    title: str,
    label: str,
    *,
    value: int,
    minimum: int,
    maximum: int,
) -> tuple[int, bool]:
    dialog = ThemedDialog(title, parent)
    prompt = QLabel(label)
    prompt.setObjectName("dialogText")
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    dialog.body_layout.addWidget(prompt)
    dialog.body_layout.addWidget(spin)
    cancel = dialog.add_button("Cancel")
    ok = dialog.add_button("OK", primary=True)
    cancel.clicked.connect(dialog.reject)
    ok.clicked.connect(dialog.accept)
    accepted = dialog.exec() == QDialog.DialogCode.Accepted
    return int(spin.value()), accepted


def show_message(parent: QWidget | None, title: str, message: str, *, danger: bool = False) -> None:
    dialog = ThemedDialog(title, parent)
    label = QLabel(message)
    label.setObjectName("dialogText")
    label.setWordWrap(True)
    dialog.body_layout.addWidget(label)
    ok = dialog.add_button("OK", primary=not danger, danger=danger)
    ok.clicked.connect(dialog.accept)
    dialog.exec()


def show_info(parent: QWidget | None, title: str, message: str) -> None:
    show_message(parent, title, message)


def show_warning(parent: QWidget | None, title: str, message: str) -> None:
    show_message(parent, title, message, danger=True)


def ask_confirm(
    parent: QWidget | None,
    title: str,
    message: str,
    *,
    danger: bool = False,
) -> bool:
    dialog = ThemedDialog(title, parent)
    label = QLabel(message)
    label.setObjectName("dialogText")
    label.setWordWrap(True)
    dialog.body_layout.addWidget(label)
    cancel = dialog.add_button("Cancel")
    ok = dialog.add_button("OK", primary=not danger, danger=danger)
    cancel.clicked.connect(dialog.reject)
    ok.clicked.connect(dialog.accept)
    return dialog.exec() == QDialog.DialogCode.Accepted


def open_file_name(
    parent: QWidget | None,
    title: str,
    name_filter: str,
) -> str:
    path, _ = QFileDialog.getOpenFileName(parent, title, "", name_filter)
    return path
