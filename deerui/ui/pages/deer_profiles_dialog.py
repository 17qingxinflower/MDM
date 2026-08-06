from __future__ import annotations

from PySide6.QtWidgets import QLabel, QLineEdit, QListWidget

from deerui.services.profile_service import validate_deer_code
from deerui.ui.themed_dialogs import ThemedDialog


class DeerProfilesDialog(ThemedDialog):
    def __init__(self, existing_codes: list[str] | None = None) -> None:
        super().__init__("Deer Profile Management", width=390)

        title = QLabel("Existing Deer Profiles")
        title.setProperty("muted", True)
        self.body_layout.addWidget(title)

        self.list_widget = QListWidget()
        self.list_widget.setMinimumHeight(210)
        self.list_widget.addItems(existing_codes or [])
        self.body_layout.addWidget(self.list_widget)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Enter ID, e.g. 2.1.6")
        self.body_layout.addWidget(self.input)
        self.list_widget.currentTextChanged.connect(self.input.setText)
        self.list_widget.itemDoubleClicked.connect(lambda item: self.accept())

        add_button = self.add_button("Use / Register", primary=True)
        add_button.clicked.connect(self.accept)
        self.input.returnPressed.connect(self.accept)

    def deer_code(self) -> str:
        return validate_deer_code(self.input.text())
