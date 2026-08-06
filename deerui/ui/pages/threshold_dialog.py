from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtWidgets import QDoubleSpinBox, QLabel

from deerui.domain.models import DEFAULT_BEHAVIOR_CLASSES
from deerui.services.app_state import ThresholdSettings
from deerui.ui.themed_dialogs import ThemedDialog


class ThresholdDialog(ThemedDialog):
    def __init__(
        self,
        settings: ThresholdSettings | None = None,
        action_names: Iterable[str] | None = None,
    ) -> None:
        super().__init__("Alert Settings", width=440)
        settings = settings or ThresholdSettings()
        names = list(action_names or self._default_action_names())

        self.body_layout.addWidget(self._label("Long behavior-duration alert (seconds)"))
        self.min_duration = QDoubleSpinBox()
        self.min_duration.setRange(0.1, 86400.0)
        self.min_duration.setValue(settings.min_duration_seconds)
        self.body_layout.addWidget(self.min_duration)

        self.body_layout.addWidget(self._label("Rapid transition alert (events/minute)"))
        self.max_transition = QDoubleSpinBox()
        self.max_transition.setRange(0.1, 999.0)
        self.max_transition.setValue(settings.max_transitions_per_minute)
        self.body_layout.addWidget(self.max_transition)

        self.body_layout.addWidget(self._label("Per-behavior duration thresholds (seconds)"))
        self.action_threshold_inputs: dict[str, QDoubleSpinBox] = {}
        for action_name in names:
            self.body_layout.addWidget(self._label(action_name))
            spin = QDoubleSpinBox()
            spin.setRange(0.1, 86400.0)
            spin.setValue(settings.action_thresholds.get(action_name, settings.min_duration_seconds))
            self.action_threshold_inputs[action_name] = spin
            self.body_layout.addWidget(spin)

        save_button = self.add_button("Save Settings", primary=True)
        save_button.clicked.connect(self.accept)

    def settings(self) -> ThresholdSettings:
        return ThresholdSettings(
            min_duration_seconds=float(self.min_duration.value()),
            max_transitions_per_minute=float(self.max_transition.value()),
            action_thresholds={
                action_name: float(spin.value())
                for action_name, spin in self.action_threshold_inputs.items()
            },
        )

    def _label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("muted", True)
        return label

    def _default_action_names(self) -> list[str]:
        return sorted({behavior.display_zh for behavior in DEFAULT_BEHAVIOR_CLASSES.values()})
