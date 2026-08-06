from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from deerui.domain.models import AlertSegment, BehaviorSegment
from deerui.ui import theme
from deerui.ui.themed_dialogs import ThemedDialog


try:  # pragma: no cover - availability is covered by runtime smoke/tests
    from matplotlib import rcParams
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
except Exception:  # noqa: BLE001
    Figure = None  # type: ignore[assignment]
    FigureCanvas = None  # type: ignore[assignment]
    MaxNLocator = None  # type: ignore[assignment]
    rcParams = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ChartSegment:
    action_key: str
    display_name: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float


class ChartDialog(ThemedDialog):
    def __init__(
        self,
        task_code: str,
        segments: list[BehaviorSegment],
        alerts: list[AlertSegment],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(f"Behavior Analytics Dashboard - {task_code}", parent=parent, width=1120)
        self.setMinimumHeight(760)
        self._canvases: list[QWidget] = []
        self._figures: list[Any] = []
        self._mpl_callback_ids: list[tuple[Any, int]] = []
        self.segments = self._normalise_segments(segments)
        self.alerts = alerts

        if Figure is None or FigureCanvas is None:
            self.body_layout.addWidget(self._label("matplotlib is not installed; the dashboard cannot be generated."))
        elif not self.segments:
            self.body_layout.addWidget(self._label("No behavior segments were recorded for this task."))
        else:
            self._configure_matplotlib()
            self.body_layout.addWidget(self._dashboard_widget())

        close = self.add_button("Close", primary=True)
        close.clicked.connect(self.accept)

    def _dashboard_widget(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(
            f"""
            QScrollArea {{
                border: none;
                background: {theme.COLOR_PANEL};
            }}
            QScrollArea > QWidget > QWidget {{
                background: {theme.COLOR_PANEL};
            }}
            """
        )
        scroll.viewport().setStyleSheet(f"background: {theme.COLOR_PANEL};")

        content = QWidget()
        content.setStyleSheet(f"background: {theme.COLOR_PANEL};")
        layout = QGridLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(20)
        layout.setVerticalSpacing(20)

        layout.addWidget(self._chart_panel("Behavior Timeline", self._timeline_figure()), 0, 0, 1, 2)
        layout.addWidget(self._chart_panel("Behavior Frequency", self._frequency_figure()), 1, 0)
        layout.addWidget(self._chart_panel("Time Share", self._proportion_figure()), 1, 1)

        if self.alerts:
            layout.addWidget(self._alert_panel(), 2, 0, 1, 2)

        scroll.setWidget(content)
        return scroll

    def _chart_panel(self, title: str, figure) -> QFrame:
        panel = QFrame()
        panel.setProperty("panel", True)
        panel.setStyleSheet(
            f"""
            QFrame[panel="true"] {{
                background: {theme.COLOR_PANEL};
                border: 1px solid {theme.COLOR_BORDER};
                border-radius: 10px;
            }}
            QLabel {{
                border: none;
                background: transparent;
            }}
            """
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)
        label = QLabel(title)
        label.setStyleSheet(f"color: {theme.COLOR_TEXT_MAIN}; font-weight: 800; font-size: 15px;")
        canvas = FigureCanvas(figure)
        self._canvases.append(canvas)
        canvas.setMinimumHeight(230)
        layout.addWidget(label)
        layout.addWidget(canvas)
        return panel

    def _timeline_figure(self):
        figure = self._figure(10.8, 2.35, left=0.08, right=0.975, bottom=0.22, top=0.90)
        ax = figure.add_subplot(111)
        first_start = min(segment.start_seconds for segment in self.segments)
        total_end = max(segment.end_seconds for segment in self.segments)
        spans = [
            (
                segment.start_seconds - first_start,
                max(segment.duration_seconds, 0.05),
                segment,
            )
            for segment in self.segments
        ]
        colors = [theme.behavior_color(segment.action_key, segment.display_name) for _, _, segment in spans]
        bars = ax.broken_barh(
            [(start, duration) for start, duration, _ in spans],
            (0.35, 0.38),
            facecolors=colors,
            edgecolors="none",
            alpha=0.95,
        )
        ax.set_xlim(0, max(total_end - first_start, 1.0))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Time (s)", color="#C7D2CC", labelpad=8)
        ax.grid(axis="x", color="#314139", alpha=0.45, linewidth=0.8)
        self._style_axis(ax)
        annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(14, -54),
            textcoords="offset points",
            bbox={
                "boxstyle": "round,pad=0.55",
                "fc": "#F8FAFC",
                "ec": "#8B5CF6",
                "lw": 1.2,
            },
            arrowprops={"arrowstyle": "->", "color": "#F8FAFC", "lw": 1},
            fontsize=9,
            color="#111827",
            ha="left",
            va="top",
        )
        annotation.set_clip_on(False)
        annotation.set_annotation_clip(False)
        annotation.set_zorder(20)
        annotation.set_visible(False)

        def on_hover(event) -> None:
            annotation.set_visible(False)
            if event.inaxes is not ax or event.xdata is None:
                figure.canvas.draw_idle()
                return
            for start, duration, segment in spans:
                if start <= event.xdata <= start + duration:
                    x_min, x_max = ax.get_xlim()
                    relative_x = (event.xdata - x_min) / max(x_max - x_min, 1e-6)
                    if relative_x > 0.72:
                        annotation.set_position((-112, -54))
                        annotation.set_ha("right")
                    else:
                        annotation.set_position((14, -54))
                        annotation.set_ha("left")
                    annotation.xy = (event.xdata, 0.62)
                    annotation.set_text(
                        "Behavior: {action}\nDuration: {duration:.1f}s\nStart: {start_text}\nEnd: {end_text}".format(
                            action=segment.display_name,
                            duration=segment.duration_seconds,
                            start_text=self._format_relative_time(start),
                            end_text=self._format_relative_time(start + duration),
                        )
                    )
                    annotation.set_visible(True)
                    break
            figure.canvas.draw_idle()

        callback_id = figure.canvas.mpl_connect("motion_notify_event", on_hover)
        self._mpl_callback_ids.append((figure, callback_id))
        return figure

    def done(self, result: int) -> None:
        self._dispose_charts()
        super().done(result)

    def _dispose_charts(self) -> None:
        for figure, callback_id in self._mpl_callback_ids:
            canvas = getattr(figure, "canvas", None)
            if canvas is None:
                continue
            try:
                canvas.mpl_disconnect(callback_id)
            except Exception:  # noqa: BLE001 - matplotlib may already be tearing down.
                pass
        self._mpl_callback_ids.clear()

        for canvas in self._canvases:
            try:
                canvas.close()
                canvas.setParent(None)
                canvas.deleteLater()
            except RuntimeError:
                pass
        self._canvases.clear()

        for figure in self._figures:
            try:
                figure.clear()
            except Exception:  # noqa: BLE001 - keep dialog teardown best-effort.
                pass
        self._figures.clear()

    def _frequency_figure(self):
        figure = self._figure(5.2, 2.9, left=0.14, right=0.96, bottom=0.24, top=0.88)
        ax = figure.add_subplot(111)
        counts = Counter((segment.action_key, segment.display_name) for segment in self.segments)
        actions = list(counts.keys())
        values = [counts[action] for action in actions]
        labels = [display_name for _, display_name in actions]
        x_positions = list(range(len(labels)))
        bars = ax.bar(
            x_positions,
            values,
            color=[theme.behavior_color(action_key, display_name) for action_key, display_name in actions],
            width=0.46,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, color="#C7D2CC", fontsize=10)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(value),
                ha="center",
                va="bottom",
                color=theme.COLOR_TEXT_MAIN,
                fontsize=9,
                fontweight="bold",
            )
        ax.set_ylabel("Count", color="#9CA3AF", labelpad=10)
        ax.set_ylim(0, max(values) + 1)
        ax.margins(x=0.18)
        if MaxNLocator is not None:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="x", rotation=0)
        self._style_axis(ax)
        return figure

    def _proportion_figure(self):
        figure = self._figure(5.2, 2.9, left=0.03, right=0.95, bottom=0.08, top=0.94)
        ax = figure.add_subplot(111)
        durations: dict[tuple[str, str], float] = defaultdict(float)
        for segment in self.segments:
            durations[(segment.action_key, segment.display_name)] += segment.duration_seconds
        actions = list(durations.keys())
        values = [durations[action] for action in actions]
        wedges, _, autotexts = ax.pie(
            values,
            labels=None,
            autopct="%1.1f%%",
            startangle=90,
            colors=[theme.behavior_color(action_key, display_name) for action_key, display_name in actions],
            pctdistance=0.72,
            wedgeprops={"width": 0.38, "edgecolor": theme.COLOR_PANEL, "linewidth": 2},
            textprops={"color": theme.COLOR_TEXT_MAIN, "fontsize": 8},
        )
        for autotext in autotexts:
            autotext.set_color(theme.COLOR_TEXT_MAIN)
            autotext.set_fontweight("bold")
        legend_labels = [
            f"{display_name}  {duration:.1f}s" for (_, display_name), duration in zip(actions, values)
        ]
        ax.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.92, 0.5),
            frameon=False,
            labelcolor=theme.COLOR_TEXT_MAIN,
            fontsize=9,
        )
        ax.set_aspect("equal")
        ax.set_facecolor(theme.COLOR_PANEL)
        return figure

    def _alert_panel(self) -> QFrame:
        panel = QFrame()
        panel.setProperty("panel", True)
        layout = QVBoxLayout(panel)
        title = QLabel("Alert Markers")
        title.setStyleSheet(f"color: {theme.COLOR_WARNING}; font-weight: 800;")
        body = QLabel("\n".join(f"- {alert.display_name or alert.alert_type}: {alert.recommendation}" for alert in self.alerts[:6]))
        body.setWordWrap(True)
        body.setStyleSheet(f"color: {theme.COLOR_TEXT_MAIN};")
        layout.addWidget(title)
        layout.addWidget(body)
        return panel

    def _label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("dialogText")
        label.setWordWrap(True)
        label.setMinimumHeight(260)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    @staticmethod
    def _normalise_segments(segments: list[BehaviorSegment]) -> list[ChartSegment]:
        result: list[ChartSegment] = []
        for segment in sorted(segments, key=lambda item: item.started_at_seconds):
            duration = max(segment.duration_seconds, segment.ended_at_seconds - segment.started_at_seconds)
            if duration <= 0:
                continue
            result.append(
                ChartSegment(
                    action_key=segment.action_key,
                    display_name=segment.display_name or segment.action_key,
                    start_seconds=segment.started_at_seconds,
                    end_seconds=segment.started_at_seconds + duration,
                    duration_seconds=duration,
                )
            )
        return result

    @staticmethod
    def _configure_matplotlib() -> None:
        if rcParams is None:
            return
        rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False

    def _figure(
        self,
        width: float,
        height: float,
        *,
        left: float = 0.08,
        right: float = 0.94,
        top: float = 0.92,
        bottom: float = 0.18,
    ):
        figure = Figure(figsize=(width, height), dpi=100, facecolor=theme.COLOR_PANEL)
        figure.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
        self._figures.append(figure)
        return figure

    @staticmethod
    def _style_axis(ax) -> None:
        ax.set_facecolor(theme.COLOR_PANEL)
        ax.tick_params(colors="#AAB7B0", labelsize=10, pad=6)
        ax.xaxis.label.set_color("#C7D2CC")
        ax.yaxis.label.set_color("#C7D2CC")
        for spine in ax.spines.values():
            spine.set_color("#55635C")
            spine.set_linewidth(0.8)

    @staticmethod
    def _format_relative_time(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remain = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{remain:02d}"
