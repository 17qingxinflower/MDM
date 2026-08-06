from __future__ import annotations

from pathlib import Path

import pandas as pd

from deerui.domain.models import AlertSegment, BehaviorSegment
from deerui.domain.time_utils import seconds_to_clock


class ExportService:
    def export_behavior_segments(self, segments: list[BehaviorSegment], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "Behavior": segment.display_name,
                "Start Time": seconds_to_clock(segment.started_at_seconds),
                "End Time": seconds_to_clock(segment.ended_at_seconds),
                "Duration (s)": round(segment.duration_seconds, 2),
            }
            for segment in segments
        ]
        pd.DataFrame(rows).to_excel(output_path, index=False)
        return output_path

    def export_alert_segments(self, alerts: list[AlertSegment], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "Alert Type": "Long behavior duration"
                if alert.alert_type == "long_duration"
                else "Frequent behavior switching",
                "Behavior": alert.display_name or "Rapid mixed-behavior changes",
                "Time Range": (
                    f"{seconds_to_clock(alert.started_at_seconds)} - "
                    f"{seconds_to_clock(alert.ended_at_seconds)}"
                ),
                "Recommended Action": alert.recommendation,
            }
            for alert in alerts
        ]
        pd.DataFrame(rows).to_excel(output_path, index=False)
        return output_path
