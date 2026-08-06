from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HistoryRecord:
    task_code: str
    deer_code: str
    mode_text: str
    source_text: str
    date_text: str
    period_text: str
    has_alert: bool
    raw_excel_path: str | None = None
    alert_excel_path: str | None = None
    task_id: int | None = None
