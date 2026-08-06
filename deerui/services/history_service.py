from __future__ import annotations

from pathlib import Path

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from deerui.domain.models import AlertSegment, BehaviorSegment
from deerui.persistence.models import (
    AlertSegmentRow,
    AnalysisTaskRow,
    BehaviorSegmentRow,
    DeerProfileRow,
)
from deerui.persistence.repositories import DailySummaryRepository
from deerui.services.history_records import HistoryRecord


class HistoryService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_recent_tasks(
        self,
        limit: int = 100,
        *,
        query: str = "",
        has_alert: bool | None = None,
    ) -> list[AnalysisTaskRow]:
        statement = select(AnalysisTaskRow)
        if query:
            like = f"%{query.strip()}%"
            statement = statement.join(DeerProfileRow).where(
                or_(
                    AnalysisTaskRow.task_code.like(like),
                    DeerProfileRow.deer_code.like(like),
                )
            )
        if has_alert is not None:
            statement = statement.where(AnalysisTaskRow.has_alert.is_(has_alert))
        statement = statement.order_by(AnalysisTaskRow.id.desc()).limit(limit)
        return list(self.session.scalars(statement))

    def list_recent_records(
        self,
        limit: int = 100,
        *,
        query: str = "",
        has_alert: bool | None = None,
    ) -> list[HistoryRecord]:
        return [
            self.to_record(row)
            for row in self.list_recent_tasks(limit, query=query, has_alert=has_alert)
        ]

    def get_task(self, task_id: int) -> AnalysisTaskRow | None:
        return self.session.get(AnalysisTaskRow, task_id)

    def delete_task(self, task_id: int, *, delete_files: bool = False) -> bool:
        task = self.get_task(task_id)
        if task is None:
            return False
        paths = [task.raw_excel_path, task.alert_excel_path]
        summary_date = task.started_at.date()
        deer_profile_id = task.deer_profile_id
        self.session.execute(delete(AlertSegmentRow).where(AlertSegmentRow.task_id == task_id))
        self.session.execute(delete(BehaviorSegmentRow).where(BehaviorSegmentRow.task_id == task_id))
        self.session.execute(delete(AnalysisTaskRow).where(AnalysisTaskRow.id == task_id))
        DailySummaryRepository(self.session).rebuild_for_task_date(summary_date, deer_profile_id)
        if delete_files:
            self._delete_existing_files(paths)
        return True

    def delete_tasks(self, task_ids: list[int], *, delete_files: bool = False) -> int:
        deleted = 0
        for task_id in task_ids:
            if self.delete_task(task_id, delete_files=delete_files):
                deleted += 1
        return deleted

    def list_behavior_segments(self, task_id: int) -> list[BehaviorSegment]:
        rows = self.session.scalars(
            select(BehaviorSegmentRow)
            .where(BehaviorSegmentRow.task_id == task_id)
            .order_by(BehaviorSegmentRow.started_at_seconds)
        ).all()
        return [
            BehaviorSegment(
                action_key=row.action_key,
                display_name=row.display_name,
                started_at_seconds=row.started_at_seconds,
                ended_at_seconds=row.ended_at_seconds,
                duration_seconds=row.duration_seconds,
                confidence=row.confidence,
            )
            for row in rows
        ]

    def list_alert_segments(self, task_id: int) -> list[AlertSegment]:
        rows = self.session.scalars(
            select(AlertSegmentRow)
            .where(AlertSegmentRow.task_id == task_id)
            .order_by(AlertSegmentRow.started_at_seconds)
        ).all()
        return [
            AlertSegment(
                alert_type=row.alert_type,
                action_key=row.action_key,
                display_name=row.display_name,
                started_at_seconds=row.started_at_seconds,
                ended_at_seconds=row.ended_at_seconds,
                duration_seconds=row.duration_seconds,
                transition_rate=row.transition_rate,
                recommendation=row.recommendation,
            )
            for row in rows
        ]

    @staticmethod
    def to_record(row: AnalysisTaskRow) -> HistoryRecord:
        date_text = row.started_at.strftime("%Y-%m-%d")
        if row.ended_at is None:
            period_text = f"{row.started_at.strftime('%H:%M')} - Running"
        else:
            period_text = f"{row.started_at.strftime('%H:%M')} - {row.ended_at.strftime('%H:%M')}"
        deer_code = row.deer_profile.deer_code if row.deer_profile else str(row.deer_profile_id)
        mode_text = {
            "realtime": "Realtime Detection",
            "playback": "Video Playback",
            "batch": "Batch Processing",
        }.get(row.mode, row.mode)
        source_text = row.camera_source.name if row.camera_source else "Local video / manual source"
        return HistoryRecord(
            task_code=row.task_code,
            deer_code=deer_code,
            mode_text=mode_text,
            source_text=source_text,
            date_text=date_text,
            period_text=period_text,
            has_alert=row.has_alert,
            raw_excel_path=row.raw_excel_path,
            alert_excel_path=row.alert_excel_path,
            task_id=row.id,
        )

    @staticmethod
    def _delete_existing_files(paths: list[str | None]) -> None:
        for raw_path in paths:
            if not raw_path:
                continue
            path = Path(raw_path)
            if path.exists() and path.is_file():
                path.unlink()
