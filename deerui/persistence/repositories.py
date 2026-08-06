from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from deerui.domain.models import AlertSegment, BehaviorSegment, TaskMode
from deerui.persistence.models import (
    AlertSegmentRow,
    AnalysisTaskRow,
    BehaviorSegmentRow,
    DailyBehaviorSummaryRow,
    DeerProfileRow,
)


class DeerProfileRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_or_create(self, deer_code: str) -> DeerProfileRow:
        row = self.session.scalar(select(DeerProfileRow).where(DeerProfileRow.deer_code == deer_code))
        if row is not None:
            return row
        row = DeerProfileRow(deer_code=deer_code)
        self.session.add(row)
        self.session.flush()
        return row


class AnalysisTaskRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        task_code: str,
        deer_profile_id: int,
        mode: TaskMode,
        started_at: datetime,
        camera_source_id: int | None = None,
    ) -> AnalysisTaskRow:
        row = AnalysisTaskRow(
            task_code=task_code,
            deer_profile_id=deer_profile_id,
            camera_source_id=camera_source_id,
            mode=mode.value,
            started_at=started_at,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def complete(
        self,
        task_id: int,
        ended_at: datetime,
        has_alert: bool,
        raw_excel_path: str | None = None,
        alert_excel_path: str | None = None,
    ) -> None:
        row = self.session.get(AnalysisTaskRow, task_id)
        if row is None:
            raise ValueError(f"Analysis task {task_id} does not exist")
        row.ended_at = ended_at
        row.has_alert = has_alert
        row.raw_excel_path = raw_excel_path
        row.alert_excel_path = alert_excel_path

    def complete_interrupted_realtime_tasks(
        self,
        fallback_ended_at: datetime,
    ) -> list[tuple[int, int]]:
        rows = self.session.scalars(
            select(AnalysisTaskRow).where(
                AnalysisTaskRow.mode == TaskMode.REALTIME.value,
                AnalysisTaskRow.ended_at.is_(None),
            )
        ).all()
        recovered: list[tuple[int, int]] = []
        for row in rows:
            max_segment_end = self.session.scalar(
                select(func.max(BehaviorSegmentRow.ended_at_seconds)).where(
                    BehaviorSegmentRow.task_id == row.id
                )
            )
            row.ended_at = self._ended_at_from_segment_seconds(
                max_segment_end,
                fallback_ended_at=fallback_ended_at,
            )
            row.has_alert = bool(row.has_alert)
            recovered.append((row.id, row.deer_profile_id))
        return recovered

    @staticmethod
    def _ended_at_from_segment_seconds(
        segment_seconds: float | None,
        *,
        fallback_ended_at: datetime,
    ) -> datetime:
        if segment_seconds is None:
            return fallback_ended_at
        try:
            value = float(segment_seconds)
        except (TypeError, ValueError):
            return fallback_ended_at
        if value < 946_684_800.0:
            return fallback_ended_at
        return datetime.fromtimestamp(value, tz=timezone.utc)


class SegmentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_behavior_segments(self, task_id: int, segments: list[BehaviorSegment]) -> None:
        for segment in segments:
            self.upsert_behavior_segment(task_id, segment)

    def upsert_behavior_segment(self, task_id: int, segment: BehaviorSegment) -> BehaviorSegmentRow:
        row = self.session.scalar(
            select(BehaviorSegmentRow).where(
                BehaviorSegmentRow.task_id == task_id,
                BehaviorSegmentRow.action_key == segment.action_key,
                BehaviorSegmentRow.started_at_seconds == segment.started_at_seconds,
            )
        )
        if row is None:
            row = BehaviorSegmentRow(
                task_id=task_id,
                action_key=segment.action_key,
                display_name=segment.display_name,
                started_at_seconds=segment.started_at_seconds,
                ended_at_seconds=segment.ended_at_seconds,
                duration_seconds=segment.duration_seconds,
                confidence=segment.confidence,
            )
            self.session.add(row)
            return row

        row.display_name = segment.display_name
        row.ended_at_seconds = segment.ended_at_seconds
        row.duration_seconds = segment.duration_seconds
        row.confidence = segment.confidence
        return row

    def add_alert_segments(self, task_id: int, alerts: list[AlertSegment]) -> None:
        self.session.add_all(
            AlertSegmentRow(
                task_id=task_id,
                alert_type=alert.alert_type,
                action_key=alert.action_key,
                display_name=alert.display_name,
                started_at_seconds=alert.started_at_seconds,
                ended_at_seconds=alert.ended_at_seconds,
                duration_seconds=alert.duration_seconds,
                transition_rate=alert.transition_rate,
                recommendation=alert.recommendation,
            )
            for alert in alerts
        )


class DailySummaryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def rebuild_for_task_date(self, summary_date: date, deer_profile_id: int) -> None:
        self.session.execute(
            delete(DailyBehaviorSummaryRow).where(
                DailyBehaviorSummaryRow.summary_date == summary_date,
                DailyBehaviorSummaryRow.deer_profile_id == deer_profile_id,
            )
        )

        rows = self.session.execute(
            select(
                BehaviorSegmentRow.action_key,
                BehaviorSegmentRow.display_name,
                func.sum(BehaviorSegmentRow.duration_seconds),
                func.count(BehaviorSegmentRow.id),
            )
            .join(AnalysisTaskRow, AnalysisTaskRow.id == BehaviorSegmentRow.task_id)
            .where(
                AnalysisTaskRow.deer_profile_id == deer_profile_id,
                func.date(AnalysisTaskRow.started_at) == summary_date,
            )
            .group_by(BehaviorSegmentRow.action_key, BehaviorSegmentRow.display_name)
        ).all()

        total_duration = sum(float(row[2] or 0) for row in rows) or 1.0
        self.session.add_all(
            DailyBehaviorSummaryRow(
                summary_date=summary_date,
                deer_profile_id=deer_profile_id,
                action_key=str(action_key),
                display_name=str(display_name),
                total_duration_seconds=float(duration or 0),
                frequency=int(frequency or 0),
                percentage=round((float(duration or 0) / total_duration) * 100, 2),
            )
            for action_key, display_name, duration, frequency in rows
        )
