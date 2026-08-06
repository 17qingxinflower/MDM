from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class DeerProfileRow(Base):
    __tablename__ = "deer_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    deer_code: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class CameraSourceRow(Base):
    __tablename__ = "camera_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AnalysisTaskRow(Base):
    __tablename__ = "analysis_tasks"
    __table_args__ = (CheckConstraint("mode IN ('realtime', 'playback', 'batch')"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    task_code: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    deer_profile_id: Mapped[int] = mapped_column(ForeignKey("deer_profiles.id"), nullable=False)
    camera_source_id: Mapped[Optional[int]] = mapped_column(ForeignKey("camera_sources.id"))
    mode: Mapped[str] = mapped_column(String(20), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    has_alert: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    raw_excel_path: Mapped[Optional[str]] = mapped_column(Text)
    alert_excel_path: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    deer_profile: Mapped[DeerProfileRow] = relationship()
    camera_source: Mapped[Optional[CameraSourceRow]] = relationship()


class BehaviorSegmentRow(Base):
    __tablename__ = "behavior_segments"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("analysis_tasks.id", ondelete="CASCADE"))
    action_key: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    started_at_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    ended_at_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AlertSegmentRow(Base):
    __tablename__ = "alert_segments"
    __table_args__ = (CheckConstraint("alert_type IN ('long_duration', 'high_transition')"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("analysis_tasks.id", ondelete="CASCADE"))
    alert_type: Mapped[str] = mapped_column(String(40), nullable=False)
    action_key: Mapped[Optional[str]] = mapped_column(Text)
    display_name: Mapped[Optional[str]] = mapped_column(Text)
    started_at_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    ended_at_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    transition_rate: Mapped[Optional[float]] = mapped_column(Float)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DailyBehaviorSummaryRow(Base):
    __tablename__ = "daily_behavior_summary"
    __table_args__ = (
        UniqueConstraint("summary_date", "deer_profile_id", "action_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    summary_date: Mapped[date] = mapped_column(Date, nullable=False)
    deer_profile_id: Mapped[int] = mapped_column(ForeignKey("deer_profiles.id"), nullable=False)
    action_key: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    total_duration_seconds: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    frequency: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    percentage: Mapped[float] = mapped_column(Float, default=0, nullable=False)
