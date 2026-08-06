from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamCapacityPolicy:
    worker_count: int
    reliable_streams_per_worker: int

    def __post_init__(self) -> None:
        if self.worker_count <= 0:
            raise ValueError("worker_count must be greater than 0")
        if self.reliable_streams_per_worker <= 0:
            raise ValueError("reliable_streams_per_worker must be greater than 0")

    @property
    def reliable_capacity(self) -> int:
        return self.worker_count * self.reliable_streams_per_worker


@dataclass(frozen=True)
class CapacityGateDecision:
    allowed: bool
    active_stream_count: int
    requested_stream_count: int
    reliable_capacity: int
    remaining_after_start: int
    reason_zh: str


def evaluate_capacity_gate(
    *,
    active_stream_count: int,
    requested_stream_count: int,
    policy: StreamCapacityPolicy,
) -> CapacityGateDecision:
    if active_stream_count < 0:
        raise ValueError("active_stream_count cannot be negative")
    if requested_stream_count <= 0:
        raise ValueError("requested_stream_count must be greater than 0")

    total_after_start = active_stream_count + requested_stream_count
    reliable_capacity = policy.reliable_capacity
    remaining = max(reliable_capacity - total_after_start, 0)
    if total_after_start <= reliable_capacity:
        return CapacityGateDecision(
            allowed=True,
            active_stream_count=active_stream_count,
            requested_stream_count=requested_stream_count,
            reliable_capacity=reliable_capacity,
            remaining_after_start=remaining,
            reason_zh=(
                f"Current streams: {active_stream_count}; requested new streams: {requested_stream_count}; "
                f"reliable capacity: {reliable_capacity} streams"
            ),
        )

    return CapacityGateDecision(
        allowed=False,
        active_stream_count=active_stream_count,
        requested_stream_count=requested_stream_count,
        reliable_capacity=reliable_capacity,
        remaining_after_start=remaining,
        reason_zh=(
            f"Current streams: {active_stream_count}; requested new streams: {requested_stream_count}; "
            f"exceeds reliable capacity: {reliable_capacity} streams"
        ),
    )
