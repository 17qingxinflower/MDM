from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil


@dataclass(frozen=True)
class StreamDescriptor:
    """A single camera/video route that can be assigned to a processing worker."""

    stream_id: str
    deer_code: str
    source: str
    target_fps: float = 5.0
    preview_fps: float = 5.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.stream_id.strip():
            raise ValueError("stream_id cannot be empty")
        if not self.deer_code.strip():
            raise ValueError("deer_code cannot be empty")
        if not self.source.strip():
            raise ValueError("source cannot be empty")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be greater than 0")
        if self.preview_fps <= 0:
            raise ValueError("preview_fps must be greater than 0")


@dataclass(frozen=True)
class WorkerDescriptor:
    """Capacity and health information for one backend processing worker."""

    worker_id: str
    device: str
    capacity: int
    last_heartbeat_at: datetime

    def __post_init__(self) -> None:
        if not self.worker_id.strip():
            raise ValueError("worker_id cannot be empty")
        if not self.device.strip():
            raise ValueError("device cannot be empty")
        if self.capacity <= 0:
            raise ValueError("capacity must be greater than 0")


@dataclass(frozen=True)
class Assignment:
    stream_id: str
    worker_id: str


@dataclass(frozen=True)
class AssignmentPlan:
    assignments: tuple[Assignment, ...]
    unassigned_stream_ids: tuple[str, ...]

    @property
    def assignments_by_worker(self) -> dict[str, tuple[str, ...]]:
        grouped: defaultdict[str, list[str]] = defaultdict(list)
        for assignment in self.assignments:
            grouped[assignment.worker_id].append(assignment.stream_id)
        return {worker_id: tuple(streams) for worker_id, streams in grouped.items()}

    @property
    def assignments_by_stream(self) -> dict[str, str]:
        return {assignment.stream_id: assignment.worker_id for assignment in self.assignments}

    @property
    def assigned_stream_count(self) -> int:
        return len(self.assignments)

    @property
    def is_complete(self) -> bool:
        return not self.unassigned_stream_ids


class StreamAssignmentPlanner:
    """Create stable stream-to-worker assignments without exceeding worker capacity."""

    def plan(
        self,
        streams: Iterable[StreamDescriptor],
        workers: Iterable[WorkerDescriptor],
    ) -> AssignmentPlan:
        active_streams = [stream for stream in streams if stream.enabled]
        worker_list = list(workers)
        loads = {worker.worker_id: 0 for worker in worker_list}
        assignments: list[Assignment] = []
        unassigned: list[str] = []

        for stream in active_streams:
            target_worker = self._least_loaded_worker_with_capacity(worker_list, loads)
            if target_worker is None:
                unassigned.append(stream.stream_id)
                continue
            assignments.append(Assignment(stream.stream_id, target_worker.worker_id))
            loads[target_worker.worker_id] += 1

        return AssignmentPlan(tuple(assignments), tuple(unassigned))

    def _least_loaded_worker_with_capacity(
        self,
        workers: list[WorkerDescriptor],
        loads: dict[str, int],
    ) -> WorkerDescriptor | None:
        candidates = [
            worker
            for worker in workers
            if loads.get(worker.worker_id, 0) < worker.capacity
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda worker: (loads[worker.worker_id], worker.worker_id))


def required_worker_count(stream_count: int, worker_capacity: int) -> int:
    if stream_count < 0:
        raise ValueError("stream_count cannot be negative")
    if worker_capacity <= 0:
        raise ValueError("worker_capacity must be greater than 0")
    if stream_count == 0:
        return 0
    return ceil(stream_count / worker_capacity)


def stale_worker_ids(
    workers: Iterable[WorkerDescriptor],
    *,
    now: datetime | None = None,
    timeout_seconds: float = 15.0,
) -> tuple[str, ...]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0")
    current = now or datetime.now(timezone.utc)
    return tuple(
        worker.worker_id
        for worker in workers
        if _seconds_since(worker.last_heartbeat_at, current) > timeout_seconds
    )


def _seconds_since(previous: datetime, current: datetime) -> float:
    if previous.tzinfo is None and current.tzinfo is not None:
        previous = previous.replace(tzinfo=current.tzinfo)
    if current.tzinfo is None and previous.tzinfo is not None:
        current = current.replace(tzinfo=previous.tzinfo)
    return (current - previous).total_seconds()
