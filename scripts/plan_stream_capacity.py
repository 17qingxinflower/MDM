from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deerui.config import load_config, resolve_inference_device
from deerui.runtime import (
    StreamAssignmentPlanner,
    StreamDescriptor,
    WorkerDescriptor,
    required_worker_count,
)


def main() -> int:
    config = load_config()
    stream_count = int(config.max_managed_streams)
    worker_capacity = int(config.worker_stream_capacity)
    worker_count = required_worker_count(stream_count, worker_capacity)
    device = resolve_inference_device(config.inference_device)
    now = datetime.now(timezone.utc)

    streams = [
        StreamDescriptor(
            stream_id=f"cam-{index + 1:03d}",
            deer_code="demo",
            source=f"rtsp://example/camera/{index + 1:03d}",
            target_fps=float(config.stream_target_fps),
            preview_fps=float(config.preview_fps),
        )
        for index in range(stream_count)
    ]
    workers = [
        WorkerDescriptor(
            worker_id=f"worker-{index + 1:02d}",
            device=device,
            capacity=worker_capacity,
            last_heartbeat_at=now,
        )
        for index in range(worker_count)
    ]
    plan = StreamAssignmentPlanner().plan(streams, workers)

    print(f"streams={stream_count}")
    print(f"target_fps_per_stream={config.stream_target_fps}")
    print(f"preview_fps_per_stream={config.preview_fps}")
    print(f"worker_capacity={worker_capacity}")
    print(f"required_workers={worker_count}")
    print(f"device={device}")
    print(f"assigned_streams={plan.assigned_stream_count}")
    print(f"unassigned_streams={len(plan.unassigned_stream_ids)}")
    for worker_id, assigned_streams in plan.assignments_by_worker.items():
        print(f"{worker_id}: {len(assigned_streams)} streams")
    return 0 if plan.is_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
