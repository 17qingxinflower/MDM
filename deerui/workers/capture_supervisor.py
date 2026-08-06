from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable

try:
    from PySide6.QtCore import QThread, Qt
except ImportError:  # pragma: no cover - dependency availability is checked at runtime
    QThread = None
    Qt = None

from deerui.workers.camera_worker import CameraWorker


FrameCallback = Callable[[str, object, float], None]
MessageCallback = Callable[[str, str], None]
FinishedCallback = Callable[[str], None]
RestartedCallback = Callable[[str, "CaptureRouteHandle"], None]


@dataclass(frozen=True)
class CaptureRouteSpec:
    route_id: str
    source: str | int
    target_fps: float
    reconnect_delay_seconds: float
    max_read_failures: int
    open_timeout_milliseconds: int
    read_timeout_milliseconds: int
    max_buffer_drain_grabs_per_frame: int = 2


@dataclass(frozen=True)
class CaptureRouteSnapshot:
    route_id: str
    source: str | int
    target_fps: float
    status: str
    frames_received: int = 0
    consecutive_failures: int = 0
    restart_count: int = 0
    started_at_monotonic: float | None = None
    last_frame_at_monotonic: float | None = None
    last_frame_timestamp_seconds: float | None = None
    last_status_message: str | None = None
    last_error_message: str | None = None


@dataclass(frozen=True)
class CaptureRouteHandle:
    route_id: str
    thread: object
    worker: object


@dataclass(frozen=True)
class _CaptureCallbacks:
    on_frame: FrameCallback
    on_status: MessageCallback | None = None
    on_error: MessageCallback | None = None
    on_finished: FinishedCallback | None = None
    on_restarted: RestartedCallback | None = None


@dataclass
class _CaptureRoute:
    spec: CaptureRouteSpec
    handle: CaptureRouteHandle
    snapshot: CaptureRouteSnapshot
    callbacks: _CaptureCallbacks
    notify_finished: bool = True


class CaptureSupervisor:
    def __init__(
        self,
        parent: object | None = None,
        *,
        thread_factory: Callable[..., object] | None = None,
        worker_factory: Callable[..., object] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._parent = parent
        self._thread_factory = thread_factory or self._default_thread_factory
        self._worker_factory = worker_factory or CameraWorker
        self._clock = clock or time.monotonic
        self._routes: dict[str, _CaptureRoute] = {}

    def start_route(
        self,
        spec: CaptureRouteSpec,
        *,
        on_frame: FrameCallback,
        on_status: MessageCallback | None = None,
        on_error: MessageCallback | None = None,
        on_finished: FinishedCallback | None = None,
        on_restarted: RestartedCallback | None = None,
    ) -> CaptureRouteHandle:
        existing = self._routes.get(spec.route_id)
        if existing is not None and existing.snapshot.status not in {"stopped", "finished"}:
            raise ValueError(f"capture route already active: {spec.route_id}")
        callbacks = _CaptureCallbacks(
            on_frame=on_frame,
            on_status=on_status,
            on_error=on_error,
            on_finished=on_finished,
            on_restarted=on_restarted,
        )

        thread = self._create_thread()
        worker = self._worker_factory(
            spec.source,
            target_fps=spec.target_fps,
            reconnect_delay_seconds=spec.reconnect_delay_seconds,
            max_read_failures=spec.max_read_failures,
            open_timeout_milliseconds=spec.open_timeout_milliseconds,
            read_timeout_milliseconds=spec.read_timeout_milliseconds,
            max_buffer_drain_grabs_per_frame=spec.max_buffer_drain_grabs_per_frame,
        )
        handle = CaptureRouteHandle(spec.route_id, thread, worker)
        route = _CaptureRoute(
            spec=spec,
            handle=handle,
            snapshot=CaptureRouteSnapshot(
                route_id=spec.route_id,
                source=spec.source,
                target_fps=spec.target_fps,
                status="starting",
                started_at_monotonic=self._clock(),
            ),
            callbacks=callbacks,
        )
        self._routes[spec.route_id] = route

        worker.moveToThread(thread)
        self._connect(thread.started, worker.run)
        self._connect(
            worker.frame_ready,
            lambda frame, timestamp, route_id=spec.route_id: self._handle_frame(
                route_id,
                frame,
                timestamp,
                callbacks.on_frame,
            ),
            direct=True,
        )
        if callbacks.on_status is not None:
            self._connect(
                worker.status,
                lambda message, route_id=spec.route_id: self._handle_status(
                    route_id,
                    message,
                    callbacks.on_status,
                ),
            )
        else:
            self._connect(
                worker.status,
                lambda message, route_id=spec.route_id: self._handle_status(
                    route_id,
                    message,
                    None,
                ),
            )
        if callbacks.on_error is not None:
            self._connect(
                worker.error,
                lambda message, route_id=spec.route_id: self._handle_error(
                    route_id,
                    message,
                    callbacks.on_error,
                ),
            )
        else:
            self._connect(
                worker.error,
                lambda message, route_id=spec.route_id: self._handle_error(
                    route_id,
                    message,
                    None,
                ),
            )
        self._connect(worker.finished, thread.quit)
        self._connect(worker.finished, worker.deleteLater)
        self._connect(
            thread.finished,
            lambda route_id=spec.route_id, route_handle=handle: self._handle_finished(
                route_id,
                route_handle,
            ),
        )
        self._connect(thread.finished, thread.deleteLater)

        thread.start()
        current = self._routes.get(spec.route_id)
        if current is not None and current.snapshot.status == "starting":
            current.snapshot = replace(current.snapshot, status="running")
        return handle

    def stop_route(self, route_id: str, *, wait_milliseconds: int = 1500) -> bool:
        route = self._routes.get(route_id)
        if route is None:
            return False
        if route.snapshot.status == "stopped":
            return False
        return self._stop_capture_route(
            route,
            wait_milliseconds=wait_milliseconds,
            notify_finished=True,
        )

    def restart_route(self, route_id: str, *, wait_milliseconds: int = 1500) -> bool:
        route = self._routes.get(route_id)
        if route is None:
            return False
        spec = route.spec
        callbacks = route.callbacks
        restart_count = route.snapshot.restart_count + 1
        self._stop_capture_route(
            route,
            wait_milliseconds=wait_milliseconds,
            notify_finished=False,
        )
        handle = self.start_route(
            spec,
            on_frame=callbacks.on_frame,
            on_status=callbacks.on_status,
            on_error=callbacks.on_error,
            on_finished=callbacks.on_finished,
            on_restarted=callbacks.on_restarted,
        )
        restarted_route = self._routes.get(route_id)
        if restarted_route is not None:
            restarted_route.snapshot = replace(
                restarted_route.snapshot,
                restart_count=restart_count,
                consecutive_failures=0,
            )
        if callbacks.on_restarted is not None:
            callbacks.on_restarted(route_id, handle)
        return True

    def mark_stale_routes(self, *, stale_after_seconds: float) -> list[str]:
        stale_routes: list[str] = []
        threshold = max(float(stale_after_seconds), 0.0)
        now = self._clock()
        for route_id, route in list(self._routes.items()):
            if route.snapshot.status not in {"starting", "running"}:
                continue
            heartbeat_at = (
                route.snapshot.last_frame_at_monotonic
                if route.snapshot.last_frame_at_monotonic is not None
                else route.snapshot.started_at_monotonic
            )
            if heartbeat_at is None:
                continue
            age_seconds = max(now - heartbeat_at, 0.0)
            if age_seconds <= threshold:
                continue
            consecutive_failures = (
                route.snapshot.consecutive_failures
                if route.snapshot.status == "stale"
                else route.snapshot.consecutive_failures + 1
            )
            route.snapshot = replace(
                route.snapshot,
                status="stale",
                consecutive_failures=consecutive_failures,
                last_status_message=f"no frame for {age_seconds:.1f}s",
            )
            stale_routes.append(route_id)
        return stale_routes

    def restart_stale_routes(
        self,
        *,
        stale_after_seconds: float,
        wait_milliseconds: int = 1500,
        max_routes: int | None = None,
    ) -> list[str]:
        stale_routes = self.mark_stale_routes(stale_after_seconds=stale_after_seconds)
        if max_routes is not None:
            stale_routes = stale_routes[: max(int(max_routes), 0)]
        restarted: list[str] = []
        for route_id in stale_routes:
            if self.restart_route(route_id, wait_milliseconds=wait_milliseconds):
                restarted.append(route_id)
        return restarted

    def restart_failed_routes(
        self,
        *,
        wait_milliseconds: int = 1500,
        max_routes: int | None = None,
    ) -> list[str]:
        failed_routes = [
            route_id
            for route_id, route in list(self._routes.items())
            if route.snapshot.status == "failed"
        ]
        if max_routes is not None:
            failed_routes = failed_routes[: max(int(max_routes), 0)]
        restarted: list[str] = []
        for route_id in failed_routes:
            if self.restart_route(route_id, wait_milliseconds=wait_milliseconds):
                restarted.append(route_id)
        return restarted

    def _stop_capture_route(
        self,
        route: _CaptureRoute,
        *,
        wait_milliseconds: int,
        notify_finished: bool,
    ) -> bool:
        route.snapshot = replace(route.snapshot, status="stopping")
        route.notify_finished = notify_finished
        worker = route.handle.worker
        thread = route.handle.thread
        stop = getattr(worker, "stop", None)
        if callable(stop):
            stop()
        if getattr(thread, "isRunning", lambda: False)():
            thread.quit()
            if not thread.wait(wait_milliseconds):
                terminate = getattr(thread, "terminate", None)
                if callable(terminate):
                    terminate()
                thread.wait(1000)
        self._handle_finished(route.spec.route_id, route.handle)
        return True

    def stop_all(self, *, wait_milliseconds: int = 1500) -> None:
        for route_id in list(self._routes):
            self.stop_route(route_id, wait_milliseconds=wait_milliseconds)

    def set_target_fps(self, route_id: str, target_fps: float) -> bool:
        route = self._routes.get(route_id)
        if route is None:
            return False
        worker = route.handle.worker
        setter = getattr(worker, "set_target_fps", None)
        if callable(setter):
            setter(target_fps)
        else:
            setattr(worker, "target_fps", target_fps)
        route.snapshot = replace(route.snapshot, target_fps=max(float(target_fps), 1.0))
        return True

    def snapshot(self, route_id: str) -> CaptureRouteSnapshot | None:
        route = self._routes.get(route_id)
        if route is None:
            return None
        return replace(route.snapshot)

    def active_route_ids(self) -> list[str]:
        return [
            route_id
            for route_id, route in self._routes.items()
            if route.snapshot.status not in {"stopped", "finished"}
        ]

    def _handle_frame(
        self,
        route_id: str,
        frame: object,
        timestamp_seconds: float,
        callback: FrameCallback,
    ) -> None:
        route = self._routes.get(route_id)
        if route is not None:
            route.snapshot = replace(
                route.snapshot,
                status="running",
                frames_received=route.snapshot.frames_received + 1,
                consecutive_failures=0,
                last_frame_at_monotonic=self._clock(),
                last_frame_timestamp_seconds=float(timestamp_seconds),
            )
        callback(route_id, frame, timestamp_seconds)

    def _handle_status(
        self,
        route_id: str,
        message: str,
        callback: MessageCallback | None,
    ) -> None:
        route = self._routes.get(route_id)
        if route is not None:
            route.snapshot = replace(route.snapshot, last_status_message=message)
        if callback is not None:
            callback(route_id, message)

    def _handle_error(
        self,
        route_id: str,
        message: str,
        callback: MessageCallback | None,
    ) -> None:
        route = self._routes.get(route_id)
        if route is not None:
            route.snapshot = replace(
                route.snapshot,
                status="failed",
                consecutive_failures=route.snapshot.consecutive_failures + 1,
                last_error_message=message,
            )
        if callback is not None:
            callback(route_id, message)

    def _handle_finished(self, route_id: str, handle: CaptureRouteHandle | None = None) -> None:
        route = self._routes.get(route_id)
        if route is None or route.snapshot.status == "stopped":
            return
        if handle is not None and route.handle is not handle:
            return
        route.snapshot = replace(route.snapshot, status="stopped")
        if route.notify_finished and route.callbacks.on_finished is not None:
            route.callbacks.on_finished(route_id)

    def _create_thread(self) -> object:
        try:
            return self._thread_factory(self._parent)
        except TypeError:
            return self._thread_factory()

    def _default_thread_factory(self, parent: object | None = None) -> object:
        if QThread is None:
            raise RuntimeError("PySide6 is required for capture threads")
        return QThread(parent)

    def _connect(self, signal, callback, *, direct: bool = False) -> None:
        if direct and Qt is not None:
            try:
                signal.connect(callback, Qt.ConnectionType.DirectConnection)
                return
            except TypeError:
                pass
        signal.connect(callback)
