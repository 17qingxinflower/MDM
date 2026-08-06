from __future__ import annotations

from collections import deque
from math import hypot
from typing import Deque


class MotionTracker:
    def __init__(self, max_points: int = 180, meters_per_pixel: float = 0.01) -> None:
        self.max_points = max_points
        self.meters_per_pixel = float(meters_per_pixel)
        self._points: Deque[tuple[int, int]] = deque(maxlen=max_points)
        self._last_point: tuple[int, int] | None = None
        self._last_time: float | None = None
        self.total_distance = 0.0
        self.current_speed = 0.0

    @property
    def trajectory(self) -> list[tuple[int, int]]:
        return list(self._points)

    @property
    def distance_m(self) -> float:
        return self.total_distance * self.meters_per_pixel

    @property
    def speed_mps(self) -> float:
        return self.current_speed * self.meters_per_pixel

    def set_meters_per_pixel(self, meters_per_pixel: float) -> None:
        self.meters_per_pixel = max(float(meters_per_pixel), 0.000001)

    def reset(self) -> None:
        self._points.clear()
        self._last_point = None
        self._last_time = None
        self.total_distance = 0.0
        self.current_speed = 0.0

    def update(self, point: tuple[int, int] | None, timestamp: float) -> None:
        if point is None:
            return

        current_point = (int(point[0]), int(point[1]))
        current_time = float(timestamp)

        if self._last_point is None or self._last_time is None:
            self._points.append(current_point)
            self._last_point = current_point
            self._last_time = current_time
            self.current_speed = 0.0
            return

        dt = current_time - self._last_time
        if dt <= 0:
            return

        distance = hypot(
            current_point[0] - self._last_point[0],
            current_point[1] - self._last_point[1],
        )
        self.current_speed = distance / dt
        self.total_distance += distance

        if distance > 0:
            self._points.append(current_point)
            self._last_point = current_point
        self._last_time = current_time
