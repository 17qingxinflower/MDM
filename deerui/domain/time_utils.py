from __future__ import annotations

from datetime import timedelta


def seconds_to_clock(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))
