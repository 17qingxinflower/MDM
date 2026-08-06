from __future__ import annotations

from collections.abc import Mapping, Sequence

from deerui.domain.models import AlertSegment, BehaviorSegment


class AlertRuleEngine:
    def __init__(
        self,
        min_duration_seconds: float = 5.0,
        max_transitions_per_minute: float = 10.0,
        action_thresholds: Mapping[str, float] | None = None,
    ) -> None:
        self.min_duration_seconds = min_duration_seconds
        self.max_transitions_per_minute = max_transitions_per_minute
        self.action_thresholds = dict(action_thresholds or {})

    def analyze(self, segments: Sequence[BehaviorSegment]) -> list[AlertSegment]:
        alerts: list[AlertSegment] = []
        alerts.extend(self._detect_long_duration(segments))
        high_transition = self._detect_high_transition(segments)
        if high_transition is not None:
            alerts.append(high_transition)
        return sorted(alerts, key=lambda alert: alert.started_at_seconds)

    def _detect_long_duration(self, segments: Sequence[BehaviorSegment]) -> list[AlertSegment]:
        alerts: list[AlertSegment] = []
        for segment in segments:
            threshold = self.action_thresholds.get(
                segment.display_name,
                self.action_thresholds.get(segment.action_key, self.min_duration_seconds),
            )
            if segment.duration_seconds >= threshold:
                alerts.append(
                    AlertSegment(
                        alert_type="long_duration",
                        action_key=segment.action_key,
                        display_name=segment.display_name,
                        started_at_seconds=segment.started_at_seconds,
                        ended_at_seconds=segment.ended_at_seconds,
                        duration_seconds=segment.duration_seconds,
                        transition_rate=None,
                        recommendation="Check for illness or stereotyped behavior",
                    )
                )
        return alerts

    def _detect_high_transition(
        self, segments: Sequence[BehaviorSegment]
    ) -> AlertSegment | None:
        if len(segments) <= 1:
            return None

        total_duration = segments[-1].ended_at_seconds - segments[0].started_at_seconds
        transition_count = len(segments) - 1
        if total_duration <= 0:
            return None

        transition_rate = (transition_count / total_duration) * 60
        if transition_rate <= self.max_transitions_per_minute:
            return None

        return AlertSegment(
            alert_type="high_transition",
            action_key=None,
            display_name="Rapid mixed-behavior changes",
            started_at_seconds=segments[0].started_at_seconds,
            ended_at_seconds=segments[-1].ended_at_seconds,
            duration_seconds=total_duration,
            transition_rate=transition_rate,
            recommendation=f"Rate {transition_rate:.1f}/min; check estrus or courtship behavior",
        )
