from __future__ import annotations

from deerui.domain.alert_rules import AlertRuleEngine
from deerui.domain.models import AlertSegment, BehaviorSegment


class AnalysisService:
    def __init__(self, alert_engine: AlertRuleEngine | None = None) -> None:
        self.alert_engine = alert_engine or AlertRuleEngine()

    def analyze_alerts(self, segments: list[BehaviorSegment]) -> list[AlertSegment]:
        return self.alert_engine.analyze(segments)
