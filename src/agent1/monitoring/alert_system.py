from typing import Dict, Any, List, Optional, Callable
import time

from ..core import AlertEvent, AlertRule
from .metrics_collector import MetricsCollector


class AlertSystem:
    """Evaluates alert rules against collected metrics and emits alert events.

    Maintains a registry of ``AlertRule`` definitions, tracks cooldown windows so
    that each rule fires at most once per period, invokes registered handler
    callbacks for every newly triggered event, and keeps an in-memory list of
    currently active (unresolved) alerts.
    """

    def __init__(self, collector: MetricsCollector) -> None:
        self._collector: MetricsCollector = collector
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: List[AlertEvent] = []
        self._last_triggered: Dict[str, float] = {}
        self._handlers: List[Callable[[AlertEvent], None]] = []

    def add_rule(self, rule: AlertRule) -> str:
        """Register an alert monitoring rule. Returns the rule name."""
        self._rules[rule.name] = rule
        return rule.name

    def remove_rule(self, name: str) -> bool:
        """Remove a registered alert rule by name."""
        if name in self._rules:
            del self._rules[name]
            return True
        return False

    def list_rules(self) -> List[AlertRule]:
        """Return all currently registered alert rules."""
        return list(self._rules.values())

    def register_handler(self, handler: Callable[[AlertEvent], None]) -> bool:
        """Register a callback invoked when an alert is triggered."""
        self._handlers.append(handler)
        return True

    def unregister_handler(self, handler: Callable[[AlertEvent], None]) -> bool:
        """Remove a previously registered alert handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)
            return True
        return False

    def active_alerts(self) -> List[AlertEvent]:
        """Return currently active (unresolved) alerts."""
        return list(self._active_alerts)

    def clear_active_alerts(self) -> None:
        """Clear all stored active alert events."""
        self._active_alerts.clear()

    def _metric_value(self, metric_name: str) -> Optional[float]:
        """Retrieve the current value of a named metric (gauge or counter)."""
        gauge: Optional[float] = self._collector.get_gauge_value(metric_name)
        if gauge is not None:
            return gauge
        snapshot: Dict[str, Any] = self._collector.snapshot()
        counters: Any = snapshot.get("counters", {})
        if metric_name in counters:
            return float(counters[metric_name])
        return None

    def _compare(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate a comparison between a measured value and a threshold."""
        if operator == "greater_than":
            return value > threshold
        if operator == "less_than":
            return value < threshold
        if operator == "equal":
            return value == threshold
        raise ValueError(f"Unknown comparison operator: {operator}")

    def evaluate(self, rules: Optional[List[AlertRule]] = None) -> List[AlertEvent]:
        """Evaluate alert rules against current metrics; returns newly triggered events."""
        rule_list: List[AlertRule] = (
            list(rules) if rules is not None else list(self._rules.values())
        )
        now: float = time.time()
        new_events: List[AlertEvent] = []

        for rule in rule_list:
            last_time: float = self._last_triggered.get(rule.name, 0.0)
            if now - last_time < rule.cooldown_seconds:
                continue

            current_value: Optional[float] = self._metric_value(rule.metric_name)
            if current_value is None:
                continue

            triggered: bool = self._compare(
                current_value, rule.threshold, rule.comparison_operator
            )
            if not triggered:
                continue

            event: AlertEvent = AlertEvent(
                rule_name=rule.name,
                triggered_at=now,
                current_value=current_value,
                threshold=rule.threshold,
                severity=rule.severity,
                message=(
                    f"Metric '{rule.metric_name}' value {current_value:.4f} "
                    f"{rule.comparison_operator} threshold {rule.threshold:.4f}"
                ),
            )
            self._active_alerts.append(event)
            self._last_triggered[rule.name] = now
            new_events.append(event)

            for handler in self._handlers:
                try:
                    handler(event)
                except Exception as exc:  # noqa: BLE001 - handlers must not crash evaluation
                    print(f"Alert handler error ({handler}): {exc}")

        return new_events


__all__: List[str] = ["AlertSystem"]