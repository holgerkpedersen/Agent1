"""Core metric and alert data structures for the monitoring subsystem.

Self-contained on purpose: the dashboard stack (collector, alerts, HTTP API)
depends only on this module plus the standard library, so it can be imported
cheaply from anywhere (REPL loop, ``--serve`` process, tests).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricData:
    name: str
    value: float
    timestamp: float
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    name: str
    metric_name: str
    threshold: float
    comparison_operator: str
    severity: str
    cooldown_seconds: int = 60


@dataclass
class AlertEvent:
    rule_name: str
    triggered_at: float
    current_value: float
    threshold: float
    severity: str
    message: str


__all__: List[str] = [
    "MetricType",
    "MetricData",
    "AlertRule",
    "AlertEvent",
]
