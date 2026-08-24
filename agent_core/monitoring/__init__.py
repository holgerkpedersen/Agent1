"""Monitoring package: metrics collection, alerting, and the web dashboard.

This is the live backend behind ``python agent.py --serve`` and the in-REPL
``--dashboard`` flag.  It was promoted from the retired ``src/agent1``
framework into ``agent_core`` so it is a first-class, always-installed
component instead of a path-hack import.
"""

from typing import Any, Awaitable, Callable, List

from .alert_system import AlertSystem
from .dashboard_api import DashboardAPIServer
from .metrics_collector import MetricsCollector
from .types import AlertEvent, AlertRule, MetricData, MetricType

AsyncMessageHandler = Callable[[Any], Awaitable[None]]
MetricHandler = Callable[[MetricData], None]
AlertHandler = Callable[[AlertEvent], None]


__all__: List[str] = [
    "AlertEvent",
    "AlertRule",
    "MetricData",
    "MetricType",
    "AsyncMessageHandler",
    "MetricHandler",
    "AlertHandler",
    "MetricsCollector",
    "DashboardAPIServer",
    "AlertSystem",
]
