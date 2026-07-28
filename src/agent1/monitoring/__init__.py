"""Monitoring package for the Agent1 system."""

from typing import Awaitable, Callable, List

from ..core import (
    AgentMessage,
    AlertEvent,
    AlertRule,
    MessageType,
    MetricData,
    MetricType,
)

from .metrics_collector import MetricsCollector
from .dashboard_api import DashboardAPIServer
from .alert_system import AlertSystem


AsyncMessageHandler = Callable[[AgentMessage], Awaitable[None]]
MetricHandler = Callable[[MetricData], None]
AlertHandler = Callable[[AlertEvent], None]


__all__: List[str] = [
    "AgentMessage",
    "AlertEvent",
    "AlertRule",
    "MessageType",
    "MetricData",
    "MetricType",
    "AsyncMessageHandler",
    "MetricHandler",
    "AlertHandler",
    "MetricsCollector",
    "DashboardAPIServer",
    "AlertSystem",
]