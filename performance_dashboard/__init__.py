"""Performance Dashboard System package.

Provides a framework for collecting, storing, analyzing and visualizing
performance metrics of commands and tasks over time.
"""

from .config import Config
from .models import MetricRecord, CommandMetric, TaskMetric
from .collectors.command_collector import CommandCollector
from .collectors.task_collector import TaskCollector
from .storage.database import Database
from .storage.metrics_store import MetricsStore
from .analytics.processor import AnalyticsProcessor
from .analytics.aggregator import Aggregator
from .visualization.dashboard import Dashboard
from .visualization.charts import ChartFactory
from .api.endpoints import APIEndpoints

__version__: str = "1.0.0"

__all__: list[str] = [
    "__version__",
    "Config",
    "MetricRecord",
    "CommandMetric",
    "TaskMetric",
    "CommandCollector",
    "TaskCollector",
    "Database",
    "MetricsStore",
    "AnalyticsProcessor",
    "Aggregator",
    "Dashboard",
    "ChartFactory",
    "APIEndpoints",
]