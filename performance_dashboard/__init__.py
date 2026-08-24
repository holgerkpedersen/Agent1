"""Performance Dashboard System package.

Provides a framework for collecting, storing, analyzing and visualizing
performance metrics of commands and tasks over time.

The root package intentionally exposes only dependency-free names.
Submodules that require optional third-party services (flask for the REST
API, redis for the metrics store) are importable directly but are NOT
imported here, so ``import performance_dashboard`` stays lightweight:

    from performance_dashboard.config import DatabaseConfig      # always works
    from performance_dashboard.storage.metrics_store import MetricsStore  # needs redis
    from performance_dashboard.api.endpoints import register_endpoints   # needs flask
"""

from typing import Any, Callable

__version__: str = "1.0.0"

__all__: list[str] = [
    "__version__",
    "CommandMetric",
    "TaskMetric",
    "PerformanceRecord",
    "TimeSeriesPoint",
    "StatisticalSummary",
    "TrendAnalysisResult",
    "DashboardFilter",
    "AlertThreshold",
    "APIResponseEnvelope",
    "DatabaseConfig",
    "PerformanceThresholds",
    "DashboardSettings",
    "CollectorIntervals",
    "ValidationRules",
    "AggregationEngine",
    "HierarchicalAggregator",
    "SummaryBuilder",
    "StatisticalSummaryGenerator",
    "TrendDetector",
    "CorrelationMapper",
    "TaskCollector",
    "RollingWindowAggregator",
    "PerformanceDatabase",
    "TimeSeriesAdapter",
]

# -- eager, dependency-free re-exports ---------------------------------------

from .config import (  # noqa: E402
    CollectorIntervals,
    DashboardSettings,
    DatabaseConfig,
    PerformanceThresholds,
    ValidationRules,
)
from .models import (  # noqa: E402
    AlertThreshold,
    APIResponseEnvelope,
    CommandMetric,
    DashboardFilter,
    PerformanceRecord,
    StatisticalSummary,
    TaskMetric,
    TimeSeriesPoint,
    TrendAnalysisResult,
)

# -- optional extras (loaded on first attribute access) ----------------------


def __getattr__(name: str) -> Any:  # PEP 562 module-level getattr
    """Lazily import heavier submodules; raise ImportError naming the dep."""
    if name in ("TaskCollector",):
        from .collectors.task_collector import TaskCollector

        return TaskCollector
    if name in ("RollingWindowAggregator",):
        from .analytics.processor import RollingWindowAggregator

        return RollingWindowAggregator
    if name in (
        "AggregationEngine", "HierarchicalAggregator", "SummaryBuilder",
        "StatisticalSummaryGenerator", "TrendDetector", "CorrelationMapper",
    ):
        from . import analytics

        return getattr(analytics, name)
    if name in ("PerformanceDatabase", "TimeSeriesAdapter"):
        from .storage.database import PerformanceDatabase, TimeSeriesAdapter

        return {"PerformanceDatabase": PerformanceDatabase,
                "TimeSeriesAdapter": TimeSeriesAdapter}[name]
    if name in ("MetricsStore",):
        from .storage.metrics_store import MetricsStore

        return MetricsStore
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def _lazy(name: str, importer: Callable[[], Any]) -> None:
    """Placeholder kept for API compatibility; no-op."""
