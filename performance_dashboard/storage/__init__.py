"""Storage package for the Performance Dashboard system.

This subpackage provides persistence and retrieval capabilities for
performance metrics collected by the dashboard's collectors. It exposes
the core storage abstractions used throughout the application so that
other modules (analytics, visualization, API) can depend on stable,
well-defined interfaces rather than concrete implementations.

Public exports:
    - DatabaseConfig   : configuration model describing database connection
      parameters and operational settings consumed by the storage layer.
    - MetricsStore     : in-memory / persistent store responsible for
      recording ``PerformanceRecord`` entries and serving filtered queries
      against them. Concrete persistence backends (e.g. SQLite, Redis) are
      expected to subclass or implement this interface.

The package intentionally keeps its surface area small: storage concerns
are encapsulated within the dedicated modules (``database.py`` and
``metrics_store.py``) while ``__init__.py`` only re-exports the most
commonly referenced types for ergonomic cross-package usage.
"""

from performance_dashboard.config import DatabaseConfig
from performance_dashboard.storage.metrics_store import MetricsStore

__all__: list[str] = ["DatabaseConfig", "MetricsStore"]