"""Storage package for the Performance Dashboard system.

This subpackage provides persistence and retrieval capabilities for
performance metrics collected by the dashboard's collectors. It exposes
the core storage abstractions used throughout the application so that
other modules (analytics, visualization, API) can depend on stable,
well-defined interfaces rather than concrete implementations.

Public exports:
    - DatabaseConfig   : configuration model describing database connection
      parameters and operational settings consumed by the storage layer.

The package intentionally keeps its surface area small: storage concerns
are encapsulated within the dedicated modules (``database.py`` and
``metrics_store.py``) while ``__init__.py`` only re-exports the
dependency-free configuration type.  ``MetricsStore`` (which requires the
optional ``redis`` package) is imported directly from
``performance_dashboard.storage.metrics_store`` when needed.
"""

from performance_dashboard.config import DatabaseConfig

__all__: list[str] = ["DatabaseConfig"]