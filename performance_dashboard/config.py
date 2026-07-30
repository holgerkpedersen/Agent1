from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection parameters."""
    host: str = "localhost"
    port: int = 5432
    username: str = "dashboard_user"
    password: str = ""
    database_name: str = "performance_db"
    pool_size: int = 10
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class PerformanceThresholds:
    """Warning and error limits for commands/tasks performance."""
    command_warning_ms: float = 500.0
    command_error_ms: float = 2000.0
    task_warning_seconds: float = 10.0
    task_error_seconds: float = 60.0
    memory_warning_mb: float = 512.0
    memory_error_mb: float = 2048.0


@dataclass(frozen=True)
class DashboardSettings:
    """Dashboard refresh intervals and display preferences."""
    refresh_interval_seconds: int = 30
    max_data_points_displayed: int = 100
    show_real_time_updates: bool = True
    theme: str = "dark"
    chart_height_pixels: int = 400


@dataclass(frozen=True)
class CollectorIntervals:
    """Data collection intervals for collectors."""
    command_collection_interval_seconds: int = 60
    task_collection_interval_seconds: int = 30
    metrics_retention_hours: int = 24
    cleanup_interval_hours: int = 1


@dataclass(frozen=True)
class ValidationRules:
    """Validation rules for collected data."""
    min_command_duration_ms: float = 0.0
    max_command_duration_ms: float = 3600000.0
    allowed_task_statuses: tuple[str, ...] = ("success", "failed", "running", "pending")
    max_metric_name_length: int = 256


DATABASE_CONFIG: Final[DatabaseConfig] = DatabaseConfig()

PERFORMANCE_THRESHOLDS: Final[PerformanceThresholds] = PerformanceThresholds()

DASHBOARD_SETTINGS: Final[DashboardSettings] = DashboardSettings()

COLLECTOR_INTERVALS: Final[CollectorIntervals] = CollectorIntervals()

VALIDATION_RULES: Final[ValidationRules] = ValidationRules()