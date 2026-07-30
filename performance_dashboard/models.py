from __future__ import annotations

from typing import TypedDict, Union


class CommandMetric(TypedDict):
    execution_time_ms: float
    memory_usage_mb: float
    cpu_utilization_percent: float
    command_name: str
    return_code: int


class TaskMetric(TypedDict):
    duration_seconds: float
    success_rate: float
    resource_consumption_units: float
    task_id: str
    status: Union[str, None]


class PerformanceRecord(TypedDict):
    timestamp: float
    record_type: Union[CommandMetric, TaskMetric]
    source_identifier: str
    metadata: dict[str, str]


class DashboardFilter(TypedDict):
    start_time: float
    end_time: float
    command_names: list[str]
    task_ids: list[str]
    minimum_cpu_utilization: float
    maximum_memory_usage_mb: float


class AlertThreshold(TypedDict):
    warning_limit: float
    error_limit: float
    metric_type: str
    comparison_operator: str


class StatisticalSummary(TypedDict):
    mean_value: float
    median_value: float
    standard_deviation: float
    percentile_95: float
    percentile_99: float
    sample_count: int


class TrendAnalysisResult(TypedDict):
    trend_direction: Union[str, None]
    slope_coefficient: float
    confidence_score: float
    time_window_seconds: float
    anomaly_detected: bool


class DashboardLayoutSpec(TypedDict):
    grid_columns: int
    chart_height_px: int
    refresh_interval_ms: int
    widget_order: list[str]
    theme_name: str


class APIResponseEnvelope(TypedDict):
    status_code: int
    message: Union[str, None]
    data: Union[dict[str, object], list[object], None]
    error_details: Union[str, None]


class TimeSeriesPoint(TypedDict):
    timestamp: float
    value: float
    series_label: str
    confidence_interval_lower: float
    confidence_interval_upper: float