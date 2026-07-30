from typing import Any, Dict, List, Optional, Union

from ..models import (
    APIResponseEnvelope,
    AlertThreshold,
    CommandMetric,
    DashboardFilter,
    PerformanceRecord,
    TaskMetric,
    TimeSeriesPoint,
    TrendAnalysisResult,
)
from ..visualization.renderers import LayoutTranslator


class MetricEncoder:
    """Base encoder providing ordered field mapping for metric serialization."""

    def encode(self, metric: Union[CommandMetric, TaskMetric]) -> Dict[str, Any]:
        if isinstance(metric, CommandMetric):
            return {
                "command_name": metric.command_name,
                "execution_time_ms": metric.execution_time_ms,
                "memory_usage_mb": metric.memory_usage_mb,
                "cpu_utilization_percent": metric.cpu_utilization_percent,
                "return_code": metric.return_code,
            }
        return {
            "task_id": metric.task_id,
            "duration_seconds": metric.duration_seconds,
            "success_rate": metric.success_rate,
            "resource_consumption_units": metric.resource_consumption_units,
            "status": metric.status,
        }


class MetricSerializer(MetricEncoder):
    """Serialize CommandMetric or TaskMetric to ordered dict preserving client expectations."""

    def serialize(self, metric: Union[CommandMetric, TaskMetric]) -> Dict[str, Any]:
        return self.encode(metric)


class TrendFormatter:
    """Base formatter for trend analysis packaging supporting statistical evidence."""

    def format(
        self, result: Optional[TrendAnalysisResult]
    ) -> Optional[Dict[str, Any]]:
        if result is None:
            return None
        return {
            "trend_direction": result.trend_direction,
            "slope_coefficient": result.slope_coefficient,
            "confidence_score": result.confidence_score,
            "time_window_seconds": result.time_window_seconds,
            "anomaly_detected": result.anomaly_detected,
        }


class TrendReportSerializer(TrendFormatter):
    """Package analytical findings including supporting statistical evidence."""

    def serialize(
        self, result: Optional[TrendAnalysisResult]
    ) -> Optional[Dict[str, Any]]:
        return self.format(result)


class ErrorDiagnosticBuilder:
    """Transform validation failures and processing errors into actionable diagnostics."""

    def build(self, error_details: Union[str, None]) -> List[Dict[str, str]]:
        if not error_details:
            return []
        return [{"category": "error", "message": error_details}]


class DashboardStateSerializer(LayoutTranslator):
    """Represent dashboard composition instructions via hierarchical structure."""

    def serialize_state(self) -> Dict[str, Any]:
        layout = self._default_layout()
        return {
            "layout_spec": {
                "grid_columns": layout.grid_columns,
                "chart_height_px": layout.chart_height_px,
                "refresh_interval_ms": layout.refresh_interval_ms,
                "widget_order": list(layout.widget_order),
                "theme_name": layout.theme_name,
            },
        }


class FilterSerializer:
    """Serialize DashboardFilter preserving temporal and metric constraints."""

    def serialize(self, filter_spec: DashboardFilter) -> Dict[str, Any]:
        return {
            "start_time": filter_spec.start_time,
            "end_time": filter_spec.end_time,
            "command_names": list(filter_spec.command_names),
            "task_ids": list(filter_spec.task_ids),
            "minimum_cpu_utilization": filter_spec.minimum_cpu_utilization,
            "maximum_memory_usage_mb": filter_spec.maximum_memory_usage_mb,
        }


class ThresholdSerializer:
    """Serialize AlertThreshold with comparison operator and limits."""

    def serialize(self, threshold: AlertThreshold) -> Dict[str, Any]:
        return {
            "metric_type": threshold.metric_type,
            "comparison_operator": threshold.comparison_operator,
            "warning_limit": threshold.warning_limit,
            "error_limit": threshold.error_limit,
        }


class RecordSerializer:
    """Serialize PerformanceRecord with embedded metric and metadata."""

    def serialize(self, record: PerformanceRecord) -> Dict[str, Any]:
        return {
            "timestamp": record.timestamp,
            "record_type": MetricSerializer().serialize(
                # type: ignore[arg-type]
                record.record_type  # Union[CommandMetric, TaskMetric]
            ),
            "source_identifier": record.source_identifier,
            "metadata": dict(record.metadata),
        }


class TimeSeriesPointSerializer:
    """Serialize TimeSeriesPoint preserving confidence interval bounds."""

    def serialize(self, point: TimeSeriesPoint) -> Dict[str, Any]:
        return {
            "timestamp": point.timestamp,
            "value": point.value,
            "series_label": point.series_label,
            "confidence_interval_lower": point.confidence_interval_lower,
            "confidence_interval_upper": point.confidence_interval_upper,
        }


class ResponseEnvelopeSerializer:
    """Standardize response envelope via APIResponseEnvelope with timestamp payload."""

    def serialize(
        self,
        status_code: int,
        message: Union[str, None],
        data: Union[Dict[str, object], List[object], None],
        error_details: Union[str, None],
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        diagnostics = ErrorDiagnosticBuilder().build(error_details)
        return {
            "status_code": status_code,
            "message": message,
            "data": data,
            "error_details": error_details,
            "timestamp": timestamp if timestamp is not None else 0.0,
            "diagnostics": diagnostics,
        }


class BatchMetricSerializer:
    """Serialize lists of metrics maintaining ordering and type safety."""

    def serialize(
        self, metrics: List[Union[CommandMetric, TaskMetric]]
    ) -> List[Dict[str, Any]]:
        return [MetricSerializer().serialize(m) for m in metrics]


class BatchRecordSerializer:
    """Serialize lists of PerformanceRecords with embedded metrics."""

    def serialize(self, records: List[PerformanceRecord]) -> List[Dict[str, Any]]:
        return [RecordSerializer().serialize(r) for r in records]