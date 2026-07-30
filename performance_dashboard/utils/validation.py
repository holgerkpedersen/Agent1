"""Input validation framework enforcing structural integrity of data streams."""
from __future__ import annotations

import math
from typing import Set

from performance_dashboard.models import (
    AlertThreshold,
    CommandMetric,
    DashboardFilter,
    PerformanceRecord,
    TaskMetric,
)


# Acceptable numeric ranges per measurement specification.
MIN_CPU_UTILIZATION: float = 0.0
MAX_CPU_UTILIZATION: float = 100.0
NON_NEGATIVE_BOUND: float = 0.0
SUCCESS_RATE_MINIMUM: float = 0.0
SUCCESS_RATE_MAXIMUM: float = 1.0
EXIT_CODE_MINIMUM: int = 0
EXIT_CODE_MAXIMUM: int = 255

GREATER_THAN_OPERATORS: Set[str] = {">", ">="}
LESS_THAN_OPERATORS: Set[str] = {"<", "<="}
EQUALITY_OPERATORS: Set[str] = {"=", "!="}


def _within_bounds(value: float, lower: float, upper: float) -> bool:
    """Return True when ``value`` lies within the inclusive [lower, upper] range."""
    return lower <= value <= upper


class MetricSchemaValidator:
    """Confirm measurements contain required fields with values in acceptable ranges."""

    def validate_command_metric(self, metric: CommandMetric) -> bool:
        if not _within_bounds(metric.execution_time_ms, NON_NEGATIVE_BOUND, math.inf):
            raise ValueError("execution_time_ms must be non-negative")
        if not _within_bounds(metric.memory_usage_mb, NON_NEGATIVE_BOUND, math.inf):
            raise ValueError("memory_usage_mb must be non-negative")
        if not _within_bounds(
            metric.cpu_utilization_percent, MIN_CPU_UTILIZATION, MAX_CPU_UTILIZATION
        ):
            raise ValueError("cpu_utilization_percent must fall within [0.0, 100.0]")
        if not metric.command_name:
            raise ValueError("command_name must be a non-empty string")
        if metric.return_code < EXIT_CODE_MINIMUM or metric.return_code > EXIT_CODE_MAXIMUM:
            raise ValueError(
                "return_code must fall within [%d, %d]"
                % (EXIT_CODE_MINIMUM, EXIT_CODE_MAXIMUM)
            )
        return True

    def validate_task_metric(self, metric: TaskMetric) -> bool:
        if not _within_bounds(metric.duration_seconds, NON_NEGATIVE_BOUND, math.inf):
            raise ValueError("duration_seconds must be non-negative")
        if not _within_bounds(
            metric.success_rate, SUCCESS_RATE_MINIMUM, SUCCESS_RATE_MAXIMUM
        ):
            raise ValueError("success_rate must fall within [0.0, 1.0]")
        if not _within_bounds(metric.resource_consumption_units, NON_NEGATIVE_BOUND, math.inf):
            raise ValueError("resource_consumption_units must be non-negative")
        if not metric.task_id:
            raise ValueError("task_id must be a non-empty string")
        return True

    def validate_performance_record(self, record: PerformanceRecord) -> bool:
        if not _within_bounds(record.timestamp, NON_NEGATIVE_BOUND, math.inf):
            raise ValueError("timestamp must be non-negative")
        inner = record.record_type
        if isinstance(inner, CommandMetric):
            self.validate_command_metric(inner)
        elif isinstance(inner, TaskMetric):
            self.validate_task_metric(inner)
        else:
            raise TypeError("record_type must be a CommandMetric or TaskMetric instance")
        return True


class FilterParameterValidator:
    """Query validator enforcing filter parameter bounds checking."""

    def validate_filter(self, filter_spec: DashboardFilter) -> bool:
        if not _within_bounds(filter_spec.start_time, NON_NEGATIVE_BOUND, math.inf):
            raise ValueError("start_time must be non-negative")
        if not _within_bounds(filter_spec.end_time, NON_NEGATIVE_BOUND, math.inf):
            raise ValueError("end_time must be non-negative")
        if filter_spec.end_time < filter_spec.start_time:
            raise ValueError(
                "end_time must not precede start_time accounting for clock drift"
            )
        if not _within_bounds(
            filter_spec.minimum_cpu_utilization, MIN_CPU_UTILIZATION, MAX_CPU_UTILIZATION
        ):
            raise ValueError("minimum_cpu_utilization must fall within [0.0, 100.0]")
        if not _within_bounds(filter_spec.maximum_memory_usage_mb, NON_NEGATIVE_BOUND, math.inf):
            raise ValueError("maximum_memory_usage_mb must be non-negative")
        return True


class ThresholdConsistencyVerifier:
    """Prevent contradictory warning/error limits during static analysis passes."""

    def verify_threshold(self, threshold: AlertThreshold) -> bool:
        if not threshold.metric_type:
            raise ValueError("metric_type must be a non-empty string")
        operator = threshold.comparison_operator
        if (
            operator not in GREATER_THAN_OPERATORS
            and operator not in LESS_THAN_OPERATORS
            and operator not in EQUALITY_OPERATORS
        ):
            raise ValueError(
                "comparison_operator must be one of >, >=, <, <=, =, !=; got %r" % operator
            )
        warning_limit = threshold.warning_limit
        error_limit = threshold.error_limit
        if not _within_bounds(warning_limit, NON_NEGATIVE_BOUND, math.inf):
            raise ValueError("warning_limit must be non-negative")
        if not _within_bounds(error_limit, NON_NEGATIVE_BOUND, math.inf):
            raise ValueError("error_limit must be non-negative")
        if operator in GREATER_THAN_OPERATORS:
            # Higher-value metrics alert past a limit; warning must precede error.
            if warning_limit >= error_limit:
                raise ValueError(
                    "for greater-than operators warning_limit must precede error_limit"
                )
        elif operator in LESS_THAN_OPERATORS:
            # Lower-value metrics alert below a limit; warning must exceed error.
            if warning_limit <= error_limit:
                raise ValueError(
                    "for less-than operators warning_limit must exceed error_limit"
                )
        else:  # Equality operators require identical limits to avoid contradictions.
            if warning_limit != error_limit:
                raise ValueError("equality operators require matching warning and error limits")
        return True


__all__: Set[str] = {
    "MetricSchemaValidator",
    "FilterParameterValidator",
    "ThresholdConsistencyVerifier",
}