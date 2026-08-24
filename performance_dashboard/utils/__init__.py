"""Utility subpackage for the performance dashboard system.

Exposes helper modules for time-related calculations and input
validation used across collectors, storage, analytics, visualization
and API layers.

Real public surface (verified against the modules):
    time_utils: normalize_timezone, compute_duration_spans,
                format_display_label, align_to_bucket, compute_bucket_range,
                is_within_interval, humanize_duration, get_current_timestamp
    validation: MetricSchemaValidator, FilterParameterValidator,
                ThresholdConsistencyVerifier, validate_command_metric,
                validate_task_metric, validate_filter,
                validate_performance_record, verify_threshold, _within_bounds
"""

from .validation import (
    FilterParameterValidator,
    MetricSchemaValidator,
    ThresholdConsistencyVerifier,
)

__all__: list[str] = [
    "FilterParameterValidator",
    "MetricSchemaValidator",
    "ThresholdConsistencyVerifier",
]
