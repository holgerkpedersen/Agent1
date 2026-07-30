from typing import Union

from performance_dashboard.models import (
    CommandMetric,
    TaskMetric,
    TimeSeriesPoint,
    TrendAnalysisResult,
    StatisticalSummary,
)
from performance_dashboard.storage.metrics_store import RollingWindowAggregator


def calculate_performance_trends(
    records: list[Union[CommandMetric, TaskMetric]],
    time_window_seconds: float = 60.0,
) -> list[TrendAnalysisResult]:
    """Identify directional changes in efficiency over time windows."""
    results: list[TrendAnalysisResult] = []

    if not records:
        return results

    aggregator = RollingWindowAggregator(threshold_seconds=int(time_window_seconds))
    summary = aggregator.aggregate_records(records)  # type: ignore[attr-defined]

    for record in records:
        scalar_value: float | None = None

        if isinstance(record, CommandMetric):
            scalar_value = record.execution_time_ms / max(record.cpu_utilization_percent, 1.0)
        elif isinstance(record, TaskMetric):
            scalar_value = record.duration_seconds / max(record.success_rate, 1e-6)

        if scalar_value is None:
            continue

        trend_result = TrendAnalysisResult(
            trend_direction="improving" if scalar_value < summary.mean_value else "degrading",
            slope_coefficient=scalar_value - summary.mean_value,
            confidence_score=max(0.0, min(1.0, 1.0 - abs(scalar_value - summary.mean_value) / max(summary.standard_deviation, 1e-6))),
            time_window_seconds=time_window_seconds,
            anomaly_detected=abs(scalar_value - summary.mean_value) > (2 * summary.standard_deviation),
        )

        results.append(trend_result)

    return results


def compute_statistical_distributions(
    records: list[Union[CommandMetric, TaskMetric]],
) -> dict[str, StatisticalSummary]:
    """Determine normal ranges and outliers for various metric types."""
    distributions: dict[str, StatisticalSummary] = {}

    if not records:
        return distributions

    aggregator = RollingWindowAggregator()

    command_metrics: list[CommandMetric] = []
    task_metrics: list[TaskMetric] = []

    for record in records:
        if isinstance(record, CommandMetric):
            command_metrics.append(record)
        elif isinstance(record, TaskMetric):
            task_metrics.append(record)

    # Command metric distribution (execution time)
    exec_times: list[float] = [m.execution_time_ms for m in command_metrics]
    cmd_summary = aggregator._statistical_summary(exec_times)
    if cmd_summary is not None:
        distributions["command_execution_time"] = cmd_summary

    # Task metric distribution (duration seconds)
    durations: list[float] = [m.duration_seconds for m in task_metrics]
    task_summary = aggregator._statistical_summary(durations)
    if task_summary is not None:
        distributions["task_duration"] = task_summary

    return distributions


def generate_comparative_reports(
    baseline_records: list[Union[CommandMetric, TaskMetric]],
    comparison_records: list[Union[CommandMetric, TaskMetric]],
) -> dict[str, Union[StatisticalSummary, TrendAnalysisResult]]:
    """Generate comparative reports between two sets of performance data."""
    report: dict[str, Union[StatisticalSummary, TrendAnalysisResult]] = {}

    baseline_dists = compute_statistical_distributions(baseline_records)
    comparison_dists = compute_statistical_distributions(comparison_records)

    for key in set(baseline_dists.keys()) & set(comparison_dists.keys()):
        baseline_summary = baseline_dists[key]
        comparison_summary = comparison_dists[key]

        trend_result = TrendAnalysisResult(
            trend_direction="improving" if comparison_summary.mean_value < baseline_summary.mean_value else "degrading",
            slope_coefficient=comparison_summary.mean_value - baseline_summary.mean_value,
            confidence_score=max(0.0, min(1.0, 1.0 - abs(comparison_summary.mean_value - baseline_summary.mean_value) / max(baseline_summary.standard_deviation, 1e-6))),
            time_window_seconds=60.0,
            anomaly_detected=abs(comparison_summary.mean_value - baseline_summary.mean_value) > (2 * baseline_summary.standard_deviation),
        )

        report[key] = trend_result

    return report


def _extract_metric_scalar(record: Union[CommandMetric, TaskMetric]) -> float | None:
    """Extract a normalized scalar value from either metric type."""
    if isinstance(record, CommandMetric):
        return record.execution_time_ms / max(record.cpu_utilization_percent, 1.0)
    elif isinstance(record, TaskMetric):
        return record.duration_seconds / max(record.success_rate, 1e-6)
    return None


def _build_trend_result(
    scalar_value: float,
    summary: StatisticalSummary,
    time_window_seconds: float = 60.0,
) -> TrendAnalysisResult:
    """Build a trend analysis result from a scalar value and statistical summary."""
    deviation_ratio: float = abs(scalar_value - summary.mean_value) / max(summary.standard_deviation, 1e-6)

    return TrendAnalysisResult(
        trend_direction="improving" if scalar_value < summary.mean_value else "degrading",
        slope_coefficient=scalar_value - summary.mean_value,
        confidence_score=max(0.0, min(1.0, 1.0 - deviation_ratio)),
        time_window_seconds=time_window_seconds,
        anomaly_detected=deviation_ratio > 2.0,
    )


def analyze_time_series_trends(points: list[TimeSeriesPoint]) -> TrendAnalysisResult | None:
    """Analyze trends from a series of time-series points."""
    if not points or len(points) < 2:
        return None

    values: list[float] = [p.value for p in points]
    aggregator = RollingWindowAggregator()
    summary = aggregator._statistical_summary(values)

    if summary is None:
        return None

    latest_value: float = points[-1].value
    earliest_value: float = points[0].value
    slope_coefficient: float = latest_value - earliest_value

    trend_direction: str | None = "improving" if slope_coefficient < 0 else "degrading"

    confidence_score: float = max(
        0.0, min(1.0, 1.0 - abs(latest_value - summary.mean_value) / max(summary.standard_deviation, 1e-6))
    )

    anomaly_detected: bool = abs(latest_value - summary.mean_value) > (2 * summary.standard_deviation)

    return TrendAnalysisResult(
        trend_direction=trend_direction,
        slope_coefficient=slope_coefficient,
        confidence_score=confidence_score,
        time_window_seconds=float(points[-1].timestamp - points[0].timestamp),
        anomaly_detected=anomaly_detected,
    )