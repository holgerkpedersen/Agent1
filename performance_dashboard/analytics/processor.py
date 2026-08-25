"""Analytics processing for performance metrics.

All metric types (:data:`CommandMetric`, :class:`TaskMetric`,
:class:`TimeSeriesPoint`) are ``TypedDict``s — plain ``dict`` objects at
runtime.  They must therefore be discriminated by key presence and accessed
by subscription; ``isinstance()`` checks against TypedDict classes raise
``TypeError`` and attribute access raises ``AttributeError``.
"""

from __future__ import annotations

from typing import Union

from performance_dashboard.models import (
    CommandMetric,
    TaskMetric,
    TimeSeriesPoint,
    TrendAnalysisResult,
    StatisticalSummary,
)

MetricRecord = Union[CommandMetric, TaskMetric]


def _is_command_metric(record: MetricRecord) -> bool:
    """True when *record* carries CommandMetric keys."""
    return "execution_time_ms" in record


def _is_task_metric(record: MetricRecord) -> bool:
    """True when *record* carries TaskMetric keys."""
    return "duration_seconds" in record


class RollingWindowAggregator:
    """Statistical summarizer over a window of metric values.

    Accepts either raw floats or mixed command/task metric dicts; in the
    latter case each record is reduced to a scalar via
    :func:`_extract_metric_scalar`.
    """

    def __init__(self, threshold_seconds: float = 60.0) -> None:
        self.threshold_seconds = threshold_seconds

    def statistical_summary(
        self, values: list[float]
    ) -> StatisticalSummary | None:
        """Compute distribution statistics; None for an empty population."""
        if not values:
            return None
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean_val = sum(sorted_vals) / n
        median_val = (
            sorted_vals[n // 2] if n % 2 == 1
            else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
        )
        variance = sum((v - mean_val) ** 2 for v in sorted_vals) / n if n > 0 else 0.0
        p95_idx = int(n * 0.95) - 1 if n >= 2 else 0
        p99_idx = int(n * 0.99) - 1 if n >= 2 else 0
        return StatisticalSummary(
            mean_value=mean_val,
            median_value=median_val,
            standard_deviation=variance ** 0.5,
            percentile_95=sorted_vals[max(p95_idx, 0)],
            percentile_99=sorted_vals[max(p99_idx, 0)],
            sample_count=n,
        )

    def aggregate_records(
        self, records: list[MetricRecord]
    ) -> StatisticalSummary | None:
        """Reduce mixed metric records to one summary over their scalars."""
        scalars = [s for s in map(_extract_metric_scalar, records)
                   if s is not None]
        return self.statistical_summary(scalars)


def calculate_performance_trends(
    records: list[MetricRecord],
    time_window_seconds: float = 60.0,
) -> list[TrendAnalysisResult]:
    """Identify directional changes in efficiency over time windows."""
    results: list[TrendAnalysisResult] = []

    if not records:
        return results

    aggregator = RollingWindowAggregator(threshold_seconds=int(time_window_seconds))
    summary = aggregator.aggregate_records(records)

    if summary is None:
        return results

    for record in records:
        scalar_value = _extract_metric_scalar(record)
        if scalar_value is None:
            continue

        results.append(_build_trend_result(scalar_value, summary, time_window_seconds))

    return results


def compute_statistical_distributions(
    records: list[MetricRecord],
) -> dict[str, StatisticalSummary]:
    """Determine normal ranges and outliers for various metric types."""
    distributions: dict[str, StatisticalSummary] = {}

    if not records:
        return distributions

    aggregator = RollingWindowAggregator()

    exec_times: list[float] = [
        m["execution_time_ms"] for m in records if _is_command_metric(m)
    ]
    cmd_summary = aggregator.statistical_summary(exec_times)
    if cmd_summary is not None:
        distributions["command_execution_time"] = cmd_summary

    durations: list[float] = [
        m["duration_seconds"] for m in records if _is_task_metric(m)
    ]
    task_summary = aggregator.statistical_summary(durations)
    if task_summary is not None:
        distributions["task_duration"] = task_summary

    return distributions


def generate_comparative_reports(
    baseline_records: list[MetricRecord],
    comparison_records: list[MetricRecord],
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


def _extract_metric_scalar(record: MetricRecord) -> float | None:
    """Extract a normalized scalar value from either metric type."""
    if _is_command_metric(record):
        cpu = max(float(record.get("cpu_utilization_percent", 0.0)), 1.0)
        return float(record["execution_time_ms"]) / cpu
    if _is_task_metric(record):
        rate = max(float(record.get("success_rate", 0.0)), 1e-6)
        return float(record["duration_seconds"]) / rate
    return None


def _build_trend_result(
    scalar_value: float,
    summary: StatisticalSummary,
    time_window_seconds: float = 60.0,
) -> TrendAnalysisResult:
    """Build a trend analysis result from a scalar value and statistical summary."""
    mean_value = float(summary["mean_value"])
    std_dev = float(summary["standard_deviation"])
    deviation_ratio: float = abs(scalar_value - mean_value) / max(std_dev, 1e-6)

    return TrendAnalysisResult(
        trend_direction="improving" if scalar_value < mean_value else "degrading",
        slope_coefficient=scalar_value - mean_value,
        confidence_score=max(0.0, min(1.0, 1.0 - deviation_ratio)),
        time_window_seconds=time_window_seconds,
        anomaly_detected=deviation_ratio > 2.0,
    )


def analyze_time_series_trends(points: list[TimeSeriesPoint]) -> TrendAnalysisResult | None:
    """Analyze trends from a series of time-series points."""
    if not points or len(points) < 2:
        return None

    values: list[float] = [float(p["value"]) for p in points]
    aggregator = RollingWindowAggregator()
    summary = aggregator.statistical_summary(values)

    if summary is None:
        return None

    latest_value: float = values[-1]
    earliest_value: float = values[0]
    slope_coefficient: float = latest_value - earliest_value

    trend_direction: str | None = "improving" if slope_coefficient < 0 else "degrading"

    mean_value = float(summary["mean_value"])
    std_dev = float(summary["standard_deviation"])
    confidence_score: float = max(
        0.0, min(1.0, 1.0 - abs(latest_value - mean_value) / max(std_dev, 1e-6))
    )

    anomaly_detected: bool = abs(latest_value - mean_value) > (2 * std_dev)

    return TrendAnalysisResult(
        trend_direction=trend_direction,
        slope_coefficient=slope_coefficient,
        confidence_score=confidence_score,
        time_window_seconds=float(len(points)),
        anomaly_detected=anomaly_detected,
    )
