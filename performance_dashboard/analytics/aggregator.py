from typing import Dict, List, Optional, Union

from ..models import (
    CommandMetric,
    TaskMetric,
    PerformanceRecord,
    StatisticalSummary,
    TimeSeriesPoint,
    TrendAnalysisResult,
    DashboardFilter,
)
from .processor import analyze_time_series_trends


class AggregationEngine:
    """Base engine for aggregating performance records."""

    def __init__(self) -> None:
        self._records: List[PerformanceRecord] = []

    def ingest(self, record: PerformanceRecord) -> bool:
        # PerformanceRecord is a TypedDict (plain dict at runtime): verify
        # shape instead of isinstance(), which raises TypeError here.
        if not isinstance(record, dict) or "record_type" not in record or "timestamp" not in record:
            return False
        self._records.append(record)
        return True

    def batch_ingest(self, records: List[PerformanceRecord]) -> int:
        count = 0
        for r in records:
            if self.ingest(r):
                count += 1
        return count


class HierarchicalAggregator(AggregationEngine):
    """Roll up individual records into grouped summaries by command/task dimensions."""

    def __init__(self) -> None:
        super().__init__()
        self._group_keys: List[str] = []

    def aggregate_by_command(
        self, filter_spec: Optional[DashboardFilter] = None
    ) -> Dict[str, StatisticalSummary]:
        groups: Dict[str, List[float]] = {}
        for record in self._records:
            rt = record["record_type"]
            if "execution_time_ms" not in rt:
                continue
            cmd_name = str(rt["command_name"])
            scalar = _extract_metric_scalar(record)
            if scalar is None:
                continue
            groups.setdefault(cmd_name, []).append(scalar)

        results: Dict[str, StatisticalSummary] = {}
        for key, values in groups.items():
            summary = _statistical_summary(values)
            if summary is not None:
                results[key] = summary
        return results

    def aggregate_by_task(
        self, filter_spec: Optional[DashboardFilter] = None
    ) -> Dict[str, StatisticalSummary]:
        groups: Dict[str, List[float]] = {}
        for record in self._records:
            rt = record["record_type"]
            if "duration_seconds" not in rt:
                continue
            task_id = str(rt["task_id"])
            scalar = _extract_metric_scalar(record)
            if scalar is None:
                continue
            groups.setdefault(task_id, []).append(scalar)

        results: Dict[str, StatisticalSummary] = {}
        for key, values in groups.items():
            summary = _statistical_summary(values)
            if summary is not None:
                results[key] = summary
        return results


class SummaryBuilder:
    """Base builder for statistical summaries."""

    def __init__(self) -> None:
        self._values: List[float] = []

    def add_value(self, value: float) -> bool:
        if not isinstance(value, (int, float)):
            return False
        self._values.append(float(value))
        return True


class StatisticalSummaryGenerator(SummaryBuilder):
    """Produce mean/median/stdev values across metric populations."""

    def generate_summary(self) -> Optional[StatisticalSummary]:
        if not self._values:
            return None
        return _statistical_summary(self._values)


class TrendDetector:
    """Detect sustained shifts in performance metrics over time windows."""

    def __init__(self) -> None:
        self._points: List[TimeSeriesPoint] = []

    def add_point(self, point: TimeSeriesPoint) -> bool:
        # TimeSeriesPoint is a TypedDict: shape check, not isinstance().
        if not isinstance(point, dict) or "value" not in point or "timestamp" not in point:
            return False
        self._points.append(point)
        return True

    def detect_trends(
        self, window_seconds: int = 600
    ) -> Optional[TrendAnalysisResult]:
        filtered = [p for p in self._points
                    if isinstance(p.get("value"), (int, float))]
        if not filtered:
            return None
        result = analyze_time_series_trends(filtered)
        if result is None:
            return None
        adjusted = TrendAnalysisResult(
            trend_direction=result["trend_direction"],
            slope_coefficient=result["slope_coefficient"],
            confidence_score=result["confidence_score"],
            time_window_seconds=float(window_seconds),
            anomaly_detected=result["anomaly_detected"],
        )
        return adjusted


class CorrelationMapper:
    """Visualize interdependencies affecting performance across dimensions."""

    def __init__(self) -> None:
        self._command_data: Dict[str, List[float]] = {}
        self._task_data: Dict[str, List[float]] = {}

    def map_correlations(
        self, records: Optional[List[PerformanceRecord]] = None
    ) -> Dict[str, float]:
        if records is not None:
            for r in records:
                scalar = _extract_metric_scalar(r)
                if scalar is None:
                    continue
                rt = r["record_type"]
                if "execution_time_ms" in rt:
                    self._command_data.setdefault(str(rt["command_name"]), []).append(scalar)
                elif "duration_seconds" in rt:
                    self._task_data.setdefault(str(rt["task_id"]), []).append(scalar)

        correlations: Dict[str, float] = {}
        for cmd_key, cmd_vals in self._command_data.items():
            if len(cmd_vals) < 2:
                continue
            mean_cmd = sum(cmd_vals) / len(cmd_vals)
            for task_key, task_vals in self._task_data.items():
                if len(task_vals) < 2:
                    continue
                mean_task = sum(task_vals) / len(task_vals)
                numerator = sum(
                    (cmd_vals[i] - mean_cmd) * (task_vals[i] - mean_task)
                    for i in range(min(len(cmd_vals), len(task_vals)))
                )
                denom_cmd = sum((v - mean_cmd) ** 2 for v in cmd_vals)
                denom_task = sum((v - mean_task) ** 2 for v in task_vals)
                if denom_cmd > 0 and denom_task > 0:
                    corr = numerator / (denom_cmd * denom_task) ** 0.5
                    correlations[f"{cmd_key}__{task_key}"] = corr
        return correlations


def _extract_metric_scalar(record: PerformanceRecord) -> Optional[float]:
    """Extract a representative scalar value from a performance record."""
    rt = record["record_type"]
    if "execution_time_ms" in rt:
        return float(rt["execution_time_ms"])
    if "duration_seconds" in rt:
        return float(rt["duration_seconds"])
    return None


def _statistical_summary(values: List[float]) -> Optional[StatisticalSummary]:
    """Compute statistical distribution values from a population."""
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean_val = sum(sorted_vals) / n
    median_val = (
        sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    )
    variance = sum((v - mean_val) ** 2 for v in sorted_vals) / n if n > 0 else 0.0
    std_dev = variance ** 0.5
    p95_idx = int(n * 0.95) - 1 if n >= 2 else 0
    p99_idx = int(n * 0.99) - 1 if n >= 2 else 0
    percentile_95 = sorted_vals[max(p95_idx, 0)]
    percentile_99 = sorted_vals[max(p99_idx, 0)]
    return StatisticalSummary(
        mean_value=mean_val,
        median_value=median_val,
        standard_deviation=std_dev,
        percentile_95=percentile_95,
        percentile_99=percentile_99,
        sample_count=n,
    )