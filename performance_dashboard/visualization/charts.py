from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

from performance_dashboard.analytics.aggregator import StatisticalSummaryGenerator
from performance_dashboard.analytics.processor import _extract_metric_scalar, analyze_time_series_trends
from performance_dashboard.models import (
    AlertThreshold,
    CommandMetric,
    PerformanceRecord,
    TaskMetric,
    TimeSeriesPoint,
    TrendAnalysisResult,
)
from performance_dashboard.utils.time_utils import format_display_label, normalize_timezone
from performance_dashboard.utils.validation import _within_bounds


def _normalize(value: float, low: float, high: float, out_min: float, out_max: float) -> float:
    span = high - low
    if span == 0.0:
        return (out_min + out_max) / 2.0
    ratio = (value - low) / span
    return out_min + ratio * (out_max - out_min)


def _points_to_path(points: Sequence[Tuple[float, float]]) -> str:
    if not points:
        return ""
    parts: List[str] = [f"M {points[0][0]:.2f},{points[0][1]:.2f}"]
    parts += [f"L {x:.2f},{y:.2f}" for x, y in list(points)[1:]]
    return " ".join(parts)


def _polygon_markup(points: Sequence[Tuple[float, float]]) -> str:
    if not points:
        return ""
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in list(points))
    return f'<polygon points="{coords}" />'


def _summarize(values: List[float]) -> Optional[StatisticalSummary]:  # noqa: F821
    if not values:
        return None
    engine = StatisticalSummaryGenerator()
    for v in values:
        engine.add_value(v)
    return engine.generate_summary()


class TemporalRenderer(ABC):
    def __init__(self, width: float = 800.0, height: float = 400.0, margin: float = 50.0) -> None:
        self.width = width
        self.height = height
        self.margin = margin

    @abstractmethod
    def render(self) -> str: ...

    def _x_for_timestamp(self, ts: float, t_low: float, t_high: float) -> float:
        return _normalize(ts, t_low, t_high, self.margin, self.width - self.margin)

    def _y_for_value(self, value: float, v_low: float, v_high: float) -> float:
        return _normalize(value, v_low, v_high, self.height - self.margin, self.margin)

    @staticmethod
    def _group(markup: str) -> str:
        return f"<g>{markup}</g>"


class TimeSeriesPlotter(TemporalRenderer):
    def __init__(
        self, points: List[TimeSeriesPoint], width: float = 800.0, height: float = 400.0, margin: float = 50.0
    ) -> None:
        super().__init__(width, height, margin)
        self.points = sorted(points, key=lambda p: p.timestamp)

    def _trend_overlay(self, t_low: float, t_high: float, v_base: float, v_low: float, v_high: float) -> str:
        result: Optional[TrendAnalysisResult] = analyze_time_series_trends(self.points)
        if result is None or result.trend_direction is None:
            return ""
        x_start = self._x_for_timestamp(t_low, t_low, t_high)
        y_base = self._y_for_value(v_base, v_low, v_high)
        trend_end_v = v_base + result.slope_coefficient * (t_high - t_low)
        x_end = self._x_for_timestamp(t_high, t_low, t_high)
        y_end = self._y_for_value(trend_end_v, v_low, v_high)
        markup: List[str] = [f'<line x1="{x_start:.2f}" y1="{y_base:.2f}" x2="{x_end:.2f}" y2="{y_end:.2f}" stroke="steelblue" stroke-width="2" />']
        if result.anomaly_detected:
            markup.append('<text x="{:.2f}" y="{:.2f}" fill="crimson">anomaly</text>'.format(x_end, y_end))
        return " ".join(markup)

    def render(self) -> str:
        if not self.points:
            return TemporalRenderer._group("")
        t_low = min(p.timestamp for p in self.points)
        t_high = max(p.timestamp for p in self.points)
        v_vals = [p.value for p in self.points]
        v_low = min(v_vals)
        v_high = max(v_vals)

        line_pts: List[Tuple[float, float]] = []
        upper_pts: List[Tuple[float, float]] = []
        lower_pts: List[Tuple[float, float]] = []
        for p in self.points:
            x = self._x_for_timestamp(p.timestamp, t_low, t_high)
            line_pts.append((x, self._y_for_value(p.value, v_low, v_high)))
            upper_pts.append((x, self._y_for_value(p.confidence_interval_upper, v_low, v_high)))
            lower_pts.append((x, self._y_for_value(p.confidence_interval_lower, v_low, v_high)))

        band = _polygon_markup(upper_pts + list(reversed(lower_pts)))
        trend = self._trend_overlay(t_low, t_high, v_vals[0], v_low, v_high)
        start_label = format_display_label(normalize_timezone(t_low))
        x_start_lbl = self._x_for_timestamp(t_low, t_low, t_high)
        label = f'<text x="{x_start_lbl:.2f}" y="{self.height - 10:.2f}" fill="black">{start_label}</text>'
        markup: List[str] = [
            '<polyline points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in line_pts) + '" fill="none" stroke="steelblue" stroke-width="1.5" />',
            f'<polygon fill="steelblue" fill-opacity="0.18" stroke="none">{band}</polygon>' if band else "",
            trend,
            label,
        ]
        return TemporalRenderer._group(" ".join(m for m in markup if m))


class HistogramBuilder(ABC):
    def __init__(self, width: float = 640.0, height: float = 320.0, margin: float = 40.0) -> None:
        self.width = width
        self.height = height
        self.margin = margin

    @abstractmethod
    def render(self) -> str: ...


class DistributionHistogram(HistogramBuilder):
    def __init__(self, values: List[float], buckets: int = 10, width: float = 640.0, height: float = 320.0, margin: float = 40.0) -> None:
        super().__init__(width, height, margin)
        self.values = values
        self.buckets = max(1, buckets)

    def build_from_records(self, records: List[PerformanceRecord]) -> "DistributionHistogram":
        extracted: List[float] = []
        for r in records:
            scalar = _extract_metric_scalar(r)
            if scalar is not None:
                extracted.append(scalar)
        return DistributionHistogram(extracted, self.buckets, self.width, self.height, self.margin)

    def render(self) -> str:
        if not self.values:
            return "<g></g>"
        v_min = min(self.values)
        v_max = max(self.values)
        span = (v_max - v_min) / self.buckets or 1.0
        counts: List[int] = [0] * self.buckets
        for v in self.values:
            idx = int((v - v_min) / span) if v < v_max else self.buckets - 1
            counts[min(idx, self.buckets - 1)] += 1
        max_count = max(counts) or 1
        summary = _summarize(self.values)

        bar_w = (self.width - 2 * self.margin) / self.buckets
        parts: List[str] = []
        for i, c in enumerate(counts):
            x = self.margin + i * bar_w
            bh = _normalize(c, 0.0, float(max_count), 0.0, self.height - 2 * self.margin)
            y = self.height - self.margin - bh
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" fill="seagreen" stroke="white" />')

        stats: List[str] = []
        if summary is not None:
            for label, val in [("mean", summary.mean_value), ("median", summary.median_value), ("p95", summary.percentile_95)]:
                stats.append(f'<text x="{self.margin:.2f}" y="{self.margin + 14 * len(stats):.2f}" fill="black">{label}={val:.2f}</text>')
        return "<g>" + " ".join(parts) + " ".join(stats) + "</g>"


class ComparativeBarChart(HistogramBuilder):
    def __init__(self, groups: Dict[str, List[float]], width: float = 640.0, height: float = 320.0, margin: float = 40.0) -> None:
        super().__init__(width, height, margin)
        self.groups = {k: v for k, v in groups.items() if v}

    def from_command_metrics(self, metrics: List[CommandMetric]) -> "ComparativeBarChart":
        grouped: Dict[str, List[float]] = {}
        for m in metrics:
            grouped.setdefault(m.command_name, []).append(m.execution_time_ms)
        return ComparativeBarChart(grouped, self.width, self.height, self.margin)

    def from_task_metrics(self, metrics: List[TaskMetric]) -> "ComparativeBarChart":
        grouped: Dict[str, List[float]] = {}
        for m in metrics:
            if m.duration_seconds is not None:  # noqa: F821
                grouped.setdefault(m.task_id, []).append(m.duration_seconds)
        return ComparativeBarChart(grouped, self.width, self.height, self.margin)

    def render(self) -> str:
        if not self.groups:
            return "<g></g>"
        means: Dict[str, float] = {}
        for name, vals in self.groups.items():
            summary = _summarize(vals)
            means[name] = summary.mean_value if summary is not None else 0.0
        max_mean = max(means.values()) or 1.0

        bar_w = (self.width - 2 * self.margin) / len(self.groups)
        ordered: List[str] = list(self.groups.keys())
        parts: List[str] = []
        prev_x: Optional[float] = None
        for i, name in enumerate(ordered):
            x = self.margin + i * bar_w
            bh = _normalize(means[name], 0.0, max_mean, 0.0, self.height - 2 * self.margin)
            y = self.height - self.margin - bh
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" fill="darkorange" stroke="white" />')
            parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{self.height - self.margin + 14:.2f}" text-anchor="middle" fill="black">{name}</text>')
            if prev_x is not None:
                delta = abs(means[name] - means[ordered[i - 1]])
                mid_x = (prev_x + x) / 2.0
                parts.append(f'<line x1="{mid_x:.2f}" y1="{self.height - self.margin:.2f}" x2="{mid_x:.2f}" y2="{y:.2f}" stroke="crimson" />')
                parts.append(f'<text x="{mid_x:.2f}" y="{self.height - self.margin + 30:.2f}" text-anchor="middle" fill="crimson">delta={delta:.2f}</text>')
            prev_x = x

        return "<g>" + " ".join(parts) + "</g>"


class AnomalyHighlightMarker(TemporalRenderer):
    def __init__(self, points: List[TimeSeriesPoint], threshold: Optional[AlertThreshold] = None, width: float = 800.0, height: float = 400.0, margin: float = 50.0) -> None:
        super().__init__(width, height, margin)
        self.points = sorted(points, key=lambda p: p.timestamp)
        self.threshold = threshold

    def render(self) -> str:
        if not self.points:
            return "<g></g>"
        t_low = min(p.timestamp for p in self.points)
        t_high = max(p.timestamp for p in self.points)
        v_vals = [p.value for p in self.points]
        v_min = min(v_vals)
        v_max = max(v_vals)

        bounds: Tuple[float, float, float] = (v_min, v_max, 0.0)
        if self.threshold is not None:
            warning = self.threshold.warning_limit
            error = self.threshold.error_limit
            summary = _summarize(v_vals)
            p95 = summary.percentile_95 if summary is not None else warning
            bounds = (p95, error, 0.0)

        markers: List[str] = []
        for p in self.points:
            x = self._x_for_timestamp(p.timestamp, t_low, t_high)
            y = self._y_for_value(p.value, v_min, v_max)
            upper_limit = bounds[1] if bounds[1] != 0.0 else float("inf")
            warning_limit = bounds[0]
            is_error = p.value > upper_limit and not _within_bounds(p.value, 0.0, upper_limit)
            is_warning = (not is_error) and p.value > warning_limit and not _within_bounds(p.value, 0.0, warning_limit)
            color = "crimson" if is_error else ("darkorange" if is_warning else "")
            if color:
                markers.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" />')

        return "<g>" + " ".join(markers) + "</g>"


class VectorFormatter(ABC):
    @abstractmethod
    def format_document(self, fragments: List[str], width: float = 800.0, height: float = 400.0, zoom: float = 1.0) -> str: ...


class SVGExporter(VectorFormatter):
    def __init__(self, width: float = 800.0, height: float = 400.0) -> None:
        self.width = width
        self.height = height

    def format_document(self, fragments: List[str], width: float = 800.0, height: float = 400.0, zoom: float = 1.0) -> str:
        doc_width = width if width > 0 else self.width
        doc_height = height if height > 0 else self.height
        view_w = doc_width / max(zoom, 0.0001)
        view_h = doc_height / max(zoom, 0.0001)
        body = " ".join(fragments)
        style = '<style>.band{fill-opacity:0.18}.line{stroke-width:1.5}</style>'
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_w:.2f} {view_h:.2f}" '
            f'preserveAspectRatio="xMidYMid meet"><g transform="scale({zoom:.4f})">{style}{body}</g></svg>'
        )


__all__: List[str] = [
    "TemporalRenderer",
    "TimeSeriesPlotter",
    "HistogramBuilder",
    "DistributionHistogram",
    "ComparativeBarChart",
    "AnomalyHighlightMarker",
    "VectorFormatter",
    "SVGExporter",
]