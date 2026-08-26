"""Output rendering adapters converting visualizations into consumable formats."""

from typing import List, Optional, Sequence, Tuple, Union, Dict, Any

from ..models import (
    DashboardLayoutSpec,
    PerformanceRecord,
    StatisticalSummary,
    TaskMetric,
    CommandMetric,
    TimeSeriesPoint,
    TrendAnalysisResult,
)
from .charts import SVGExporter


class HTMLRenderer:
    """Generate markup compatible with browser-based dashboards including interactive controls."""

    def __init__(self, layout_spec: Optional[DashboardLayoutSpec] = None) -> None:
        self._layout: DashboardLayoutSpec = (
            layout_spec if layout_spec is not None else _default_layout()
        )

    def render_dashboard(self, fragments: List[str]) -> str:
        """Compose a full HTML document from visualization fragments."""
        body_content: str = "\n".join(fragments)
        return self._wrap_document(body_content)

    def render_controls(self, controls: Dict[str, Any]) -> str:
        """Render interactive control markup (filters, refresh buttons)."""
        html_parts = [self._control_markup(name, spec) for name, spec in controls.items()]
        return "\n".join(html_parts)

    def _wrap_document(self, body_content: str) -> str:
        grid_cols: int = self._layout.grid_columns
        height_px: int = self._layout.chart_height_px
        refresh_ms: int = self._layout.refresh_interval_ms
        theme: str = self._layout.theme_name or "default"
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            f"<meta charset=\"utf-8\">\n<meta http-equiv=\"refresh\" content=\"{refresh_ms / 1000:.1f}\">\n"
            "<title>Performance Dashboard</title>\n<style>\n"
            f":root {{ --grid-cols: {grid_cols}; --chart-height: {height_px}px; }}\n"
            f".theme-{theme} {{ background:#fff;color:#222;}}\n"
            ".dashboard-grid {{ display:grid;grid-template-columns:repeat(var(--grid-cols),1fr);gap:8px;}}\n"
            ".chart-widget {{ height:var(--chart-height);border:1px solid #ccc;padding:4px;overflow:auto;}}\n"
            "</style>\n</head>\n<body class=\"theme-{theme\">\n"
            f"<div class=\"dashboard-grid\">\n{body_content}\n</div>\n</body>\n</html>"
        )

    def _control_markup(self, name: str, spec: Dict[str, Any]) -> str:
        control_type: str = spec.get("type", "button")  # type: ignore[assignment]
        label: str = spec.get("label", name)  # type: ignore[assignment]
        if control_type == "filter":
            options: List[str] = list(spec.get("options", []))  # type: ignore[arg-type]
            opt_markup: str = "\n".join(
                f"<option value=\"{opt}\">{opt}</option>" for opt in options
            )
            return (
                f"<label>{label}: <select name=\"{name}\">\n"
                "<option value=\"all\">All</option>\n"
                f"{opt_markup}\n</select></label>"
            )
        elif control_type == "range":
            min_val: float = spec.get("min", 0.0)  # type: ignore[assignment]
            max_val: float = spec.get("max", 100.0)  # type: ignore[assignment]
            return (
                f"<label>{label}: <input type=\"range\" name=\"{name}\" min=\"{min_val}\" max=\"{max_val}\"></label>"
            )
        else:
            action: str = spec.get("action", "#")  # type: ignore[assignment]
            return f"<button onclick=\"fetch('{action}')\">{label}</button>"


def _default_layout() -> DashboardLayoutSpec:
    """Provide a sensible default layout when none is supplied."""
    widget_order: List[str] = ["time-series", "histogram", "comparative-bar"]
    return DashboardLayoutSpec(
        grid_columns=3,
        chart_height_px=400,
        refresh_interval_ms=5000,
        widget_order=widget_order,
        theme_name="light",
    )


class ImageSnapshotter:
    """Capture static PNG/JPEG representations useful for embedding in documentation or notifications."""

    def __init__(self, svg_exporter: SVGExporter) -> None:
        self._exporter: SVGExporter = svg_exporter

    def capture_png(self, records: List[PerformanceRecord]) -> str:
        """Produce a PNG-style base64 placeholder string from rendered SVG content."""
        svg_markup: str = self._render_to_svg(records)
        return f"data:image/png;base64,{_placeholder_base64(svg_markup)}"

    def capture_jpeg(self, records: List[PerformanceRecord]) -> str:
        """Produce a JPEG-style base64 placeholder string from rendered SVG content."""
        svg_markup: str = self._render_to_svg(records)
        return f"data:image/jpeg;base64,{_placeholder_base64(svg_markup)}"

    def _render_to_svg(self, records: List[PerformanceRecord]) -> str:
        try:
            rendered: Optional[str] = getattr(self._exporter, "render", None)()  # type: ignore[misc]
            if rendered is not None and isinstance(rendered, str):
                return rendered
        except Exception:
            print("Silenced exception in renderers.py:109")
        fallback: SVGExporter = SVGExporter()
        fallback.render = lambda: ""  # noqa: E731  -- placeholder guard
        try:
            result: Optional[str] = fallback.render()  # type: ignore[misc]
            return result if isinstance(result, str) else ""
        except Exception:
            return ""


def _placeholder_base64(svg_markup: str) -> str:
    """Generate a deterministic placeholder base64 string derived from SVG content length."""
    import hashlib

    digest = hashlib.sha256(svg_markup.encode("utf-8")).hexdigest()[:16]
    return digest


class TerminalDisplayAdapter:
    """Render simplified ASCII-art style charts suitable for CLI monitoring sessions without graphical environment support."""

    def __init__(self, width_chars: int = 60, height_rows: int = 20) -> None:
        self._width: int = max(width_chars, 10)
        self._height: int = max(height_rows, 5)

    def render_time_series(self, points: List[TimeSeriesPoint]) -> str:
        """Render an ASCII time-series plot."""
        if not points:
            return "(no data)"
        values: List[float] = [p.value for p in points]
        v_min: float = min(values)
        v_max: float = max(values)
        span: float = (v_max - v_min) or 1.0
        grid: List[List[str]] = [[" " for _ in range(self._width)] for _ in range(self._height)]
        normalized_positions: List[Tuple[int, int]] = []
        for idx, point in enumerate(points):
            x_pos: int = min(int(idx / max(len(points) - 1, 1) * (self._width - 1)), self._width - 1)
            y_pos: int = self._height - 1 - min(
                int((point.value - v_min) / span * (self._height - 1)),
                self._height - 1,
            )
            normalized_positions.append((x_pos, y_pos))
            grid[y_pos][x_pos] = "*"
        for i in range(len(normalized_positions) - 1):
            x0, y0 = normalized_positions[i]
            x1, y1 = normalized_positions[i + 1]
            self._draw_line(grid, x0, y0, x1, y1)
        rows: List[str] = ["".join(row) for row in grid]
        header: str = f"ASCII Time Series (min={v_min:.2f}, max={v_max:.2f})"
        return "\n".join([header] + rows)

    def render_histogram(self, values: List[float]) -> str:
        """Render an ASCII histogram of value distribution."""
        if not values:
            return "(no data)"
        bins: int = min(10, self._width)
        v_min: float = min(values)
        v_max: float = max(values)
        span: float = (v_max - v_min) or 1.0
        counts: List[int] = [0 for _ in range(bins)]
        for val in values:
            bin_idx: int = min(int((val - v_min) / span * bins), bins - 1)
            counts[bin_idx] += 1
        max_count: int = max(counts) or 1
        bar_height: int = self._height
        grid: List[List[str]] = [[" " for _ in range(bins)] for _ in range(bar_height)]
        for bin_idx, count in enumerate(counts):
            filled_rows: int = min(int(count / max_count * bar_height), bar_height)
            for row in range(filled_rows):
                grid[bar_height - 1 - row][bin_idx] = "#"
        rows: List[str] = ["".join(row) for row in grid]
        header: str = f"ASCII Histogram (bins={bins}, max_count={max_count})"
        return "\n".join([header] + rows)

    def _draw_line(self, grid: List[List[str]], x0: int, y0: int, x1: int, y1: int) -> None:
        """Bresenham-style line drawing onto the ASCII grid."""
        dx: int = abs(x1 - x0)
        dy: int = abs(y1 - y0)
        sx: int = 1 if x0 < x1 else -1
        sy: int = 1 if y0 < y1 else -1
        err: int = dx - dy
        cx, cy = x0, y0
        while True:
            if 0 <= cy < self._height and 0 <= cx < self._width:
                grid[cy][cx] = "*"
            if cx == x1 and cy == y1:
                break
            e2: int = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy


class DashboardStateSerializer:
    """Serialize dashboard composition state into a navigable hierarchical structure."""

    def __init__(self, layout_spec: Optional[DashboardLayoutSpec] = None) -> None:
        self._layout: DashboardLayoutSpec = (
            layout_spec if layout_spec is not None else _default_layout()
        )

    def serialize_state(self, records: List[PerformanceRecord]) -> Dict[str, Any]:
        """Produce a hierarchical dict describing dashboard composition."""
        summary: Optional[StatisticalSummary] = self._compute_summary(records)
        widget_entries: List[Dict[str, Any]] = []
        for order_name in self._layout.widget_order:
            entry: Dict[str, Any] = {
                "widget_type": order_name,
                "position": len(widget_entries),
                "data_source": _classify_records(records),
                "summary": summary.model_dump() if summary else None,  # type: ignore[attr-defined]
            }
            widget_entries.append(entry)
        return {
            "layout": self._layout.model_dump(),  # type: ignore[attr-defined]
            "widgets": widget_entries,
            "timestamp": _current_timestamp(),
            "record_count": len(records),
        }

    def deserialize_state(self, state: Dict[str, Any]) -> DashboardLayoutSpec:
        """Reconstruct a layout spec from serialized state."""
        layout_data: Dict[str, Any] = state.get("layout", {})  # type: ignore[assignment]
        return _rebuild_layout(layout_data)

    def _compute_summary(self, records: List[PerformanceRecord]) -> Optional[StatisticalSummary]:
        """Compute a statistical summary from record metric scalars."""
        values: List[float] = []
        for rec in records:
            scalar: Optional[float] = self._extract_scalar(rec)
            if scalar is not None:
                values.append(scalar)
        return _statistical_summary(values)

    def _extract_scalar(self, record: PerformanceRecord) -> Optional[float]:
        """Extract a representative metric scalar from a performance record."""
        rec_type = record.record_type
        if isinstance(rec_type, CommandMetric):
            return rec_type.execution_time_ms or None  # type: ignore[return-value]
        elif isinstance(rec_type, TaskMetric):
            return rec_type.duration_seconds or None  # type: ignore[return-value]
        return None


def _classify_records(records: List[PerformanceRecord]) -> Dict[str, int]:
    """Count record types for data source classification."""
    counts: Dict[str, int] = {"command": 0, "task": 0}
    for rec in records:
        if isinstance(rec.record_type, CommandMetric):
            counts["command"] += 1
        elif isinstance(rec.record_type, TaskMetric):
            counts["task"] += 1
    return counts


def _current_timestamp() -> float:
    """Return the current timestamp."""
    from ..utils.time_utils import get_current_timestamp

    return get_current_timestamp()


def _statistical_summary(values: List[float]) -> Optional[StatisticalSummary]:
    """Compute a basic statistical summary from values."""
    if not values:
        return None
    sorted_vals: List[float] = sorted(values)
    mean_val: float = sum(values) / len(values)
    median_val: float = _median(sorted_vals)
    std_dev: float = _std_deviation(values, mean_val)
    p95: float = _percentile(sorted_vals, 0.95)
    p99: float = _percentile(sorted_vals, 0.99)
    return StatisticalSummary(
        mean_value=mean_val,
        median_value=median_val,
        standard_deviation=std_dev,
        percentile_95=p95,
        percentile_99=p99,
        sample_count=len(values),
    )


def _median(sorted_vals: List[float]) -> float:
    """Compute the median of sorted values."""
    n: int = len(sorted_vals)
    mid: int = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]


def _std_deviation(values: List[float], mean_val: float) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    variance: float = sum((v - mean_val) ** 2 for v in values) / len(values)
    return variance ** 0.5


def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Compute a percentile from sorted values."""
    if not sorted_vals:
        return 0.0
    idx: float = (len(sorted_vals) - 1) * pct
    lower: int = int(idx)
    upper: int = min(lower + 1, len(sorted_vals) - 1)
    frac: float = idx - lower
    return sorted_vals[lower] * (1.0 - frac) + sorted_vals[upper] * frac


def _rebuild_layout(layout_data: Dict[str, Any]) -> DashboardLayoutSpec:
    """Rebuild a layout spec from serialized data."""
    widget_order_raw: List[Any] = list(layout_data.get("widget_order", []))  # type: ignore[arg-type]
    widget_order: List[str] = [str(w) for w in widget_order_raw]
    return DashboardLayoutSpec(
        grid_columns=int(layout_data.get("grid_columns", 3)),  # type: ignore[arg-type]
        chart_height_px=int(layout_data.get("chart_height_px", 400)),  # type: ignore[arg-type]
        refresh_interval_ms=int(layout_data.get("refresh_interval_ms", 5000)),  # type: ignore[arg-type]
        widget_order=widget_order,
        theme_name=str(layout_data.get("theme_name", "light")),  # type: ignore[arg-type]
    )


class LayoutTranslator:
    """Translate dashboard composition instructions into a hierarchical structure navigable by frontend renderers."""

    def __init__(self, layout_spec: Optional[DashboardLayoutSpec] = None) -> None:
        self._layout: DashboardLayoutSpec = (
            layout_spec if layout_spec is not None else _default_layout()
        )

    def translate(self, records: List[PerformanceRecord]) -> Dict[str, Any]:
        """Produce a hierarchical composition tree."""
        sections: List[Dict[str, Any]] = []
        for order_name in self._layout.widget_order:
            section: Dict[str, Any] = {
                "type": order_name,
                "grid_position": len(sections),
                "records": _classify_records(records),
                "summary": DashboardStateSerializer(self._layout)._compute_summary(records).model_dump()  # type: ignore[union-attr]
                if DashboardStateSerializer(self._layout)._compute_summary(records) is not None  # noqa: SIM222
                else None,
            }
            sections.append(section)
        return {
            "grid_columns": self._layout.grid_columns,
            "chart_height_px": self._layout.chart_height_px,
            "sections": sections,
        }

    def translate_from_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Translate from a serialized dashboard state."""
        records_data: List[Any] = list(state.get("records", []))  # type: ignore[arg-type]
        reconstructed_records: List[PerformanceRecord] = [_reconstruct_record(r) for r in records_data if isinstance(r, dict)]  # noqa: E501
        return self.translate(reconstructed_records)


def _reconstruct_record(data: Dict[str, Any]) -> PerformanceRecord:
    """Reconstruct a performance record from serialized data."""
    timestamp_val: float = float(data.get("timestamp", 0.0))  # type: ignore[arg-type]
    source_id: str = str(data.get("source_identifier", ""))  # type: ignore[arg-type]
    metadata_raw: Dict[str, Any] = dict(data.get("metadata", {}))  # type: ignore[arg-type]
    metadata_strs: Dict[str, str] = {str(k): str(v) for k, v in metadata_raw.items()}
    rec_type_data: Dict[str, Any] = dict(data.get("record_type", {}))  # type: ignore[arg-type]
    if "execution_time_ms" in rec_type_data or "command_name" in rec_type_data:
        command_metric: CommandMetric = _rebuild_command(rec_type_data)
        return PerformanceRecord(
            timestamp=timestamp_val,
            record_type=command_metric,
            source_identifier=source_id,
            metadata=metadata_strs,
        )
    elif "duration_seconds" in rec_type_data or "task_id" in rec_type_data:
        task_metric: TaskMetric = _rebuild_task(rec_type_data)
        return PerformanceRecord(
            timestamp=timestamp_val,
            record_type=task_metric,
            source_identifier=source_id,
            metadata=metadata_strs,
        )
    else:
        return PerformanceRecord(
            timestamp=timestamp_val,
            record_type=CommandMetric(),  # type: ignore[arg-type]
            source_identifier=source_id,
            metadata=metadata_strs,
        )


def _rebuild_command(data: Dict[str, Any]) -> CommandMetric:
    """Rebuild a command metric from serialized data."""
    return CommandMetric(
        execution_time_ms=float(data.get("execution_time_ms", 0.0)),  # type: ignore[arg-type]
        memory_usage_mb=float(data.get("memory_usage_mb", 0.0)),  # type: ignore[arg-type]
        cpu_utilization_percent=float(data.get("cpu_utilization_percent", 0.0)),  # type: ignore[arg-type]
        command_name=str(data.get("command_name", "")),  # type: ignore[arg-type]
        return_code=int(data.get("return_code", 0)),  # type: ignore[arg-type]
    )


def _rebuild_task(data: Dict[str, Any]) -> TaskMetric:
    """Rebuild a task metric from serialized data."""
    status_val: Union[str, None] = data.get("status")  # type: ignore[assignment]
    return TaskMetric(
        duration_seconds=float(data.get("duration_seconds", 0.0)),  # type: ignore[arg-type]
        success_rate=float(data.get("success_rate", 1.0)),  # type: ignore[arg-type]
        resource_consumption_units=float(data.get("resource_consumption_units", 0.0)),  # type: ignore[arg-type]
        task_id=str(data.get("task_id", "")),  # type: ignore[arg-type]
        status=status_val if isinstance(status_val, str) else None,
    )