from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

# Severity level constants for theme application
SEVERITY_NORMAL = "normal"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"


class DashboardLayoutSpec(TypedDict):
    """Hierarchical structure specification navigable frontend renderers."""

    layout: dict[str, Any]
    widgets: list[dict[str, Any]]
    theme: dict[str, Any]
    responsive: dict[str, Any]


@dataclass
class ChartPanel:
    """Represents a single chart panel in the dashboard."""

    id: str
    importance: int  # Higher = more important placement priority
    group: str  # Logical grouping key
    data_source: Optional[str] = None
    dimensions: tuple[int, int] = (400, 300)
    content: dict[str, Any] = field(default_factory=dict)


@dataclass
class WidgetState:
    """Tracks synchronization state for visualization widgets."""

    widget_id: str
    data_source: Optional[str]
    last_update: float
    dirty: bool = True


class LayoutManager:
    """Arranges chart panels according importance logical grouping."""

    def __init__(self) -> None:
        self._panels: list[ChartPanel] = []
        self._grid_rows: int = 0
        self._grid_cols: int = 0

    def add_panel(self, panel: ChartPanel) -> None:
        """Add a chart panel to the layout manager."""
        self._panels.append(panel)

    def arrange(self) -> dict[str, Any]:
        """Arrange panels by importance within logical groups.

        Returns hierarchical layout structure for frontend renderers.
        """
        # Sort panels by importance (descending), then group them logically
        sorted_panels = sorted(
            self._panels, key=lambda p: (-p.importance, p.group)
        )

        # Determine grid dimensions based on panel count and max width
        total_width = sum(p.dimensions[0] for p in sorted_panels) or 1200

        self._grid_cols = max(1, min(len(sorted_panels), math.ceil(total_width / 400)))
        self._grid_rows = max(
            1, math.ceil(len(sorted_panels) / self._grid_cols)
        )

        # Build hierarchical layout with group-based nesting
        groups: dict[str, list[dict[str, Any]]] = {}
        for idx, panel in enumerate(sorted_panels):
            row = idx // self._grid_cols
            col = idx % self._grid_cols
            placement: dict[str, Any] = {
                "id": panel.id,
                "row": row,
                "col": col,
                "importance": panel.importance,
                "dimensions": list(panel.dimensions),
                "data_source": panel.data_source,
                "content": panel.content,
            }
            groups.setdefault(panel.group, []).append(placement)

        return {
            "grid_rows": self._grid_rows,
            "grid_cols": self._grid_cols,
            "groups": groups,
            "panels": [p.id for p in sorted_panels],
        }


class WidgetCoordinator:
    """Synchronizes updates multiple visualization elements sharing data source."""

    def __init__(self) -> None:
        self._states: dict[str, WidgetState] = {}
        self._source_watchers: dict[str, list[str]] = {}

    def register_widget(self, widget_id: str, data_source: Optional[str]) -> None:
        """Register a visualization widget with its data source."""
        state = WidgetState(
            widget_id=widget_id,
            data_source=data_source,
            last_update=0.0,
        )
        self._states[widget_id] = state
        if data_source is not None:
            watchers = self._source_watchers.setdefault(data_source, [])
            if widget_id not in watchers:
                watchers.append(widget_id)

    def notify_update(self, data_source: str, timestamp: float) -> list[str]:
        """Notify all widgets sharing a data source of an update.

        Returns list of affected widget IDs that need refresh.
        """
        affected = self._source_watchers.get(data_source, [])
        for wid in affected:
            state = self._states[wid]
            if state.last_update < timestamp or state.dirty:
                state.last_update = timestamp
                state.dirty = True
        return list(affected)

    def mark_clean(self, widget_id: str) -> None:
        """Mark a widget as synchronized (no pending updates)."""
        state = self._states.get(widget_id)
        if state is not None:
            state.dirty = False


class ThemeApplier:
    """Applies consistent color schemes typography reflecting severity levels."""

    # Severity-based color palettes and typography settings
    _SEVERITY_COLORS: dict[str, dict[str, str]] = {
        SEVERITY_NORMAL: {
            "primary": "#4A90D9",
            "secondary": "#6FA8DC",
            "accent": "#357ABD",
            "background": "#F5F7FA",
        },
        SEVERITY_WARNING: {
            "primary": "#FFC107",
            "secondary": "#FFD54F",
            "accent": "#FFA000",
            "background": "#FFF8E1",
        },
        SEVERITY_ERROR: {
            "primary": "#D32F2F",
            "secondary": "#EF5350",
            "accent": "#C62828",
            "background": "#FFEBEE",
        },
    }

    _SEVERITY_TYPOGRAPHY: dict[str, dict[str, str]] = {
        SEVERITY_NORMAL: {"font_weight": "400", "text_color": "#333333"},
        SEVERITY_WARNING: {"font_weight": "500", "text_color": "#8B4513"},
        SEVERITY_ERROR: {"font_weight": "600", "text_color": "#D32F2F"},
    }

    def apply_theme(
        self, severity: str = SEVERITY_NORMAL, overrides: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Apply consistent color scheme and typography based on severity.

        Returns theme specification for frontend renderers.
        """
        colors = dict(self._SEVERITY_COLORS.get(severity, self._SEVERITY_COLORS[SEVERITY_NORMAL]))
        typography = dict(
            self._SEVERITY_TYPOGRAPHY.get(severity, self._SEVERITY_TYPOGRAPHY[SEVERITY_NORMAL])
        )

        theme: dict[str, Any] = {
            "colors": colors,
            "typography": typography,
            "severity": severity,
        }

        if overrides is not None:
            theme.update(overrides)

        return theme


class ResponsiveAdapter:
    """Adjusts presentation fidelity display dimensions viewing contexts."""

    # Breakpoints for different viewing contexts
    _BREAKPOINTS: dict[str, tuple[int, int]] = {
        "mobile": (320, 568),
        "tablet": (768, 1024),
        "desktop": (1200, 800),
        "wide": (1920, 1080),
    }

    def adapt(
        self, dimensions: tuple[int, int], fidelity_hint: Optional[str] = None
    ) -> dict[str, Any]:
        """Adjust presentation based on display dimensions and viewing context.

        Returns responsive specification for frontend renderers.
        """
        width, height = dimensions

        # Determine viewing context from dimensions
        context = "desktop"
        if width <= self._BREAKPOINTS["mobile"][0]:
            context = "mobile"
        elif width <= self._BREAKPOINTS["tablet"][0]:
            context = "tablet"
        elif width >= self._BREAKPOINTS["wide"][0]:
            context = "wide"

        # Calculate fidelity level: high for wide, medium for desktop/tablet, low for mobile
        fidelity_levels = {"high": 3, "medium": 2, "low": 1}
        base_fidelity = {
            "mobile": "low",
            "tablet": "medium",
            "desktop": "medium",
            "wide": "high",
        }

        fidelity = fidelity_hint or base_fidelity.get(context, "medium")
        detail_level = fidelity_levels.get(fidelity, 2)

        # Adjust dimensions for context constraints
        constrained_width = min(width, self._BREAKPOINTS[context][0]) if width else width
        scale_factor = constrained_width / (self._BREAKPOINTS["desktop"][0] or 1)

        return {
            "context": context,
            "fidelity": fidelity,
            "detail_level": detail_level,
            "scale_factor": round(scale_factor, 2),
            "dimensions": [constrained_width, height],
            "breakpoints": dict(self._BREAKPOINTS),
        }


class DashboardRenderer:
    """Navigates frontend renderers using hierarchical layout spec."""

    def __init__(self) -> None:
        self.layout_manager = LayoutManager()
        self.widget_coordinator = WidgetCoordinator()
        self.theme_applier = ThemeApplier()
        self.responsive_adapter = ResponsiveAdapter()

    def build_spec(
        self,
        panels: list[ChartPanel],
        display_dimensions: tuple[int, int] = (1200, 800),
        severity: str = SEVERITY_NORMAL,
        fidelity_hint: Optional[str] = None,
    ) -> DashboardLayoutSpec:
        """Build complete dashboard layout specification for frontend renderers."""
        # Register all panels with coordinator
        for panel in panels:
            self.layout_manager.add_panel(panel)
            if panel.data_source is not None:
                self.widget_coordinator.register_widget(
                    f"widget_{panel.id}", panel.data_source
                )

        layout = self.layout_manager.arrange()
        theme = self.theme_applier.apply_theme(severity=severity)
        responsive = self.responsive_adapter.adapt(
            display_dimensions, fidelity_hint=fidelity_hint
        )

        widgets: list[dict[str, Any]] = []
        for panel in panels:
            widget_spec: dict[str, Any] = {
                "id": f"widget_{panel.id}",
                "panel_id": panel.id,
                "data_source": panel.data_source,
                "group": panel.group,
                "importance": panel.importance,
            }
            widgets.append(widget_spec)

        spec: DashboardLayoutSpec = {
            "layout": layout,
            "widgets": widgets,
            "theme": theme,
            "responsive": responsive,
        }
        return spec


__all__ = [
    "DashboardLayoutSpec",
    "ChartPanel",
    "WidgetState",
    "LayoutManager",
    "WidgetCoordinator",
    "ThemeApplier",
    "ResponsiveAdapter",
    "DashboardRenderer",
    "SEVERITY_NORMAL",
    "SEVERITY_WARNING",
    "SEVERITY_ERROR",
]