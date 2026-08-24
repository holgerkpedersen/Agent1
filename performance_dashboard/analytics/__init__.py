"""Analytics subpackage for the performance dashboard.

Provides metric processing, aggregation utilities and trend analysis
components used to derive insights from collected performance records.

Real public surface (verified against the modules):
    processor: calculate_performance_trends, compute_statistical_distributions,
               generate_comparative_reports, analyze_time_series_trends,
               RollingWindowAggregator
    aggregator: AggregationEngine, HierarchicalAggregator, SummaryBuilder,
                StatisticalSummaryGenerator, TrendDetector, CorrelationMapper
"""

from .processor import (
    RollingWindowAggregator,
    analyze_time_series_trends,
    calculate_performance_trends,
    compute_statistical_distributions,
    generate_comparative_reports,
)
from .aggregator import (
    AggregationEngine,
    CorrelationMapper,
    HierarchicalAggregator,
    StatisticalSummaryGenerator,
    SummaryBuilder,
    TrendDetector,
)

__all__: list[str] = [
    "RollingWindowAggregator",
    "analyze_time_series_trends",
    "calculate_performance_trends",
    "compute_statistical_distributions",
    "generate_comparative_reports",
    "AggregationEngine",
    "CorrelationMapper",
    "HierarchicalAggregator",
    "StatisticalSummaryGenerator",
    "SummaryBuilder",
    "TrendDetector",
]

__version__: str = "1.0.0"
