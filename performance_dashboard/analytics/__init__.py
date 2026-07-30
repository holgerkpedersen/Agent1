"""Analytics subpackage for the performance dashboard.

Provides metric processing, aggregation utilities and trend analysis
components used to derive insights from collected performance records.
"""

from .processor import AnalyticsProcessor
from .aggregator import RollingAggregator

__all__: list[str] = [
    "AnalyticsProcessor",
    "RollingAggregator",
]

__version__: str = "1.0.0"