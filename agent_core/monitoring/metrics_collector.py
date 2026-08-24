from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from .types import MetricData, MetricType


class MetricsCollector:
    """Collects and stores performance metrics for monitoring and alerting."""

    def __init__(self) -> None:
        self._metrics_store: deque[MetricData] = deque(maxlen=10000)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock: threading.Lock = threading.Lock()

    def increment_counter(
        self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None
    ) -> None:
        with self._lock:
            self._counters[name] += value
            metric_data = MetricData(
                name=name,
                value=self._counters[name],
                timestamp=time.time(),
                metric_type=MetricType.COUNTER,
                tags=tags or {},
            )
            self._metrics_store.append(metric_data)

    def set_gauge(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        with self._lock:
            self._gauges[name] = value
            metric_data = MetricData(
                name=name,
                value=value,
                timestamp=time.time(),
                metric_type=MetricType.GAUGE,
                tags=tags or {},
            )
            self._metrics_store.append(metric_data)

    def record_histogram(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        with self._lock:
            samples = self._histograms[name]
            samples.append(value)
            if len(samples) > 100:
                del samples[:-100]
            metric_data = MetricData(
                name=name,
                value=value,
                timestamp=time.time(),
                metric_type=MetricType.HISTOGRAM,
                tags=tags or {},
            )
            self._metrics_store.append(metric_data)

    def record_timer(
        self, name: str, duration: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        with self._lock:
            metric_data = MetricData(
                name=name,
                value=duration,
                timestamp=time.time(),
                metric_type=MetricType.TIMER,
                tags=tags or {},
            )
            self._metrics_store.append(metric_data)

    def get_metrics(
        self,
        name_filter: Optional[str] = None,
        type_filter: Optional[MetricType] = None,
    ) -> List[MetricData]:
        with self._lock:
            filtered: List[MetricData] = []
            for metric in list(self._metrics_store):
                if name_filter is not None and not metric.name.startswith(name_filter):
                    continue
                if type_filter is not None and metric.metric_type != type_filter:
                    continue
                filtered.append(metric)
        return filtered

    def get_counter_value(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    def get_gauge_value(self, name: str) -> Optional[float]:
        with self._lock:
            return self._gauges.get(name)

    def histogram_summary(self, name: str) -> Dict[str, float]:
        with self._lock:
            samples = list(self._histograms.get(name, []))
            if not samples:
                return {"count": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0}
            count = float(len(samples))
            return {
                "count": count,
                "min": min(samples),
                "max": max(samples),
                "mean": sum(samples) / count,
            }

    def reset(self) -> None:
        with self._lock:
            self._metrics_store.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histogram_samples": {k: list(v) for k, v in self._histograms.items()},
                "metric_count": len(self._metrics_store),
                "timestamp": time.time(),
            }


__all__: List[str] = ["MetricsCollector"]
