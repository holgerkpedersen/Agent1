"""Metrics storage module for the Performance Dashboard.

This module provides a unified interface to store and retrieve metrics from various sources,
including Prometheus, InfluxDB, and custom metric collectors. It supports multiple backends,
caching strategies, and query optimization techniques.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

import redis.asyncio as aioredis  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Constants
# ---------------------------------------------------------------------------

class MetricType(str, Enum):
    """Supported metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class StorageBackend(str, Enum):
    """Available storage backends."""
    PROMETHEUS = "prometheus"
    INFLUXDB = "influxdb"
    REDIS = "redis"
    MEMORY = "memory"
    FILESYSTEM = "filesystem"


DEFAULT_CACHE_TTL: int = 300  # seconds (5 minutes)
MAX_QUERY_RESULTS: int = 10_000
METRIC_NAME_PATTERN: str = r"^[a-zA-Z_:][a-zA-Z0-9_:]*$"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": dict(self.labels),
        }


@dataclass
class MetricSeries:
    """A series of metric data points for a given metric name and labels."""
    name: str
    type: MetricType = MetricType.GAUGE
    points: List[MetricPoint] = field(default_factory=list)

    def add_point(self, point: MetricPoint) -> None:
        self.points.append(point)

    def sort_points(self) -> None:
        """Sort points by timestamp in ascending order."""
        self.points.sort(key=lambda p: p.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "points": [p.to_dict() for p in self.points],
        }


@dataclass
class QueryRequest:
    """Represents a query request to the metrics store."""
    metric_name: str
    start_time: datetime
    end_time: datetime
    step: timedelta = field(default_factory=lambda: timedelta(seconds=60))
    labels: Dict[str, str] = field(default_factory=dict)
    backend: StorageBackend = StorageBackend.PROMETHEUS
    cache_ttl: int = DEFAULT_CACHE_TTL

    def validate(self) -> None:
        """Validate the query request parameters."""
        if self.start_time >= self.end_time:
            raise ValueError("start_time must be before end_time")
        if self.step.total_seconds() <= 0:
            raise ValueError("step must have a positive duration")

    @property
    def cache_key(self) -> str:
        """Generate a deterministic cache key for this query."""
        sorted_labels = json.dumps(self.labels, sort_keys=True)
        return (
            f"metrics:{self.backend.value}:{self.metric_name}:"
            f"{self.start_time.isoformat()}:{self.end_time.isoformat()}:"
            f"{int(self.step.total_seconds())}:{sorted_labels}"
        )


@dataclass
class QueryResult:
    """Represents the result of a metrics query."""
    series_list: List[MetricSeries] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_points(self) -> int:
        return sum(len(s.points) for s in self.series_list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "series": [s.to_dict() for s in self.series_list],
            "metadata": dict(self.metadata),
            "total_points": self.total_points,
        }


# ---------------------------------------------------------------------------
# Abstract Backend Interface
# ---------------------------------------------------------------------------

class MetricsBackend(ABC):
    """Abstract base class for metrics storage backends."""

    @abstractmethod
    async def query(
        self, request: QueryRequest
    ) -> QueryResult:
        """Execute a metric query and return results.

        Args:
            request: The validated query request.

        Returns:
            A :class:`QueryResult` containing the fetched data.
        """

    @abstractmethod
    async def write(
        self, series: MetricSeries
    ) -> None:
        """Persist a metric series to the backend.

        Args:
            series: The metric series to store.
        """

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources used by this backend."""


# ---------------------------------------------------------------------------
# Concrete Backends
# ---------------------------------------------------------------------------

class PrometheusBackend(MetricsBackend):
    """Prometheus metrics backend using the HTTP API."""

    def __init__(self, endpoint: str = "http://localhost:9090") -> None:
        self.endpoint = endpoint.rstrip("/")
        self._session: Optional[Any] = None  # aiohttp session placeholder

    async def query(self, request: QueryRequest) -> QueryResult:
        logger.debug("Querying Prometheus at %s for metric '%s'", self.endpoint, request.metric_name)

        # Simulated implementation — in production this would use aiohttp
        result = QueryResult(metadata={"backend": "prometheus", "endpoint": self.endpoint})

        if not self._session:
            logger.warning("Prometheus backend session not initialized; returning empty results")
            return result

        query_str = request.metric_name
        if request.labels:
            label_parts = [f'{k}="{v}"' for k, v in sorted(request.labels.items())]
            query_str += "{" + ",".join(label_parts) + "}"

        # Example actual call structure (commented):
        # url = f"{self.endpoint}/api/v1/query_range"
        # params = {
        #     "query": query_str,
        #     "start": request.start_time.timestamp(),
        #     "end": request.end_time.timestamp(),
        #     "step": int(request.step.total_seconds()),
        # }
        # async with self._session.get(url, params=params) as resp:
        #     data = await resp.json()

        # Simulate some sample points for demonstration purposes
        simulated_points: List[MetricPoint] = []
        current_time = request.start_time
        while current_time <= request.end_time and len(simulated_points) < MAX_QUERY_RESULTS:
            value = float(hash((request.metric_name, str(current_time))) % 1000) / 10.0
            simulated_points.append(MetricPoint(timestamp=current_time, value=value))
            current_time += request.step

        series = MetricSeries(name=request.metric_name, type=MetricType.GAUGE, points=simulated_points)
        result.series_list.append(series)
        return result

    async def write(self, series: MetricSeries) -> None:
        logger.info("Writing %d points to Prometheus for metric '%s'", len(series.points), series.name)
        # In a real implementation this would POST to /api/v1/write or use remote_write protocol.
        pass

    async def close(self) -> None:
        if self._session:
            await self._session.close()  # type: ignore[misc]


class InfluxDBBackend(MetricsBackend):
    """InfluxDB v2 metrics backend."""

    def __init__(self, org: str, bucket: str, token: str, url: str = "http://localhost:8086") -> None:
        self.org = org
        self.bucket = bucket
        self.token = token
        self.url = url.rstrip("/")
        self._client: Optional[Any] = None  # influxdb_client.AsyncClient placeholder

    async def query(self, request: QueryRequest) -> QueryResult:
        logger.debug("Querying InfluxDB at %s for metric '%s'", self.url, request.metric_name)

        result = QueryResult(metadata={"backend": "influxdb", "org": self.org, "bucket": self.bucket})

        if not self._client:
            logger.warning("InfluxDB client not initialized; returning empty results")
            return result

        # Simulated flux query (commented for production use):
        # from influxdb_client import InfluxDBClient, QueryApi
        # flux_query = f'''
        #   from(bucket: "{self.bucket}")
        #     |> range(start: {request.start_time.isoformat()}, stop: {request.end_time.isoformat()})
        #     |> filter(fn: (r) => r._measurement == "{request.metric_name}")
        # '''

        simulated_points = []
        current_time = request.start_time
        while current_time <= request.end_time and len(simulated_points) < MAX_QUERY_RESULTS:
            value = float(hash((request.metric_name, str(current_time))) % 500) / 10.0
            simulated_points.append(MetricPoint(timestamp=current_time, value=value))
            current_time += request.step

        series = MetricSeries(name=request.metric_name, type=MetricType.GAUGE, points=simulated_points)
        result.series_list.append(series)
        return result

    async def write(self, series: MetricSeries) -> None:
        logger.info("Writing %d points to InfluxDB for metric '%s'", len(series.points), series.name)
        # In production: self._client.write_api().write(bucket=self.bucket, org=self.org, record=...)
        pass

    async def close(self) -> None:
        if self._client:
            await self._client.close()  # type: ignore[misc]


class RedisBackend(MetricsBackend):
    """Redis-based metrics backend for caching and short-term storage."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self.redis_url = redis_url
        self._client: Optional[aioredis.Redis] = None  # type: ignore[type-arg]

    async def _ensure_client(self) -> aioredis.Redis:  # type: ignore[type-arg]
        if self._client is None:
            self._client = aioredis.from_url(self.redis_url)  # type: ignore[assignment]
        return self._client

    async def query(self, request: QueryRequest) -> QueryResult:
        client = await self._ensure_client()
        logger.debug("Querying Redis for cache key '%s'", request.cache_key)

        cached_data = await client.get(request.cache_key)  # type: ignore[union-attr]
        result = QueryResult(metadata={"backend": "redis"})

        if cached_data is not None:
            try:
                deserialized = pickle.loads(cached_data)
                for series_dict in deserialized.get("series", []):
                    points = [
                        MetricPoint(
                            timestamp=datetime.fromisoformat(p["timestamp"]),
                            value=p["value"],
                            labels=dict(p["labels"]),
                        )
                        for p in series_dict["points"]
                    ]
                    mt = MetricType(series_dict.get("type", "gauge"))
                    result.series_list.append(MetricSeries(name=series_dict["name"], type=mt, points=points))
                logger.debug("Cache hit for key '%s'", request.cache_key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to deserialize Redis cache data: %s", exc)

        return result

    async def write(self, series: MetricSeries) -> None:
        client = await self._ensure_client()
        serialized = pickle.dumps(series.to_dict())
        key = f"metric:{series.name}"
        await client.setex(key, DEFAULT_CACHE_TTL, serialized)  # type: ignore[union-attr]
        logger.debug("Wrote metric series '%s' to Redis with TTL=%ds", series.name, DEFAULT_CACHE_TTL)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()


class MemoryBackend(MetricsBackend):
    """In-memory metrics backend suitable for testing and small-scale usage."""

    def __init__(self) -> None:
        self._store: Dict[str, MetricSeries] = {}
        self._lock = threading.Lock()

    async def query(self, request: QueryRequest) -> QueryResult:
        logger.debug("Querying memory backend for metric '%s'", request.metric_name)

        result = QueryResult(metadata={"backend": "memory"})

        with self._lock:
            series = self._store.get(request.metric_name)
            if series is None:
                return result

            filtered_points = [
                p for p in series.points
                if request.start_time <= p.timestamp <= request.end_time
                and all(p.labels.get(k) == v for k, v in request.labels.items())
            ]

        new_series = MetricSeries(name=request.metric_name, type=series.type, points=filtered_points)
        result.series_list.append(new_series)
        return result

    async def write(self, series: MetricSeries) -> None:
        with self._lock:
            existing = self._store.get(series.name)
            if existing is None:
                self._store[series.name] = MetricSeries(
                    name=series.name, type=series.type, points=list(series.points)
                )
            else:
                existing.add_point(*series.points)  # noqa: F841 — placeholder logic

        logger.debug("Wrote %d points to memory backend for metric '%s'", len(series.points), series.name)

    async def close(self) -> None:
        with self._lock:
            self._store.clear()


class FilesystemBackend(MetricsBackend):
    """Filesystem-based metrics backend using JSON files."""

    def __init__(self, base_dir: Union[str, Path] = "./metrics_data") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def query(self, request: QueryRequest) -> QueryResult:
        file_path = self._file_for_metric(request.metric_name)
        logger.debug("Querying filesystem at %s for metric '%s'", str(file_path), request.metric_name)

        result = QueryResult(metadata={"backend": "filesystem", "path": str(file_path)})

        if not file_path.exists():
            return result

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:  # noqa: BLE001
            logger.error("Failed to read metrics file %s: %s", str(file_path), exc)
            return result

        for series_dict in data.get("series", []):
            points = [
                MetricPoint(
                    timestamp=datetime.fromisoformat(p["timestamp"]),
                    value=p["value"],
                    labels=dict(p["labels"]),
                )
                for p in series_dict.get("points", [])
            ]
            mt = MetricType(series_dict.get("type", "gauge"))
            result.series_list.append(MetricSeries(name=series_dict["name"], type=mt, points=points))

        return result

    async def write(self, series: MetricSeries) -> None:
        file_path = self._file_for_metric(series.name)
        logger.debug("Writing metric '%s' to filesystem at %s", series.name, str(file_path))

        existing_data = {"series": []}
        if file_path.exists():
            try:
                existing_data = json.loads(file_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):  # noqa: BLE001
                logger.warning("Silenced exception in metrics_store.py:424")

        series_dict = series.to_dict()
        found_existing = False
        for i, s in enumerate(existing_data["series"]):
            if s["name"] == series.name and s.get("type") == series.type.value:
                existing_data["series"][i]["points"].extend([p.to_dict() for p in series.points])
                found_existing = True
                break

        if not found_existing:
            existing_data["series"].append(series_dict)

        file_path.write_text(json.dumps(existing_data, indent=2), encoding="utf-8")

    async def close(self) -> None:
        logger.debug("Filesystem backend closed (no persistent resources)")

    def _file_for_metric(self, metric_name: str) -> Path:
        safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in metric_name)
        return self.base_dir / f"{safe_name}.json"


# ---------------------------------------------------------------------------
# Cache Layer (Decorator-based LRU + TTL cache)
# ---------------------------------------------------------------------------

class MetricsCache:
    """A hybrid LRU/TTL cache layer wrapping backend queries."""

    def __init__(self, max_size: int = 1024, default_ttl: int = DEFAULT_CACHE_TTL) -> None:
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def _is_expired(self, entry_time: float, ttl: int) -> bool:
        return time.monotonic() - entry_time > ttl

    async def get_cached(
        self, cache_key: str, ttl: Optional[int] = None
    ) -> Optional[QueryResult]:
        effective_ttl = ttl if ttl is not None else self.default_ttl
        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is None:
                return None
            value, timestamp = entry
            if self._is_expired(timestamp, effective_ttl):
                del self._cache[cache_key]
                return None
            return value

    async def set_cached(
        self, cache_key: str, result: QueryResult, ttl: Optional[int] = None
    ) -> None:
        effective_ttl = ttl if ttl is not None else self.default_ttl
        with self._lock:
            # Enforce LRU eviction when exceeding max_size
            if len(self._cache) >= self.max_size and cache_key not in self._cache:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[cache_key] = (result, time.monotonic())


# ---------------------------------------------------------------------------
# Main Metrics Store Facade
# ---------------------------------------------------------------------------

class MetricsStore:
    """Main facade for querying and storing metrics across multiple backends.

    This class manages backend selection, caching, concurrency control, and error handling.
    """

    def __init__(
        self,
        primary_backend: StorageBackend = StorageBackend.PROMETHEUS,
        fallback_backends: Optional[List[StorageBackend]] = None,
        cache_enabled: bool = True,
        redis_url: str = "redis://localhost:6379/0",
    ) -> None:
        self.primary_backend_name = primary_backend
        self.fallback_backend_names = fallback_backends or []
        self.cache_enabled = cache_enabled

        # Initialize backends lazily to avoid unnecessary connections during tests.
        self._backends: Dict[StorageBackend, MetricsBackend] = {}
        self._redis_url = redis_url
        self._cache_layer = MetricsCache() if cache_enabled else None

        self._initialized = False
        self._init_lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        """Initialize all backends synchronously (called internally)."""
        with self._init_lock:
            if self._initialized:
                return

            backend_instances: Dict[StorageBackend, MetricsBackend] = {
                StorageBackend.PROMETHEUS: PrometheusBackend(),
                StorageBackend.INFLUXDB: InfluxDBBackend(
                    org="my-org", bucket="metrics", token="secret-token"
                ),
                StorageBackend.REDIS: RedisBackend(self._redis_url),
                StorageBackend.MEMORY: MemoryBackend(),
                StorageBackend.FILESYSTEM: FilesystemBackend("./metrics_data"),
            }

            requested_names = {self.primary_backend_name, *self.fallback_backend_names}
            for name in requested_names:
                if name not in backend_instances:
                    logger.warning("Unknown backend '%s'; skipping initialization", name)
                    continue
                self._backends[name] = backend_instances[name]

            self._initialized = True

    async def _get_backend(self, backend_name: StorageBackend) -> Optional[MetricsBackend]:
        """Retrieve a backend instance, initializing if necessary."""
        if not self._initialized:
            # Run synchronous initialization in executor to avoid blocking event loop unnecessarily.
            await asyncio.get_event_loop().run_in_executor(None, self._ensure_initialized)

        return self._backends.get(backend_name)

    async def query(self, request: QueryRequest) -> QueryResult:
        """Query metrics from the primary backend with fallback and caching support."""
        request.validate()

        # Check cache first if enabled.
        cached_result = None
        if self.cache_enabled and self._cache_layer is not None:
            cached_result = await self._cache_layer.get_cached(request.cache_key, ttl=request.cache_ttl)

        if cached_result is not None:
            logger.debug("Cache hit for query '%s'", request.metric_name)
            return cached_result

        # Try primary backend.
        backend_names_to_try = [self.primary_backend_name] + self.fallback_backend_names
        last_error: Optional[Exception] = None

        for backend_name in backend_names_to_try:
            backend = await self._get_backend(backend_name)
            if backend is None:
                continue

            try:
                result = await asyncio.wait_for(backend.query(request), timeout=30.0)
                logger.debug("Successfully queried '%s' via %s", request.metric_name, backend_name.value)

                # Cache successful results.
                if self.cache_enabled and self._cache_layer is not None:
                    await self._cache_layer.set_cached(request.cache_key, result, ttl=request.cache_ttl)

                return result
            except asyncio.TimeoutError as exc:
                logger.warning("Timeout querying backend %s for metric '%s'", backend_name.value, request.metric_name)
                last_error = exc
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Error querying backend %s for metric '%s': %s",
                    backend_name.value,
                    request.metric_name,
                    exc,
                )
                last_error = exc

        if last_error is not None:
            raise RuntimeError(
                f"All backends failed to query metric '{request.metric_name}'"
            ) from last_error

        logger.warning("No available backend for querying metric '%s'", request.metric_name)
        return QueryResult(metadata={"status": "no_backend_available"})

    async def write(self, series: MetricSeries) -> None:
        """Write a metric series to the primary backend."""
        if not self._initialized:
            await asyncio.get_event_loop().run_in_executor(None, self._ensure_initialized)

        backend = self._backends.get(self.primary_backend_name)
        if backend is None:
            raise RuntimeError(f"Primary backend '{self.primary_backend_name}' not available")

        try:
            await asyncio.wait_for(backend.write(series), timeout=30.0)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to write metric '%s': %s", series.name, exc)
            raise RuntimeError(f"Write failed for metric '{series.name}'") from exc

    async def close(self) -> None:
        """Close all initialized backends."""
        if not self._initialized:
            return

        tasks = []
        for backend in self._backends.values():
            tasks.append(asyncio.create_task(backend.close()))

        await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug("All metrics backends closed")


# ---------------------------------------------------------------------------
# Convenience Functions and Decorators
# ---------------------------------------------------------------------------

def async_retry(max_attempts: int = 3, delay_seconds: float = 1.0):
    """Decorator to retry an async function with exponential backoff."""
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.error("Retry exhausted for %s after %d attempts", func.__name__, max_attempts)
                        raise
                    sleep_time = delay_seconds * (2 ** (attempts - 1))
                    logger.warning(
                        "Attempt %d/%d failed for %s; retrying in %.1fs: %s",
                        attempts,
                        max_attempts,
                        func.__name__,
                        sleep_time,
                        exc,
                    )
                    await asyncio.sleep(sleep_time)

        return wrapper

    return decorator


@lru_cache(maxsize=256)
def normalize_metric_name(name: str) -> str:
    """Normalize a metric name by stripping whitespace and validating format."""
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Metric name cannot be empty")

    # Replace invalid characters with underscores.
    import re  # noqa: PLC0415 — local import for clarity
    normalized = re.sub(r"[^a-zA-Z0-9_:]", "_", cleaned)
    return normalized


def build_query_request(
    metric_name: str,
    start_time: datetime,
    end_time: datetime,
    step_seconds: int = 60,
    labels: Optional[Dict[str, str]] = None,
    backend: StorageBackend = StorageBackend.PROMETHEUS,
) -> QueryRequest:
    """Helper function to construct a :class:`QueryRequest` with defaults."""
    return QueryRequest(
        metric_name=normalize_metric_name(metric_name),
        start_time=start_time,
        end_time=end_time,
        step=timedelta(seconds=step_seconds),
        labels=dict(labels or {}),
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Module-level singleton instance (optional convenience)
# ---------------------------------------------------------------------------

_default_store: Optional[MetricsStore] = None
_singleton_lock = threading.Lock()


def get_default_metrics_store(
    primary_backend: StorageBackend = StorageBackend.PROMETHEUS,
    cache_enabled: bool = True,
) -> MetricsStore:
    """Return a singleton :class:`MetricsStore` instance."""
    global _default_store  # noqa: PLW0603

    with _singleton_lock:
        if _default_store is None or not isinstance(_default_store, MetricsStore):
            _default_store = MetricsStore(
                primary_backend=primary_backend, cache_enabled=cache_enabled
            )

    return _default_store


# ---------------------------------------------------------------------------
# Example Usage / Smoke Test (when run directly)
# ---------------------------------------------------------------------------

async def _example_usage() -> None:
    """Demonstrates basic usage of the metrics store."""
    store = MetricsStore(
        primary_backend=StorageBackend.MEMORY, cache_enabled=True
    )

    now = datetime.utcnow()
    request = build_query_request(
        metric_name="cpu_utilization",
        start_time=now - timedelta(minutes=10),
        end_time=now,
        step_seconds=60,
        labels={"host": "server-01"},
        backend=StorageBackend.MEMORY,
    )

    # Write some data points.
    series = MetricSeries(
        name="cpu_utilization", type=MetricType.GAUGE,
        points=[MetricPoint(timestamp=now - timedelta(minutes=i), value=float(i * 10)) for i in range(5)],
    )
    await store.write(series)

    # Query them back.
    result = await store.query(request)
    print(f"Queried {result.total_points} points from backend '{request.backend}'")

    await store.close()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(_example_usage())