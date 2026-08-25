from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any

from performance_dashboard.models import PerformanceRecord, TimeSeriesPoint
from performance_dashboard.config import DatabaseConfig


T = TypeVar("T", bound=Any)


class BaseDatabase(ABC):
    """Abstract base interface for database operations."""

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...


class PerformanceDatabase(BaseDatabase, Generic[T]):
    """Abstract interface for metric storage operations."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._connected = False

    @abstractmethod
    def insert_record(self, record: PerformanceRecord) -> bool: ...

    @abstractmethod
    def query_records(
        self, start_time: float, end_time: float, source_identifier: str | None = None
    ) -> list[PerformanceRecord]: ...

    @abstractmethod
    def delete_records(self, before_timestamp: float) -> int: ...

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected


class TimeSeriesDB(ABC):
    """Abstract base for temporal record handling."""

    @abstractmethod
    def insert_time_series_point(self, point: TimeSeriesPoint) -> bool: ...

    @abstractmethod
    def query_time_series(
        self, series_label: str, start_time: float, end_time: float
    ) -> list[TimeSeriesPoint]: ...


class TimeSeriesAdapter(TimeSeriesDB, Generic[T]):
    """Specialized handling for temporal performance records."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config

    @abstractmethod
    def insert_time_series_point(self, point: TimeSeriesPoint) -> bool: ...

    @abstractmethod
    def query_time_series(
        self, series_label: str, start_time: float, end_time: float
    ) -> list[TimeSeriesPoint]: ...


class BatchProcessor(ABC):
    """Abstract base for batch processing operations."""

    @abstractmethod
    def process_batch(self, items: list[T]) -> int: ...


class BatchInsertManager(BatchProcessor, Generic[T]):
    """Optimize bulk writes during high-frequency collection periods."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._buffer: list[PerformanceRecord] = []

    @abstractmethod
    def process_batch(self, items: list[T]) -> int: ...

    def add_record(self, record: PerformanceRecord) -> bool:
        self._buffer.append(record)
        return True

    def flush_buffer(self) -> int:
        count = len(self._buffer)
        if count > 0:
            inserted = self.process_batch(list(self._buffer))
            self._buffer.clear()
            return inserted
        return 0

    def buffer_size(self) -> int:
        return len(self._buffer)