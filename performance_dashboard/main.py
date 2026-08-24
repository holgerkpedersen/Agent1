from __future__ import annotations

import os
import signal
import threading
import time
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")

from performance_dashboard.config import (
    CollectorIntervals,
    DashboardSettings,
    DatabaseConfig,
    PerformanceThresholds,
)
from performance_dashboard.models import PerformanceRecord, TimeSeriesPoint
from performance_dashboard.storage.database import (
    BaseDatabase,
    BatchInsertManager,
    BatchProcessor,
    PerformanceDatabase,
    TimeSeriesAdapter,
    TimeSeriesDB,
)
from performance_dashboard.collectors.command_collector import (
    normalize_metric_formats,
    parse_execution_logs,
)
from performance_dashboard.collectors.task_collector import TaskCollector
from performance_dashboard.analytics.aggregator import AggregationEngine


class SystemBootstrapper:
    """Bootstrap system initialization coordinating component startup sequence."""

    def __init__(self) -> None:
        self._config: Optional[DatabaseConfig] = DatabaseConfig()
        self._settings: Optional[DashboardSettings] = DashboardSettings()
        self._intervals: Optional[CollectorIntervals] = CollectorIntervals()
        self._thresholds: Optional[PerformanceThresholds] = PerformanceThresholds()
        self._database: Optional[PerformanceDatabase[Any]] = None
        self._time_series_db: Optional[TimeSeriesAdapter[Any]] = None
        self._task_collector: Optional[TaskCollector] = TaskCollector()
        self._aggregator: Optional[AggregationEngine] = AggregationEngine()
        self._shutdown_event: threading.Event = threading.Event()

    def _read_interval(self, env_key: str, default_seconds: float) -> float:
        raw = os.environ.get(env_key)
        if raw is None:
            return default_seconds
        try:
            return float(raw)
        except ValueError:
            return default_seconds

    def _instantiate_database(self) -> bool:
        self._database = PerformanceDatabase[Any](config=self._config)  # type: ignore[arg-type]
        if not isinstance(self._database, BaseDatabase):
            return False
        self._time_series_db = TimeSeriesAdapter[Any](config=self._config)  # type: ignore[arg-type]
        if not isinstance(self._time_series_db, TimeSeriesDB):
            return False
        return True

    def _start_collector_threads(self) -> bool:
        if self._task_collector is None or self._aggregator is None:
            return False
        self._shutdown_event.clear()
        task_thread = threading.Thread(target=self._collection_sweep, name="collector-task", daemon=True)
        command_thread = threading.Thread(target=self._command_sweep, name="collector-command", daemon=True)
        self._task_collector_thread: Optional[threading.Thread] = task_thread
        self._command_collector_thread: Optional[threading.Thread] = command_thread
        task_thread.start()
        command_thread.start()
        return True

    def _collection_sweep(self) -> None:
        interval = self._read_interval("COLLECTION_INTERVAL_SECONDS", 60.0)
        while not self._shutdown_event.is_set():
            db = self._database
            if db is None or not db.is_connected():
                time.sleep(interval)
                continue
            point = TimeSeriesPoint(
                timestamp=time.time(),
                value=self._task_collector.monitor_task_progress("runtime") if self._task_collector else 0.0,
                series_label="task-progress",
                confidence_interval_lower=0.0,
                confidence_interval_upper=1.0,
            )
            db.insert_time_series_point(point)
            time.sleep(interval)

    def _command_sweep(self) -> None:
        interval = self._read_interval("COMMAND_INTERVAL_SECONDS", 30.0)
        while not self._shutdown_event.is_set():
            db = self._database
            if db is None or not db.is_connected():
                time.sleep(interval)
                continue
            raw_log = parse_execution_logs("command runtime sweep")
            metric = normalize_metric_formats(raw_log)
            if metric is None:
                time.sleep(interval)
                continue
            record = PerformanceRecord(
                timestamp=time.time(),
                record_type=metric,
                source_identifier="command",
                metadata={"sweep": "periodic"},
            )
            db.insert_record(record)
            agg = self._aggregator
            if agg is not None:
                agg.ingest(record)
            time.sleep(interval)

    def initialize_components(self) -> bool:
        configured = self._config is not None and self._settings is not None and self._intervals is not None
        db_ready = self._instantiate_database()
        if db_ready and self._database is not None:
            db_ready = self._database.connect()
        threads_ok = self._start_collector_threads()
        return configured and db_ready and threads_ok


class JobScheduler(Generic[T]):
    """Periodic registration respecting Generic[T] bound=Any parameterization."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Tuple[float, Callable[..., Any]]] = {}
        self._lock: threading.Lock = threading.Lock()
        self._last_run: Dict[str, float] = {}

    def register_job(self, name: str, interval_seconds: float, callback: Callable[..., Any]) -> bool:
        with self._lock:
            if not isinstance(name, str) or interval_seconds <= 0.0:
                return False
            self._jobs[name] = (interval_seconds, callback)
            self._last_run[name] = time.time()
            return True

    def schedule_periodic_jobs(self) -> bool:
        collection_interval = float(os.environ.get("COLLECTION_INTERVAL_SECONDS", "60"))
        cache_interval = float(os.environ.get("CACHE_MAINTENANCE_INTERVAL_SECONDS", "120"))
        analytics_interval = float(os.environ.get("ANALYTICS_INTERVAL_SECONDS", "300"))
        registered = True
        registered &= self.register_job("metric-collection-sweep", collection_interval, lambda: None)
        registered &= self.register_job("cache-maintenance-routine", cache_interval, lambda: None)
        registered &= self.register_job("analytics-summary-job", analytics_interval, lambda: None)
        return registered

    def run_once(self) -> int:
        executed = 0
        now = time.time()
        with self._lock:
            snapshot: List[Tuple[str, Tuple[float, Callable[..., Any]]]] = list(self._jobs.items())
        for name, (interval, callback) in snapshot:
            last = self._last_run.get(name, 0.0)
            if now - last >= interval:
                try:
                    callback()
                except Exception:
                    pass
                executed += 1
                with self._lock:
                    self._last_run[name] = now
        return executed

    def readiness_probe(self) -> bool:
        with self._lock:
            scheduled = len(self._jobs) > 0
        return scheduled


class DashboardRuntime:
    """Establishes runtime execution loop managing periodic operations."""

    def __init__(self) -> None:
        self._bootstrapper: SystemBootstrapper = SystemBootstrapper()
        self._scheduler: JobScheduler[Any] = JobScheduler[Any]()
        self._loop_interval: float = 1.0
        self._shutdown_event: threading.Event = self._bootstrapper._shutdown_event

    def _cache_maintenance(self) -> int:
        db = self._bootstrapper._database
        tsdb = self._bootstrapper._time_series_db
        preserved = 0
        if db is not None and db.is_connected():
            preserved += db.flush_buffer()
        retention = float(os.environ.get("RETENTION_THRESHOLD_SECONDS", "86400"))
        cutoff = time.time() - retention
        if tsdb is not None:
            preserved += tsdb.delete_records(cutoff)  # type: ignore[arg-type]
        return preserved

    def _analytics_summary(self) -> Optional[Any]:
        agg = self._bootstrapper._aggregator
        db = self._bootstrapper._database
        if agg is None or db is None or not db.is_connected():
            return None
        summary = agg.generate_summary()
        if summary is None:
            return None
        return summary

    def schedule_periodic_jobs(self) -> bool:
        ok = self._scheduler.schedule_periodic_jobs()
        collection_interval = float(os.environ.get("COLLECTION_INTERVAL_SECONDS", "60"))
        cache_interval = float(os.environ.get("CACHE_MAINTENANCE_INTERVAL_SECONDS", "120"))
        analytics_interval = float(os.environ.get("ANALYTICS_INTERVAL_SECONDS", "300"))
        ok &= self._scheduler.register_job("metric-collection-sweep", collection_interval, self._bootstrapper._collection_sweep)
        ok &= self._scheduler.register_job("cache-maintenance-routine", cache_interval, self._cache_maintenance)
        ok &= self._scheduler.register_job("analytics-summary-job", analytics_interval, self._analytics_summary)
        return ok

    def handle_shutdown_signals(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, lambda _signum, _frame: self._shutdown_event.set())  # type: ignore[arg-type]
            except (ValueError, OSError):
                pass
        db = self._bootstrapper._database
        tsdb = self._bootstrapper._time_series_db
        if db is not None and db.is_connected():
            db.flush_buffer()
        if db is not None:
            db.disconnect()
        if tsdb is not None:
            tsdb.disconnect()  # type: ignore[arg-type]
        self._shutdown_event.set()
        for thread in (self._bootstrapper._task_collector_thread, self._bootstrapper._command_collector_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=5.0)

    def health_status(self) -> Dict[str, bool]:
        db = self._bootstrapper._database
        tsdb = self._bootstrapper._time_series_db
        liveness = (
            self._bootstrapper._config is not None
            and self._bootstrapper._settings is not None
            and self._bootstrapper._intervals is not None
            and db is not None
            and tsdb is not None
        )
        readiness = (
            liveness
            and db is not None
            and db.is_connected()  # type: ignore[union-attr]
            and not self._shutdown_event.is_set()
            and self._scheduler.readiness_probe()
        )
        return {"liveness": bool(liveness), "readiness": bool(readiness)}

    def run_loop(self) -> int:
        initialized = self._bootstrapper.initialize_components()
        if not initialized:
            return 1
        scheduled = self.schedule_periodic_jobs()
        if not scheduled:
            return 2
        self.handle_shutdown_signals.__self__  # noqa: B018 - ensure handler installed
        while not self._shutdown_event.is_set():
            self._scheduler.run_once()
            time.sleep(self._loop_interval)
        self.handle_shutdown_signals()
        return 0


def health_status() -> Dict[str, bool]:
    """Expose liveness/readiness indicators for deployment orchestration platforms."""
    runtime = DashboardRuntime()
    if not runtime._bootstrapper.initialize_components():
        return {"liveness": False, "readiness": False}
    return runtime.health_status()


def main() -> int:
    runtime = DashboardRuntime()
    exit_code = runtime.run_loop()
    signal.signal(signal.SIGTERM, lambda _signum, _frame: None)  # type: ignore[arg-type]
    signal.signal(signal.SIGINT, lambda _signum, _frame: None)  # type: ignore[arg-type]
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())