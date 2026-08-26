"""Task performance collector module.

Tracks performance characteristics of scheduled/background tasks and workflows,
including completion times, intermediate milestones, subtask aggregation, and
dependency correlation for bottleneck identification.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Union

from performance_dashboard.models import TaskMetric
from performance_dashboard.utils.time_utils import get_current_timestamp
from performance_dashboard.utils.validation import MetricSchemaValidator, validate_task_metric

logger = logging.getLogger(__name__)


class TaskCollector:
    """Collects and aggregates task-level performance metrics."""

    def __init__(self) -> None:
        self._validator: MetricSchemaValidator = MetricSchemaValidator()
        self._milestones: dict[str, list[float]] = defaultdict(list)
        self._subtask_metrics: dict[str, list[TaskMetric]] = defaultdict(list)
        self._dependency_graph: dict[str, set[str]] = defaultdict(set)

    def monitor_task_progress(self, task_id: str, milestone_name: str | None = None) -> float:
        """Record completion times and intermediate milestones for long-running processes.

        Args:
            task_id: Unique identifier for the tracked task.
            milestone_name: Optional name of an intermediate milestone checkpoint.

        Returns:
            The current timestamp at which progress was recorded.
        """
        timestamp: float = get_current_timestamp()
        self._milestones[task_id].append(timestamp)
        if milestone_name is not None:
            logger.debug("Milestone '%s' reached for task '%s' at %.3f", milestone_name, task_id, timestamp)
        else:
            logger.info("Task '%s' progress checkpoint recorded at %.3f", task_id, timestamp)
        return timestamp

    def aggregate_subtask_metrics(self, task_id: str, subtask_metric: TaskMetric) -> TaskMetric | None:
        """Combine metrics from component operations within complex tasks.

        Args:
            task_id: Unique identifier for the parent task.
            subtask_metric: A TaskMetric instance representing a component operation.

        Returns:
            An aggregated TaskMetric combining all recorded subtasks, or None if validation fails.
        """
        is_valid: bool = validate_task_metric(self._validator, subtask_metric)
        if not is_valid:
            logger.warning("Invalid subtask metric for task '%s'; skipping aggregation", task_id)
            return None

        self._subtask_metrics[task_id].append(subtask_metric)
        aggregated: TaskMetric = self._compute_aggregate(task_id)
        logger.debug(
            "Aggregated %d subtasks for '%s': duration=%.3fs, success_rate=%.2f%%",
            len(self._subtask_metrics[task_id]), task_id, aggregated["duration_seconds"], aggregated["success_rate"] * 100.0,
        )
        return aggregated

    def _compute_aggregate(self, task_id: str) -> TaskMetric:
        """Compute an aggregate TaskMetric from recorded subtasks for a given task."""
        subtask_list: list[TaskMetric] = self._subtask_metrics.get(task_id, [])
        if not subtask_list:
            return TaskMetric(
                duration_seconds=0.0, success_rate=1.0, resource_consumption_units=0.0, task_id=task_id, status="pending",
            )

        total_duration: float = sum(m["duration_seconds"] for m in subtask_list)
        avg_success_rate: float = sum(m["success_rate"] for m in subtask_list) / len(subtask_list)
        max_resource_consumption: float = max(m["resource_consumption_units"] for m in subtask_list)

        # Determine overall status from per-subtask statuses.
        statuses: list[str | None] = [m.get("status") for m in subtask_list
                                      if m.get("status") is not None]
        overall_status: str | None = self._resolve_overall_status(statuses)

        return TaskMetric(
            duration_seconds=total_duration, success_rate=avg_success_rate, resource_consumption_units=max_resource_consumption, task_id=task_id, status=overall_status,
        )

    def _resolve_overall_status(self, statuses: list[str]) -> str | None:
        """Resolve the overall task status from individual subtask statuses."""
        if not statuses:
            return "pending"

        # Priority: latest status wins, then any failure, then any running.
        last = statuses[-1]
        if last == "failed":
            return "failed"
        if last == "running":
            return "running"
        if last == "completed":
            return "completed"
        has_failure = any(s == "failed" for s in statuses)
        has_running = any(s == "running" for s in statuses)
        if has_failure:
            return "failed"
        if has_running:
            return "running"
        return "completed"

    def correlate_dependency_performance(
        self, task_id: str, dependency_task_ids: list[str], bottleneck_threshold_ms: float = 500.0,
    ) -> dict[str, Union[TaskMetric, bool]]:
        """Link executions identifying bottlenecks via Union type branch resolution using pattern matching constructs.

        Args:
            task_id: Unique identifier for the primary task being analyzed.
            dependency_task_ids: List of task IDs that the primary task depends on.
            bottleneck_threshold_ms: Threshold in milliseconds to flag a subtask as a bottleneck.

        Returns:
            A dictionary mapping each dependency task ID to either its aggregated TaskMetric or a boolean indicating bottleneck status.
        """
        results: dict[str, Union[TaskMetric, bool]] = {}
        for dep_id in dependency_task_ids:
            self._dependency_graph[task_id].add(dep_id)
            subtask_metrics: list[TaskMetric] = self._subtask_metrics.get(dep_id, [])

            if not subtask_metrics:
                results[dep_id] = False
                logger.debug("No metrics recorded for dependency '%s'", dep_id)
                continue

            longest_duration_ms: float = subtask_metrics[0]["duration_seconds"] * 1000.0
            is_bottleneck: bool = longest_duration_ms > bottleneck_threshold_ms
            if is_bottleneck:
                results[dep_id] = True
                logger.warning(
                    "Dependency '%s' identified as bottleneck (>%.1fms)",
                    dep_id, bottleneck_threshold_ms,
                )
            else:
                aggregate_metric: TaskMetric | None = self._compute_aggregate(dep_id)
                if aggregate_metric is not None:
                    results[dep_id] = aggregate_metric
                else:
                    results[dep_id] = False

        logger.info("Correlated %d dependencies for task '%s'", len(dependency_task_ids), task_id)
        return results