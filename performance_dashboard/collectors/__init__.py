"""Performance data collectors package.

Provides modules for gathering command-level and task-level metrics
from runtime execution sources. Each collector produces validated
``CommandMetric`` or ``TaskMetric`` records ready for persistence.

``command_collector`` exposes module-level functions
(``collect_command_metrics``, ``parse_execution_logs``,
``normalize_metric_formats``, ``build_performance_record``);
``task_collector`` exposes the ``TaskCollector`` class.
"""

from .task_collector import TaskCollector

__all__: list[str] = [
    "TaskCollector",
]