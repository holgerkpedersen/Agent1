"""Performance data collectors package.

Provides modules for gathering command-level and task-level metrics
from runtime execution sources. Each collector produces validated
``CommandMetric`` or ``TaskMetric`` records ready for persistence.
"""

from .command_collector import CommandCollector
from .task_collector import TaskCollector

__all__: list[str] = [
    "CommandCollector",
    "TaskCollector",
]