"""Orchestration package: dependency graphs and the task scheduler.

Moved from the retired ``src/agent1.orchestration`` namespace into
``agent_core``.  The unused ``workflow_engine`` module was not carried over.
"""

from typing import List

from .dependency_graph import DependencyGraph
from .task_scheduler import TaskScheduler, TaskExecutor
from .types import AgentMessage, MessageType, TaskNode, TaskStatus

__all__: List[str] = [
    "AgentMessage",
    "MessageType",
    "TaskNode",
    "TaskStatus",
    "TaskExecutor",
    "DependencyGraph",
    "TaskScheduler",
]
