from typing import Dict, List, Optional, Any, Callable, Awaitable, Set, Tuple
import asyncio
import time

from ..core import TaskNode, TaskStatus, AgentMessage, MessageType


# Async executor signature for individual tasks (typically backed by plugins).
TaskExecutor = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


class DependencyGraph:
    """Tracks task nodes and their dependency relationships."""

    def __init__(self) -> None:
        self._tasks: Dict[str, TaskNode] = {}
        # Maps a dependency id to the ids that depend on it.
        self._dependents: Dict[str, Set[str]] = {}
        # Maps a task id to its direct dependencies.
        self._dependencies: Dict[str, Set[str]] = {}

    def add_task(self, node: TaskNode) -> None:
        if node.task_id in self._tasks:
            raise ValueError(f"Duplicate task id '{node.task_id}'")
        for dep in node.dependencies:
            if dep not in self._tasks:
                raise ValueError(
                    f"Dependency '{dep}' of task '{node.task_id}' does not exist"
                )
        self._tasks[node.task_id] = node
        self._dependencies[node.task_id] = set(node.dependencies)
        for dep in node.dependencies:
            self._dependents.setdefault(dep, set()).add(node.task_id)

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        return self._tasks.get(task_id)

    def update_status(self, task_id: str, status: TaskStatus) -> None:
        node = self._tasks.get(task_id)
        if node is None:
            raise ValueError(f"Unknown task '{task_id}'")
        node.status = status

    def get_ready_tasks(self) -> List[TaskNode]:
        """Return pending tasks whose dependencies are all resolved."""
        ready: List[TaskNode] = []
        for tid, node in self._tasks.items():
            if node.status != TaskStatus.PENDING:
                continue
            deps_resolved = True
            for dep in node.dependencies:
                dep_node = self._tasks.get(dep)
                if dep_node is None or dep_node.status not in (
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                ):
                    deps_resolved = False
                    break
            if deps_resolved:
                ready.append(node)
        return ready

    def topological_order(self) -> List[str]:
        order: List[str] = []
        visited: Set[str] = set()
        temp: Set[str] = set()

        def visit(tid: str) -> None:
            if tid in visited:
                return
            if tid in temp:
                raise ValueError("Cycle detected in dependency graph")
            temp.add(tid)
            for dep in self._dependencies.get(tid, set()):
                visit(dep)
            temp.discard(tid)
            visited.add(tid)
            order.append(tid)

        for tid in list(self._tasks):
            visit(tid)
        return order

    def all_tasks(self) -> List[TaskNode]:
        return list(self._tasks.values())


class TaskScheduler:
    """Dependency-aware scheduler that dispatches tasks to executors."""

    def __init__(self, executors: Optional[Dict[str, TaskExecutor]] = None) -> None:
        self._graph: DependencyGraph = DependencyGraph()
        self._executors: Dict[str, TaskExecutor] = dict(executors or {})
        self._results: Dict[str, Dict[str, Any]] = {}
        # task_id -> (due_time, agent_id)
        self._scheduled: Dict[str, Tuple[float, str]] = {}
        self._handlers: List[Callable[[AgentMessage], None]] = []
        self._running: bool = False
        self._loop_task: Optional[asyncio.Task[None]] = None

    def register_executor(self, task_id: str, executor: TaskExecutor) -> None:
        self._executors[task_id] = executor

    def unregister_executor(self, task_id: str) -> bool:
        if task_id in self._executors:
            del self._executors[task_id]
            return True
        return False

    def add_task(self, node: TaskNode) -> None:
        self._graph.add_task(node)

    def schedule_task(
        self,
        task_id: str,
        agent_id: Optional[str] = None,
        delay_seconds: float = 0.0,
    ) -> bool:
        node = self._graph.get_task(task_id)
        if node is None or node.status != TaskStatus.PENDING:
            return False
        due_time = time.monotonic() + max(delay_seconds, 0.0)
        self._scheduled[task_id] = (due_time, agent_id or "")
        return True

    def cancel_schedule(self, task_id: str) -> bool:
        if task_id in self._scheduled:
            del self._scheduled[task_id]
            return True
        return False

    def subscribe(self, handler: Callable[[AgentMessage], None]) -> bool:
        if handler not in self._handlers:
            self._handlers.append(handler)
            return True
        return False

    def unsubscribe(self, handler: Callable[[AgentMessage], None]) -> bool:
        if handler in self._handlers:
            self._handlers.remove(handler)
            return True
        return False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        loop = asyncio.get_event_loop()
        self._loop_task = loop.create_task(self._monitor())

    async def stop(self) -> None:
        self._running = False
        if self._loop_task is not None:
            try:
                await asyncio.wait_for(self._loop_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        self._loop_task = None

    async def _monitor(self) -> None:
        while self._running:
            now = time.monotonic()
            due_ids: List[str] = []
            for tid, (due_time, agent_id) in list(self._scheduled.items()):
                if due_time <= now and tid not in due_ids:
                    node = self._graph.get_task(tid)
                    if node is None or node.status != TaskStatus.PENDING:
                        continue
                    deps_satisfied = all(
                        dep_node.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                        for dep_node in (self._graph.get_task(dep) for dep in node.dependencies)
                        if dep_node is not None
                    )
                    if deps_satisfied:
                        due_ids.append(tid)
            for tid in due_ids:
                agent_id = self._scheduled[tid][1]
                asyncio.create_task(self._execute_task(tid, agent_id))
                del self._scheduled[tid]
            await asyncio.sleep(0.05)

    async def _execute_task(self, task_id: str, agent_id: str) -> None:
        node = self._graph.get_task(task_id)
        if node is None:
            return
        executor = self._executors.get(task_id)
        input_data: Dict[str, Any] = {}
        for dep in node.dependencies:
            result = self._results.get(dep)
            if result is not None:
                input_data[dep] = result

        try:
            if executor is None:
                outcome: Dict[str, Any] = {
                    "success": False,
                    "status": "no_executor",
                }
            else:
                outcome = await executor(input_data)
            self._results[task_id] = outcome
            node.status = TaskStatus.COMPLETED
        except Exception as exc:  # noqa: BLE001 - surface failures to callers
            self._results[task_id] = {
                "success": False,
                "error": str(exc),
                "status": "failed",
            }
            node.status = TaskStatus.FAILED

        self._publish_status(task_id, agent_id)

    def _publish_status(self, task_id: str, sender_id: str) -> None:
        if not self._handlers:
            return
        node = self._graph.get_task(task_id)
        if node is None:
            return
        result = self._results.get(task_id) or {}
        message = AgentMessage(
            sender_id=sender_id,
            receiver_id=None,
            message_type=MessageType.STATUS_UPDATE,
            content={"task_id": task_id, "status": node.status.value, "result": result},
            timestamp=time.monotonic(),
        )
        for handler in self._handlers:
            try:
                handler(message)
            except Exception as exc:  # noqa: BLE001 - isolate handler failures
                pass

    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._results.get(task_id)

    def pending_count(self) -> int:
        return sum(
            1 for node in self._graph.all_tasks() if node.status == TaskStatus.PENDING
        )

    def scheduled_count(self) -> int:
        return len(self._scheduled)

    def snapshot(self) -> Dict[str, Any]:
        tasks_state: List[Dict[str, Any]] = []
        for node in self._graph.all_tasks():
            tasks_state.append(
                {
                    "task_id": node.task_id,
                    "name": node.name,
                    "status": node.status.value,
                    "priority": node.priority,
                    "assigned_agent": node.assigned_agent,
                    "dependencies": list(node.dependencies),
                }
            )
        return {
            "tasks": tasks_state,
            "scheduled": [tid for tid in self._scheduled],
            "results": dict(self._results),
            "topological_order": self._graph.topological_order(),
        }


__all__: List[str] = ["DependencyGraph", "TaskScheduler"]