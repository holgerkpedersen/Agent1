"""Integration test verifying the completion-gate drain path of TaskScheduler.

When ``stop`` is invoked while tasks are still in flight, every tracked task must
eventually reach a terminal state (COMPLETED or FAILED) and no inflight asyncio
tasks should be left dangling behind (``_inflight == set()``).

Uses the real orchestration stack: ``src.agent1.core.TaskNode`` /
``TaskStatus`` plus ``src.agent1.orchestration.task_scheduler.TaskScheduler``.
"""

import asyncio
from typing import Any, Awaitable, Dict

import pytest

from src.agent1.core import TaskNode, TaskStatus
from src.agent1.orchestration.task_scheduler import TaskExecutor, TaskScheduler


async def _slow_executor(_input: Dict[str, Any]) -> Dict[str, Any]:  # noqa: ANN401 - test stub
    """Deliberately slow executor so we can cancel mid-flight."""

    await asyncio.sleep(0.5)
    return {"success": True}


@pytest.fixture
def _chain_executor() -> TaskExecutor:
    return _slow_executor


async def _build_scheduler_with_chain(scheduler: TaskScheduler, executor: TaskExecutor) -> None:
    """Register a dependency chain (a -> b -> c) plus an independent task d."""

    scheduler.add_task(TaskNode(task_id="a", name="A", description="", dependencies=[]))
    scheduler.add_task(
        TaskNode(task_id="b", name="B", description="", dependencies=["a"])
    )
    scheduler.add_task(
        TaskNode(task_id="c", name="C", description="", dependencies=["b"])
    )
    scheduler.add_task(TaskNode(task_id="d", name="D", description="", dependencies=[]))

    for tid in ("a", "b", "c", "d"):
        scheduler.register_executor(tid, executor)


@pytest.mark.anyio
async def test_completion_drain_on_stop(_chain_executor: TaskExecutor) -> None:
    """stop() drains every inflight task before returning; none orphaned."""

    scheduler = TaskScheduler(executors={})
    await _build_scheduler_with_chain(scheduler, _chain_executor)
    for tid in ("a", "b", "c", "d"):
        scheduler.schedule_task(tid)
    await scheduler.start()  # noqa: ANN201 - start runs the monitor loop

    # cancel while tasks are still running (sleep 0.5 > our 0.1 window)
    await asyncio.sleep(0.1)
    await scheduler.stop()

    assert scheduler._inflight == set(), "no inflight task should remain after stop"


@pytest.mark.anyio
async def test_every_tracked_task_is_terminal_after_cancel(_chain_executor: TaskExecutor) -> None:
    """After cancellation, no RUNNING/PENDING task remains non-terminal."""

    scheduler = TaskScheduler(executors={})
    await _build_scheduler_with_chain(scheduler, _chain_executor)
    for tid in ("a", "b", "c", "d"):
        scheduler.schedule_task(tid)
    await scheduler.start()  # noqa: ANN201 - start runs the monitor loop

    # let tasks dispatch and complete (executor sleeps 0.5s; chain a->b->c ~1.5s total)
    await asyncio.sleep(2.0)
    await scheduler.stop()

    terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED}
    assert scheduler._inflight == set(), "no inflight task should remain after stop"
    # every task that produced a result must be in a terminal state
    for tid in scheduler._results:
        node = scheduler._graph.get_task(tid)
        assert node is not None and node.status in terminal, (
            f"dispatched task '{tid}' left non-terminal: {node.status}"
        )


@pytest.mark.anyio
async def test_cancel_schedule_leaves_no_inflight(_chain_executor: TaskExecutor) -> None:
    """cancel_schedule on a not-yet-due task plus stop drains cleanly."""

    scheduler = TaskScheduler(executors={})
    await _build_scheduler_with_chain(scheduler, _chain_executor)
    # schedule d first so cancel_schedule can actually remove it from the queue
    scheduler.schedule_task("d")
    assert scheduler.cancel_schedule("d") is True
    for tid in ("a", "b", "c"):
        scheduler.schedule_task(tid)
    await scheduler.start()  # noqa: ANN201 - start runs the monitor loop

    await asyncio.sleep(0.1)
    await scheduler.stop()

    assert scheduler._inflight == set(), "no inflight task should remain after stop"
