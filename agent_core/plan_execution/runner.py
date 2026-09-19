"""Execute a parsed plan via isolated subagents (role-gated, dependency-aware).

Wires the dormant :mod:`agent_core.orchestration` stack into the live agent:
the rich :class:`DependencyGraph` validates acyclicity before scheduling, and
:class:`TaskScheduler` dispatches each task to an isolated
:class:`~agent_core.subagent.SubAgent` (own conversation, shared filesystem,
role-gated tools).  A plan-mode parent still caps every child read-only, so
``plan exec`` must first switch to build mode (enforced by the command).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from agent_core.orchestration import DependencyGraph
from agent_core.orchestration.dependency_graph import CycleError
from agent_core.orchestration.task_scheduler import TaskExecutor, TaskScheduler
from agent_core.orchestration.types import TaskNode, TaskStatus

from .parser import PlanTask


async def _default_executor(
    agent: Any, task: PlanTask, input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Real executor: spawn an isolated subagent for *task* and run it."""
    sub = agent.spawn_subagent(name=f"exec-{task.id}", role=task.role)
    desc = task.description
    if input_data:
        parts = []
        for dep_id, res in input_data.items():
            summary = res.get("summary") or res.get("result") or ""
            parts.append(f"- {dep_id}: {summary}")
        desc = f"{desc}\n\nUpstream results from dependencies:\n" + "\n".join(parts)
    result = await sub.respond(desc)
    return {
        "success": True,
        "result": result,
        "summary": sub.get_context_summary(max_messages=3),
    }


def build_and_validate_graph(
    tasks: List[PlanTask],
) -> Tuple[DependencyGraph, List[str]]:
    """Build the dependency graph and return ``(graph, topological_order)``.

    Raises :class:`CycleError` on a dependency cycle (with the offending path)
    and :class:`ValueError` when a task references an unknown dependency.
    """
    graph = DependencyGraph()
    for t in tasks:
        graph.add_node(t.id, task_type=t.role)
    for t in tasks:
        for dep in t.depends_on:
            if not graph.has_node(dep):
                raise ValueError(
                    f"Task '{t.id}' depends on unknown task '{dep}'"
                )
            # add_dependency(dependent, dependency): dependency -> dependent.
            graph.add_dependency(t.id, dep)
    order = graph.topological_order()  # raises CycleError on a cycle
    return graph, order


async def run_plan(
    agent: Any,
    tasks: List[PlanTask],
    *,
    executor_factory: Optional[Callable[[PlanTask], TaskExecutor]] = None,
    dry_run: bool = False,
    plan_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Schedule and execute *tasks* through isolated subagents.

    *executor_factory* lets tests inject fake executors (no LLM).  Returns the
    scheduler :meth:`~agent_core.orchestration.task_scheduler.TaskScheduler.snapshot`.

    When *plan_dir* is provided, task completion state and dependency results are
    persisted to disk so a subsequent session can resume via ``plan_step``:
      - ``dep_<id>_result.json`` — written after each completed/failed task,
        consumed by downstream tasks' ``input_data``.
      - ``plan_execution_report.md`` — appended with one line per terminal task::

            - [<task_id>] <completed|failed> (role: <role>)

    This mirrors the filesystem contract that ``_nlp_plan_step`` already reads,
    making batch execution (``plan_start``) resumable across sessions (#108).
    """
    scheduler = TaskScheduler()
    for t in tasks:
        scheduler.add_task(
            TaskNode(
                task_id=t.id,
                name=t.id,
                description=t.description,
                dependencies=list(t.depends_on),
            )
        )

    # Capture each task's resolved input_data on the snapshot so callers
    # (and the execution report) can see how dependency results flowed in.
    captured_inputs: dict[str, dict[str, Any]] = {}

    def _factory(t: PlanTask) -> TaskExecutor:
        base = executor_factory(t) if executor_factory is not None \
            else lambda input_data: _default_executor(agent, t, input_data)

        async def wrapper(input_data: dict[str, Any]) -> dict[str, Any]:
            captured_inputs[t.id] = dict(input_data)
            return await base(input_data)

        return wrapper

    for t in tasks:
        scheduler.register_executor(t.id, _factory(t))
    for t in tasks:
        scheduler.schedule_task(t.id)

    # ── Persistence: write dep results + execution report per terminal task (#108) ──
    plan_dir_path = Path(plan_dir) if plan_dir else None
    if plan_dir_path is not None:
        def _persist_handler(msg: Any) -> None:
            """Subscribe to scheduler status updates; persist on completion/failure."""
            content = msg.content or {}
            tid = content.get("task_id")
            result = content.get("result", {}) or {}
            node_status = content.get("status")  # 'completed' | 'failed' | ...
            if not tid or node_status not in ("completed", "failed"):
                return

            task_obj = next((t for t in tasks if t.id == tid), None)
            role = task_obj.role if task_obj else ""

            # Write dependency result so downstream plan_step can consume it.
            dep_path = plan_dir_path / f"dep_{tid}_result.json"
            try:
                dep_path.write_text(
                    json.dumps(result, default=str), encoding="utf-8",
                )
            except Exception:
                pass

            # Append one line to the execution report (mirrors _nlp_plan_step).
            status_label = "completed" if node_status == "completed" else "failed"
            entry = f"- [{tid}] {status_label} (role: {role})\n"
            try:
                with open(plan_dir_path / "plan_execution_report.md", "a", encoding="utf-8") as f:
                    f.write(entry)
            except Exception:
                pass

        scheduler.subscribe(_persist_handler)

    if dry_run:
        snap = scheduler.snapshot()
        for rec in snap["tasks"]:
            rec["input_data"] = dict(captured_inputs.get(rec["task_id"], {}))
        return snap

    await scheduler.start()
    # Wait until nothing is pending, scheduled, or still in flight.
    while (
        scheduler.pending_count() > 0
        or scheduler.scheduled_count() > 0
        or scheduler._inflight
    ):
        await asyncio.sleep(0.02)
    await scheduler.stop()
    snap = scheduler.snapshot()
    for rec in snap["tasks"]:
        rec["input_data"] = dict(captured_inputs.get(rec["task_id"], {}))
    return snap
