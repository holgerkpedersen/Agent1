"""Unit tests for the plan-execution parser and graph builder.

No LLM / no live agent required: the executor seam (``run_plan``) is exercised
with a fake ``executor_factory`` so the dependency scheduling is fully tested
in isolation.
"""
from __future__ import annotations

import pytest

from agent_core.plan_execution import (
    PlanTask,
    build_and_validate_graph,
    parse_plan_tasks,
    run_plan,
)
from agent_core.orchestration.dependency_graph import CycleError


# ── parser: markdown ────────────────────────────────────────────────────
def test_md_parse_single_task_default_role():
    tasks = parse_plan_tasks("## Tasks\n- [T1] write the thing\n", fmt="md")
    assert tasks == [PlanTask(id="T1", description="write the thing", role="implementer")]


def test_md_parse_multi_with_role_and_deps():
    md = (
        "## Tasks\n"
        "- [T0] explore the module (role: planner)\n"
        "- [T1] implement feature (role: implementer) deps: T0\n"
        "- [T2] add tests depends: T1\n"
    )
    tasks = parse_plan_tasks(md, fmt="md")
    assert tasks[0].role == "planner"
    assert tasks[1].depends_on == ["T0"]
    assert tasks[2].role == "implementer"  # default when omitted
    assert tasks[2].depends_on == ["T1"]


def test_md_parse_no_tasks_block_returns_empty():
    tasks = parse_plan_tasks("# Plan\nNo tasks here.\n", fmt="md")
    assert tasks == []


def test_md_parse_id_without_brackets():
    tasks = parse_plan_tasks("## Tasks\n- T1: do it\n", fmt="md")
    assert len(tasks) == 1 and tasks[0].id == "T1"


# ── parser: json ───────────────────────────────────────────────────────
def test_json_parse_basic():
    txt = (
        '{"tasks":[{"id":"T1","description":"do it","role":"implementer",'
        '"depends_on":["T0"]}]}'
    )
    tasks = parse_plan_tasks(txt, fmt="json")
    assert tasks == [
        PlanTask(id="T1", description="do it", role="implementer", depends_on=["T0"])
    ]


def test_json_parse_missing_description_raises():
    import json
    with pytest.raises(ValueError):
        parse_plan_tasks('{"tasks":[{"id":"T1"}]}', fmt="json")


# ── graph builder ──────────────────────────────────────────────────────
def test_build_graph_topological_order_respects_deps():
    tasks = [
        PlanTask(id="T0", description="a"),
        PlanTask(id="T1", description="b", depends_on=["T0"]),
        PlanTask(id="T2", description="c", depends_on=["T1"]),
    ]
    graph, order = build_and_validate_graph(tasks)
    assert order.index("T0") < order.index("T1") < order.index("T2")
    assert graph.node_count() == 3


def test_build_graph_cycle_raises():
    tasks = [
        PlanTask(id="A", description="x", depends_on=["B"]),
        PlanTask(id="B", description="y", depends_on=["A"]),
    ]
    with pytest.raises(CycleError):
        build_and_validate_graph(tasks)


def test_build_graph_unknown_dependency_raises():
    tasks = [PlanTask(id="A", description="x", depends_on=["GHOST"])]
    with pytest.raises(ValueError):
        build_and_validate_graph(tasks)


# ── run_plan with a fake executor (no LLM) ─────────────────────────────
async def _fake_executor_factory(collect):
    def make(task: PlanTask) -> object:
        async def exec(input_data):
            collect.append((task.id, sorted(input_data.keys())))
            return {"success": True, "result": f"done {task.id}"}
        return exec
    return make


@pytest.mark.anyio
async def test_run_plan_orders_and_completes():
    calls: list[tuple[str, list[str]]] = []
    tasks = [
        PlanTask(id="T0", description="a"),
        PlanTask(id="T1", description="b", depends_on=["T0"]),
        PlanTask(id="T2", description="c", depends_on=["T1"]),
    ]
    snapshot = await run_plan(
        agent=None, tasks=tasks,
        executor_factory=await _fake_executor_factory(calls),
    )
    # All three reached a terminal state.
    assert all(t["status"] == "completed" for t in snapshot["tasks"])
    # Dependency results flowed into the child executor's input_data.
    by_id = {cid: deps for cid, deps in calls}
    assert by_id["T0"] == []
    assert by_id["T1"] == ["T0"]
    assert by_id["T2"] == ["T1"]


@pytest.mark.anyio
async def test_run_plan_dry_run_spawns_nothing():
    calls: list[tuple[str, list[str]]] = []
    tasks = [PlanTask(id="T1", description="a")]
    snapshot = await run_plan(
        agent=None, tasks=tasks, dry_run=True,
        executor_factory=await _fake_executor_factory(calls),
    )
    assert calls == []
    assert snapshot["tasks"][0]["status"] == "pending"
