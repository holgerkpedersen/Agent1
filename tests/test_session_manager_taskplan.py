"""Guard the session-manager test plan: it must stay machine-executable.

``docs/plans/2026-10-31-session-manager-test.md`` is not prose - it is the
input to ``plan exec``, which pushes its ``## Tasks`` block through
:func:`agent_core.plan_execution.runner.build_and_validate_graph`.  That gate
is strict and fails the WHOLE plan, not one step:

* ``ValueError`` when a task names a dependency that is not itself a task, and
* ``CycleError`` when dependencies form a cycle.

This test parses the real document with the real parser, asserts the exact
task ids, rejects roles unknown to the executor through ``get_role``, and runs
the real graph gate - so a broken dependency fails here, not at execution time.

Pure parsing + graph building: no LLM, no live agent, no workspace mutation.
"""

from __future__ import annotations

import pathlib

from agent_core.plan_execution import build_and_validate_graph, parse_plan_tasks
from agent_core.subagent_roles import get_role

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PLAN = REPO_ROOT / "docs" / "plans" / "2026-10-31-session-manager-test.md"

#: The steps the plan runs, in declaration order.
EXPECTED_TASK_IDS = (
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "T8",
    "T9",
    "T10",
    "T11",
    "T12",
)

#: Dependency edges the plan promises (unit suites before the command suite,
#: the guard after both, review last).
ORDER_INVARIANTS = {
    "manager unit tests before the clear-command suite": ("T1", "T4"),
    "stats display before end-to-end run": ("T5", "T7"),
    "confirmation loop before end-to-end run": ("T6", "T7"),
    "end-to-end before isolation": ("T7", "T8"),
    "both suites before the plan guard": ("T3", "T9"),
    "plan guard before the coverage run": ("T9", "T10"),
    "coverage run before documentation": ("T10", "T11"),
    "coverage run before review": ("T10", "T12"),
}


def _plan_text() -> str:
    assert PLAN.is_file(), f"plan missing: {PLAN}"
    return PLAN.read_text(encoding="utf-8")


def _tasks():
    tasks = parse_plan_tasks(_plan_text(), fmt="md")
    assert tasks, (
        f"{PLAN.name} parsed to 0 tasks - the '## Tasks' block is missing, "
        "mis-indented, or not in '- [id] description (role: x) deps: y' form"
    )
    return tasks


def test_plan_exists():
    assert PLAN.is_file(), f"plan missing: {PLAN}"


def test_tasks_block_parses_to_expected_ids():
    """The parser sees exactly the intended steps - no more, no fewer."""
    assert [t.id for t in _tasks()] == list(EXPECTED_TASK_IDS)


def test_no_duplicate_task_ids():
    ids = [t.id for t in _tasks()]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, f"duplicate task ids: {dupes}"


def test_every_task_has_a_real_description():
    for t in _tasks():
        assert len(t.description.strip()) >= 20, (
            f"task {t.id} has a stub description: {t.description!r}"
        )


def test_roles_are_known_to_the_executor():
    """``plan exec`` rejects unknown roles before running anything."""
    bad = [(t.id, t.role) for t in _tasks() if get_role(t.role) is None]
    assert not bad, f"tasks with unknown roles: {bad}"


def test_unknown_roles_are_rejected_by_get_role():
    """The role lookup the executor uses answers None for a made-up role."""
    assert get_role("tester") is not None
    assert get_role("documenter") is not None
    assert get_role("reviewer") is not None
    assert get_role("not-a-real-role") is None


def test_no_self_dependency():
    for t in _tasks():
        assert t.id not in t.depends_on, f"task {t.id} depends on itself"


def test_plan_passes_the_real_execution_gate():
    """The exact call ``plan_start`` makes: unknown dep raises, cycle raises."""
    tasks = _tasks()
    graph, order = build_and_validate_graph(tasks)

    assert graph.node_count() == len(tasks)
    assert len(order) == len(tasks), f"order {len(order)} != tasks {len(tasks)}"
    assert set(order) == {t.id for t in tasks}


def test_declared_ordering_invariants_hold():
    tasks = _tasks()
    _, order = build_and_validate_graph(tasks)
    pos = {tid: i for i, tid in enumerate(order)}
    for label, (before, after) in ORDER_INVARIANTS.items():
        assert pos[before] < pos[after], f"{label}: {before} must run before {after}"


def test_review_is_a_sink_and_coverage_feeds_it():
    """T11 (document) and T12 (review) are the plan's only sinks, both fed by T10.

    ``topological_order`` returns dependencies first, so every sink has no
    outgoing dependency edges in the graph.
    """
    tasks = {t.id: t for t in _tasks()}
    graph, order = build_and_validate_graph(list(tasks.values()))

    # The graph edge direction is dependency -> dependent (runner.py), so a
    # node with no outgoing edges is a sink: nothing depends on it.
    sinks = sorted(graph.leaves())
    assert sinks == ["T11", "T12"], f"unexpected sinks: {sinks}"

    assert "T10" in tasks["T12"].depends_on
    assert "T10" in tasks["T11"].depends_on
    # Dependencies come first in the order, so the review runs after T10.
    assert order.index("T10") < order.index("T12")
