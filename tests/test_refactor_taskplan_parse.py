"""Guard the agent.py refactor taskplan: it must stay machine-executable.

``docs/refactor_agent_py_taskplan.md`` is not prose — it is the input to
``plan exec``, which resolves it through ``latest_run_dir()/plan_proposed.md``
or an explicit plan file and then runs it through
:func:`agent_core.plan_execution.runner.build_and_validate_graph`.

That gate is strict and fails the WHOLE plan, not just one step:

* ``ValueError`` when a task names a dependency that is not itself a task in
  the list, and
* ``CycleError`` when dependencies form a cycle.

The real regression this guards
-------------------------------
Steps ``t01`` (baseline + module-surface guard) and ``t02`` (bootstrap
extraction) were already COMPLETED and committed (b45d5c8, 7dac445), so they
are deliberately absent from the ``## Tasks`` block. But ``t03``-``t06`` still
carried ``deps: t01`` and ``t08``/``t09`` carried ``deps: t02``. Those six
dangling references made ``plan_start`` abort before running anything with
``Task 't03' depends on unknown task 't01'``.

This test parses the real document with the real parser and pushes it through
the real graph gate, so a future edit that adds a dependency on a completed or
renamed step fails here instead of at execution time.

Pure parsing + graph building: no LLM, no live agent, no workspace mutation.
"""
from __future__ import annotations

import pathlib

import pytest

from agent_core.plan_execution import build_and_validate_graph, parse_plan_tasks
from agent_core.subagent_roles import get_role

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TASKPLAN = REPO_ROOT / "docs" / "refactor_agent_py_taskplan.md"

#: Steps that are already DONE and committed, so they must NOT appear in the
#: ``## Tasks`` block and must never be named as a dependency.
COMPLETED_STEPS = ("t01", "t02")

#: The remaining, still-to-do steps, in the order the plan runs them.
EXPECTED_TASK_IDS = (
    "t03",
    "t04",
    "t05",
    "t06",
    "t07",
    "t08",
    "t09",
    "t10",
    "t11",
    "t11b",
    "t12",
    "t13",
    "t14",
    "t15",
)


def _plan_text() -> str:
    assert TASKPLAN.is_file(), f"taskplan missing: {TASKPLAN}"
    return TASKPLAN.read_text(encoding="utf-8")


def _tasks():
    tasks = parse_plan_tasks(_plan_text(), fmt="md")
    assert tasks, (
        f"{TASKPLAN.name} parsed to 0 tasks — the '## Tasks' block is missing, "
        "mis-indented, or not in '- [id] description (role: x) deps: y' form"
    )
    return tasks


def test_taskplan_exists():
    assert TASKPLAN.is_file(), f"taskplan missing: {TASKPLAN}"


def test_tasks_block_parses_to_expected_ids():
    """The parser sees exactly the intended steps — no more, no fewer."""
    assert [t.id for t in _tasks()] == list(EXPECTED_TASK_IDS)


def test_every_task_has_a_real_description():
    for t in _tasks():
        assert len(t.description.strip()) >= 20, (
            f"task {t.id} has a stub description: {t.description!r}"
        )


def test_no_duplicate_task_ids():
    ids = [t.id for t in _tasks()]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, f"duplicate task ids: {dupes}"


def test_roles_are_known_to_the_executor():
    """``plan exec`` rejects unknown roles before running anything."""
    bad = [(t.id, t.role) for t in _tasks() if get_role(t.role) is None]
    assert not bad, f"tasks with unknown roles: {bad}"


def test_completed_steps_are_not_relisted_or_depended_on():
    """Regression: deps on the already-done t01/t02 aborted plan_start.

    They are not tasks any more, so naming them is an unknown dependency.
    """
    tasks = _tasks()
    ids = {t.id for t in tasks}

    relisted = [s for s in COMPLETED_STEPS if s in ids]
    assert not relisted, (
        f"completed step(s) {relisted} are listed as tasks again — remove them "
        "from the '## Tasks' block"
    )

    for t in tasks:
        dangling = [d for d in t.depends_on if d in COMPLETED_STEPS]
        assert not dangling, (
            f"task {t.id} still depends on completed step(s) {dangling}; "
            "completed steps are not in the task list, so the graph gate "
            "aborts with 'depends on unknown task'"
        )


def test_no_self_dependency():
    for t in _tasks():
        assert t.id not in t.depends_on, f"task {t.id} depends on itself"


def test_plan_passes_the_real_execution_gate():
    """The exact call ``plan_start`` makes: unknown dep -> ValueError, cycle -> CycleError."""
    tasks = _tasks()
    graph, order = build_and_validate_graph(tasks)

    assert graph.node_count() == len(tasks)
    assert len(order) == len(tasks), f"topological order {len(order)} != tasks {len(tasks)}"
    assert set(order) == {t.id for t in tasks}


def test_phase_ordering_invariants_hold():
    """The move order the taskplan promises (each step needs its dependencies first)."""
    tasks = _tasks()
    _, order = build_and_validate_graph(tasks)
    pos = {tid: i for i, tid in enumerate(order)}

    invariants = {
        "shell_intel before system_prompt": ("t05", "t07"),
        "llm client before repl": ("t08", "t10"),
        "signals before repl": ("t09", "t10"),
        "agent_files before agent_effects": ("t11", "t11b"),
        "agent_effects before nlp handlers": ("t11b", "t12"),
        "nlp handlers before plan/prompt/memory": ("t12", "t13"),
        "plan/prompt/memory before turn loop": ("t13", "t14"),
        "turn loop before consolidation": ("t14", "t15"),
    }
    for label, (before, after) in invariants.items():
        assert pos[before] < pos[after], f"{label}: {before} must run before {after}"

    assert order[-1] == "t15", f"consolidation must run last, got {order[-1]}"


def test_consolidation_depends_on_everything():
    """t15 is the final sweep, so nothing may be left un-run when it starts."""
    tasks = {t.id: t for t in _tasks()}
    t15 = tasks["t15"]
    missing = sorted(set(EXPECTED_TASK_IDS) - {"t15"} - set(t15.depends_on))
    assert not missing, f"t15 does not depend on: {missing}"
    assert t15.role == "integrator"
