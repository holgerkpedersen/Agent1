"""Plan execution: parse a proposed plan and run it via isolated subagents.

This is the executor half of the plan-mode workflow (plan #13 / #14 handoff):
a planner produces a ``.docs/<ts>/plan_proposed.md`` (persisted by
``Agent._persist_plan_answer``), and ``plan exec`` turns its ``## Tasks`` block
into dependency-aware work dispatched to role-gated, isolated subagents.
"""
from __future__ import annotations

from .parser import PlanTask, parse_plan_tasks
from .runner import build_and_validate_graph, run_plan

__all__: list[str] = [
    "PlanTask",
    "parse_plan_tasks",
    "build_and_validate_graph",
    "run_plan",
]
