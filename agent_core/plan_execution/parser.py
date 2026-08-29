"""Parse a proposed plan into an ordered, dependency-aware task list.

The contract between the planner (plan mode -> ``plan_proposed.md``) and the
executor (``plan exec``) is a ``## Tasks`` markdown block, or an explicit
``--tasks <json>`` file.  Both are parsed into :class:`PlanTask` objects so the
executor stays deterministic and testable (no LLM re-extraction of free text).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List


#: Role assumed for a task that does not name one explicitly.
DEFAULT_ROLE = "implementer"

_TASK_LINE_RE = re.compile(
    r"^\s*[-*]\s*"
    r"(?:\[(?P<id>[A-Za-z0-9_\-]+)\]|(?P<id2>[A-Za-z0-9_\-]+))"
    r"\s*[:\.)]?\s*"
    r"(?P<desc>.+?)"
    r"(?:\s*\(role:\s*(?P<role>[a-z_]+)\))?"
    r"(?:\s*[,;]?\s*(?:deps?|depends):\s*(?P<deps>[A-Za-z0-9_\-,\s]+))?"
    r"\s*$",
    re.IGNORECASE,
)

_TASKS_HEADING_RE = re.compile(r"^#{2,3}\s*tasks?\b", re.IGNORECASE)


@dataclass
class PlanTask:
    """One executable unit of a plan."""

    id: str
    description: str
    role: str = DEFAULT_ROLE
    depends_on: List[str] = field(default_factory=list)


def parse_tasks_from_json(text: str) -> List[PlanTask]:
    """Parse a JSON plan: ``{"tasks":[{"id","description","role?","depends_on?"}]}``."""
    data = json.loads(text)
    raw_tasks = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(raw_tasks, list):
        raise ValueError("JSON plan must contain a 'tasks' array")
    out: List[PlanTask] = []
    for i, item in enumerate(raw_tasks):
        if not isinstance(item, dict):
            raise ValueError(f"task #{i} is not an object")
        tid = str(item.get("id") or "").strip()
        if not tid:
            raise ValueError(f"task #{i} missing 'id'")
        desc = str(item.get("description") or "").strip()
        if not desc:
            raise ValueError(f"task '{tid}' missing 'description'")
        role = str(item.get("role") or DEFAULT_ROLE).strip() or DEFAULT_ROLE
        deps = item.get("depends_on") or []
        if isinstance(deps, str):
            deps = [d.strip() for d in deps.split(",") if d.strip()]
        out.append(
            PlanTask(
                id=tid,
                description=desc,
                role=role,
                depends_on=[str(d).strip() for d in deps if str(d).strip()],
            )
        )
    return out


def parse_tasks_from_markdown(text: str) -> List[PlanTask]:
    """Parse the ``## Tasks`` block of a markdown plan into tasks.

    Returns ``[]`` when no ``## Tasks``/``### Task`` heading is present, so the
    caller can decide how to react (error vs. empty plan).
    """
    lines = text.splitlines()
    start: int | None = None
    for i, line in enumerate(lines):
        if _TASKS_HEADING_RE.match(line):
            start = i + 1
            break
    if start is None:
        return []

    block: List[str] = []
    for line in lines[start:]:
        if line.startswith("#"):
            break
        block.append(line)

    out: List[PlanTask] = []
    for line in block:
        m = _TASK_LINE_RE.match(line)
        if not m:
            continue
        tid = (m.group("id") or m.group("id2") or "").strip()
        if not tid:
            continue
        desc = (m.group("desc") or "").strip().rstrip(".")
        if not desc:
            continue
        role = (m.group("role") or DEFAULT_ROLE).strip().lower() or DEFAULT_ROLE
        deps_raw = m.group("deps")
        deps = (
            [d.strip() for d in deps_raw.split(",") if d.strip()]
            if deps_raw
            else []
        )
        out.append(
            PlanTask(id=tid, description=desc, role=role, depends_on=deps)
        )
    return out


def parse_plan_tasks(text: str, *, fmt: str = "auto") -> List[PlanTask]:
    """Parse *text* into a list of :class:`PlanTask`.

    *fmt* is ``"auto"`` (try JSON first, fall back to markdown), ``"json"`` or
    ``"md"``.  Returns ``[]`` when no tasks are found (caller decides how to
    react).
    """
    if fmt == "json":
        return parse_tasks_from_json(text)
    if fmt == "md":
        return parse_tasks_from_markdown(text)
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return parse_tasks_from_json(text)
        except (ValueError, json.JSONDecodeError):
            return parse_tasks_from_markdown(text)
    return parse_tasks_from_markdown(text)
