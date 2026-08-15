"""Timestamped document folders for workflow-generated markdown files.

Workflow pipeline docs (spec / analysis / plan / entities / tasks) used to be
written to the workspace root (``project_*.md``), polluting the repo.  They
now live in ``.docs/<YYYY-MM-DD_HH-MM-SS>/`` — one folder per run, so
repeated runs stay apart.  ``.docs/`` is git-ignored (see .gitignore).

Readers (implement / fix / decide) use :func:`find_doc` to locate the newest
copy: latest run folder first, workspace root second (legacy files).
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

DOCS_DIR_NAME = ".docs"

#: Document names produced by the workflow pipeline (used by readers).
WORKFLOW_DOC_NAMES = (
    "project_spec.md",
    "project_features.md",
    "project_analysis.md",
    "project_plan.md",
    "project_entities.md",
    "project_tasks.md",
)


def run_stamp(now: datetime | None = None) -> str:
    """Timestamp folder name ``YYYY-MM-DD_HH-MM-SS`` (lexicographically sortable)."""
    return (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")


def docs_root(workspace: str | Path) -> Path:
    """The ``.docs`` directory under *workspace* (created if missing)."""
    root = Path(workspace) / DOCS_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_run_dir(workspace: str | Path, now: datetime | None = None) -> Path:
    """Create and return a fresh ``.docs/<stamp>/`` run folder.

    If two runs start in the same second the folder gets a ``_2``, ``_3`` ...
    suffix so runs never share a folder.
    """
    stamp = run_stamp(now)
    run = docs_root(workspace) / stamp
    n = 1
    while run.exists():
        n += 1
        run = docs_root(workspace) / f"{stamp}_{n}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def latest_run_dir(workspace: str | Path) -> Path | None:
    """Most recent ``.docs/<stamp>/`` folder, or ``None`` when there are none.

    Folder names are timestamps, so lexicographic order is chronological.
    """
    root = Path(workspace) / DOCS_DIR_NAME
    if not root.is_dir():
        return None
    runs = [d for d in root.iterdir() if d.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda d: d.name)


def find_doc(workspace: str | Path, name: str) -> str | None:
    """Locate *name*: newest run folder first, then the workspace root (legacy).

    Returns a path string, or ``None`` when the doc exists in neither place.
    """
    run = latest_run_dir(workspace)
    if run is not None:
        cand = run / name
        if cand.is_file():
            return str(cand)
    legacy = Path(workspace) / name
    if legacy.is_file():
        return str(legacy)
    return None


def find_input(workspace: str | Path, path: str) -> str:
    """Resolve an input argument.

    *path* is interpreted against the WORKSPACE: relative inputs are joined
    with *workspace* FIRST (never the process CWD), then checked; the newest
    ``.docs`` run folder (or root) copy of its basename is used as fallback.
    Always returns an absolute path; when nothing is found, the resolved
    absolute path is returned so the caller's own "file not found" error
    still fires.
    """
    ws = str(Path(workspace).resolve())
    abs_path = path if os.path.isabs(path) else os.path.join(ws, path)
    if os.path.exists(abs_path):
        return abs_path
    found = find_doc(ws, os.path.basename(abs_path))
    return found if found else abs_path


def resolve_output(workspace: str | Path, out_arg: str, sibling_of: str | None = None) -> str:
    """Resolve a command output argument.

    - Explicit paths (relative with a directory, or absolute) are respected
      as given.
    - Bare filenames (no directory component) are routed into a run folder:
      the folder containing *sibling_of* when it lives in one (so a manual
      plan/entities/taskplan pipeline stays in a single run folder), else a
      fresh ``.docs/<stamp>/`` folder.
    """
    if os.path.isabs(out_arg) or os.path.dirname(out_arg):
        return out_arg
    if sibling_of:
        parent = Path(sibling_of).parent
        if parent.is_dir() and parent.parent.name == DOCS_DIR_NAME:
            return str(parent / out_arg)
    return str(new_run_dir(workspace) / out_arg)
