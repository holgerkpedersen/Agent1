"""Regression guards for dead assignments (ruff F841) and walrus shadowing.

Two distinct defects motivated this file.

1. Dead assignments. 24 locals were assigned and never read. Several looked
   like real intent -- `greenfield`, `effective_ttl`, a timing `t`, a
   `before` tree snapshot -- so they read as live control flow while actually
   disabling whatever check they were meant to drive. F841 catches these.

2. A walrus operator used inside a call argument in
   `tests/test_autonomous_rationale.py`:

       out = build_repair_rationale(STUCK_REPEAT_ID := STUCK_REPEAT_REPAIR_ID, summary)

   Two facts, both verified empirically in this repo's Python 3.12, shape the
   test for this:

   * A walrus inside a function binds a **function-local** name. It does NOT
     rebind the module global (an explicit `global` statement is required for
     that). So this was shadowing, not cross-test global corruption.
   * F841 flags an unused walrus target, so the "dead local" half of the
     defect was already covered by the lint gate below. The half F841 misses
     is the case where the local IS subsequently read: F841 then stays
     silent, yet every later reference in that function silently resolves to
     the local instead of the module-level constant. That is the blind spot
     `test_no_walrus_shadows_a_module_level_name` covers.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that are not part of the linted source tree.
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages",
    "build", "dist", ".eggs",
}


def _iter_py_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return out


def _module_level_names(tree: ast.Module) -> set[str]:
    """Names bound at module scope, including inside module-level control flow."""
    names: set[str] = set()

    def collect(body: list[ast.stmt]) -> None:
        for st in body:
            if isinstance(st, ast.Assign):
                for t in st.targets:
                    names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
            elif isinstance(st, (ast.AnnAssign, ast.AugAssign)):
                names.update(n.id for n in ast.walk(st.target) if isinstance(n, ast.Name))
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(st.name)
            elif isinstance(st, (ast.Import, ast.ImportFrom)):
                for a in st.names:
                    names.add((a.asname or a.name).split(".")[0])
            elif isinstance(st, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                collect(st.body)
                collect(getattr(st, "orelse", []) or [])
                if isinstance(st, ast.Try):
                    for h in st.handlers:
                        collect(h.body)
                    collect(st.finalbody)

    collect(tree.body)
    return names


def _shadowing_walruses(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, name) for every walrus inside a function or lambda whose
    target shadows a module-level name.

    Unlike an explicit `global` statement, such a walrus creates a *local*
    binding. Any later reference to that name in the same function resolves
    to the local, silently bypassing the module-level constant or import.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - in-tree sources are always valid
        return []

    module_names = _module_level_names(tree)
    hits: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name):
                if sub.target.id in module_names:
                    hits.append((sub.lineno, sub.target.id))
    return hits


@pytest.mark.skipif(
    importlib.util.find_spec("ruff") is None, reason="ruff is not installed"
)
def test_no_dead_local_assignments_in_tree() -> None:
    """F841: no local variable may be assigned and never read.

    A dead assignment is not cosmetic. `greenfield = True` in workflow_cmd
    and `effective_ttl = ...` in metrics_store both read as live control
    flow; if the name were ever read, the bug would be a behaviour change
    rather than noise.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--output-format=concise",
         "--select", "F841", "."],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        "F841 dead local assignments reintroduced:\n" + proc.stdout + proc.stderr
    )


def test_no_walrus_shadows_a_module_level_name() -> None:
    """`:=` inside a function must not target a module-level name.

    F841 misses this whenever the local is read again. Passing a walrus as a
    call argument is especially easy to do by accident, and the result reads
    as a plain reference to the constant.
    """
    offenders: list[str] = []
    for path in _iter_py_files(REPO_ROOT):
        for lineno, name in _shadowing_walruses(path):
            rel = path.relative_to(REPO_ROOT).as_posix()
            offenders.append(f"{rel}:{lineno} shadows module-level {name!r} via :=")

    assert not offenders, "walrus shadowing a global:\n" + "\n".join(offenders)


def test_autonomous_rationale_passes_the_constant_unchanged() -> None:
    """The original call site stays a plain read of the constant.

    Checked through the AST rather than a substring match, so reformatting
    the call does not break the test.
    """
    path = REPO_ROOT / "tests" / "test_autonomous_rationale.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    call_args: list[ast.expr] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_repair_rationale"
        ):
            call_args.extend(node.args)

    assert call_args, "no build_repair_rationale(...) call found"
    for arg in call_args:
        assert not isinstance(arg, ast.NamedExpr), (
            f"build_repair_rationale called with a walrus expression: "
            f"{ast.dump(arg)}"
        )
