"""Regression guard: no statement may follow a terminator inside its own block.

Defect
------
``agent_core/llm/llama_server.py`` built the model-load failure path like
this::

    last_err = f"HTTP {None}"
    for model_ref in candidates:
        status, body = _http_json(...)
        ...
        last_err = (body.get("error", {}).get("message")
                    if isinstance(body, dict) else str(body)) or f"HTTP {status}"
    return False, last_err
    err = body.get("error", {}).get("message") if isinstance(body, dict) else str(body)
    return False, err or f"HTTP {status}"

The two statements after ``return`` can never execute -- mypy reports them as
``[unreachable]``. They are leftovers from the refactor that introduced
``last_err``, so the failure message they would build is dead too: the value
the caller actually sees always comes from ``last_err``.

The hazard is that dead statements read as live control flow. The trailing
``return`` *looks* like the real failure path, so someone "fixing" the HTTP
error message there would change nothing at runtime while believing they had.

Why a repository-wide scan
--------------------------
A repo-wide AST walk (the analysis that found this defect) reported exactly
one hit, so the guard can be enforced globally rather than pinned to a single
file: any new dead tail anywhere fails the build.

``else`` / ``elif`` / ``finally`` / ``except`` bodies are separate AST lists,
so they are scanned as their own blocks instead of being misread as code that
continues past a terminator.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Generated, vendored, cached or backup trees -- not maintained source.
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".pytest_tmp",
    "site-packages",
    "build",
    "dist",
    ".eggs",
    "backups",
    "agent_framework.egg-info",
    ".opencode",
    ".poolside",
}

TERMINATORS = (ast.Return, ast.Raise, ast.Continue, ast.Break)


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def _statement_lists(tree: ast.AST):
    """Every straight-line statement list in *tree*.

    Walks all nodes (so ``ExceptHandler.body`` and ``MatchCase.body`` are
    reached too) and yields the three list-valued statement fields. For a
    ``Try`` node that means the ``try``, ``else`` and ``finally`` clauses are
    reported as three independent blocks, never as one continuing tail.
    """
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and stmts:
                yield node, field, stmts


def find_dead_tails(path: Path) -> list[str]:
    """``file:line`` descriptions of statements that follow a terminator."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []

    findings: list[str] = []
    for node, _field, stmts in _statement_lists(tree):
        terminator_at = None
        for index, stmt in enumerate(stmts):
            if isinstance(stmt, TERMINATORS):
                terminator_at = index
                break
        if terminator_at is None:
            continue
        dead = stmts[terminator_at + 1:]
        if dead:
            first = dead[0]
            last = dead[-1]
            # Synthetic files handed in by tests live outside REPO_ROOT.
            try:
                display = str(path.relative_to(REPO_ROOT))
            except ValueError:
                display = str(path)
            findings.append(
                f"{display}:{node.lineno}: "
                f"{type(stmts[terminator_at]).__name__} at line "
                f"{stmts[terminator_at].lineno} is followed by unreachable "
                f"statement(s) {first.lineno}-{last.end_lineno}"
            )
    return findings


def test_no_statements_follow_a_terminator():
    """Dead tails are unreachable code and hide the real control flow."""
    findings: list[str] = []
    for path in _iter_python_files():
        findings.extend(find_dead_tails(path))

    assert not findings, (
        "Unreachable statement(s) after return/raise/continue/break -- delete "
        "the dead tail or move the logic above the terminator:\n  "
        + "\n  ".join(findings)
    )


def test_guard_actually_sees_the_source_tree():
    """The scan must not silently pass by finding no files to check."""
    files = _iter_python_files()
    assert len(files) > 50, f"scan covered only {len(files)} files"

    # The guarded module is real source, not filtered out as cache/venv.
    assert any(
        p.name == "llama_server.py" and "llm" in p.parts for p in files
    ), "llama_server.py was not scanned -- SKIP_DIRS is too broad"


def test_detects_a_constructed_dead_tail(tmp_path):
    """Meta-check: the detector fires on a synthetic offender."""
    bad = tmp_path / "offender.py"
    bad.write_text(
        "def f(x):\n"
        "    if x:\n"
        "        return 1\n"
        "        y = 2\n"
        "    return 3\n",
        encoding="utf-8",
    )
    findings = find_dead_tails(bad)
    assert len(findings) == 1
    # `y = 2` sits on line 4, directly under `return 1` on line 3.
    assert "Return at line 3 is followed by unreachable statement(s) 4-4" in findings[0]


def test_else_and_finally_bodies_are_scanned_separately(tmp_path):
    """A statement after `return` inside `finally` is dead; a clause after an
    `if`-`return` is not, and must not be reported as continuation."""
    dead = tmp_path / "dead_finally.py"
    dead.write_text(
        "def f():\n"
        "    try:\n"
        "        pass\n"
        "    finally:\n"
        "        return 1\n"
        "        after = 2\n",
        encoding="utf-8",
    )
    assert len(find_dead_tails(dead)) == 1

    alive = tmp_path / "alive_else.py"
    alive.write_text(
        "def f(x):\n"
        "    if x:\n"
        "        return 1\n"
        "    else:\n"
        "        return 2\n",
        encoding="utf-8",
    )
    assert find_dead_tails(alive) == []
