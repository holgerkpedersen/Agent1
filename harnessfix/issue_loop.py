"""Issue resolution engine — detectors, verifiers, and the autonomous fix path.

Two detectors back Phase 0:

* ``duplication`` — a ``try`` with two handlers catching the same exception, or
  a broad ``except`` that is not the last handler (so later handlers are dead
  code). This is exactly the merge-introduced bug that motivated the system.
* ``best-effort-except`` — an inline ``except Exception: logger.<level>(...,
  traceback.format_exc())`` "log-and-swallow" block. These are the recurring
  copies the autonomous agent should collapse into the shared ``_suppress_and_log``
  context manager.

A detector doubles as the issue's **verifier**: once it reports nothing for the
issue's files, the issue is resolved. The collector seeds the ledger from these
detectors and is idempotent (stable ids).

The fix path reuses the existing ``fix`` command (so AGENTS.md invariants on
file writes hold) and the existing ``harnessfix.gates`` — it does NOT invent a
new gate or a new editor.
"""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import Any, Callable

import harnessfix.gates as gates
from harnessfix import issues as issue_store

REPO_ROOT = issue_store.REPO_ROOT

# Directories never scanned (runtime state, generated artifacts, vendored,
# caches). The pytest-collection crash on reports/harnessfix/generated/*.py is
# why generated dirs are excluded — we parse source, never import it.
_SKIP_DIRS = frozenset({
    ".git", "reports", "backups", ".docs", "node_modules", "__pycache__",
    "generated", ".venv", "venv", "build", "dist", ".pytest_cache",
    ".mypy_cache", "site-packages",
})


def _iter_py_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return out


def _except_type_names(handler: "ast.ExceptHandler") -> set[str]:
    if not handler.type:
        return {""}
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    names: set[str] = set()
    for t in types:
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, ast.Attribute):
            names.add(ast.unparse(t))
    return names


def _handler_body_uses_traceback(handler: "ast.ExceptHandler") -> bool:
    for node in ast.walk(handler):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "format_exc":
                return True
    return False


def find_duplicate_handlers(files: list[Path]) -> list[dict[str, Any]]:
    """Detect duplicate / unreachable ``except`` handlers (category=duplication)."""
    findings: list[dict[str, Any]] = []
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            handlers = getattr(node, "handlers", None)
            if not handlers:
                continue
            seen: set[str] = set()
            for idx, h in enumerate(handlers):
                for name in _except_type_names(h):
                    broad = name in ("", "Exception", "BaseException")
                    if broad and idx != len(handlers) - 1:
                        findings.append(_finding(
                            f, h.lineno, "duplication",
                            "broad except shadows later handlers (dead code)",
                        ))
                    if name in seen:
                        findings.append(_finding(
                            f, h.lineno, "duplication",
                            f"exception class {name!r} caught by more than one handler",
                        ))
                    seen.add(name)
    return findings


def find_log_swallow_excepts(files: list[Path]) -> list[dict[str, Any]]:
    """Detect inline log-and-swallow broad excepts (category=best-effort-except)."""
    findings: list[dict[str, Any]] = []
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            handlers = getattr(node, "handlers", None)
            if not handlers:
                continue
            for h in handlers:
                names = _except_type_names(h)
                if names & {"Exception", "BaseException", ""} and (
                    _handler_body_uses_traceback(h)
                ):
                    findings.append(_finding(
                        f, h.lineno, "best-effort-except",
                        "inline log-and-swallow except should use the shared "
                        "_suppress_and_log context manager",
                    ))
    return findings


def _finding(f: Path, lineno: int, category: str, what: str) -> dict[str, Any]:
    # Store repo-relative, POSIX-style paths so the committed .issues.json is
    # portable (stable ids across machines, not tied to an absolute checkout).
    try:
        rel = f.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel = f.as_posix()
    loc = f"{rel}:{lineno}"
    return {
        "file": str(rel),
        "line": lineno,
        "category": category,
        "locations": [loc],
        "title": f"{category} in {rel}:{lineno}",
        "evidence": what,
        "suggested_approach": (
            "wrap the try body in `with _suppress_and_log(...):` (or a "
            "parameterized variant) instead of an inline `except Exception: "
            "logger.<level>(..., traceback.format_exc())`"
            if category == "best-effort-except"
            else "remove the duplicate / unreachable except handler"
        ),
    }


def _issue_files(issue: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
    for loc in issue.get("locations", []):
        path = loc.rsplit(":", 1)[0]
        p = Path(path)
        # Locations are repo-relative; resolve against REPO_ROOT so the
        # verifier/generator find the real file regardless of cwd.
        if not p.is_absolute():
            p = REPO_ROOT / p
        out.append(p)
    # de-dupe, keep order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def verify_issue(issue: dict[str, Any], files: list[Path] | None = None) -> bool:
    """True when the issue's detector reports nothing for its files.

    Acceptance check: an issue is only resolved once the *same* detector that
    raised it no longer flags its locations — so a patch can't "pass tests"
    while the problem persists.
    """
    files = files if files is not None else _issue_files(issue)
    if issue["category"] == "duplication":
        return not find_duplicate_handlers(files)
    if issue["category"] == "best-effort-except":
        return not find_log_swallow_excepts(files)
    # Unknown category: cannot verify — leave unresolved (human attention).
    return False


def collect_issues(path: Path | None = None) -> int:
    """Scan the repo and seed new issues; idempotent. Returns count added."""
    issues = issue_store.load_issues(path)
    files = _iter_py_files(REPO_ROOT)
    added = 0
    for finder, category in (
        (find_duplicate_handlers, "duplication"),
        (find_log_swallow_excepts, "best-effort-except"),
    ):
        for fnd in finder(files):
            issue = issue_store.make_issue(
                category,
                fnd["title"],
                fnd["locations"],
                severity="low",
                evidence=fnd["evidence"],
                suggested_approach=fnd["suggested_approach"],
            )
            if issue_store.upsert(issues, issue):
                added += 1
    if added:
        issue_store.save_issues(issues, path)
    return added


def _default_generate(issue: dict[str, Any], agent: Any) -> bool:
    """Real fix path: reuse the ``fix`` command on the issue's files.

    Routes through ``agent_core.commands.fix_cmd`` so the existing py_compile
    gate and the AGENTS.md file-safety invariants (#1 never delete a
    pre-existing file, #2 wholesale-rewrite guard, #3 [FILE:] name match) all
    apply. Constructing the Agent is cheap (no LLM connection until a patch is
    requested).
    """
    from agent_core.commands.fix_cmd import FixCommand

    args = ["--yes", "--desc", issue.get("suggested_approach") or issue["title"]]
    args.extend(str(p) for p in _issue_files(issue))
    return bool(asyncio.run(FixCommand().execute(args, agent)))


def resolve_issue(
    issue: dict[str, Any],
    agent: Any,
    *,
    level_cap: int = issue_store.DEFAULT_AUTONOMY_LEVEL,
    generate_fn: Callable[[dict[str, Any], Any], bool] | None = None,
    run_benchmark: bool = False,
    model: str | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """Attempt one issue. Returns a summary dict (verdict + accepted + files).

    Fail-closed: any ambiguity returns a non-accepted verdict and leaves the
    ledger + tree untouched (the driver reverts via its git checkpoint). Never
    commits — staging/commit is the driver's job so it can scope to the issue's
    files plus ``.issues.json``.
    """
    if int(issue.get("autonomy_level", issue_store.DEFAULT_AUTONOMY_LEVEL)) > level_cap:
        return {"verdict": "level_too_high", "accepted": False, "issue_id": issue["id"]}

    files = _issue_files(issue)
    issues = issue_store.load_issues()

    # Idempotency: if already clean, resolve without touching the tree.
    if verify_issue(issue, files):
        issue_store.resolve(
            issues, issue["id"], "resolved", note="verifier passed before fix"
        )
        issue_store.save_issues(issues)
        return {
            "verdict": "already_resolved",
            "accepted": False,
            "issue_id": issue["id"],
        }

    gen = generate_fn or _default_generate
    try:
        applied = bool(gen(issue, agent))
    except Exception:  # noqa: BLE001 - generation failure is a soft stop, not a crash
        applied = False
    if not applied:
        return {
            "verdict": "generate_failed",
            "accepted": False,
            "issue_id": issue["id"],
        }

    if not verify_issue(issue, files):
        return {"verdict": "verify_failed", "accepted": False, "issue_id": issue["id"]}

    tests_passed, _ = gates.run_test_gate()
    sec_passed, _ = gates.run_security_gate()
    bench_ok = True
    if run_benchmark and int(issue.get("autonomy_level", 0)) >= 2:
        bench_ok = gates.run_benchmark_gate(model, profile) is not None
    if not (tests_passed and sec_passed and bench_ok):
        return {"verdict": "gates_failed", "accepted": False, "issue_id": issue["id"]}

    issue_store.resolve(
        issues, issue["id"], "resolved", note="autonomous fix passed gates"
    )
    issue_store.save_issues(issues)
    return {
        "verdict": "accepted",
        "accepted": True,
        "issue_id": issue["id"],
        "autonomy_level": int(issue.get("autonomy_level", 0)),
        "files": [str(p) for p in files],
    }
