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
import re
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

        def visit(node: ast.AST, func_name: str | None) -> None:
            fname = func_name
            if isinstance(node, ast.FunctionDef):
                fname = node.name
            handlers = getattr(node, "handlers", None)
            # The canonical `_suppress_and_log` context manager IS the sink;
            # never flag its own body (would seed an unfixable issue).
            if handlers is not None and fname != "_suppress_and_log":
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
            for child in ast.iter_child_nodes(node):
                visit(child, fname)

        visit(tree, None)
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

    Returns ``True`` only if the generator actually changed one of the issue's
    files. A no-op run (the model emitted only ``[READ:]`` directives, declined,
    or produced the same content) must report ``False`` so the caller raises
    ``generate_failed`` instead of a misleading ``verify_failed`` (2026-08-28).
    """
    from agent_core.commands.fix_cmd import FixCommand

    files = _issue_files(issue)
    args = ["--yes", "--desc", issue.get("suggested_approach") or issue["title"]]
    args.extend(str(p) for p in files)
    asyncio.run(FixCommand().execute(args, agent))
    return _tree_changed(files)


def _tree_changed(files: list[Any]) -> bool:
    """True if any of *files* differs from HEAD in the working tree.

    Uses ``git status --porcelain`` so both modified and newly-created files are
    caught. Best-effort: returns False on any subprocess error.
    """
    import subprocess

    for f in files:
        try:
            res = subprocess.run(
                ["git", "status", "--porcelain", "--", str(f)],
                capture_output=True,
                text=True,
            )
        except Exception:
            return False
        if res.stdout.strip():
            return True
    return False


_LOG_LEVELS = frozenset({"debug", "info", "warning", "error", "critical", "exception"})


def _handler_log_call(handler: ast.ExceptHandler) -> tuple[str, str] | None:
    """Return (label, logger_expr_source) for a qualifying handler's leading log
    call, or None.

    Handles the canonical best-effort handler whose FIRST statement is a log call
    that includes the traceback::

        except Exception [as e]:
            logger.<level>(<msg>, traceback.format_exc())
            ...optional further fallback statements...

    ``logger_expr_source`` is the unparsed logger object (e.g. ``logger``) so the
    rewrite can emit ``<logger>.exception(label)`` and preserve the local logger.
    Anything that doesn't start with that exact log call returns None so we fail
    closed (e.g. handlers that reference the bound exception, or branch on it).
    """
    if not handler.body:
        return None
    stmt = handler.body[0]
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None
    call = stmt.value
    if not (isinstance(call.func, ast.Attribute) and call.func.attr in _LOG_LEVELS):
        return None
    # Must actually log the traceback via format_exc().
    if not any(
        isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute)
        and a.func.attr == "format_exc"
        for a in ast.walk(call)
    ):
        return None
    if not call.args:
        return None
    msg = call.args[0]
    if not isinstance(msg, ast.Constant) or not isinstance(msg.value, str):
        return None
    # If the handler binds the exception, it must not be referenced anywhere in
    # the body (it would be undefined inside the generated `with` block).
    if getattr(handler, "name", None):
        for n in ast.walk(handler):
            if isinstance(n, ast.Name) and n.id == handler.name:
                return None
    label = msg.value
    label = re.sub(r"\\n%s$", "", label)
    label = re.sub(r"%s$", "", label)
    label = re.sub(r"\\n$", "", label)  # avoid a doubled newline with the sink's own
    try:
        logger_expr = ast.unparse(call.func.value)
    except Exception:
        logger_expr = "logger"
    return label, logger_expr


def _extract_suppress_label(handler: ast.ExceptHandler) -> str | None:
    info = _handler_log_call(handler)
    return info[0] if info else None


def _try_qualifies(try_node: "ast.Try | ast.TryStar") -> bool:
    if len(try_node.handlers) != 1 or getattr(try_node, "orelse", None) or getattr(try_node, "finalbody", None):
        return False
    h = try_node.handlers[0]
    names = _except_type_names(h)
    if not (names & {"Exception", "BaseException", ""}):
        return False
    if not _handler_body_uses_traceback(h):
        return False
    return _handler_log_call(h) is not None


def _rewrite_log_swallow(src: str) -> tuple[str, int]:
    """Convert qualifying log-swallow excepts to ``with _suppress_and_log(label):``.

    Returns (new_source, count). Fails closed: only the canonical single-statement
    handler shape is rewritten; anything else is left untouched.
    """
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return src, 0
    sep = "\r\n" if "\r\n" in src else "\n"
    lines = src.splitlines(keepends=True)

    targets: list[ast.Try | ast.TryStar] = []

    def visit(node: ast.AST, func_name: str | None) -> None:
        fname = func_name
        if isinstance(node, ast.FunctionDef):
            fname = node.name
        if isinstance(node, (ast.Try, ast.TryStar)) and fname != "_suppress_and_log":
            if _try_qualifies(node):
                targets.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child, fname)

    visit(tree, None)
    if not targets:
        return src, 0

    # Apply bottom-up so earlier edits don't shift later line numbers.
    targets.sort(key=lambda n: n.lineno, reverse=True)
    replaced: list[tuple[int, int]] = []
    for node in targets:
        start = node.lineno - 1
        end = node.end_lineno  # exclusive
        assert end is not None
        # Skip if contained in an already-replaced range (nested tries).
        if any(r[0] <= start and end <= r[1] for r in replaced):
            continue
        h = node.handlers[0]
        info = _handler_log_call(h)
        if info is None:
            continue
        label, logger_expr = info
        base_line = lines[node.lineno - 1]
        base_indent = base_line[: len(base_line) - len(base_line.lstrip())]
        if len(h.body) == 1:
            # Canonical single-statement handler: swap the whole `try/except`
            # for `with _suppress_and_log(label): <try body>`. The handler's log
            # call is dropped (the context manager logs the traceback instead),
            # so on success the body returns before any logging; on exception the
            # label logs-and-swallows — identical original `except` semantics.
            try_body = lines[node.lineno : h.lineno - 1]
            nonblank = [b for b in try_body if b.strip()]
            common = min(len(b) - len(b.lstrip()) for b in nonblank) if nonblank else 0
            new_body: list[str] = []
            for b in try_body:
                if not b.strip():
                    new_body.append("")
                else:
                    new_body.append(base_indent + "    " + b[common:].rstrip("\r\n"))
            block = base_indent + "with _suppress_and_log(" + repr(label) + "):"
            for bl in new_body:
                block += sep + bl
            block += sep  # trailing newline so the following line isn't merged
            lines[start:end] = [block]
        else:
            # Multi-statement handler: keep `except Exception:` (so fallback
            # statements like `return ""` still run on exception) but replace only
            # the leading log call with `<logger>.exception(label)`. `logger.exception`
            # logs the *current* exception with its traceback and needs no
            # `traceback.format_exc()` call, so the detector no longer flags it and
            # the original exception is still captured — without changing control flow.
            stmt0 = h.body[0]
            s0 = stmt0.lineno - 1
            e0 = stmt0.end_lineno  # exclusive
            repl = base_indent + "    " + f"{logger_expr}.exception({repr(label)})" + sep
            lines[s0:e0] = [repl]
        replaced.append((start, end))

    if not replaced:
        return src, 0
    new_src = "".join(lines)
    if new_src == src:
        return src, 0
    return new_src, len(replaced)


def _ensure_suppress_import(src: str) -> str:
    """Add `from agent_core.suppress_log import _suppress_and_log` if not present."""
    if re.search(r"\bdef _suppress_and_log\b|\bfrom agent_core\.suppress_log import _suppress_and_log\b", src):
        return src
    lines = src.splitlines(keepends=True)
    sep = "\r\n" if "\r\n" in src else "\n"
    ins = 0
    for i, l in enumerate(lines):
        if l.lstrip().startswith("from __future__"):
            ins = i + 1
            break
    lines[ins:ins] = [f"from agent_core.suppress_log import _suppress_and_log{sep}"]
    return "".join(lines)


def _resolve_log_swallow(issue: dict[str, Any], agent: Any) -> bool:
    """Deterministic, AST-based fix for `best-effort-except` issues.

    Converts every qualifying ``except Exception: logger.<level>(msg,
    traceback.format_exc())`` into ``with _suppress_and_log(label):`` across the
    issue's files — no LLM guesswork, so it is complete and reproducible. The
    shared ``_suppress_and_log`` is imported where needed. Returns True only if a
    file actually changed.
    """
    files = _issue_files(issue)
    changed = False
    for fp in files:
        if not fp.exists():
            continue
        src = fp.read_text(encoding="utf-8")
        new_src, n = _rewrite_log_swallow(src)
        if n == 0:
            continue
        new_src = _ensure_suppress_import(new_src)
        try:
            compile(new_src, str(fp), "exec")
        except SyntaxError:
            continue
        fp.write_text(new_src, encoding="utf-8")
        changed = True
    return changed


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

    gen = generate_fn or (
        _resolve_log_swallow if issue.get("category") == "best-effort-except" else _default_generate
    )
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
