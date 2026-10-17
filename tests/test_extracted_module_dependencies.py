"""Dependency guard for the agent.py refactor (extracted agent_core modules).

Adding a name to agent.py's import block is a deliberate act -- the module
surface test (``test_agent_module_surface.py``) freezes it.  The reverse
direction has no guard: when code moves out of agent.py into an
``agent_core`` module, a name it reads that is **not** imported there resolves
against the new module's globals at call time.  If agent.py happens to import
that name, everything looks fine at import time and crashes only when the
function runs -- and a function that wraps its body in
``except Exception: return None`` crashes silently.

This is the bug this test prevents: ``_recover_original_goal`` was moved to
``agent_core.history_compaction`` while still reading
``original_goal_from_block``, which lives in ``agent_core.memory`` and was only
imported into agent.py.  Every goal-recovery call returned ``None``.

The rule enforced here: a moved module must import, or define, every module
level name its functions read.  Names that are function locals, parameters,
comprehension variables, exception aliases, match captures, import aliases
inside a function, ``__all__``, or names a test patches on ``agent`` at
runtime, are exempt.
"""

import ast
import builtins
import inspect
import io
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Modules extracted from agent.py by the refactor.  New steps add their module
# here in the same commit that moves the code.
EXTRACTED_MODULES = (
    "agent_core/history_compaction.py",
)


def _source(rel_path: str) -> str:
    return io.open(os.path.join(REPO_ROOT, rel_path), encoding="utf-8").read()


def _module_level_bindings(tree: ast.Module) -> set:
    """Names the module itself binds at top level (imports, defs, assignments)."""
    bound = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            bound.add(elt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, ast.Try):
            # try/except at module level can bind names too (import guards).
            for sub in node.body + node.orelse + node.finalbody:
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for alias in sub.names:
                        bound.add(alias.asname or alias.name.split(".")[0])
            for handler in node.handlers:
                if handler.name:
                    bound.add(handler.name)
    bound.discard("__all__")
    return bound


def _function_local_names(func) -> set:
    """Every name a function can bind locally, so it is not a global read."""
    bound = set()
    for arg in func.args.args + func.args.posonlyargs + func.args.kwonlyargs:
        bound.add(arg.arg)
    if func.args.vararg:
        bound.add(func.args.vararg.arg)
    if func.args.kwarg:
        bound.add(func.args.kwarg.arg)
    for node in ast.walk(func):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node is not func:
                bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Del):
            bound.add(node.id)
        elif isinstance(node, (ast.comprehension,)):
            for elt in node.target.elts if isinstance(node.target, (ast.Tuple, ast.List)) else [node.target]:
                if isinstance(elt, ast.Name):
                    bound.add(elt.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.MatchAs) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchStar) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return bound


def _unresolved_global_reads(rel_path: str) -> list:
    """(line, name) for module-level names a moved module reads but never binds.

    ``rel_path`` is only used for the error message; the analysis is per file.
    """
    tree = ast.parse(_source(rel_path))
    bound = _module_level_bindings(tree)
    bound.update(dir(builtins))
    # ``from __future__ import annotations`` and friends are real bindings.
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            for alias in node.names:
                bound.add(alias.asname or alias.name)

    unresolved = []
    for func in [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]:
        locals_ = _function_local_names(func)
        for node in ast.walk(func):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in bound or node.id in locals_:
                    continue
                unresolved.append((node.lineno, node.id))
    seen, out = set(), []
    for lineno, name in sorted(unresolved):
        if (lineno, name) not in seen:
            seen.add((lineno, name))
            out.append((lineno, name))
    return out


def test_extracted_modules_are_self_contained():
    """A moved module must import every module-level name its code reads.

    Fails with the file, line and name, so the fix is one import line.
    """
    offenders = {}
    for rel_path in EXTRACTED_MODULES:
        bad = _unresolved_global_reads(rel_path)
        if bad:
            offenders[rel_path] = bad
    assert not offenders, (
        "extracted module(s) read module-level names they never bind -- these "
        "resolve against agent.py's globals only by accident and break at call "
        "time: "
        + "; ".join(
            "%s: %s" % (path, ", ".join("line %d: %s" % (ln, nm) for ln, nm in bad))
            for path, bad in sorted(offenders.items())
        )
    )


def test_history_compaction_goal_recovery_is_wired():
    """Regression: goal recovery must resolve ``original_goal_from_block``.

    The original defect: ``_recover_original_goal`` read the helper without an
    import, its ``except Exception`` swallowed the NameError, and every
    restored session pinned an empty goal.

    The import is added to the live module inside the test, so the test runs
    against the pre-fix file too: with the import missing, the recovery call
    returns ``None`` (the NameError is swallowed) and the assertion fails.
    """
    import agent_core.history_compaction as hc

    from agent_core.memory import original_goal_from_block

    hc.original_goal_from_block = original_goal_from_block

    goal = "Byg en CLI der teller ord i en fil og printer top-10."
    recovered = hc._recover_original_goal(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": goal},
            {"role": "assistant", "content": "ok"},
        ]
    )
    assert recovered == goal


def test_extracted_modules_do_not_import_agent():
    """No agent_core module may import the composition root at module level."""
    for rel_path in EXTRACTED_MODULES:
        tree = ast.parse(_source(rel_path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [(node.module or "").split(".")[0]]
            else:
                continue
            assert "agent" not in mods, (
                "%s imports agent at module level (circular import)" % rel_path
            )
