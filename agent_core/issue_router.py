"""Issue router: classify an issue/task description into a subagent role.

Pure function, no LLM, no I/O — deterministic keyword scoring so it is
trivially unit-testable and costs nothing per call.  Used as the default
routing step when the main agent (or a human) wants to know which role
should handle an issue; ambiguous input falls back to ``planner`` (triage),
which is the one role whose job is to decompose unclear work.

Scoring: every rule contributes one point per distinct keyword hit
(word-boundary matched, case-insensitive); the highest score wins, ties
break by rule order (most specific first).  Zero hits ⇒ planner.
"""
from __future__ import annotations

import re

from agent_core.subagent_roles import get_role

#: (role, keywords) in priority order — earlier rules win ties.
_ROUTE_RULES: tuple[tuple[str, frozenset[str]], ...] = (
    # Security first: a "bug" that is actually a leaked secret must not be
    # routed to the debugger who might paste it into a fix.
    ("security", frozenset({
        "secret", "secrets", "api key", "apikey", "password", "token leak",
        "credential", "vulnerability", "vulnerable", "cve", "exploit",
        "injection", "path traversal", "sanitize", "unsanitized",
    })),
    ("debugger", frozenset({
        "traceback", "stack trace", "exception", "typeerror", "valueerror",
        "attributeerror", "keyerror", "nameerror", "importerror",
        "runtime error", "crash", "crashes", "failing", "fails", "failure",
        "broken", "regressed", "wrong result", "incorrect output",
        "does not work", "doesn't work", "bug",
    })),
    ("tester", frozenset({
        "unit test", "unittest", "pytest", "test suite", "tests fail",
        "coverage", "regression test", "red-to-green", "write tests",
        "add tests", "untested",
    })),
    ("integrator", frozenset({
        "git commit", "commit message", "merge conflict", "rebase",
        "pull request", "push", "branch", "changelog entry", "tag release",
        "conventional commit",
    })),
    ("documenter", frozenset({
        "readme", "usage.md", "architecture.md", "changes.md", "docstring",
        "documentation drift", "docs are stale", "update docs", "document the",
    })),
    ("researcher", frozenset({
        "which version", "latest version", "upstream", "library docs",
        "external api", "breaking change in", "is there a library",
        "pypi", "npm equivalent",
    })),
    ("reviewer", frozenset({
        "code review", "pre-commit", "before committing", "review my diff",
        "review the diff", "sanity check", "second pair of eyes",
        "naming collision", "lint issue",
    })),
    ("implementer", frozenset({
        "implement", "add feature", "new feature", "create module",
        "create class", "create function", "write module", "scaffold",
        "extend the", "add support for", "refactor",
    })),
)
_WORD_RE = re.compile(r"[a-z0-9'<>/]+")


def route_with_score(issue_text: str) -> tuple[str, int]:
    """Return ``(role_name, score)`` for *issue_text*.

    Never raises; unknown/empty input routes to ``planner`` with score 0.
    The returned role is always a valid registry name.
    """
    lowered = (issue_text or "").lower()
    tokens = set(_WORD_RE.findall(lowered))
    best_role, best_score = "planner", 0
    for role_name, keywords in _ROUTE_RULES:
        score = 0
        for kw in keywords:
            if " " in kw or "?" in kw or "<" in kw:
                # Multi-word phrase — substring match on the lowered text.
                if kw in lowered:
                    score += 1
            elif kw in tokens:
                score += 1
        if score > best_score:
            best_role, best_score = role_name, score
    return best_role, best_score


def route_issue(issue_text: str) -> str:
    """Role name that should handle *issue_text* (fallback: ``planner``)."""
    return route_with_score(issue_text)[0]


def routing_table() -> str:
    """Human-readable table of the routing rules (for REPL help output)."""
    lines = ["issue -> role routing:"]
    for role_name, keywords in _ROUTE_RULES:
        spec = get_role(role_name)
        title = spec.title if spec else role_name
        sample = ", ".join(sorted(keywords)[:4])
        lines.append(f"  {role_name:<12} ({title}): {sample}, ...")
    lines.append("  planner      (Planner/Triage): anything else / ambiguous")
    return "\n".join(lines)


__all__ = ["route_issue", "route_with_score", "routing_table"]
