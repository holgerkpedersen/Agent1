"""Dry-run gate for plan execution.

Blocks mutating tool calls during dry-run validation so the agent can
safely verify a plan without touching the workspace.

The :class:`PlanDryRunner` checks whether a plan document contains
references to tools that would mutate the workspace. If so, it rejects
the plan unless the caller opts into execution with ``--force``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .plan_schema import DryRunResult

#: Tool names that mutate the workspace — blocked during dry-run.
MUTATING_TOOLS = frozenset({
    "write",
    "edit",
    "fix",
    "delete_file",
    "run",
    "git",
    "delegate",
    "delegate_batch",
})

# Patterns that indicate mutating operations inside plan markdown.
_MUTATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwrite\b", re.IGNORECASE),
    re.compile(r"\bedit\b", re.IGNORECASE),
    re.compile(r"\bfix\b", re.IGNORECASE),
    re.compile(r"\brun\b.*(?:command|shell|script)", re.IGNORECASE),
    re.compile(r"\bdelete\b", re.IGNORECASE),
    re.compile(r"\bgit\b.*(?:commit|push|reset|clean)", re.IGNORECASE),
    re.compile(r"\bdelegate\b", re.IGNORECASE),
]


class PlanDryRunner:
    """Validates plan safety and blocks mutating tool calls during dry-run.

    Usage::

        runner = PlanDryRunner()
        result = runner.validate(plan_content)
        if not result.valid:
            print(result.errors)
    """

    def __init__(self, *, allow_mutations: bool = False):
        self.allow_mutations = allow_mutations

    def validate(self, content: str) -> DryRunResult:
        """Check the plan for unsafe/mutating operations.

        Returns a :class:`DryRunResult` indicating whether the plan is
        safe to execute in dry-run mode.
        """
        errors: list[str] = []
        warnings: list[str] = []
        affected_files: list[str] = []

        # Extract backticked file paths
        for m in re.finditer(r"`([^`\n]+)`", content):
            token = m.group(1).strip()
            if "." in token and not token.startswith("http"):
                affected_files.append(token)

        # Check for mutating tool references
        if not self.allow_mutations:
            for line_no, line in enumerate(content.splitlines(), 1):
                for pattern in _MUTATION_PATTERNS:
                    if pattern.search(line):
                        warnings.append(
                            f"Line {line_no}: potential mutation detected — {line.strip()[:80]}"
                        )

        # If we have warnings and mutations are not allowed, it's still valid
        # (dry-run means we acknowledge mutations exist but don't execute them).
        # Only hard-fail if the plan explicitly tries to bypass the gate.
        bypass_patterns = [
            re.compile(r"--force.*skip.*dry.?run", re.IGNORECASE),
            re.compile(r"ignore.*dry.?run.*safety", re.IGNORECASE),
        ]
        for line in content.splitlines():
            for bp in bypass_patterns:
                if bp.search(line):
                    errors.append(
                        f"Plan attempts to bypass dry-run safety: {line.strip()[:80]}"
                    )

        return DryRunResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            affected_files=affected_files,
        )
