"""Plan schema enums, dataclasses, and markdown validation constants.

This module centralizes all structural definitions for the plan workflow:
* :class:`PlanStatus` and :class:`PlanTransition` state machine enums.
* :class:`PlanMetadata` for YAML/JSON plan headers.
* :class:`DryRunResult` and :class:`DecisionGateResult` return types.
* :class:`PlanLogEntry` for JSONL audit logging.
* :func:`validate_plan_markdown` for structural schema compliance.
"""
from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any

# ── Enums ────────────────────────────────────────────────────────────────

class PlanStatus(enum.Enum):
    """Lifecycle status of a plan document."""
    PROPOSED = "proposed"
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"

class PlanTransition(enum.Enum):
    """Permitted state transitions."""
    SUBMIT = "submit"
    START = "start"
    FINISH = "finish"
    ERROR = "error"

# ── Dataclasses ──────────────────────────────────────────────────────────

@dataclass
class PlanMetadata:
    """Deserialized header from a plan markdown file."""
    id: str
    title: str
    status: PlanStatus = PlanStatus.PROPOSED
    author: str = "agent"
    tags: List[str] = field(default_factory=list)

@dataclass
class DryRunResult:
    """Outcome of a plan dry-run validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    affected_files: List[str] = field(default_factory=list)

@dataclass
class DecisionGateResult:
    """Outcome of checking a plan against decisions.json constraints."""
    passed: bool
    violations: List[str] = field(default_factory=list)
    metadata: Optional[dict[str, Any]] = None

@dataclass
class PlanLogEntry:
    """JSONL audit log entry for plan lifecycle events."""
    plan_id: str
    timestamp: str
    transition: PlanTransition
    status: PlanStatus
    details: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert enum values to strings for JSON serialization
        if isinstance(d["transition"], PlanTransition):
            d["transition"] = d["transition"].value
        if isinstance(d["status"], PlanStatus):
            d["status"] = d["status"].value
        return d

# ── Constants ────────────────────────────────────────────────────────────

# Protected filenames/regex patterns for plan audit trails
PROTECTED_PLAN_FILENAMES: frozenset[str] = frozenset({
    "plan_proposed.md",
    ".plans.json"
})

# ── Markdown Validation ─────────────────────────────────────────────────

_REQUIRED_HEADINGS = [
    "# Proposed Plan",
    "## Tasks"
]

def validate_plan_markdown(content: str) -> tuple[bool, List[str]]:
    """Verify that plan markdown follows the required schema."""
    errors = []
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    
    # Check required headings
    found_headings = set()
    for line in lines:
        if line.startswith("#"):
            found_headings.add(line)
            
    for heading in _REQUIRED_HEADINGS:
        if not any(h.startswith(heading) for h in found_headings):
            errors.append(f"Missing required heading: {heading}")
            
    # Check for at least one task
    if "- [" not in content and "- " not in content:
        if "* [" not in content and "* " not in content:
            errors.append("No tasks found (expected lines starting with '- ' or '* ')")
            
    return len(errors) == 0, errors
