"""Decision gate for plan execution.

Checks proposed plans against decisions.json constraint entries to ensure
proposed changes don't violate architectural decisions or security constraints.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from .plan_schema import DecisionGateResult


class PlanDecisionGate:
    """Validates a plan against architectural decisions before execution.

    Usage::

        gate = PlanDecisionGate(workspace_root=Path("."))
        result = gate.validate(plan_content)
        if not result.passed:
            for v in result.violations:
                print(v)
    """

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.decisions_path = workspace_root / ".decisions.json"
        self._decisions: Optional[List[dict]] = None

    def _load_decisions(self) -> List[dict]:
        """Load and cache decisions from .decisions.json."""
        if self._decisions is not None:
            return self._decisions

        if not self.decisions_path.exists():
            self._decisions = []
            return self._decisions

        try:
            with open(self.decisions_path, "r", encoding="utf-8") as f:
                self._decisions = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._decisions = []

        return self._decisions

    def validate(self, plan_content: str) -> DecisionGateResult:
        """Check the plan against all active architectural decisions.

        Returns a DecisionGateResult indicating whether the plan passes
        all decision gates or lists violations.
        """
        decisions = self._load_decisions()
        violations: List[str] = []

        # Extract paths referenced in the plan
        plan_paths = self._extract_plan_paths(plan_content)
        plan_text_lower = plan_content.lower()

        for decision in decisions:
            decision_id = decision.get("id", "unknown")
            affected_files = decision.get("affected_files", [])
            tags = decision.get("tags", [])
            title = decision.get("title", "")

            # Check 1: If plan touches files guarded by a decision, verify intent
            for affected in affected_files:
                if self._path_matches(affected, plan_paths):
                    # Decision 002: Self-modification guard
                    if "self-modification-prevention" in tags:
                        if "security" in plan_text_lower or "guard" in plan_text_lower:
                            violations.append(
                                f"Decision {decision_id} ({title}): Plan attempts to modify security-guarded path '{affected}'"
                            )

                    # Decision 001: Schema validation must remain
                    if "pydantic-schema" in tags or "input-validation" in tags:
                        if "json.loads" in plan_content and "pydantic" not in plan_content:
                            violations.append(
                                f"Decision {decision_id} ({title}): Plan uses raw json.loads instead of Pydantic schema validation"
                            )

            # Check 2: Ensure no plan contradicts resolved decisions
            resolved_by = decision.get("resolved_by")
            if resolved_by and resolved_by.lower() in plan_text_lower:
                violations.append(
                    f"Decision {decision_id} ({title}): Plan references resolved decision '{resolved_by}'"
                )

        return DecisionGateResult(
            passed=len(violations) == 0,
            violations=violations,
            metadata={"decisions_checked": len(decisions)},
        )

    def _extract_plan_paths(self, content: str) -> set[str]:
        """Extract file paths mentioned in the plan."""
        paths = set()
        for m in re.finditer(r"`([^`\n]+)`", content):
            token = m.group(1).strip()
            if "." in token and not token.startswith("http"):
                paths.add(token)
        return paths

    def _path_matches(self, decision_path: str, plan_paths: set[str]) -> bool:
        """Check if a decision-guarded path matches any plan path."""
        decision_norm = decision_path.lower().replace("\\", "/")
        for plan_path in plan_paths:
            plan_norm = plan_path.lower().replace("\\", "/")
            if decision_norm in plan_norm or plan_norm in decision_norm:
                return True
        return False
