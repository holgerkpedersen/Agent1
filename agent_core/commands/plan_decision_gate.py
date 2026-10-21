"""Decision gate for plan execution.

Checks proposed plans against decisions.json constraint entries to ensure
proposed changes don't violate architectural decisions or security constraints.

Matching is boundary-aware, never raw substring
----------------------------------------------
Two substring bugs used to make a perfectly safe plan unexecutable, because a
failed gate aborts the whole run:

* a backticked bare extension such as ``.py`` (prose like "487 ``.py`` files")
  matched *every* decision-guarded ``*.py`` path, since ``".py" in
  "agent_core/file_system.py"`` is True. Extensions are not paths, so they are
  no longer collected (see :func:`_is_path_like`).
* a three-character decision id such as ``007`` matched inside an unrelated
  line number (``3007``), flagging "references resolved decision '007'".
  Ids now match only on a token boundary (see :func:`_references_decision`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Any

from .plan_schema import DecisionGateResult

#: A backticked token that is only a file extension (``.py``, ``.md``).
_EXTENSION_ONLY_RE = re.compile(r"^\.\w+$")

#: Characters that may be part of a decision id, used to keep ``007`` from
#: matching inside ``3007`` while still matching "decision 007".
_ID_BOUNDARY = r"[0-9A-Za-z_-]"


def _is_path_like(token: str) -> bool:
    """True when *token* plausibly names a file, not just an extension.

    Rejects ``.py`` (an extension is not a path), blank/whitespace tokens and
    URLs. This is what stops prose like "487 ``.py`` files" from being read as
    a reference to every guarded ``.py`` file.
    """
    token = token.strip()
    if not token or re.search(r"\s", token):
        return False
    if token.startswith("http") or "." not in token:
        return False
    return not _EXTENSION_ONLY_RE.match(token)


def _normalize_path(path: str) -> str:
    """Lowercase, forward-slash, strip a leading ``./``; keep a trailing slash.

    The trailing slash is preserved because it distinguishes a guarded
    *directory* (which covers everything beneath it) from a guarded *file*.
    """
    norm = path.strip().strip("`").lower().replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def _references_decision(decision_id: str, plan_text_lower: str) -> bool:
    """True when *decision_id* appears in the plan as a whole token.

    Prevents ``007`` from matching inside ``3007`` (or any longer id/number).
    """
    if not decision_id:
        return False
    pattern = rf"(?<!{_ID_BOUNDARY}){re.escape(decision_id.lower())}(?!{_ID_BOUNDARY})"
    return re.search(pattern, plan_text_lower) is not None


class PlanDecisionGate:
    """Validates a plan against architectural decisions before execution.

    Usage::

        gate = PlanDecisionGate(workspace_root=Path("."))
        result = gate.validate(plan_content)
        if not result.passed:
            for v in result.violations:
                print(v)
    """

    def __init__(self, workspace_root: Path | str):
        self.workspace_root = Path(workspace_root)
        self.decisions_path = self.workspace_root / ".decisions.json"
        self._decisions: Optional[List[dict[str, Any]]] = None

    def _load_decisions(self) -> List[dict[str, Any]]:
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
                        if (
                            "json.loads" in plan_content
                            and "pydantic" not in plan_content
                        ):
                            violations.append(
                                f"Decision {decision_id} ({title}): Plan uses raw json.loads instead of Pydantic schema validation"
                            )

            # Check 2: Ensure no plan contradicts resolved decisions
            resolved_by = decision.get("resolved_by")
            if resolved_by and _references_decision(str(resolved_by), plan_text_lower):
                violations.append(
                    f"Decision {decision_id} ({title}): Plan references resolved decision '{resolved_by}'"
                )

        return DecisionGateResult(
            passed=len(violations) == 0,
            violations=violations,
            metadata={"decisions_checked": len(decisions)},
        )

    def _extract_plan_paths(self, content: str) -> set[str]:
        """Extract file paths mentioned in the plan.

        Backticked tokens that are only an extension (``.py``) are skipped:
        they name a file *type*, not a file, and would otherwise match every
        guarded path ending in that extension.
        """
        paths = set()
        for m in re.finditer(r"`([^`\n]+)`", content):
            token = m.group(1).strip()
            if _is_path_like(token):
                paths.add(token)
        return paths

    def _path_matches(self, decision_path: str, plan_paths: set[str]) -> bool:
        """Check if a decision-guarded path matches any plan path.

        Matches on path-component boundaries, so a guarded ``agent.py`` does
        not match ``not_agent.py`` and a guarded ``security/`` directory does
        not match ``security_notes.md``.
        """
        decision_norm = _normalize_path(decision_path)
        if not decision_norm:
            return False

        for plan_path in plan_paths:
            plan_norm = _normalize_path(plan_path)
            if not plan_norm:
                continue
            if decision_norm == plan_norm:
                return True
            # A guarded DIRECTORY covers every path beneath it.
            if decision_norm.endswith("/") and plan_norm.startswith(decision_norm):
                return True
            # A path mentioned under a guarded directory, or vice versa.
            if plan_norm.endswith("/") and decision_norm.startswith(plan_norm):
                return True
            # Same file named at different depths, e.g. `agent.py` vs
            # `pkg/agent.py` — require the shorter path to be a suffix at a
            # component boundary.
            if plan_norm.endswith("/" + decision_norm) or decision_norm.endswith(
                "/" + plan_norm
            ):
                return True
        return False
