"""Issue ledger — systematic, autonomous-handled work items.

This is deliberately *separate* from ``.decisions.json``: decisions record the
*why* (architectural choices), issues record the *what* (concrete, locatable
work to fix or improve). An issue may reference a decision via ``decision_ref``
but the two ledgers never overlap.

The ledger is committed at the repo root as ``.issues.json`` so the autonomous
agent and the human reviewer see the identical work list on every machine.
Writes are atomic (temp file + rename) because three writers touch it: the
collector (seeds new findings), the ``issue`` REPL command (triage/status), and
the autonomous driver (outcome).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ISSUES_PATH = REPO_ROOT / ".issues.json"

# autonomy_level: 0 = human-only, 1 = auto-safe (tests + security gates),
# 2 = benchmark-required (explicit human promotion). New issues default to 1.
DEFAULT_AUTONOMY_LEVEL = 1
VALID_AUTONOMY_LEVELS = (0, 1, 2)

RESOLVED_STATUSES = ("resolved", "wontfix")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(category: str, locations: list[str]) -> str:
    """Stable id from (category, locations) so the collector is idempotent."""
    loc = "+".join(sorted(locations))
    cat = re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-")
    digest = re.sub(r"[^A-Za-z0-9]", "_", loc)[:80]
    return f"iss-{cat}-{digest}"


def load_issues(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or ISSUES_PATH
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return []
    return data if isinstance(data, list) else []


def save_issues(issues: list[dict[str, Any]], path: Path | None = None) -> None:
    """Atomic write: temp file + os.replace so a crash can't corrupt the ledger."""
    p = path or ISSUES_PATH
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(issues, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def find_by_id(issues: list[dict[str, Any]], issue_id: str) -> dict[str, Any] | None:
    for it in issues:
        if it.get("id") == issue_id:
            return it
    return None


def make_issue(
    category: str,
    title: str,
    locations: list[str],
    *,
    severity: str = "low",
    evidence: str = "",
    suggested_approach: str = "",
    decision_ref: str | None = None,
    autonomy_level: int = DEFAULT_AUTONOMY_LEVEL,
) -> dict[str, Any]:
    return {
        "id": _slug(category, locations),
        "title": title,
        "category": category,
        "severity": severity,
        "locations": list(locations),
        "status": "open",
        "autonomy_level": autonomy_level,
        "evidence": evidence,
        "suggested_approach": suggested_approach,
        "decision_ref": decision_ref,
        "repair_ref": None,
        "created_at": _now(),
        "resolved_at": None,
    }


def upsert(issues: list[dict[str, Any]], issue: dict[str, Any]) -> bool:
    """Add ``issue`` only if its id is absent.

    Never overwrites a human-set ``status``/``autonomy_level`` — idempotent for
    repeated collector runs. Returns True if a new issue was added.
    """
    if find_by_id(issues, issue["id"]) is not None:
        return False
    issues.append(issue)
    return True


def resolve(
    issues: list[dict[str, Any]],
    issue_id: str,
    disposition: str = "resolved",
    note: str = "",
) -> bool:
    it = find_by_id(issues, issue_id)
    if it is None:
        return False
    it["status"] = disposition
    it["resolved_at"] = _now() if disposition in RESOLVED_STATUSES else None
    if note:
        it["evidence"] = f"{it.get('evidence', '')}\n[note] {note}".strip()
    return True


def promote(
    issues: list[dict[str, Any]], issue_id: str, level: int,
) -> tuple[bool, str]:
    if level not in VALID_AUTONOMY_LEVELS:
        return (
            False,
            f"invalid autonomy_level {level!r}; expected {VALID_AUTONOMY_LEVELS}",
        )
    it = find_by_id(issues, issue_id)
    if it is None:
        return False, f"no such issue {issue_id!r}"
    it["autonomy_level"] = level
    return True, f"promoted {issue_id} to autonomy_level={level}"


def open_issues(
    issues: list[dict[str, Any]], *, max_level: int | None = None,
) -> list[dict[str, Any]]:
    """Open issues eligible for the autonomous driver.

    ``max_level`` caps by autonomy_level (the driver's ``AGENT_AUTONOMY_LEVEL``).
    Deferred/wontfix are excluded; resolved are excluded.
    """
    out = []
    for it in issues:
        if it.get("status") in ("resolved", "wontfix", "deferred"):
            continue
        lvl = int(it.get("autonomy_level", DEFAULT_AUTONOMY_LEVEL))
        if max_level is not None and lvl > max_level:
            continue
        out.append(it)
    return out
