"""Discrepancy detection — compare executed actions against recorded decisions.

Phase 2 of the autonomous-decision plan (``.decisions_plan.md``): the agent
records what it *decided* in ``.decisions.json``, and this module reports
where an actual tool call diverged from those decisions.

Three discrepancy kinds are detected:

``blocked_action``
    The call was refused — by plan mode's read-only executor gate or by the
    self-modification guard — yet it targeted a decision-guarded path.  The
    guardrail held, but the agent tried.

``contradiction``
    A call that a decision explicitly forbids actually SUCCEEDED.  The
    decision is being violated in practice, so this is critical.

``unresolved_context``
    The action touches files of a decision that still carries open
    contradictions — acting on unsettled ground.

The module reads decisions only (never writes them); the JSONL ledger is
written by :func:`agent_core.decisions.record_discrepancy`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_core.decisions import (
    STATUS_ACTIVE,
    _canonical_rel,
    _decision_path,
)

#: ``decision_id`` used for discrepancies that no ledger entry owns (the
#: session mode itself refused the call).
PLAN_MODE_DECISION_ID = "plan-mode"

#: Tag marking a decision as a self-modification guard: the decision's stated
#: paths are forbidden write targets, not merely "files this decision touched".
GUARD_TAGS = frozenset({"self-modification-prevention"})

#: Sequence rating: agent params that carry a work-target path.
_PATH_ARG_KEYS = (
    "path",
    "file",
    "filepath",
    "filename",
    "target",
    "files",
    "filenames",
    "source",
    "destination",
)

#: A path-like token in decision prose (``agent_core/security/``).
_PATH_TOKEN_RE = re.compile(r"(?<![\w/`])((?:[\w.-]+/)+[\w.-]*)")

#: Tool results that mean "the guard said no" — the workspace did NOT change.
_REFUSAL_RE = re.compile(
    r"(?i)\b(?:refus\w*|blocked|rejected|denied|not\s+allowed|forbidden)\b"
)

#: Plan mode's executor rejection (see :data:`agent_core.modes._PLAN_REJECTION`).
_PLAN_REJECTION_RE = re.compile(r"^\s*\[plan mode\]")


@dataclass
class Discrepancy:
    """One divergence between an action and a recorded decision."""

    decision_id: str
    decision_title: str
    action_tool: str
    action_args: dict[str, Any]
    discrepancy_type: str  # blocked_action | unresolved_context | contradiction
    severity: str  # warning | error | critical
    suggestion: str | None = None

    def __str__(self) -> str:  # pragma: no cover — display helper
        return (
            f"#{self.decision_id} {self.discrepancy_type} "
            f"({self.severity}): {self.action_tool}"
        )


@dataclass
class _Targets:
    """Matching surface extracted from one decision."""

    paths: list[str] = field(default_factory=list)
    is_guard: bool = False


# ── Decision loading (cached on file mtime/size) ───────────────────────────

_cache: dict[str, tuple[float, int, list[dict[str, Any]]]] = {}


def clear_decision_cache() -> None:
    """Drop the parsed-decision cache (tests / after an edit to the ledger)."""
    _cache.clear()


def _active_decisions(workspace: str | Path) -> list[dict[str, Any]]:
    """Active decisions, re-parsed only when the ledger file changed.

    ``_execute_tool_call`` calls this on EVERY tool call; the workspace ledger
    holds 100+ decisions, so a stat-guarded cache keeps the hook off the hot
    path.  A missing/corrupt ledger yields ``[]`` (never an exception: the
    check must not break tool execution).

    The JSON is read directly instead of through
    :func:`agent_core.decisions.load_decisions` on purpose: that loader
    normalizes ``affected_files`` against the filesystem and DROPS entries
    that do not currently exist, which would make the guard depend on the
    checkout (a fresh clone has no ``agent_core/security/`` retry target and
    no stale-file noise filter).  Here the recorded paths are taken as
    written — they describe intent, not disk state.
    """
    fp = _decision_path(workspace)
    try:
        stat = fp.stat()
    except OSError:
        return []
    key = str(fp)
    cached = _cache.get(key)
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]
    try:
        parsed = json.loads(fp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    active: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or STATUS_ACTIVE)
        if status == STATUS_ACTIVE:
            active.append(item)
    _cache[key] = (stat.st_mtime, stat.st_size, active)
    return active


# ── Matching helpers ───────────────────────────────────────────────────────


def _rel(workspace: str | Path, raw: str) -> str | None:
    """Lexical workspace-relative form of *raw* (no existence requirement).

    ``normalize_affected_files`` drops paths that are absent from disk, which
    would make the guard depend on the checkout; here the comparison is
    against the *action*, so a missing file must still match its decision.
    """
    text = str(raw).strip().replace("\\", "/")
    if not text:
        return None
    rel = _canonical_rel(workspace, text)
    if rel:
        return rel
    return text.lstrip("./") or None


def _target_paths(workspace: str | Path, decision: dict[str, Any]) -> _Targets:
    """Paths a decision speaks for, plus whether it is a write guard.

    For guard decisions the stated paths are *forbidden write targets*; the
    decision prose is mined as well, because a fresh checkout has no
    ``agent_core/security/`` directory on disk and the recorded
    ``affected_files`` entries then normalize away to nothing.
    """
    is_guard = bool(GUARD_TAGS & {str(t) for t in decision.get("tags") or []})
    raw_paths = [str(p) for p in decision.get("affected_files") or [] if str(p).strip()]
    if is_guard:
        prose = " ".join(
            str(decision.get(k) or "") for k in ("title", "context", "decision")
        )
        raw_paths.extend(m.group(1) for m in _PATH_TOKEN_RE.finditer(prose))
    paths: list[str] = []
    for raw in raw_paths:
        rel = _rel(workspace, raw)
        if rel and rel not in paths:
            paths.append(rel)
    return _Targets(paths=paths, is_guard=is_guard)


def _action_paths(
    workspace: str | Path, tool_name: str, args: dict[str, Any]
) -> list[str]:
    """Workspace-relative paths the action touches.

    Path-valued arguments are used directly; string arguments (a ``run``
    command, a ``git`` argument list) are matched *against* the decision paths
    later, so a shell command naming a guarded file is caught too.
    """
    out: list[str] = []
    for key in _PATH_ARG_KEYS:
        value = args.get(key)
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, (list, tuple)):
            continue
        for item in values:
            if not isinstance(item, str):
                continue
            rel = _rel(workspace, item)
            if rel and rel not in out:
                out.append(rel)
    return out


def _arg_text(args: dict[str, Any]) -> str:
    """All string arguments joined — used for substring path matching."""
    return " ".join(
        str(v) for v in args.values() if isinstance(v, str)
    ).replace("\\", "/")


def _covering(target: str, paths: list[str]) -> str | None:
    """The decision path that *target* equals or sits under (dir-prefix match)."""
    for path in paths:
        if target == path or target.startswith(path.rstrip("/") + "/"):
            return path
    return None


def _is_read_only_tool(tool_name: str) -> bool:
    """True for tools verified read-only (the plan-mode allowlist).

    Reusing :data:`agent_core.modes.PLAN_MODE_TOOLS` keeps one definition of
    "cannot change the workspace" instead of a second, drifting list.
    """
    from agent_core.modes import PLAN_MODE_TOOLS

    return tool_name.lower() in PLAN_MODE_TOOLS


# ── Public entry point ─────────────────────────────────────────────────────


def check_decision_compliance(
    workspace: str | Path,
    tool_name: str,
    args: dict[str, Any],
    result: str,
    affected_files: list[str] | None = None,
) -> list[Discrepancy]:
    """Report where *action* diverges from the workspace's active decisions.

    *affected_files* (optional) are the files already known to have been
    mutated this turn; they are checked through the same guards.

    Returns ``[]`` when the action is compliant — the common case, which the
    caller must treat as "nothing to record".
    """
    result_text = str(result or "")
    plan_rejected = bool(_PLAN_REJECTION_RE.match(result_text))
    refused = plan_rejected or bool(_REFUSAL_RE.search(result_text))
    read_only = _is_read_only_tool(tool_name)

    out: list[Discrepancy] = []
    if plan_rejected:
        out.append(
            Discrepancy(
                decision_id=PLAN_MODE_DECISION_ID,
                decision_title="Plan mode is read-only",
                action_tool=tool_name,
                action_args=dict(args),
                discrepancy_type="blocked_action",
                severity="warning",
                suggestion=(
                    "The session is read-only; present the change as a plan and "
                    "switch back with 'mode build' before mutating."
                ),
            )
        )

    # A plan-mode rejection already explains this call; do not double-log the
    # same attempt against every guarded decision it happened to touch.
    if plan_rejected:
        return out

    actions = _action_paths(workspace, tool_name, args)
    for extra in affected_files or []:
        rel = _rel(workspace, str(extra))
        if rel and rel not in actions:
            actions.append(rel)
    arg_text = _arg_text(args)

    for decision in _active_decisions(workspace):
        targets = _target_paths(workspace, decision)
        if not targets.paths:
            continue
        hit = next(
            (p for p in actions if _covering(p, targets.paths) is not None), None
        )
        if hit is None:
            if read_only or not arg_text:
                continue
            hit = next((p for p in targets.paths if p in arg_text), None)
        if hit is None:
            continue

        decision_id = str(decision.get("id") or "?")
        title = str(decision.get("title") or "")

        if targets.is_guard:
            if read_only:
                # Reading a guarded tree is explicitly allowed.
                continue
            out.append(
                Discrepancy(
                    decision_id=decision_id,
                    decision_title=title,
                    action_tool=tool_name,
                    action_args=dict(args),
                    discrepancy_type="blocked_action" if refused else "contradiction",
                    severity="warning" if refused else "critical",
                    suggestion=(
                        "The guard rejected this call; do not retry the same path "
                        "— ask the user before changing a self-modification-guarded "
                        "tree."
                        if refused
                        else f"Decision #{decision_id} forbids writing '{hit}', yet "
                        "the call reported success — revert the change or supersede "
                        "the decision with 'decide'."
                    ),
                )
            )
        elif decision.get("contradictions") and not read_only:
            out.append(
                Discrepancy(
                    decision_id=decision_id,
                    decision_title=title,
                    action_tool=tool_name,
                    action_args=dict(args),
                    discrepancy_type="unresolved_context",
                    severity="warning",
                    suggestion=(
                        f"Decision #{decision_id} still has open contradictions; "
                        "confirm the intended behaviour before changing its files."
                    ),
                )
            )
    return out
