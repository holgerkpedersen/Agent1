"""Decision collection — aggregate what the agent decided on its own.

Phase 3 of the autonomous-decision plan (card ``9bc4327f``, plan
``.decisions_plan.md``).  Phases 1 and 2 already produce the raw material on
every tool call:

* Phase 1 (:mod:`agent_core.decisions`) names the choices a call took without
  being asked and appends them to ``.autonomous_decisions.jsonl``, promoting
  the new ones into ``.decisions.json``.
* Phase 2 (:mod:`agent_core.discrepancy`) compares a call against the active
  decisions and appends each divergence to ``.discrepancies.jsonl``.

This module reads those two ledgers and turns them into one review-ready
picture, which is what the Phase 4 improvement pipeline consumes.

Two entry points, two different jobs:

:meth:`DecisionCollector.collect_turn_decisions`
    Per-turn extraction.  Re-runs the Phase 1 detector over the turn's tool
    calls so a turn can be summarised *without* depending on the log having
    been flushed.  It is a pure reader: promotion stays the tool hook's job,
    so calling this can never duplicate a ledger entry.

:meth:`DecisionCollector.aggregate_periodic`
    Review.  Folds the observation log, the discrepancy ledger and any
    per-turn findings into counts grouped by tag, discrepancy type and
    severity, bounded by an optional ``since`` window.

Bookkeeping only: nothing here mutates a decision, and an I/O failure is
logged and swallowed so a review can never break a tool call.
"""

from __future__ import annotations

import logging
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_core.decisions import (
    AUTONOMOUS_LOG_FILE,
    DISCREPANCIES_FILE,
    _decision_path,
    detect_autonomous_decisions,
    load_autonomous_log,
    load_discrepancies,
)

logger = logging.getLogger(__name__)


def _parse_stamp(value: Any) -> datetime | None:
    """Parse a stored ISO timestamp, or ``None`` when it is absent/invalid."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    # A naive stamp would raise on comparison with an aware bound.
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _normalise_since(since: datetime | str | None) -> datetime | None:
    """Coerce *since* (datetime, ISO string or ``None``) to an aware datetime."""
    if since is None:
        return None
    if isinstance(since, datetime):
        return (
            since.replace(tzinfo=timezone.utc)
            if since.tzinfo is None
            else since
        )
    return _parse_stamp(since)


def _split_calls(
    tools_called: list[Any],
) -> list[tuple[str, dict[str, Any], str]]:
    """Normalise ``tools_called`` entries to ``(tool_name, args, result)``.

    A turn is recorded as ``(name, args)`` when the caller kept no result, and
    as ``(name, args, result)`` when it did.  The result matters: a call the
    guard refused decided nothing, so the detector must see it.
    """
    out: list[tuple[str, dict[str, Any], str]] = []
    for entry in tools_called or []:
        if not isinstance(entry, (tuple, list)) or not entry:
            continue
        name = str(entry[0] or "")
        args = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
        result = entry[2] if len(entry) > 2 and entry[2] is not None else ""
        out.append((name, dict(args), str(result)))
    return out


def _turn_user_input(turn_messages: list[dict[str, Any]]) -> str:
    """The most recent user message of the turn, used as the "was it asked?" ref.

    The Phase 1 detector treats a path the user named as *their* instruction,
    so this must be the user's own text — never the assistant's.
    """
    for message in reversed(list(turn_messages or [])):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return " ".join(part for part in parts if part)
    return ""


def _as_record(discrepancy: Any) -> dict[str, Any]:
    """A discrepancy as a plain dict (accepts ``Discrepancy`` or a dict)."""
    if isinstance(discrepancy, dict):
        return dict(discrepancy)
    return dict(vars(discrepancy))


def _discrepancy_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Identity of one discrepancy, for de-duplication.

    Used to tell an in-turn record that Phase 2 already appended to the ledger
    apart from one the caller only handed over in memory — counting both would
    double the review numbers.  The action's argument payload is included as a
    stable string so two genuinely different actions never collapse into one.
    """
    args = record.get("action_args")
    try:
        args_key = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover — exotic payload
        args_key = str(args)
    return (
        str(record.get("decision_id") or ""),
        str(record.get("discrepancy_type") or ""),
        str(record.get("action_tool") or ""),
        args_key,
        str(record.get("suggestion") or ""),
    )


class DecisionCollector:
    """Collects and organises decisions from multiple sources.

    Holds no state beyond the workspace: every method reads the ledgers fresh,
    so two collectors over the same workspace never disagree.
    """

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self.decisions_file = _decision_path(self.workspace)
        self.discrepancies_file = self.workspace / DISCREPANCIES_FILE
        self.log_file = self.workspace / AUTONOMOUS_LOG_FILE
        #: Discrepancies handed to :meth:`collect_turn_decisions` during this
        #: collector's lifetime, keyed by :func:`_discrepancy_key` so a repeat
        #: of the same finding is stored once.
        self._turn_discrepancies: dict[tuple[str, ...], dict[str, Any]] = {}

    # ── per turn ─────────────────────────────────────────────────────────

    def collect_turn_decisions(
        self,
        turn_messages: list[dict[str, Any]],
        tools_called: list[Any],
        discrepancies: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract autonomous decisions from one conversation turn.

        Each call is passed through the Phase 1 detector with the turn's user
        text as the "was this asked for?" reference, so a path the user named
        is never reported as the agent's own choice.

        *discrepancies* keeps this the single place a turn is catalogued: the
        entries are not filtered here (a discrepancy is already evidence of a
        divergence), but they are validated so the caller learns about a
        malformed record.

        Returns candidate decision dicts, newest-last, without duplicates.
        Promotion is deliberately not attempted — see the module docstring.
        """
        user_input = _turn_user_input(turn_messages)
        found: list[dict[str, Any]] = []
        seen: set[str] = set()
        for name, args, result in _split_calls(tools_called):
            try:
                candidates = detect_autonomous_decisions(
                    self.workspace, name, args, result, user_input=user_input
                )
            except Exception:  # noqa: BLE001 — collection must never raise
                logger.debug(
                    "decision detection failed for tool %r", name, exc_info=True
                )
                continue
            for candidate in candidates:
                title = str(candidate.get("title") or "")
                if not title or title in seen:
                    continue
                seen.add(title)
                found.append(candidate)

        for entry in discrepancies or []:
            record = _as_record(entry)
            if not record.get("discrepancy_type"):
                logger.debug("ignoring discrepancy without a type: %r", record)
                continue
            self._turn_discrepancies.setdefault(_discrepancy_key(record), record)
        return found

    # ── periodic review ──────────────────────────────────────────────────

    def aggregate_periodic(
        self,
        since: datetime | str | None = None,
        group_by_tag: bool = True,
    ) -> dict[str, Any]:
        """Aggregate collected decisions and discrepancies for review.

        *since* bounds the review window (an observation without a parsable
        timestamp is kept: it is not evidence of being old).  *group_by_tag*
        fills ``by_tag``; pass ``False`` to skip the grouping and keep the
        report cheap.

        Returns a summary suitable for feeding into the improvement pipeline:
        ``total``, ``promoted``/``unpromoted``, ``by_tag`` and a nested
        ``discrepancies`` block with its own ``total``, ``by_type`` and
        ``by_severity``.  An empty workspace yields zeros, never an error.
        """
        bound = _normalise_since(since)

        observations: list[dict[str, Any]] = []
        for entry in load_autonomous_log(self.workspace):
            stamp = _parse_stamp(entry.get("detected_at"))
            if bound is not None and stamp is not None and stamp < bound:
                continue
            observations.append(entry)

        by_tag: Counter[str] = Counter()
        promoted = 0
        unpromoted = 0
        if group_by_tag:
            for entry in observations:
                tags = entry.get("tags")
                if isinstance(tags, str):
                    tags = [tags]
                for tag in {str(t) for t in (tags or []) if str(t).strip()}:
                    by_tag[tag] += 1
        for entry in observations:
            if entry.get("decision_id") is None:
                unpromoted += 1
            else:
                promoted += 1

        disc_by_type: Counter[str] = Counter()
        disc_by_severity: Counter[str] = Counter()
        disc_total = 0

        ledger = load_discrepancies(self.workspace)
        # Identity of what the ledger already holds, so an in-turn record that
        # Phase 2 appended is counted once — the ledger copy wins.
        persisted = {_discrepancy_key(entry) for entry in ledger}
        pending = [
            record
            for key, record in self._turn_discrepancies.items()
            if key not in persisted
        ]

        for entry in [*ledger, *pending]:
            stamp = _parse_stamp(entry.get("detected_at"))
            if bound is not None and stamp is not None and stamp < bound:
                continue
            disc_total += 1
            disc_type = str(entry.get("discrepancy_type") or "unknown")
            disc_by_type[disc_type] += 1
            disc_by_severity[str(entry.get("severity") or "unknown")] += 1

        return {
            "workspace": str(self.workspace),
            "total": len(observations),
            "promoted": promoted,
            "unpromoted": unpromoted,
            "by_tag": dict(by_tag) if group_by_tag else {},
            "discrepancies": {
                "total": disc_total,
                "by_type": dict(disc_by_type),
                "by_severity": dict(disc_by_severity),
            },
        }
