"""Phase 0 - reader: parse trace JSONL back into validated events.

Schema-validates and size-limits parsed trace files (decision #022), so a
corrupt or hostile trace can never crash the HTIR compiler.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tracing import KINDS, LAYERS

_MAX_EVENTS = 10_000
_MAX_BYTES = 50 * 1024 * 1024


class TraceValidationError(ValueError):
    """Raised when a file is not a well-formed, valid trace."""


def read_trace(path: Path | str) -> list[dict[str, Any]]:
    """Parse a trace JSONL file into a list of validated event records."""
    p = Path(path)
    if not p.is_file():
        raise TraceValidationError(f"trace file not found: {p}")
    if p.stat().st_size > _MAX_BYTES:
        raise TraceValidationError(f"trace file too large: {p}")

    events: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            if len(events) >= _MAX_EVENTS:
                raise TraceValidationError(f"trace exceeds {_MAX_EVENTS} events: {p}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceValidationError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise TraceValidationError(f"{p}:{lineno}: event is not an object")
            if record.get("kind") not in KINDS:
                raise TraceValidationError(
                    f"{p}:{lineno}: unknown event kind: {record.get('kind')!r}"
                )
            if record.get("layer") not in LAYERS:
                raise TraceValidationError(
                    f"{p}:{lineno}: missing/invalid layer facet: {record.get('layer')!r}"
                )
            events.append(record)
    return events


def task_id_of(path: Path | str) -> str:
    """Return the task_id recorded in a trace file (from its first event)."""
    events = read_trace(path)
    if not events:
        raise TraceValidationError(f"empty trace: {path}")
    return str(events[0].get("task_id", ""))
