"""Per-file execution history for chat — plan item #2's zero-dep RAG half.

The trace corpus (``reports/traces/*.jsonl``) and the structured execution
ledger (``reports/history/executions.jsonl``) already know everything this
workspace has ever *done* to a file — what was read, edited, run, and which
of those attempts errored.  ``harnessfix/history.py`` indexes all of it for
implement/fix; the conversational loop never saw any of it.

This module closes that gap with no new dependencies: it extracts the files
a user message mentions, asks ``harnessfix.history.file_history`` (the real
matcher — a directory arg counts as history for its direct children) what
happened to them, and renders one compact **RECENT FILE NOTES** block.

The contract mirrors :mod:`agent_core.habits`, the sibling feature that made
user-model history visible:

* **Marker contract** — the block starts with ``FILE_HISTORY_MARKER``;
  ``agent._strip_dynamic_system_blocks`` strips it every turn and
  ``Agent._file_history_block`` rebuilds it, so a long session never
  accumulates stale notes.
* **Empty means empty** — no query, no path tokens, or no history renders
  the empty string: the prompt stays byte-identical to the pre-feature
  baseline on an untouched workspace.
* **Prompt budget, not a database dump** — at most ``MAX_FILES`` files,
  ``PER_FILE`` events each, ``LINE_CAP`` rendered lines in total; errors
  (weight 0) sort before mutations, which sort before plain runs and reads
  (see ``history.HistoryEvent.weight``).
* **Fail-open** — every public function returns an empty result on any
  internal failure; a broken ledger must never kill a chat turn.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

#: Exact prefix of the injected block — also the marker
#: ``agent._strip_dynamic_system_blocks`` uses to rebuild it every turn.
FILE_HISTORY_MARKER = "\n\nRECENT FILE NOTES"

#: How many files mentioned in one message get a lookup (mention order).
MAX_FILES = 3

#: Events rendered per file (the history layer sorts errors first).
PER_FILE = 2

#: Hard cap on rendered body lines — the block is a hint, not an export.
LINE_CAP = 10

#: A path-like token: dirs with slashes, or a file name with a known
#: extension.  Same shape as ``agent_core.habits._PATH_RE`` so both features
#: agree on what "the user mentioned a file" means.
_PATH_RE = re.compile(
    r"(?:[\w.@-]+/)+[\w.@-]+|[\w.@-]+\.(?:py|md|json|toml|ya?ml|txt|cfg|ini)\b",
    re.I,
)


def extract_file_paths(text: Any, max_files: int = MAX_FILES) -> list[str]:
    """Path tokens mentioned in *text*, in mention order.

    Backslashes are normalized to forward slashes (Windows paths are the
    common case), duplicates collapse case-insensitively, and the result is
    capped at *max_files*.  Never raises: a non-string or any internal
    failure yields ``[]``.
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return []
        normalized = text.replace("\\", "/")
        out: list[str] = []
        seen: set[str] = set()
        for match in _PATH_RE.finditer(normalized):
            token = match.group(0).strip("./")
            if not token:
                continue
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(token)
            if len(out) >= max(1, int(max_files)):
                break
        return out
    except Exception:  # noqa: BLE001 - contract: extraction never raises
        logger.exception("File-path extraction unavailable:\n")
        return []


def file_notes_block(
    query: Any,
    workspace: Any,
    *,
    max_files: int = MAX_FILES,
    per_file: int = PER_FILE,
    line_cap: int = LINE_CAP,
) -> str:
    """The RECENT FILE NOTES block for one user message (``""`` when nothing).

    *query* is the current user message; *workspace* is where ``reports/``
    lives (resolved by the caller, normally ``Agent._effective_ws_dir``).
    Renders at most *max_files* file sections, each with at most *per_file*
    events, capped at *line_cap* body lines.  Empty string when the message
    mentions no files, the workspace has no history for them, or anything
    goes wrong — never an exception, never a stray header.
    """
    try:
        paths = extract_file_paths(query, max_files=max_files)
        if not paths:
            return ""

        from harnessfix.history import file_history  # local import: stdlib-only at import time

        body: list[str] = []
        for target in paths[:max(1, int(max_files))]:
            events = file_history(target, str(workspace), limit=max(1, int(per_file)))
            if not events:
                continue
            body.append(f"- {target}: {len(events)} past event(s)")
            for ev in events:
                when = datetime.fromtimestamp(ev.ts).strftime("%m-%d %H:%M") if ev.ts else "?"
                body.append(f"  {ev.tool} {ev.kind} [{str(ev.ref)[:8]}] {when}: {ev.summary}")
            if len(body) >= line_cap:
                break

        if not body:
            return ""
        block = FILE_HISTORY_MARKER + "\n" + "\n".join(body[:line_cap]) + "\n"
        assert block.startswith(FILE_HISTORY_MARKER)  # marker contract (see module doc)
        return block
    except Exception:  # noqa: BLE001 - contract: rendering never raises
        logger.exception("File-history block unavailable:\n")
        return ""


__all__ = [
    "FILE_HISTORY_MARKER",
    "LINE_CAP",
    "MAX_FILES",
    "PER_FILE",
    "extract_file_paths",
    "file_notes_block",
]
