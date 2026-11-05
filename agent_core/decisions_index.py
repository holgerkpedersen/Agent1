"""Compact decision index for RAG-style constraint injection.

Builds a cacheable index from ``.decisions.json`` and retrieves only the
decisions relevant to the current turn's files/tags/query — instead of
broadcasting all ~106 active decisions into every prompt.

Design decisions
----------------
* **Summaries, not full text.**  Each decision contributes one line (~15 tokens)
  to the system prompt; full text stays in ``.decisions.json`` for human review.
* **File + tag inverted indexes.**  Exact file match (weight 2) and tag overlap
  (weight 1) give deterministic, instant retrieval with no external dependencies.
* **Cache auto-rebuild.**  Index is rebuilt when ``.decisions.json`` mtime changes;
  stale or corrupted cache falls through transparently.

Usage
-----
>>> from agent_core.decisions_index import load_or_build_index, retrieve_decisions
>>> index = load_or_build_index(".")
>>> results = retrieve_decisions(index, files=["agent.py"], top_k=10)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_core.decisions import (
    STATUS_ACTIVE,
    load_decisions,
)

#: Cache file location (gitignored alongside other reports/history caches).
CACHE_PATH = Path("reports/history/decisions_index.json")

#: Maximum number of decisions returned per retrieval call.
DEFAULT_TOP_K = 10

# ── Index schema ──────────────────────────────────────────────────────────────

#: Version bump when the index structure changes (triggers rebuild).
INDEX_VERSION = 1


def _build_summary(decision: dict[str, Any]) -> str:
    """One-line actionable constraint from a decision record.

    Priority: ``decision`` field > title + rationale fallback.
    Capped at 200 chars to keep the summary compact (~15 tokens).
    """
    if decision.get("status") != STATUS_ACTIVE:
        return ""
    text = decision.get("decision", "") or decision.get("title", "")
    # Truncate mid-sentence is fine — the model only needs the constraint.
    return (text[:200] + "\u2026") if len(text) > 200 else text


def _relativize_path(workspace: Path, path: str) -> str | None:
    """Workspace-relative canonical form (forward slashes), or None when the
    path does not resolve under the workspace.

    Unlike :func:`agent_core.decisions.normalize_affected_files`, this does NOT
    require the file to exist on disk — the index stores all declared paths so
    retrieval works even for deleted/moved files.
    """
    # Normalize both paths to use forward slashes and resolve consistently.
    ws_str = str(workspace).replace("\\", "/")
    p_str = str(path).replace("\\", "/")
    p = Path(p_str)
    if not p.is_absolute():
        p = Path(ws_str) / p
    try:
        # Resolve both to absolute, then compute relative path.
        resolved = p.resolve()
        ws_resolved = Path(ws_str).resolve()
        rel = resolved.relative_to(ws_resolved)
        return rel.as_posix()
    except (ValueError, OSError):
        return None


def build_index(workspace: str | Path) -> dict[str, Any]:
    """Build a compact index from ``.decisions.json``.

    Returns the index dict (also writes to :data:`CACHE_PATH`).
    Only **active** decisions are indexed; superseded/archived/void records
    are excluded because they are resolved and no longer authoritative.
    """
    workspace = Path(workspace).resolve()
    ledger_path = workspace / ".decisions.json"

    if not ledger_path.is_file():
        return {"version": INDEX_VERSION, "summaries": {}, "file_index": {},
                "tag_index": {}}

    decisions = load_decisions(workspace)
    summaries: dict[str, str] = {}
    file_index: dict[str, set[str]] = {}
    tag_index: dict[str, set[str]] = {}

    for d in decisions:
        if d.get("status") != STATUS_ACTIVE:
            continue
        did = str(d["id"])
        summary = _build_summary(d)
        if not summary:
            continue
        summaries[did] = summary

        # File index: map each affected file -> decision IDs
        for fpath in d.get("affected_files", []):
            canon = _relativize_path(workspace, fpath)
            if canon:
                file_index.setdefault(canon, set()).add(did)

        # Tag index: map each tag -> decision IDs
        for tag in d.get("tags", []):
            tag_index.setdefault(tag, set()).add(did)

    index = {
        "version": INDEX_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ledger_mtime": os.path.getmtime(ledger_path),
        "summaries": summaries,
        "file_index": {k: sorted(v) for k, v in file_index.items()},
        "tag_index": {k: sorted(v) for k, v in tag_index.items()},
    }

    # Write cache (best-effort; never breaks callers if it fails).
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass  # cache write failure is non-fatal

    return index


def load_or_build_index(workspace: str | Path) -> dict[str, Any]:
    """Load cached index; rebuild if ledger changed or cache missing.

    Rebuild triggers when:
    - Cache file does not exist
    - ``ledger_mtime`` in cache differs from current ``.decisions.json`` mtime
    - Index version mismatch (schema change)
    - Cache JSON is corrupted / unparseable
    """
    workspace = Path(workspace).resolve()
    ledger_path = workspace / ".decisions.json"

    # Try to load and validate cache
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if (raw.get("version") == INDEX_VERSION
                and raw.get("ledger_mtime") == os.path.getmtime(ledger_path)):
            return raw
    except (FileNotFoundError, KeyError, ValueError, OSError):
        pass

    # Cache miss or stale -> rebuild from ledger
    return build_index(workspace)


def retrieve_decisions(
    index: dict[str, Any],
    files: list[str] | None = None,
    tags: list[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, str]]:
    """Retrieve relevant decisions ranked by relevance score.

    Parameters
    ----------
    index : dict
        Index returned by :func:`build_index` or :func:`load_or_build_index`.
    files : list[str] | None
        Files being worked on this turn (workspace-relative paths).
    tags : list[str] | None
        Tags to match against decision tags.
    top_k : int
        Maximum number of decisions to return.

    Returns
    -------
    list[dict[str, str]]
        Each entry is ``{"id": ..., "title": ..., "summary": ...}`` -> compact
        enough for system-prompt injection (~15 tokens per decision).
    """
    matching: dict[str, int] = {}  # id -> score

    # -- File-based retrieval (weight 2) ------------------------------------
    if files:
        for fpath in files:
            basename = Path(fpath).name
            rel_path = fpath.lstrip("./")
            for path_key, dec_ids in index.get("file_index", {}).items():
                # Exact relative-path match (highest precision)
                if rel_path == path_key:
                    for did in dec_ids:
                        matching[did] = matching.get(did, 0) + 2
                # Basename fallback -- catches decisions that reference just the
                # file name without directory prefix.
                elif basename == Path(path_key).name:
                    for did in dec_ids:
                        matching[did] = matching.get(did, 0) + 1

    # -- Tag-based retrieval (weight 1) -------------------------------------
    if tags:
        for tag in tags:
            if tag in index.get("tag_index", {}):
                for did in index["tag_index"][tag]:
                    matching[did] = matching.get(did, 0) + 1

    # -- Rank and return top-K ----------------------------------------------
    ranked = sorted(matching.items(), key=lambda x: -x[1])[:top_k]
    summaries = index.get("summaries", {})
    titles: dict[str, str] = {}  # lazy load from ledger for title field

    results = []
    for did, _score in ranked:
        summary = summaries.get(did, "")
        if not summary:
            continue
        # Load title lazily (only needed for display)
        if did not in titles:
            titles[did] = _load_title_by_id(did)
        results.append({
            "id": did,
            "title": titles.get(did, ""),
            "summary": summary,
        })

    return results


def _load_title_by_id(did: str) -> str:
    """Load a single decision's title from the ledger (lazy)."""
    try:
        decisions = load_decisions(".")
        for d in decisions:
            if str(d.get("id")) == did:
                return d.get("title", "")
    except Exception:
        pass
    return ""
