"""History-assisted implementation — past executions for implement/fix.

Queries the trace corpus (``reports/traces/*.jsonl``) and the structured
execution ledger (``reports/history/executions.jsonl``) and renders compact
PAST EXECUTION NOTES blocks for implement/fix prompts.

Context (2026-08-19): implement/fix injected past *decisions* into prompts
but nothing fed past tool results/errors back in, even though ~69 traces
with ~6000 events existed.  The trace index is built lazily and cached per
process; the ledger is small and re-read per query.  All query functions are
read-only — they cannot affect implement's write/cascade invariants.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .reader import TraceValidationError, read_trace

HISTORY_SUBDIR = "history"
EXECUTIONS_FILE = "executions.jsonl"
SUMMARY_CAP = 140

_MUTATING_TOOLS = frozenset({"write", "edit", "apply_patch", "fix", "create_file"})
_READ_TOOLS = frozenset({"read", "search", "list_files", "diff", "analyze"})
_PATH_KEYS = ("path", "file", "filename", "target", "fpath")


@dataclass(frozen=True)
class HistoryEvent:
    """One past execution relevant to a file.

    ``weight`` orders importance for the prompt: 0 = errors, 1 = mutations,
    2 = command runs, 3 = read-only probes.
    """

    source: str
    ref: str
    ts: float
    tool: str
    kind: str
    summary: str
    weight: int
    files: tuple[str, ...]


def _norm(path: str) -> str:
    return str(path).replace("\\", "/").lower()


def _matches_arg(arg: str, target: str) -> bool:
    """True when a trace/ledger path *arg* touches *target*.

    A directory arg only matches a file directly inside it — a search across
    the whole workspace root must not count as history for every file.
    """
    a = _norm(arg).rstrip("/")
    t = _norm(target).rstrip("/")
    if not a or not t:
        return False
    if a == t or a.endswith("/" + t) or t.endswith("/" + a):
        return True
    if t.startswith(a + "/"):
        t_parent = t.rsplit("/", 1)[0] if "/" in t else ""
        return bool(t_parent) and t_parent == a
    return False


def _paths_from_args(args: Any) -> list[str]:
    if not isinstance(args, dict):
        return []
    out: list[str] = []
    for key in _PATH_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value)
        elif isinstance(value, list):
            out.extend(v for v in value if isinstance(v, str) and v.strip())
    return out


def _summarize(tool: str, text: str, is_error: bool, args: dict[str, Any]) -> str:
    if tool == "run":
        cmd = str(args.get("command", ""))
        return f"run: {cmd[:SUMMARY_CAP]}"
    if is_error:
        return text[:SUMMARY_CAP]
    if tool in _MUTATING_TOOLS:
        return f"{tool} ok"
    if tool in _READ_TOOLS:
        path = next((v for k, v in args.items() if k in _PATH_KEYS and isinstance(v, str)), "")
        return f"{tool} {Path(path).name if path else ''}".strip()
    return f"{tool} ok"


def _parse_trace_event(ev: dict[str, Any]) -> HistoryEvent | None:
    kind = ev.get("kind")
    if kind not in ("tool_result", "tool_error"):
        return None
    tool = str(ev.get("tool", "?"))
    try:
        args = json.loads(ev.get("args_hash", "{}"))
    except (ValueError, TypeError):
        args = {}
    if not isinstance(args, dict):
        args = {}
    affected = [str(f) for f in (ev.get("affected_files") or [])]
    is_error = kind == "tool_error"
    if is_error:
        text = f"{ev.get('exception', '')}: {str(ev.get('message', ''))[:SUMMARY_CAP]}"
    else:
        text = str(ev.get("result", ""))
    weight = 0 if is_error else (1 if tool in _MUTATING_TOOLS else (2 if tool == "run" else 3))
    files = tuple(dict.fromkeys(_norm(p) for p in _paths_from_args(args) + affected))
    return HistoryEvent(
        source="trace",
        ref=str(ev.get("task_id", "")),
        ts=float(ev.get("ts") or 0),
        tool=tool,
        kind="error" if is_error else "result",
        summary=_summarize(tool, text, is_error, args),
        weight=weight,
        files=files,
    )


_TRACE_CACHE: tuple[tuple[tuple[str, int, int], ...], list[HistoryEvent]] | None = None


def _trace_events(workspace: str) -> list[HistoryEvent]:
    global _TRACE_CACHE
    trace_dir = Path(workspace) / "reports" / "traces"
    if trace_dir.is_dir():
        sig = tuple(
            sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in trace_dir.glob("*.jsonl"))
        )
    else:
        sig = ()
    if _TRACE_CACHE is not None and _TRACE_CACHE[0] == sig:
        return _TRACE_CACHE[1]
    events: list[HistoryEvent] = []
    if trace_dir.is_dir():
        for path in sorted(trace_dir.glob("*.jsonl")):
            try:
                for ev in read_trace(path):
                    parsed = _parse_trace_event(ev)
                    if parsed is not None:
                        events.append(parsed)
            except TraceValidationError:
                continue
    _TRACE_CACHE = (sig, events)
    return events


def _run_events(workspace: str) -> list[HistoryEvent]:
    ledger = Path(workspace) / "reports" / HISTORY_SUBDIR / EXECUTIONS_FILE
    if not ledger.is_file():
        return []
    events: list[HistoryEvent] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        files: list[str] = []
        for f in rec.get("files") or []:
            if isinstance(f, str):
                files.append(f)
            elif isinstance(f, dict) and isinstance(f.get("path"), str):
                files.append(f["path"])
        ts = float(rec.get("ts") or 0)
        command = str(rec.get("command", "run"))
        outcome = str(rec.get("outcome", ""))
        note = str(rec.get("note", ""))
        summary = f"{command}: {outcome}" + (f" — {note}" if note else "")
        events.append(
            HistoryEvent(
                source="run",
                ref=str(rec.get("id", str(ts))),
                ts=ts,
                tool=command,
                kind="execute",
                summary=summary[:SUMMARY_CAP],
                weight=2,
                files=tuple(dict.fromkeys(_norm(f) for f in files)),
            )
        )
    return events


def clear_history_cache() -> None:
    """Drop the process-level trace index (tests use this)."""
    global _TRACE_CACHE
    _TRACE_CACHE = None


def history_root(start: str) -> str | None:
    """Nearest ancestor of *start* that contains a reports/ data dir.

    implement passes the workspace root directly; fix resolves it by walking
    up from the failing file so the same corpus/ledger is found either way.
    """
    current = os.path.abspath(str(start))
    while True:
        if os.path.isdir(os.path.join(current, "reports", "traces")) or os.path.isdir(
            os.path.join(current, "reports", HISTORY_SUBDIR)
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def file_history(target: str, workspace: str, limit: int = 4) -> list[HistoryEvent]:
    """Past executions touching *target*, most important first."""
    targets = {_norm(target)}
    try:
        targets.add(_norm(os.path.join(workspace, target)))
    except (TypeError, ValueError):
        print("Silenced exception in history.py:228")
    t_key = min(targets)
    events: list[HistoryEvent] = []
    seen: set[tuple[str, str, str]] = set()
    for ev in _trace_events(workspace) + _run_events(workspace):
        if not any(_matches_arg(f, t) for t in targets for f in ev.files):
            continue
        key = (ev.ref, ev.tool, t_key)
        if key in seen:
            continue
        seen.add(key)
        events.append(ev)
    events.sort(key=lambda e: (e.weight, -e.ts))
    return events[:limit]


def _append_wiki_notes(query: str, workspace: str) -> str:
    """Best-effort append of COMPILED KNOWLEDGE (wiki) block to a history note.

    Wiki is additive context — failure here never affects implement/fix's
    write/cascade invariants.  Returns "" when the wiki module or pages are
    unavailable.
    """
    try:
        from .wiki import format_wiki_notes, WIKI_PATH
    except Exception:
        return ""
    if not query:
        return ""
    # Resolve the wiki file relative to the workspace root (not repo root),
    # so tests pointing at a temp dir find their own wiki.  Fall back to the
    # module-level WIKI_PATH when no per-workspace wiki exists.
    wpath = Path(workspace) / "reports" / "wiki" / "wiki.jsonl"
    if not wpath.is_file():
        wpath = WIKI_PATH
    try:
        return format_wiki_notes(query, k=3, path=wpath) or ""
    except Exception:
        print("Silenced exception in history.py:_append_wiki_notes")
        return ""


def format_file_history(target: str, workspace: str, limit: int = 4) -> str:
    """PAST EXECUTION NOTES block for one file (empty string when none).

    Appends a COMPILED KNOWLEDGE (wiki) block when the wiki has relevant pages
    for this file — additive context that never affects write/cascade invariants.
    """
    events = file_history(target, workspace, limit)
    lines: list[str] = []
    if not events:
        return "" + _append_wiki_notes(f"{target}\n{workspace}", workspace)
    lines.append(f"\n## PAST EXECUTION NOTES — {len(events)} event(s) for {target}")
    for ev in events:
        when = datetime.fromtimestamp(ev.ts).strftime("%m-%d %H:%M") if ev.ts else "?"
        lines.append(f"  - [{ev.source} {ev.ref[:8]}] {when} {ev.tool} {ev.kind}: {ev.summary}")
    return "\n".join(lines) + _append_wiki_notes(target, workspace)


def format_batch_history(files: list[str], workspace: str, per_file: int = 2, line_cap: int = 10) -> str:
    """PAST EXECUTION NOTES block for a batch of files (empty string when none)."""
    lines: list[str] = []
    for target in files:
        events = file_history(target, workspace, per_file)
        if events:
            lines.append(f"- {target}: {len(events)} past event(s)")
            lines += [f"    {ev.tool} {ev.kind} [{ev.ref[:8]}]: {ev.summary}" for ev in events]
            if len(lines) >= line_cap:
                lines.append("  ...")
                break
    if not lines:
        return _append_wiki_notes("\n".join(files), workspace)
    return "\n## PAST EXECUTION NOTES\n" + "\n".join(lines) + _append_wiki_notes(
        "\n".join(files), workspace
    )


def append_execution(workspace: str, command: str, files: list[Any], outcome: str = "ok", note: str = "") -> None:
    """Append one structured execution record to the workspace ledger."""
    ledger = Path(workspace) / "reports" / HISTORY_SUBDIR / EXECUTIONS_FILE
    ledger.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "ts": time.time(),
        "id": f"{datetime.now():%Y%m%d_%H%M%S}",
        "command": command,
        "outcome": outcome,
        "files": files,
        "note": note,
    }
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")