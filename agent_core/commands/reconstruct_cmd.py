"""Reconstruct files from JSONL trace logs.

Scans trace files in ``reports/traces/`` for ``write`` and ``edit`` tool
operations, groups them by target file path, and replays them in timestamp
order to recreate the final state of each file.

It also offers a step-through **replay** of a single trace so a human can audit
exactly what the agent did and why — the interactive companion to the review
ledger (``review show <task>``).  Replay is read-only by default: it never
writes into the workspace unless ``--apply`` is given (and then only with a
y/N confirm, mirroring ``save_file_py``).

Usage::

    reconstruct [--start <file.jsonl>] [--end <file.jsonl>]
                [--workspace <path>] [--search <query>]
                [--dry-run] [--force]

    reconstruct replay <task> [--workspace <path>]
                     [--from <n>] [--event <kind>] [--save-prefix <dir>]
                     [--apply]
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from .base import Command, read_input, show_file_diff, save_file_py, stop_requested

if TYPE_CHECKING:
    from agent import Agent

_TRACES_DIR = os.path.join("reports", "traces")


class _FileOp:
    """A single write or edit operation extracted from a trace."""

    __slots__ = ("ts", "kind", "path", "content", "old_text", "new_text", "source")

    def __init__(
        self,
        ts: float,
        kind: str,
        path: str,
        content: str = "",
        old_text: str = "",
        new_text: str = "",
        source: str = "",
    ) -> None:
        self.ts = ts
        self.kind = kind  # "write" or "edit"
        self.path = path
        self.content = content
        self.old_text = old_text
        self.new_text = new_text
        self.source = source  # originating jsonl filename


def _parse_trace(filepath: str) -> list[_FileOp]:
    """Extract write/edit operations from a single JSONL file."""
    ops: list[_FileOp] = []
    basename = os.path.basename(filepath)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") != "tool_result":
                    continue
                tool = rec.get("tool", "")
                if tool not in ("write", "edit"):
                    continue
                ts = rec.get("ts", 0.0)
                affected = rec.get("affected_files", [])
                if not affected:
                    continue
                # Parse the args from the misnamed "args_hash" field.
                args_raw = rec.get("args_hash", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    continue
                path = args.get("path", "")
                if not path:
                    continue
                # Normalise backslashes for consistency.
                path = path.replace("\\", "/")
                if tool == "write":
                    ops.append(_FileOp(
                        ts=ts,
                        kind="write",
                        path=path,
                        content=args.get("content", ""),
                        source=basename,
                    ))
                elif tool == "edit":
                    ops.append(_FileOp(
                        ts=ts,
                        kind="edit",
                        path=path,
                        old_text=args.get("old_text", ""),
                        new_text=args.get("new_text", ""),
                        source=basename,
                    ))
    except OSError as exc:
        print(f"  Warning: could not read {filepath}: {exc}")
    return ops


def _apply_edit(content: str, old_text: str, new_text: str) -> tuple[str, bool]:
    """Apply a single edit (old_text -> new_text) to content.

    Returns (new_content, success).  On failure the content is unchanged.
    """
    if old_text not in content:
        return content, False
    return content.replace(old_text, new_text, 1), True


def _collect_files(traces_dir: str) -> list[str]:
    """Return sorted list of .jsonl files in traces_dir, by mtime."""
    if not os.path.isdir(traces_dir):
        return []
    files = []
    for name in os.listdir(traces_dir):
        if name.endswith(".jsonl"):
            files.append(os.path.join(traces_dir, name))
    files.sort(key=lambda p: os.path.getmtime(p))
    return files


def _resolve_range(
    files: list[str], start: str | None, end: str | None
) -> list[str]:
    """Slice the file list to the [start, end] inclusive range."""
    start_idx = 0
    end_idx = len(files)
    if start:
        basenames = [os.path.basename(f) for f in files]
        try:
            start_idx = basenames.index(start)
        except ValueError:
            # Try matching as a prefix.
            for i, bn in enumerate(basenames):
                if bn.startswith(start):
                    start_idx = i
                    break
            else:
                print(f"  Warning: start file '{start}' not found; using all files.")
    if end:
        basenames = [os.path.basename(f) for f in files]
        try:
            end_idx = basenames.index(end) + 1
        except ValueError:
            for i, bn in enumerate(basenames):
                if bn.startswith(end):
                    end_idx = i + 1
                    break
            else:
                print(f"  Warning: end file '{end}' not found; using all files.")
    return files[start_idx:end_idx]


# ── Step-through replay (time-travel audit) ───────────────────────────────
#
# Companion to the review ledger (harnessfix/review.py + review_cmd.py).  A
# ReplaySession loads one trace and lets a human walk its event stream
# event-by-event, seeing each action's effect on file state.  State is
# re-derived from scratch up to the cursor on every move, so prev/next are
# exact inverses — no incremental drift.

#: In real traces, write/edit are tool_result events whose *tool* field is
#: "write"/"edit" (kind is always tool_call/tool_result/tool_error).  Only a
#: tool_result counts as an applied mutation — a tool_call alone (no result)
#: or a tool_error means the write never landed on disk.
_FILE_MUT_TOOLS = frozenset({"write", "edit"})


def _is_applied_file_mut(ev: dict[str, Any]) -> bool:
    return ev.get("kind") == "tool_result" and ev.get("tool") in _FILE_MUT_TOOLS


def _is_file_mut_tool(ev: dict[str, Any]) -> bool:
    """True for any write/edit tool event (call/result/error) — used by
    describe() to render the op, and by next_file_event() to preview it."""
    return ev.get("tool") in _FILE_MUT_TOOLS


#: Cap for the inline preview of an llm_response's text.
_LLM_PREVIEW_CAP = 400


def _shorten(text: str, cap: int) -> str:
    text = (text or "").replace("\r\n", "\n").strip()
    if len(text) <= cap:
        return text
    return text[:cap] + " ..."


class ReplaySession:
    """Deterministic step-through replay of one trace's event stream.

    Pure and side-effect-free except for reading the trace file.  File state
    is always re-derived from events[0..cursor], so stepping back is exact.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = _read_events(path)
        self.cursor = 0
        self._meta = next(
            (e for e in self.events if e.get("kind") == "task_begin"), {}
        )

    # ── meta ──────────────────────────────────────────────────────────
    @property
    def task_id(self) -> str:
        return str(self._meta.get("task_id", "") or (self.events[0].get("task_id", "") if self.events else ""))

    @property
    def model(self) -> str:
        return str(self._meta.get("model", ""))

    @property
    def profile(self) -> str:
        return str(self._meta.get("profile", ""))

    @property
    def prompt(self) -> str:
        return str(self._meta.get("user_input", ""))

    @property
    def outcome(self) -> str:
        for e in reversed(self.events):
            if e.get("kind") == "loop_end":
                return str(e.get("outcome", ""))
        return ""

    @property
    def affected_files(self) -> list[str]:
        files: list[str] = []
        for e in self.events:
            for f in e.get("affected_files") or []:
                if f and f not in files:
                    files.append(f)
        return files

    # ── navigation ────────────────────────────────────────────────────
    def next(self) -> bool:
        if self.cursor < len(self.events) - 1:
            self.cursor += 1
            return True
        return False

    def prev(self) -> bool:
        if self.cursor > 0:
            self.cursor -= 1
            return True
        return False

    def goto(self, idx: int) -> None:
        if not self.events:
            self.cursor = 0
            return
        self.cursor = max(0, min(idx, len(self.events) - 1))

    def current_event(self) -> dict[str, Any] | None:
        if not self.events or self.cursor < 0 or self.cursor >= len(self.events):
            return None
        return self.events[self.cursor]

    # ── state derivation ──────────────────────────────────────────────
    def file_state_at_cursor(self, path: str) -> str | None:
        """File content reflecting only events up to and including cursor.

        Returns None if the file has not been written by that point.
        """
        norm = path.replace("\\", "/")
        content: str | None = None
        for ev in self.events[: self.cursor + 1]:
            if not _is_applied_file_mut(ev):
                continue
            args = _parse_args(ev)
            p = str(args.get("path", "")).replace("\\", "/")
            if ev.get("tool") == "write":
                if p == norm:
                    content = str(args.get("content", ""))
            else:  # edit
                if p == norm and content is not None:
                    new_content, ok = _apply_edit(
                        content,
                        str(args.get("old_text", "")),
                        str(args.get("new_text", "")),
                    )
                    if ok:
                        content = new_content
        return content

    def next_file_event(self, path: str | None = None) -> dict[str, Any] | None:
        """The next write/edit event at or after the cursor (optionally for a
        specific file).  Used to preview the *upcoming* mutation."""
        norm = path.replace("\\", "/") if path else None
        for ev in self.events[self.cursor + 1 :]:
            if not _is_file_mut_tool(ev):
                continue
            if norm is None:
                return ev
            args = _parse_args(ev)
            p = str(args.get("path", "")).replace("\\", "/")
            if p == norm:
                return ev
        return None

    def final_state(self) -> dict[str, str]:
        """Final reconstructed content of every touched file (all events)."""
        state: dict[str, str] = {}
        for ev in self.events:
            if not _is_applied_file_mut(ev):
                continue
            args = _parse_args(ev)
            p = str(args.get("path", "")).replace("\\", "/")
            if ev.get("tool") == "write":
                if p:
                    state[p] = str(args.get("content", ""))
            else:  # edit
                if p and p in state:
                    new_content, ok = _apply_edit(
                        state[p],
                        str(args.get("old_text", "")),
                        str(args.get("new_text", "")),
                    )
                    if ok:
                        state[p] = new_content
        return state

    # ── rendering ─────────────────────────────────────────────────────
    def describe(self, ev: dict[str, Any], *, show_content: bool = True) -> str:
        """One-line-plus description of an event for the audit view."""
        kind = ev.get("kind", "?")
        layer = ev.get("layer", "")
        ts = ev.get("ts", 0.0)
        idx = self.events.index(ev) if ev in self.events else -1
        header = f"[{idx}] {kind} ({layer}) @ {ts:.3f}"
        if _is_file_mut_tool(ev):
            args = _parse_args(ev)
            p = str(args.get("path", "")).replace("\\", "/")
            if ev.get("tool") == "write":
                content = str(args.get("content", ""))
                line = f"{header}\n    write {p} ({len(content)} bytes)"
                if show_content:
                    line += f"\n{_indent(_shorten(content, 200), '    | ')}"
                return line
            # edit
            old = str(args.get("old_text", ""))
            new = str(args.get("new_text", ""))
            line = f"{header}\n    edit {p}"
            if show_content:
                line += (
                    f"\n{_indent(_shorten(old, 200), '    - ')}"
                    f"\n{_indent(_shorten(new, 200), '    + ')}"
                )
            return line
        if kind == "llm_response":
            text = _shorten(str(ev.get("text", "")), _LLM_PREVIEW_CAP)
            return f"{header}\n{_indent(text, '    > ')}"
        if kind == "tool_call":
            tool = ev.get("tool", "")
            args = _parse_args(ev)
            return f"{header}\n    -> call {tool} {_shorten(json.dumps(args, ensure_ascii=False), 160)}"
        if kind == "tool_result":
            tool = ev.get("tool", "")
            result = _shorten(str(ev.get("result", "")), 200)
            return f"{header}\n    <- {tool} result: {result}"
        if kind == "tool_error":
            tool = ev.get("tool", "")
            err = _shorten(str(ev.get("error", ev.get("result", ""))), 200)
            return f"{header}\n    ! {tool} error: {err}"
        if kind == "guard_triggered":
            guard = ev.get("guard", "")
            note = _shorten(str(ev.get("note", "")), 200)
            return f"{header}\n    ! guard {guard}: {note}"
        if kind == "loop_end":
            outcome = ev.get("outcome", "")
            return f"{header}\n    # loop_end outcome={outcome}"
        if kind == "task_begin":
            prompt = _shorten(self.prompt, 200)
            return f"{header}\n    task_begin model={self.model} profile={self.profile}\n    prompt: {prompt}"
        if kind == "step_start":
            it = ev.get("iteration", "?")
            return f"{header}\n    step_start iteration={it}"
        return header


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + ln for ln in text.split("\n"))


def _parse_args(ev: dict[str, Any]) -> dict[str, Any]:
    raw = ev.get("args_hash", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if isinstance(raw, str) else {}
    except json.JSONDecodeError:
        return {}


def _read_events(path: str) -> list[dict[str, Any]]:
    """Read a trace JSONL file into a list of event dicts (lenient)."""
    events: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    events.append(rec)
    except OSError as exc:
        print(f"  Warning: could not read {path}: {exc}")
    return events


def _resolve_task_path(
    traces_dir: str, task: str
) -> str | None:
    """Resolve a task id / filename / prefix to a .jsonl path under traces_dir."""
    if os.path.isabs(task) and task.endswith(".jsonl") and os.path.isfile(task):
        return task
    base = os.path.basename(task)
    if base.endswith(".jsonl") and os.path.isfile(os.path.join(traces_dir, base)):
        return os.path.join(traces_dir, base)
    # prefix match on basename stem
    for name in sorted(os.listdir(traces_dir)) if os.path.isdir(traces_dir) else []:
        if name.endswith(".jsonl") and name.startswith(base.split(".")[0]):
            return os.path.join(traces_dir, name)
    return None


class ReconstructCommand(Command):
    """Reconstruct files from JSONL trace logs."""

    @property
    def name(self) -> str:
        return "reconstruct"

    @property
    def help_text(self) -> str:
        return (
            "reconstruct [--start <file>] [--end <file>] [--workspace <path>] "
            "[--search <query>] [--dry-run] [--force]  |  "
            "reconstruct replay <task> [--from <n>] [--event <kind>] "
            "[--save-prefix <dir>] [--apply]"
        )

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        # ── Dispatch subcommands ────────────────────────────────────────
        if args and args[0].lower() == "replay":
            return await self._cmd_replay(args[1:], agent)

        # ── Parse flags ─────────────────────────────────────────────────
        start_file: str | None = None
        end_file: str | None = None
        workspace: str = agent.workspace
        search: str | None = None
        dry_run = False
        force = False

        i = 0
        while i < len(args):
            if args[i] == "--start" and i + 1 < len(args):
                start_file = args[i + 1]
                i += 2
            elif args[i] == "--end" and i + 1 < len(args):
                end_file = args[i + 1]
                i += 2
            elif args[i] == "--workspace" and i + 1 < len(args):
                workspace = args[i + 1]
                i += 2
            elif args[i] == "--search" and i + 1 < len(args):
                search = args[i + 1]
                i += 2
            elif args[i] == "--dry-run":
                dry_run = True
                i += 1
            elif args[i] == "--force":
                force = True
                i += 1
            else:
                self.error(f"Unknown flag: {args[i]}")
                return True

        # ── Discover and filter trace files ─────────────────────────────
        traces_dir = os.path.join(workspace, _TRACES_DIR)
        if not os.path.isdir(traces_dir):
            self.error(f"Traces directory not found: {traces_dir}")
            return True

        all_files = _collect_files(traces_dir)
        if not all_files:
            print("  No .jsonl trace files found.")
            return True

        selected = _resolve_range(all_files, start_file, end_file)
        print(f"  Scanning {len(selected)} trace file(s)...")

        # ── Parse all selected traces ──────────────────────────────────
        all_ops: list[_FileOp] = []
        for fp in selected:
            all_ops.extend(_parse_trace(fp))
        if not all_ops:
            print("  No write/edit operations found in selected traces.")
            return True

        # ── Filter by --search ──────────────────────────────────────────
        if search:
            all_ops = [op for op in all_ops if search in op.path]
            if not all_ops:
                print(f"  No operations match '{search}'.")
                return True

        print(f"  Found {len(all_ops)} write/edit operation(s).")

        # ── Group by path, sort by timestamp ────────────────────────────
        by_path: dict[str, list[_FileOp]] = defaultdict(list)
        for op in all_ops:
            by_path[op.path].append(op)
        for path in by_path:
            by_path[path].sort(key=lambda o: o.ts)

        # ── Compute final state per file (last-write + subsequent edits)
        file_results: dict[str, str] = {}  # path -> final content
        skipped = 0
        applied = 0
        edit_warnings = 0

        for path, ops in by_path.items():
            # Find the last write operation.
            last_write_idx = -1
            for idx, op in enumerate(ops):
                if op.kind == "write":
                    last_write_idx = idx

            if last_write_idx == -1:
                # No write found — file must already exist; we can't create it
                # from edits alone.
                print(f"  SKIP: {path} — no write operation found (edits only, file must exist)")
                skipped += len(ops)
                continue

            # Base content from last write.
            content = ops[last_write_idx].content
            applied += 1

            # Apply subsequent edits in order.
            for op in ops[last_write_idx + 1 :]:
                if op.kind != "edit":
                    continue
                new_content, ok = _apply_edit(content, op.old_text, op.new_text)
                if ok:
                    content = new_content
                    applied += 1
                else:
                    print(
                        f"  WARNING: {os.path.basename(path)} edit from "
                        f"{op.source} — old_text not found, skipped"
                    )
                    edit_warnings += 1
                    skipped += 1

            file_results[path] = content

        # ── Write results ───────────────────────────────────────────────
        written = 0
        for path, content in file_results.items():
            abs_path = path
            # If path is relative, resolve against workspace.
            if not os.path.isabs(path):
                abs_path = os.path.join(workspace, path)

            if dry_run:
                print(f"  DRY-RUN: would write {abs_path} ({len(content)} bytes)")
                continue

            if os.path.exists(abs_path) and not force:
                # Read existing and compare.
                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        existing = f.read()
                    if existing == content:
                        print(f"  UNCHANGED: {abs_path}")
                        continue
                except (OSError, UnicodeDecodeError):
                    print("Silenced exception in reconstruct_cmd.py:309")

            parent = os.path.dirname(abs_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  WRITTEN: {abs_path} ({len(content)} bytes)")
            written += 1

        # ── Summary ─────────────────────────────────────────────────────
        print(f"\n  Done: {len(file_results)} file(s), "
              f"{applied} op(s) applied, {skipped} skipped, "
              f"{edit_warnings} warning(s)")
        if dry_run:
            print("  (dry-run — no files were written)")
        elif written:
            print(f"  {written} file(s) written to {workspace}")

        return True

    # ── replay subcommand ────────────────────────────────────────────────
    #
    # Time-travel audit: walk one trace event-by-event.  Read-only by default;
    # --apply writes the reconstructed file set (with y/N confirm) but never
    # silently mutates the real workspace.  Non-tty input (CI/tests) prints the
    # whole timeline once and exits, so output stays deterministic.

    async def _cmd_replay(self, args: list[str], agent: "Agent") -> bool:
        workspace = agent.workspace
        traces_dir = os.path.join(workspace, _TRACES_DIR)

        task: str | None = None
        from_idx: int | None = None
        event_filter: str | None = None
        save_prefix: str | None = None
        apply = False

        i = 0
        while i < len(args):
            a = args[i]
            if not task and not a.startswith("--"):
                task = a
                i += 1
            elif a == "--workspace" and i + 1 < len(args):
                workspace = args[i + 1]
                traces_dir = os.path.join(workspace, _TRACES_DIR)
                i += 2
            elif a == "--from" and i + 1 < len(args):
                try:
                    from_idx = int(args[i + 1])
                except ValueError:
                    self.error(f"--from expects an integer, got {args[i + 1]!r}")
                    return True
                i += 2
            elif a == "--event" and i + 1 < len(args):
                event_filter = args[i + 1]
                i += 2
            elif a == "--save-prefix" and i + 1 < len(args):
                save_prefix = args[i + 1]
                i += 2
            elif a == "--apply":
                apply = True
                i += 1
            else:
                self.error(f"Unknown replay flag: {a}")
                return True

        if not task:
            self.error("Usage: reconstruct replay <task> [--from <n>] [--event <kind>] [--save-prefix <dir>] [--apply]")
            return True
        if not os.path.isdir(traces_dir):
            self.error(f"Traces directory not found: {traces_dir}")
            return True

        trace_path = _resolve_task_path(traces_dir, task)
        if trace_path is None:
            self.error(f"Trace not found for task {task!r} in {traces_dir}")
            return True

        session = ReplaySession(trace_path)
        if not session.events:
            self.error(f"No events in trace: {trace_path}")
            return True

        print(f"\n  == REPLAY {session.task_id} ==")
        print(f"  model={session.model or '-'} profile={session.profile or '-'} "
              f"outcome={session.outcome or '-'}")
        print(f"  prompt: {_shorten(session.prompt, 160) or '(none)'}")
        print(f"  events: {len(session.events)}  |  affected files: "
              f"{len(session.affected_files)}")
        print(f"  Companion: `review show {session.task_id}` (review ledger)")
        if session.affected_files:
            print("  files:")
            for f in session.affected_files:
                print(f"    - {f}")

        # Optional starting point / event filter.
        if from_idx is not None:
            session.goto(from_idx)
        if event_filter:
            idx = next(
                (k for k, e in enumerate(session.events) if e.get("kind") == event_filter),
                None,
            )
            if idx is None:
                self.error(f"No event of kind {event_filter!r} in trace.")
                return True
            session.goto(idx)

        # Non-interactive (piped/CI): dump the whole timeline, then (optionally)
        # apply.  --apply is honored in both modes.
        if not sys.stdin.isatty():
            print("\n  -- timeline (non-interactive dump) --")
            for ev in session.events:
                if event_filter and ev.get("kind") != event_filter:
                    continue
                print("  " + session.describe(ev, show_content=True).replace("\n", "\n  "))
            print(f"\n  End of replay ({len(session.events)} events).")
            if apply:
                self._replay_apply(session, save_prefix or workspace, agent)
            return True

        # Interactive loop.
        print("\n  Step controls: n/next  p/prev  g <n>/goto  f <path>  "
              "d <path>  l/ledger  s/stop")
        print("  (empty = next; 's' quits)\n")
        cur = session.current_event()
        if cur is not None:
            print("  " + session.describe(cur).replace("\n", "\n  "))

        while True:
            if stop_requested():
                break
            resp = read_input("  replay> ").strip().lower()
            if resp in ("s", "stop", "q", "quit", "abort", "x"):
                break
            if resp == "" or resp in ("n", "next"):
                if not session.next():
                    print("  (end of trace)")
            elif resp in ("p", "prev"):
                if not session.prev():
                    print("  (start of trace)")
            elif resp.startswith("g") or resp.startswith("goto"):
                parts = resp.replace("goto", "").split()
                if len(parts) >= 2 and parts[0] in ("", "g"):
                    try:
                        session.goto(int(parts[1]))
                    except ValueError:
                        print("  usage: g <n>")
                elif len(parts) == 1:
                    try:
                        session.goto(int(parts[0]))
                    except ValueError:
                        print("  usage: g <n>")
                else:
                    print("  usage: g <n>")
            elif resp.startswith("f") or resp.startswith("file"):
                arg = resp.split(maxsplit=1)[1].strip() if " " in resp else ""
                self._replay_show_file(session, arg)
            elif resp.startswith("d") or resp.startswith("diff"):
                arg = resp.split(maxsplit=1)[1].strip() if " " in resp else ""
                self._replay_show_diff(session, arg)
            elif resp in ("l", "ledger"):
                print(f"  ledger: review show {session.task_id}  "
                      f"(outcome={session.outcome or '-'})")
            else:
                print("  unknown: n/p/g/f/d/l/s")
                continue
            cur = session.current_event()
            if cur is not None:
                print("  " + session.describe(cur).replace("\n", "\n  "))

        # Optional export of reconstructed files.
        if apply:
            self._replay_apply(session, save_prefix or workspace, agent)
        print("\n  Replay ended.")
        return True

    def _replay_show_file(self, session: ReplaySession, path: str) -> None:
        if not path:
            print("  usage: f <path-or-stem>")
            return
        norm = path.replace("\\", "/")
        # Match by exact path or by basename stem.
        candidates = [
            f for f in session.affected_files
            if f.replace("\\", "/") == norm
            or f.replace("\\", "/").endswith("/" + norm)
            or os.path.basename(f).lower() == os.path.basename(norm).lower()
        ]
        if not candidates:
            print(f"  no such file in trace: {path}")
            return
        for c in candidates:
            content = session.file_state_at_cursor(c)
            if content is None:
                print(f"  {c}: not yet written at cursor {session.cursor}")
            else:
                print(f"  {c} ({len(content)} bytes) @ cursor {session.cursor}:")
                for ln in content.split("\n"):
                    print(f"    | {ln}")

    def _replay_show_diff(self, session: ReplaySession, path: str) -> None:
        nxt = session.next_file_event(path or None)
        if nxt is None:
            print("  no upcoming file mutation" + (f" for {path}" if path else ""))
            return
        args = _parse_args(nxt)
        p = str(args.get("path", "")).replace("\\", "/")
        current = session.file_state_at_cursor(p) or ""
        if nxt.get("kind") == "write":
            print(f"  upcoming write -> {p} ({len(str(args.get('content', '')))} bytes)")
            show_file_diff(os.path.basename(p), current, str(args.get("content", "")))
        else:
            new, ok = _apply_edit(
                current,
                str(args.get("old_text", "")),
                str(args.get("new_text", "")),
            )
            if not ok:
                print(f"  upcoming edit on {p}: old_text not present at cursor")
                show_file_diff(
                    os.path.basename(p), current,
                    current.replace(str(args.get("old_text", "")), str(args.get("new_text", "")), 1)
                    if str(args.get("old_text", "")) in current else current,
                )
            else:
                print(f"  upcoming edit -> {p}")
                show_file_diff(os.path.basename(p), current, new)

    def _replay_apply(
        self, session: ReplaySession, prefix: str, agent: "Agent"
    ) -> None:
        """Write the reconstructed file set (opt-in --apply).  Never writes
        into the real workspace unless prefix is the workspace and the user
        confirms each file via save_file_py."""
        final = session.final_state()
        if not final:
            print("  nothing to reconstruct.")
            return
        print(f"  --apply: writing {len(final)} reconstructed file(s) "
              f"under prefix {prefix}")
        for p, content in final.items():
            # Resolve target: if prefix is the workspace and path is absolute,
            # write in place; otherwise stage under prefix.
            target = p
            if prefix and prefix != agent.workspace:
                target = os.path.join(prefix, os.path.basename(p))
            save_file_py(target, content, auto_yes=False)
