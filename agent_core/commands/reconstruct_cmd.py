"""Reconstruct files from JSONL trace logs.

Scans trace files in ``reports/traces/`` for ``write`` and ``edit`` tool
operations, groups them by target file path, and replays them in timestamp
order to recreate the final state of each file.

Usage::

    reconstruct [--start <file.jsonl>] [--end <file.jsonl>]
                [--workspace <path>] [--search <query>]
                [--dry-run] [--force]
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from .base import Command

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


class ReconstructCommand(Command):
    """Reconstruct files from JSONL trace logs."""

    @property
    def name(self) -> str:
        return "reconstruct"

    @property
    def help_text(self) -> str:
        return (
            "reconstruct [--start <file>] [--end <file>] [--workspace <path>] "
            "[--search <query>] [--dry-run] [--force]"
        )

    async def execute(self, args: list[str], agent: "Agent") -> bool:
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
                    pass

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
