"""Tests for the reconstruct command."""
import json
import os
import tempfile

import pytest

from agent_core.commands.reconstruct_cmd import (
    _FileOp,
    _apply_edit,
    _collect_files,
    _parse_trace,
    _resolve_range,
    ReconstructCommand,
)


# ── Helper to build a minimal JSONL trace ──────────────────────────────────

def _write_trace(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_tool_result(
    tool: str,
    args: dict,
    *,
    ts: float = 1000.0,
    affected: list[str] | None = None,
    source: str = "test.py",
) -> dict:
    return {
        "task_id": "abc123",
        "ts": ts,
        "correlation_id": "",
        "kind": "tool_result",
        "layer": "tool_interface",
        "iteration": 0,
        "tool": tool,
        "args_hash": json.dumps(args),
        "tc_id": "",
        "duplicate": False,
        "duration_s": 0.01,
        "result": "ok",
        "affected_files": affected or [],
    }


# ── Unit tests: _apply_edit ────────────────────────────────────────────────

class TestApplyEdit:
    def test_basic_replacement(self):
        result, ok = _apply_edit("hello world", "world", "earth")
        assert ok
        assert result == "hello earth"

    def test_old_text_not_found(self):
        result, ok = _apply_edit("hello world", "xyz", "abc")
        assert not ok
        assert result == "hello world"

    def test_replaces_first_occurrence_only(self):
        result, ok = _apply_edit("a b a b", "a", "X")
        assert ok
        assert result == "X b a b"

    def test_single_occurrence_replaced(self):
        result, ok = _apply_edit("aXbXc", "X", "Y")
        assert ok
        assert result == "aYbXc"


# ── Unit tests: _parse_trace ───────────────────────────────────────────────

class TestParseTrace:
    def test_extracts_write_ops(self, tmp_path):
        trace = tmp_path / "trace.jsonl"
        rec = _make_tool_result(
            "write",
            {"path": "C:/Dev/foo.py", "content": "print('hello')"},
            ts=100.0,
            affected=["C:\\Dev\\foo.py"],
        )
        _write_trace(str(trace), [rec])
        ops = _parse_trace(str(trace))
        assert len(ops) == 1
        assert ops[0].kind == "write"
        assert ops[0].path == "C:/Dev/foo.py"
        assert ops[0].content == "print('hello')"

    def test_extracts_edit_ops(self, tmp_path):
        trace = tmp_path / "trace.jsonl"
        target = "C:/Dev/foo.py"
        rec = _make_tool_result(
            "edit",
            {
                "path": target,
                "old_text": "old",
                "new_text": "new",
            },
            ts=200.0,
            affected=[target],
        )
        _write_trace(str(trace), [rec])
        ops = _parse_trace(str(trace))
        assert len(ops) == 1
        assert ops[0].kind == "edit"
        assert ops[0].old_text == "old"
        assert ops[0].new_text == "new"

    def test_skips_non_write_edit_tools(self, tmp_path):
        trace = tmp_path / "trace.jsonl"
        rec = _make_tool_result("read", {"path": "foo.py"})
        _write_trace(str(trace), [rec])
        ops = _parse_trace(str(trace))
        assert len(ops) == 0

    def test_skips_tool_call_events(self, tmp_path):
        trace = tmp_path / "trace.jsonl"
        rec = {
            "kind": "tool_call",
            "tool": "write",
            "args_hash": json.dumps({"path": "x.py", "content": "y"}),
            "ts": 100.0,
        }
        _write_trace(str(trace), [rec])
        ops = _parse_trace(str(trace))
        assert len(ops) == 0

    def test_normalizes_backslashes(self, tmp_path):
        trace = tmp_path / "trace.jsonl"
        target = "C:\\Dev\\foo.py"
        rec = _make_tool_result(
            "write",
            {"path": target, "content": "x"},
            affected=[target],
        )
        _write_trace(str(trace), [rec])
        ops = _parse_trace(str(trace))
        assert ops[0].path == "C:/Dev/foo.py"


# ── Unit tests: _resolve_range ─────────────────────────────────────────────

class TestResolveRange:
    def _make_files(self, tmp_path, names):
        files = []
        for name in names:
            p = tmp_path / name
            p.write_text("x")
            files.append(str(p))
            # Ensure different mtimes.
            os.utime(str(p), (1000 + len(files), 1000 + len(files)))
        return files

    def test_no_start_no_end_returns_all(self, tmp_path):
        files = self._make_files(tmp_path, ["a.jsonl", "b.jsonl", "c.jsonl"])
        result = _resolve_range(files, None, None)
        assert len(result) == 3

    def test_start_only(self, tmp_path):
        files = self._make_files(tmp_path, ["a.jsonl", "b.jsonl", "c.jsonl"])
        result = _resolve_range(files, "b.jsonl", None)
        assert len(result) == 2
        assert os.path.basename(result[0]) == "b.jsonl"

    def test_end_only(self, tmp_path):
        files = self._make_files(tmp_path, ["a.jsonl", "b.jsonl", "c.jsonl"])
        result = _resolve_range(files, None, "b.jsonl")
        assert len(result) == 2
        assert os.path.basename(result[-1]) == "b.jsonl"

    def test_start_and_end(self, tmp_path):
        files = self._make_files(tmp_path, ["a.jsonl", "b.jsonl", "c.jsonl", "d.jsonl"])
        result = _resolve_range(files, "b.jsonl", "c.jsonl")
        assert len(result) == 2
        assert os.path.basename(result[0]) == "b.jsonl"
        assert os.path.basename(result[1]) == "c.jsonl"

    def test_prefix_match(self, tmp_path):
        files = self._make_files(tmp_path, ["abc123.jsonl", "def456.jsonl", "ghi789.jsonl"])
        result = _resolve_range(files, "def", None)
        assert len(result) == 2
        assert os.path.basename(result[0]) == "def456.jsonl"


# ── Unit tests: _collect_files ─────────────────────────────────────────────

class TestCollectFiles:
    def test_returns_jsonl_files_sorted_by_mtime(self, tmp_path):
        (tmp_path / "b.jsonl").write_text("x")
        (tmp_path / "a.jsonl").write_text("y")
        (tmp_path / "c.txt").write_text("z")
        os.utime(str(tmp_path / "a.jsonl"), (100, 100))
        os.utime(str(tmp_path / "b.jsonl"), (200, 200))
        files = _collect_files(str(tmp_path))
        assert len(files) == 2
        assert files[0].endswith("a.jsonl")
        assert files[1].endswith("b.jsonl")

    def test_empty_dir(self, tmp_path):
        assert _collect_files(str(tmp_path)) == []

    def test_nonexistent_dir(self):
        assert _collect_files("/nonexistent/path") == []


# ── Integration test: full reconstruct pipeline ────────────────────────────

class TestReconstructPipeline:
    def test_write_then_edit_produces_final_content(self, tmp_path):
        """write creates file, edit modifies it — final state is correct."""
        trace1 = tmp_path / "t1.jsonl"
        trace2 = tmp_path / "t2.jsonl"
        target = str(tmp_path / "workspace" / "foo.py").replace("\\", "/")

        _write_trace(str(trace1), [
            _make_tool_result(
                "write",
                {"path": target, "content": "line1\nline2\n"},
                ts=100.0,
                affected=[target],
            ),
        ])
        _write_trace(str(trace2), [
            _make_tool_result(
                "edit",
                {
                    "path": target,
                    "old_text": "line2",
                    "new_text": "LINE2",
                },
                ts=200.0,
                affected=[target],
            ),
        ])

        # Parse and merge
        ops = _parse_trace(str(trace1)) + _parse_trace(str(trace2))
        ops.sort(key=lambda o: o.ts)

        # Find last write
        last_write = [o for o in ops if o.kind == "write"][-1]
        content = last_write.content

        # Apply subsequent edits
        for op in ops:
            if op.kind == "edit" and op.ts > last_write.ts:
                content, _ = _apply_edit(content, op.old_text, op.new_text)

        assert content == "line1\nLINE2\n"

    def test_last_write_wins(self, tmp_path):
        """When multiple writes exist, only the last write + edits matter."""
        ops = [
            _FileOp(ts=100, kind="write", path="x.py", content="v1"),
            _FileOp(ts=200, kind="write", path="x.py", content="v2"),
            _FileOp(ts=300, kind="write", path="x.py", content="v3"),
        ]
        # Last write is v3
        last_write = ops[-1]
        assert last_write.content == "v3"

    def test_edit_after_last_write_applied(self, tmp_path):
        """Edits after the last write are applied to the write content."""
        content = "hello world"
        ops = [
            _FileOp(ts=100, kind="write", path="x.py", content="hello world"),
            _FileOp(ts=200, kind="edit", path="x.py", old_text="world", new_text="earth"),
        ]
        last_write_idx = 0
        content = ops[last_write_idx].content
        for op in ops[last_write_idx + 1:]:
            if op.kind == "edit":
                content, _ = _apply_edit(content, op.old_text, op.new_text)
        assert content == "hello earth"

    def test_edit_before_last_write_ignored(self, tmp_path):
        """Edits before the last write are ignored (write overwrites them)."""
        content = "final"
        ops = [
            _FileOp(ts=100, kind="edit", path="x.py", old_text="x", new_text="y"),
            _FileOp(ts=200, kind="write", path="x.py", content="final"),
        ]
        last_write_idx = 1
        content = ops[last_write_idx].content
        for op in ops[last_write_idx + 1:]:
            if op.kind == "edit":
                content, _ = _apply_edit(content, op.old_text, op.new_text)
        assert content == "final"


# ── CLI integration test ───────────────────────────────────────────────────

class TestReconstructCommand:
    def test_dry_run_writes_nothing(self, tmp_path, capsys):
        """--dry-run reports what would be written but does not write."""
        ws = tmp_path / "workspace"
        ws.mkdir()
        traces = ws / "reports" / "traces"
        traces.mkdir(parents=True)

        target = str((ws / "out.py")).replace("\\", "/")
        _write_trace(str(traces / "t1.jsonl"), [
            _make_tool_result(
                "write",
                {"path": target, "content": "print(1)"},
                affected=[target],
            ),
        ])

        cmd = ReconstructCommand()

        class FakeAgent:
            workspace = str(ws)

        import asyncio
        result = asyncio.run(
            cmd.execute(["--dry-run"], FakeAgent())
        )
        assert result is True
        assert not os.path.exists(str(ws / "out.py"))  # nothing written

    def test_missing_traces_dir(self, tmp_path, capsys):
        """Reports error when traces dir doesn't exist."""
        cmd = ReconstructCommand()
        class FakeAgent:
            workspace = str(tmp_path / "nonexistent")

        import asyncio
        asyncio.run(
            cmd.execute([], FakeAgent())
        )
        captured = capsys.readouterr()
        assert "not found" in captured.out
