"""Regression tests for the history-assisted implementation query layer.

Pins harnessfix/history.py: the trace-index + execution-ledger queries and
the PAST EXECUTION NOTES formatters that implement/fix inject into prompts.
The real-corpus test below is skipped when reports/traces/ is absent.
"""
import json
import os

import pytest

from harnessfix.history import (
    HISTORY_SUBDIR,
    append_execution,
    clear_history_cache,
    file_history,
    format_batch_history,
    format_file_history,
    history_root,
)

TRACE_DIR = "reports"
TRACE_SUB = "traces"


def _trace(tmp_path, name: str, events: list[dict]) -> object:
    trace_dir = tmp_path / TRACE_DIR / TRACE_SUB
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / name
    path.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    return path


def _result_event(task: str, tool: str, args: dict, result: str = "", affected=None, ts: float = 100.0) -> dict:
    return {
        "task_id": task,
        "ts": ts,
        "kind": "tool_result",
        "layer": "tool_interface",
        "tool": tool,
        "args_hash": json.dumps(args),
        "result": result,
        "affected_files": affected or [],
    }


def _error_event(task: str, tool: str, args: dict, exc: str, msg: str, ts: float = 100.0) -> dict:
    return {
        "task_id": task,
        "ts": ts,
        "kind": "tool_error",
        "layer": "tool_interface",
        "tool": tool,
        "args_hash": json.dumps(args),
        "exception": exc,
        "message": msg,
    }


class TestPathMatching:
    def test_absolute_arg_matches_relative_target(self):
        from harnessfix.history import _matches_arg

        assert _matches_arg("C:/Dev/Agent1/agent_core/commands/workflow_cmd.py", "agent_core/commands/workflow_cmd.py")
        assert _matches_arg("C:/Dev/Agent1/agent_core/commands/workflow_cmd.py", "workflow_cmd.py")
        assert not _matches_arg("C:/Dev/Agent1/agent_core/commands/workflow_cmd.py", "agent_core/commands/implement_cmd.py")

    def test_mixed_separators_normalized(self):
        from harnessfix.history import _matches_arg

        assert _matches_arg("C:\\Dev\\Agent1/agent_core/commands/x.py", "agent_core/commands/x.py")

    def test_directory_arg_matches_direct_child_only(self):
        from harnessfix.history import _matches_arg

        assert _matches_arg("C:/Dev/Agent1/agent_core/commands", "C:/Dev/Agent1/agent_core/commands/workflow_cmd.py")
        assert not _matches_arg("C:/Dev/Agent1", "C:/Dev/Agent1/agent_core/commands/workflow_cmd.py")

    def test_relative_target_suffix_against_relative_arg(self):
        from harnessfix.history import _matches_arg

        assert _matches_arg("agent_core/commands/x.py", "commands/x.py")


class TestTraceParsing:
    def test_read_event_uses_path_arg(self, tmp_path):
        p = _trace(
            tmp_path,
            "t1.jsonl",
            [_result_event("t1", "read", {"path": "C:/Dev/Agent1/agent.py"}, result="content", ts=1)],
        )
        from harnessfix.history import _parse_trace_event, _trace_events

        ev = _parse_trace_event(json.loads(p.read_text(encoding="utf-8").splitlines()[0]))
        assert ev is not None
        assert ev.tool == "read"
        assert ev.weight == 3
        assert any(f.endswith("agent.py") for f in ev.files)
        assert ev.kind == "result"

    def test_error_event_weight_zero_and_message(self, tmp_path):
        p = _trace(
            tmp_path,
            "t2.jsonl",
            [
                _error_event(
                    "t2", "read", {"path": "C:/Dev/Agent1/agent.py"},
                    "PermissionError", "Permission denied: agent.py",
                )
            ],
        )
        from harnessfix.history import _parse_trace_event

        ev = _parse_trace_event(json.loads(p.read_text(encoding="utf-8").splitlines()[0]))
        assert ev is not None
        assert ev.kind == "error"
        assert ev.weight == 0
        assert "Permission denied" in ev.summary

    def test_edit_with_affected_files_matches(self, tmp_path):
        p = _trace(
            tmp_path,
            "t3.jsonl",
            [
                _result_event(
                    "t3", "edit", {"new_text": "x"}, affected=["C:/Dev/Agent1/agent_core/commands/y.py"], ts=2
                )
            ],
        )
        from harnessfix.history import _parse_trace_event

        ev = _parse_trace_event(json.loads(p.read_text(encoding="utf-8").splitlines()[0]))
        assert ev is not None
        assert ev.weight == 1
        assert any("y.py" in f for f in ev.files)

    def test_run_event_summary_holds_command(self, tmp_path):
        p = _trace(
            tmp_path,
            "t4.jsonl",
            [_result_event("t4", "run", {"command": "python -m pytest tests/test_x.py"}, ts=3)],
        )
        from harnessfix.history import _parse_trace_event

        ev = _parse_trace_event(json.loads(p.read_text(encoding="utf-8").splitlines()[0]))
        assert ev is not None
        assert ev.weight == 2
        assert "pytest" in ev.summary


class TestFileHistory:
    def test_filters_and_sorts_by_weight(self, tmp_path):
        _trace(
            tmp_path,
            "h1.jsonl",
            [
                _result_event("h1", "read", {"path": "C:/Dev/Agent1/a.py"}, ts=1),
                _result_event("h1", "edit", {"path": "C:/Dev/Agent1/a.py"}, affected=["C:/Dev/Agent1/a.py"], ts=2),
                _error_event("h1", "write", {"path": "C:/Dev/Agent1/a.py"}, "OSError", "disk full", ts=3),
            ],
        )
        ws = str(tmp_path)
        events = file_history("a.py", ws)
        assert [e.weight for e in events] == [0, 1, 3]
        assert [e.kind for e in events] == ["error", "result", "result"]

    def test_limits_and_dedupes_chunk_reads(self, tmp_path):
        _trace(
            tmp_path,
            "h2.jsonl",
            [
                _result_event("h2", "read", {"path": "C:/Dev/Agent1/b.py", "offset": 1}, ts=1),
                _result_event("h2", "read", {"path": "C:/Dev/Agent1/b.py", "offset": 101}, ts=2),
            ],
        )
        events = file_history("b.py", str(tmp_path), limit=1)
        assert len(events) == 1

    def test_irrelevant_file_not_returned(self, tmp_path):
        _trace(tmp_path, "h3.jsonl", [_result_event("h3", "read", {"path": "C:/Dev/Agent1/c.py"})])
        assert file_history("d.py", str(tmp_path)) == []


class TestFormatters:
    def test_empty_history_returns_empty_string(self, tmp_path):
        assert format_file_history("missing.py", str(tmp_path)) == ""
        assert format_batch_history(["missing.py"], str(tmp_path)) == ""

    def test_format_file_history_renders_entries(self, tmp_path):
        _trace(
            tmp_path,
            "f1.jsonl",
            [_error_event("f1", "read", {"path": "C:/Dev/Agent1/f1.jsonl"}, "TimeoutExpired", "command timed out", ts=5)],
        )
        block = format_file_history("f1.jsonl", str(tmp_path))
        assert "PAST EXECUTION NOTES" in block
        assert "error" in block

    def test_format_batch_history_multi_file_and_cap(self, tmp_path):
        for i, name in enumerate(["x1.py", "x2.py"]):
            _trace(tmp_path, f"b{i}.jsonl", [_result_event(f"b{i}", "write", {"path": f"C:/Dev/Agent1/{name}"}, affected=[f"C:/Dev/Agent1/{name}"])])
        block = format_batch_history(["x1.py", "x2.py"], str(tmp_path), per_file=1, line_cap=3)
        assert "PAST EXECUTION NOTES" in block
        assert "x1.py" in block
        assert "..." in block


class TestExecutionLedger:
    def test_append_and_read_roundtrip(self, tmp_path):
        ws = str(tmp_path)
        append_execution(
            ws, "implement",
            [{"path": "agent_core/commands/z.py", "status": "written"}],
            outcome="ok", note="1 file",
        )
        events = file_history("agent_core/commands/z.py", ws)
        assert len(events) == 1
        assert events[0].source == "run"
        assert events[0].tool == "implement"
        assert events[0].kind == "execute"
        assert "ok" in events[0].summary
        ledger = tmp_path / TRACE_DIR / HISTORY_SUBDIR / "executions.jsonl"
        assert ledger.is_file()

    def test_ledger_empty_when_absent(self, tmp_path):
        assert file_history("z.py", str(tmp_path)) == []

    def test_corrupt_ledger_line_skipped(self, tmp_path):
        ledger = tmp_path / TRACE_DIR / HISTORY_SUBDIR / "executions.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text("{not json}\n", encoding="utf-8")
        assert file_history("z.py", str(tmp_path)) == []


class TestHistoryRoot:
    def test_finds_reports_root_by_walking_up(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (tmp_path / TRACE_DIR / TRACE_SUB).mkdir(parents=True)
        root = history_root(str(deep))
        assert root == os.path.abspath(str(tmp_path))

    def test_none_when_no_reports_dir(self, tmp_path):
        assert history_root(str(tmp_path)) is None


class TestRealCorpus:
    def test_workflow_cmd_history_from_real_traces(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isdir(os.path.join(here, "reports", TRACE_SUB)):
            pytest.skip("real trace corpus absent")
        events = file_history("agent_core/commands/workflow_cmd.py", here, limit=10)
        assert len(events) >= 1
        assert all(ev.files for ev in events)


def teardown_module():
    clear_history_cache()