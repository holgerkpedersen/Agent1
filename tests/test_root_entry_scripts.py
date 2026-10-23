"""Tests for the root entry scripts that the agent_core-scoped coverage
config never touches: agent_dashboard.py, sync_kanban_to_agent1.py,
fix_kanban_queue.py.

These scripts hardcode C:\\Dev\\Kanban paths; the tests pin that imports
are side-effect free, that the queue processing is isolated (monkeypatched
paths + stubbed bridge), and that the dashboard CLI surface works from the
repo root.
"""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Import a root-level module by name (repo root is on sys.path via conftest)."""
    if (ROOT / f"{name}.py").exists():
        return importlib.import_module(name)
    raise ImportError(f"{name}.py not importable from repo root")


def _public_functions(mod) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }


class TestImportPurity:
    """Importing a root entry script must have no side effects."""

    @pytest.mark.parametrize(
        "name", ["agent_dashboard", "sync_kanban_to_agent1", "fix_kanban_queue"]
    )
    def test_module_imports_cleanly(self, name: str) -> None:
        mod = _load(name)
        assert mod.__file__ and Path(mod.__file__).name == f"{name}.py"

    @pytest.mark.parametrize(
        "name", ["agent_dashboard", "sync_kanban_to_agent1", "fix_kanban_queue"]
    )
    def test_import_does_not_run_main(self, name: str) -> None:
        """A module-level entry call would make every import a side effect."""
        mod = _load(name)
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        entry_calls = [
            node
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", "") in ("main", "sync", "fix_queue")
        ]
        guard = [
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and any(
                isinstance(c, ast.Compare) and getattr(c.left, "id", "") == "__name__"
                for c in ast.walk(node.test)
            )
        ]
        assert not entry_calls, f"{name}.py calls its entry point at module level"
        assert guard, f"{name}.py has no __main__ guard"


class TestSyncKanbanToAgent1:
    def test_entry_point_exists(self) -> None:
        mod = _load("sync_kanban_to_agent1")
        assert "sync" in _public_functions(mod)

    def test_missing_queue_file_reports_without_crashing(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """No queue file -> clear error message, no traceback, no real path touched."""
        mod = _load("sync_kanban_to_agent1")
        monkeypatch.setattr(mod, "KANBAN_QUEUE_FILE", tmp_path / "missing.jsonl")
        mod.sync()
        out = capsys.readouterr().out
        assert "not found" in out.lower()

    def test_processes_all_ops_with_stubbed_bridge(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """card_create/update/move/delete each dispatch to the bridge; unknown
        ops are skipped; the summary line reports the counts."""
        mod = _load("sync_kanban_to_agent1")
        calls: list[str] = []

        def _fake(fn, ok=True):
            def _impl(_payload):
                calls.append(fn)
                return ok

            return _impl

        monkeypatch.setattr(mod, "_apply_card_create", _fake("create"))
        monkeypatch.setattr(mod, "_apply_card_update", _fake("update"))
        monkeypatch.setattr(mod, "_apply_card_move", _fake("move"))
        monkeypatch.setattr(mod, "_apply_card_delete", _fake("delete"))

        queue = tmp_path / "messages.jsonl"
        messages = [
            {"op": "card_create", "payload": {"id": "c1", "title": "t"}},
            {"op": "card_update", "source_id": "c1", "payload": {"title": "t2"}},
            {"op": "card_move", "source_id": "c1", "payload": {"column": "done"}},
            {"op": "card_delete", "source_id": "c1"},
            {"op": "bogus_op"},
        ]
        queue.write_text("\n".join(json.dumps(m) for m in messages), encoding="utf-8")
        monkeypatch.setattr(mod, "KANBAN_QUEUE_FILE", queue)

        mod.sync()

        out = capsys.readouterr().out
        assert calls == ["create", "update", "move", "delete"]
        assert "bogus_op" in out, "unknown op must be reported as skipped"
        assert "Processed: 5" in out
        assert "Successes: 4" in out
        assert "Failures:  0" in out

    def test_bridge_failure_is_counted_not_raised(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """A bridge function returning False counts as a failure; an exception
        is caught and counted — one bad message must not abort the sync."""
        mod = _load("sync_kanban_to_agent1")

        def _boom(_msg):
            raise RuntimeError("bridge exploded")

        monkeypatch.setattr(mod, "_apply_card_update", _boom)
        monkeypatch.setattr(mod, "_apply_card_create", lambda _p: False)

        queue = tmp_path / "messages.jsonl"
        messages = [
            {"op": "card_create", "payload": {"id": "c1"}},
            {"op": "card_update", "source_id": "c1", "payload": {}},
        ]
        queue.write_text("\n".join(json.dumps(m) for m in messages), encoding="utf-8")
        monkeypatch.setattr(mod, "KANBAN_QUEUE_FILE", queue)

        mod.sync()

        out = capsys.readouterr().out
        assert "Successes: 0" in out
        assert "Failures:  2" in out
        assert "bridge exploded" in out


class TestFixKanbanQueue:
    def test_entry_point_exists(self) -> None:
        mod = _load("fix_kanban_queue")
        assert "fix_queue" in _public_functions(mod)

    def test_missing_queue_file_reports_without_crashing(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        mod = _load("fix_kanban_queue")
        monkeypatch.setattr(mod, "KANBAN_QUEUE_FILE", tmp_path / "missing.jsonl")
        mod.fix_queue()
        out = capsys.readouterr().out
        assert "not found" in out.lower()

    def test_no_template_reports_and_writes_nothing(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """Without a card_update template for the target id, the queue must
        be left byte-identical."""
        mod = _load("fix_kanban_queue")
        queue = tmp_path / "messages.jsonl"
        original = (
            json.dumps(
                {"op": "card_create", "source_id": "zzz", "payload": {}, "ts": 1}
            )
            + "\n"
        )
        queue.write_text(original, encoding="utf-8")
        monkeypatch.setattr(mod, "KANBAN_QUEUE_FILE", queue)
        monkeypatch.setattr(mod, "TARGET_ID", "zzz")

        mod.fix_queue()

        out = capsys.readouterr().out
        assert "No card_update found" in out
        assert queue.read_text(encoding="utf-8") == original

    def test_prepends_card_create_and_resets_offset(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """The fix inserts a card_create built from the template at the head
        of the queue and resets the offset file to 0."""
        mod = _load("fix_kanban_queue")
        queue = tmp_path / "messages.jsonl"
        offset = tmp_path / "offset.txt"
        template = {
            "op": "card_update",
            "source_id": "target123",
            "payload": {"title": "card"},
            "ts": 42,
        }
        queue.write_text(json.dumps(template) + "\n", encoding="utf-8")
        offset.write_text("31", encoding="utf-8")
        monkeypatch.setattr(mod, "KANBAN_QUEUE_FILE", queue)
        monkeypatch.setattr(mod, "OFFSET_FILE", offset)
        monkeypatch.setattr(mod, "TARGET_ID", "target123")

        mod.fix_queue()

        out = capsys.readouterr().out
        assert "offset reset to 0" in out.lower()
        assert offset.read_text(encoding="utf-8") == "0"
        lines = [
            json.loads(line)
            for line in queue.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(lines) == 2
        created = lines[0]
        assert created["op"] == "card_create"
        assert created["source_id"] == "target123"
        assert created["payload"] == template["payload"]
        assert created["processed"] is False


class TestAgentDashboard:
    def test_entry_points_exist(self) -> None:
        mod = _load("agent_dashboard")
        names = set(_public_functions(mod))
        assert {"run_dashboard_server", "start_dashboard_thread"} <= names

    def test_cli_help_exits_zero(self) -> None:
        """`python agent_dashboard.py --help` must work from the repo root."""
        proc = subprocess.run(
            [sys.executable, str(ROOT / "agent_dashboard.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(ROOT),
        )
        assert proc.returncode == 0, (
            f"--help exited {proc.returncode}: {proc.stderr[:500]}"
        )
        assert "usage" in proc.stdout.lower()

    def test_dashboard_port_parses_cli(self, monkeypatch) -> None:
        mod = _load("agent_dashboard")
        for argv, expected in [
            (["agent_dashboard.py"], 8081),
            (["agent_dashboard.py", "--port", "9000"], 9000),
            (["agent_dashboard.py", "--port=9001"], 9001),
            (["agent_dashboard.py", "--port", "notanint"], 8081),
        ]:
            monkeypatch.setattr(sys, "argv", argv)
            assert mod._dashboard_port() == expected
