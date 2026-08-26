"""String-collision guard tests: repairs must not break test assertions."""
from __future__ import annotations

import json
from pathlib import Path

from harnessfix import gates
from harnessfix.loop import run_loop
from harnessfix.repairs.collisions import find_test_collisions
from harnessfix.repairs.tool_interface import _NEW, _OLD
from harnessfix.tracing import KIND_LOOP_END, KIND_TOOL_ERROR, TraceWriter


def _write_tool_error_trace(traces_dir: Path, task_id: str) -> None:
    writer = TraceWriter(task_id=task_id, directory=traces_dir)
    writer.emit(
        {
            "kind": KIND_TOOL_ERROR,
            "layer": "tool_interface",
            "exception": "ValidationError",
            "message": "schema validation failed for path",
        }
    )
    writer.emit(
        {"kind": KIND_LOOP_END, "layer": "lifecycle", "outcome": "completed", "termination_reason": "answer"}
    )
    writer.close()


def _loop_source() -> str:
    return Path("agent_core/llm/tool_loop.py").read_text(encoding="utf-8")


def _write_asserting_test(tests_dir: Path, content: str, name: str = "test_x.py") -> None:
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / name).write_text(content, encoding="utf-8")


class TestFindTestCollisions:
    def test_detects_assertions_pinning_the_old_string(self, tmp_path):
        tests_dir = tmp_path / "tests"
        _write_asserting_test(
            tests_dir,
            'def test_error_fed_back():\n    assert "Tool error: boom" in content\n',
        )
        hits = find_test_collisions(("Tool error: ",), tests_dir)
        assert len(hits) == 1
        hit = hits[0]
        assert hit.path.name == "test_x.py"
        assert hit.line == 2
        assert "Tool error: boom" in hit.snippet
        assert hit.fragment == "Tool error: "
        assert hit.to_dict()["file"].endswith("test_x.py")

    def test_no_fragments_or_missing_dir_returns_empty(self, tmp_path):
        assert find_test_collisions((), tmp_path) == []
        assert find_test_collisions(("Tool error: ",), tmp_path / "nope") == []

    def test_reports_every_fragment_and_file(self, tmp_path):
        tests_dir = tmp_path / "tests"
        _write_asserting_test(tests_dir, 'a = "Tool error: x"\n', "a.py")
        _write_asserting_test(tests_dir, 'b = "Tool error: y"\n', "b.py")
        hits = find_test_collisions(("Tool error: ",), tests_dir)
        assert len(hits) == 2

    def test_guard_fixture_files_excluded_by_default(self, tmp_path):
        """Decision #051: the guard's own fixture tests contain the runtime
        fragments as literals, so they are excluded by default — otherwise
        every repair would self-block."""
        from harnessfix.repairs.collisions import GUARD_TEST_FILENAMES

        assert "test_harnessfix_collisions.py" in GUARD_TEST_FILENAMES
        tests_dir = tmp_path / "tests"
        _write_asserting_test(
            tests_dir, 'a = "Tool error: x"\n', "test_harnessfix_collisions.py"
        )
        assert find_test_collisions(("Tool error: ",), tests_dir) == []
        # Opting out of the exclusion still finds it.
        hits = find_test_collisions(("Tool error: ",), tests_dir, exclude_files=frozenset())
        assert len(hits) == 1
        assert hits[0].path.name == "test_harnessfix_collisions.py"


class TestLoopCollisionGuard:
    def test_repair_skipped_when_tests_assert_the_string(self, tmp_path, monkeypatch):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        _write_tool_error_trace(traces_dir, "tr1")

        tests_dir = tmp_path / "tests"
        _write_asserting_test(
            tests_dir,
            'def test_tool_error():\n    assert "Tool error: boom" in msg["content"]\n',
        )
        monkeypatch.setattr(
            "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR", tests_dir
        )
        # The collision guard fires before the gate, so get_baseline_failures
        # must still be stubbed to avoid a real (slow) suite run.
        monkeypatch.setattr(gates, "get_baseline_failures", lambda *a, **k: frozenset())

        out = tmp_path / "out"
        summary = run_loop(traces_dir, approve=True, model=None, output_dir=out)

        assert summary["verdict"] == "skipped_test_collision"
        assert summary["collisions"], summary
        assert summary["collisions"][0]["line"] == 2
        # The repair was never applied — no gate run, no revert.
        assert _OLD in _loop_source()
        assert _NEW not in _loop_source()
        persisted = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert persisted["verdict"] == "skipped_test_collision"

    def test_repair_proceeds_when_tests_are_clean(self, tmp_path, monkeypatch):
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        _write_tool_error_trace(traces_dir, "tr2")

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()  # no assertions on the affected string
        monkeypatch.setattr(
            "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR", tests_dir
        )
        monkeypatch.setattr(gates, "get_baseline_failures", lambda *a, **k: frozenset())
        monkeypatch.setattr(gates, "run_test_gate", lambda *a, **k: (True, "passed"))
        monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
        monkeypatch.setattr(gates, "run_benchmark_gate", lambda model, profile=None: None)

        from harnessfix.repairs.tool_interface import revert

        out = tmp_path / "out"
        try:
            summary = run_loop(traces_dir, approve=True, model=None, output_dir=out)
            assert summary["verdict"] == "accepted"
            assert _NEW in _loop_source()
        finally:
            revert()
        assert _OLD in _loop_source()

    def test_guard_fixture_hits_are_ignored_and_recorded(self, tmp_path, monkeypatch):
        """Decision #051: matches inside the guard's OWN test files no longer
        block the repair; they are reported as ignored_guard_test_hits."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        _write_tool_error_trace(traces_dir, "tr3")

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        _write_asserting_test(
            tests_dir, 'a = "Tool error: x"\n', "test_harnessfix_collisions.py"
        )
        monkeypatch.setattr(
            "harnessfix.repairs.collisions.DEFAULT_TESTS_DIR", tests_dir
        )
        monkeypatch.setattr(gates, "get_baseline_failures", lambda *a, **k: frozenset())
        monkeypatch.setattr(gates, "run_test_gate", lambda *a, **k: (True, "passed"))
        monkeypatch.setattr(gates, "run_security_gate", lambda: (True, "ok"))
        monkeypatch.setattr(gates, "run_benchmark_gate", lambda model, profile=None: None)

        from harnessfix.repairs.tool_interface import revert

        out = tmp_path / "out"
        try:
            summary = run_loop(traces_dir, approve=True, model=None, output_dir=out)
            assert summary["verdict"] == "accepted"
            assert summary["ignored_guard_test_hits"] >= 1
            assert "collisions" not in summary
            assert _NEW in _loop_source()
        finally:
            revert()
        assert _OLD in _loop_source()
