"""Regression tests: the write/edit verification gate runs ruff, not just py_compile.

History: ``_verify_file`` only ran ``py_compile``, so the agent could write a
file that parses but violates the very lint rules the pre-commit hook and CI
block on.  The violation then surfaced minutes later as a red build instead of
immediately, while the agent still had the file in hand.

Now every ``.py`` write/edit reports ``[verify] py_compile …`` (first line,
unchanged) followed by ``[verify] ruff …``.

Two invariants are load-bearing and asserted here:

1. ``[verify] py_compile`` stays the FIRST line — ``turn_loop`` scans for that
   exact marker to build the post-mutation self-review note.
2. A missing/broken ruff NEVER fails a write.  Ruff is a ``[dev]`` extra, so an
   end-user install legitimately has no linter.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import agent_core.agent_effects as agent_effects
from agent_core.agent_effects import (
    RUFF_TIMEOUT_S,
    _format_ruff_violations,
    _lint_file,
    _verify_file,
)

# An undefined name (F821) and an unused import (F401) in one file: both are
# in the E/F/W set CI blocks on, so either one must show up in the summary.
DIRTY_SOURCE = "import os\n\n\nprint(undefined_name_xyz)\n"
CLEAN_SOURCE = "x = 1\n"


class _FakeAgent:
    """Minimal stand-in for the Agent attributes ``_verify_file`` reads."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = str(workspace)
        self._nlp_workspace = None


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    return tmp_path


def _verify(workdir: Path, path: Path) -> str:
    agent = _FakeAgent(workdir)
    return asyncio.run(_verify_file(agent, str(path)))


class TestVerifyRunsRuff:
    def test_clean_file_reports_both_gates(self, workdir: Path) -> None:
        target = workdir / "clean_mod.py"
        target.write_text(CLEAN_SOURCE, encoding="utf-8")
        out = _verify(workdir, target)
        assert "[verify] py_compile ✓" in out
        assert "[verify] ruff ✓" in out

    def test_py_compile_marker_stays_first_line(self, workdir: Path) -> None:
        """``turn_loop._mutating_files_this_turn`` depends on this ordering."""
        target = workdir / "clean_mod.py"
        target.write_text(CLEAN_SOURCE, encoding="utf-8")
        out = _verify(workdir, target)
        assert out.splitlines()[0] == "[verify] py_compile ✓"

    def test_lint_violation_is_reported_with_code(
        self, workdir: Path
    ) -> None:
        target = workdir / "dirty_mod.py"
        target.write_text(DIRTY_SOURCE, encoding="utf-8")
        out = _verify(workdir, target)
        # It parses, so py_compile still passes...
        assert "[verify] py_compile ✓" in out
        # ...but the lint gate must catch what CI would block on.
        assert "[verify] ruff ✗" in out
        assert "F401" in out or "E303" in out
        assert "dirty_mod.py:" in out

    def test_syntax_error_still_reports_py_compile_first(
        self, workdir: Path
    ) -> None:
        target = workdir / "broken_mod.py"
        target.write_text("def f(:\n", encoding="utf-8")
        out = _verify(workdir, target)
        assert out.splitlines()[0].startswith("[verify] py_compile ✗")

    def test_no_ruff_line_for_non_python_file(self, workdir: Path) -> None:
        target = workdir / "notes.txt"
        target.write_text("hello\n", encoding="utf-8")
        assert _lint_file(str(target)) == ""

    def test_missing_ruff_degrades_to_no_line(
        self, monkeypatch: pytest.MonkeyPatch, workdir: Path
    ) -> None:
        """An end-user install without the [dev] extra must not fail writes."""
        monkeypatch.setattr(agent_effects, "_ruff_command", lambda path: None)
        target = workdir / "clean_mod.py"
        target.write_text(CLEAN_SOURCE, encoding="utf-8")
        assert _lint_file(str(target)) == ""
        out = _verify(workdir, target)
        assert out == "[verify] py_compile ✓"

    def test_broken_ruff_binary_never_raises(
        self, monkeypatch: pytest.MonkeyPatch, workdir: Path
    ) -> None:
        target = workdir / "clean_mod.py"
        target.write_text(CLEAN_SOURCE, encoding="utf-8")
        monkeypatch.setattr(
            agent_effects, "_ruff_command", lambda path: ["definitely-not-a-real-ruff"]
        )
        out = _lint_file(str(target))
        assert out.startswith("[verify] ruff could not run")

    def test_ruff_child_is_wall_clock_capped(self) -> None:
        """Mirrors the fix_cmd guard: no unbounded child in a turn."""
        assert RUFF_TIMEOUT_S <= 60


class TestFormatRuffViolations:
    def test_parses_concise_output(self) -> None:
        stdout = (
            "agent_core/x.py:3:1: F401 `os` imported but unused\n"
            "agent_core/x.py:9:80: E501 Line too long (91 > 88)\n"
            "Found 2 errors.\n"
        )
        items = _format_ruff_violations(stdout)
        assert items == [
            "x.py:3: F401 `os` imported but unused",
            "x.py:9: E501 Line too long (91 > 88)",
        ]

    def test_ignores_summary_and_blank_lines(self) -> None:
        assert _format_ruff_violations("All checks passed!\n\n") == []


class TestRealWritePathRunsRuff:
    """End-to-end through the actual Agent write handler."""

    def test_write_tool_output_contains_ruff_line(self, tmp_path: Path) -> None:
        from agent import Agent

        bot = Agent(workspace=str(tmp_path))
        target = tmp_path / "probe_written.py"
        out = asyncio.run(
            bot._execute_tool_call(
                "write", {"path": str(target), "content": CLEAN_SOURCE}
            )
        )
        assert "[verify] py_compile ✓" in out
        assert "[verify] ruff ✓" in out
        assert target.exists()

    def test_write_of_dirty_file_surfaces_violation(
        self, tmp_path: Path
    ) -> None:
        from agent import Agent

        bot = Agent(workspace=str(tmp_path))
        target = tmp_path / "probe_dirty.py"
        out = asyncio.run(
            bot._execute_tool_call(
                "write", {"path": str(target), "content": DIRTY_SOURCE}
            )
        )
        assert "[verify] ruff ✗" in out

    def test_mutation_marker_extraction_still_works(
        self, tmp_path: Path
    ) -> None:
        """The two-line verify output must not break self-review extraction."""
        from agent import Agent

        bot = Agent(workspace=str(tmp_path))
        target = tmp_path / "probe_extract.py"
        out = asyncio.run(
            bot._execute_tool_call(
                "write", {"path": str(target), "content": CLEAN_SOURCE}
            )
        )
        bot._turn_start_index = 0
        bot._chat_history.append({"role": "tool", "content": out})
        files = bot._mutating_files_this_turn()
        assert len(files) == 1
        assert Path(files[0]).name == "probe_extract.py"
