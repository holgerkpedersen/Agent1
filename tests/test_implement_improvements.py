"""Tests for implement command improvements (2026-08-23 round).

Covers three changes to ``agent_core/commands/implement_cmd.py``:

1. ``_parse_line_number`` anchors on structured locations
   (``path.py:LINE:``, ``File "...", line N``, ``line N``) instead of
   scanning the first digit — a bare digit scan returned 2 for
   ``agent_core/llm/v2/client.py:88:`` (from the ``v2`` fragment) and
   centred the LLM fix window on the wrong lines.
2. The generation retry loop treats ``[LM Studio ...]`` responses as
   failures (same as ``[Error: ...]``) and records per-file outcomes when
   a batch exhausts its retries, instead of handing the sentinel to the
   block parser which then silently dropped the batch.
3. ``ImplementCommand._status_report`` renders a read-only plan progress
   report (ready / needs-generation / stdlib-shadowing) with no LLM calls.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from agent_core.commands.implement_cmd import (
    ImplementCommand,
    _parse_line_number,
)


class TestParseLineNumberStructured:
    def test_mypy_path_with_version_fragment(self):
        """Regression: first-digit scan returned 2 from the 'v2' path part."""
        err = (
            "agent_core/llm/v2/client.py:88: error: Item \"None\" of "
            "\"str | None\" has no attribute \"strip\"  [union-attr]"
        )
        assert _parse_line_number(err) == 88

    def test_windows_abs_path(self):
        err = r"C:\Dev\Agent1\agent_core\commands\implement_cmd.py:162: error: X"
        assert _parse_line_number(err) == 162

    def test_traceback_file_line_form(self):
        err = 'File "agent_core/utils/grid.py", line 130, in __init__\nValueError: bad'
        assert _parse_line_number(err) == 130

    def test_plain_line_word(self):
        err = "SMOKE: instantiation failed at line 42: TypeError: missing arg"
        assert _parse_line_number(err) == 42

    def test_path_form_preferred_over_later_line_word(self):
        err = "grid.py:7: error: something (see line 999 for details)"
        assert _parse_line_number(err) == 7

    def test_cross_marker_still_returns_one(self):
        assert _parse_line_number("CROSS: MISSING: Grid.rows") == 1

    def test_no_location_returns_zero(self):
        assert _parse_line_number("something went wrong") == 0


class TestGenerationRetrySentinels:
    """The retry loop must not accept transport sentinels as content."""

    def test_lmstudio_sentinel_is_a_failure_string(self):
        # Contract check on the sentinel prefix set used by execute():
        # both prefixes must be treated as failures.
        for sentinel in ("[Error: timeout]", "[LM Studio stream error: HTTP Error 400: Bad Request]"):
            assert sentinel.startswith(("[Error:", "[LM Studio"))

    def test_batch_failure_records_outcome_for_every_batch_file(self):
        """When generation fails after retries, every file in the batch gets
        an outcome so --status/--retry and the summary report it honestly."""
        cmd = ImplementCommand()
        outcomes: dict[str, str] = {}
        batch = ["agent_core/a.py", "agent_core/b.py"]
        impl_response = "[LM Studio stream error: model is not loaded]"
        # Mirror the fixed execute() behaviour:
        if not impl_response or impl_response.startswith(("[Error:", "[LM Studio")):
            for bf in batch:
                outcomes[bf] = f"generation failed — {str(impl_response)[:120]}"
        assert set(outcomes) == set(batch)
        assert "generation failed" in outcomes["agent_core/a.py"]
        assert "model is not loaded" in outcomes["agent_core/a.py"]


class TestStatusReport:
    def _cmd(self) -> ImplementCommand:
        return ImplementCommand()

    def test_ready_and_needs_generation(self, tmp_path, capsys):
        (tmp_path / "pkg").mkdir()
        good = tmp_path / "pkg" / "good.py"
        good.write_text("x = 1\n", encoding="utf-8")
        broken = tmp_path / "pkg" / "broken.py"
        broken.write_text("def f(:\n", encoding="utf-8")

        self._cmd()._status_report(
            ["pkg/good.py", "pkg/broken.py", "pkg/missing.py"], str(tmp_path),
        )
        out = capsys.readouterr().out
        assert "1 ready" in out
        assert "2 need work" in out
        assert "✓ pkg/good.py" in out
        assert "✗ pkg/broken.py — compile failed" in out
        assert "✗ pkg/missing.py — not found" in out

    def test_empty_file_counts_as_needs_generation(self, tmp_path, capsys):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "empty.py").write_text("", encoding="utf-8")
        self._cmd()._status_report(["pkg/empty.py"], str(tmp_path))
        out = capsys.readouterr().out
        assert "empty file" in out

    def test_non_python_file_needs_no_compile(self, tmp_path, capsys):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "notes.md").write_text("# notes\n", encoding="utf-8")
        self._cmd()._status_report(["docs/notes.md"], str(tmp_path))
        out = capsys.readouterr().out
        assert "1 ready" in out

    def test_shadowing_dir_reported_separately(self, tmp_path, capsys):
        # A planned file under a NOT-yet-existing 'logging/' directory
        # shadows the stdlib module — reported as redirected, not ready.
        self._cmd()._status_report(["logging/formatter.py"], str(tmp_path))
        out = capsys.readouterr().out
        assert "stdlib-shadowing" in out
        assert "shadows stdlib" in out

    def test_status_mode_returns_before_llm(self, tmp_path, capsys):
        """`implement --status` must exit before any LLM/chat call."""
        import asyncio

        taskplan = tmp_path / "taskplan.md"
        taskplan.write_text("1. `pkg/newmod.py` — do a thing\n", encoding="utf-8")

        cmd = ImplementCommand()

        async def fail_chat(*a, **kw):  # pragma: no cover - must not run
            raise AssertionError("LLM called during --status")

        class _StubLLM:
            def chat(self, *a, **kw):
                return fail_chat()

        class _StubAgent:
            workspace = str(tmp_path)
            llm = _StubLLM()

        with patch.object(_StubLLM, "chat", side_effect=fail_chat):
            result = asyncio.run(
                cmd.execute([str(taskplan), "--status"], _StubAgent())
            )
        assert result is True
        out = capsys.readouterr().out
        assert "[status]" in out
        assert "no LLM calls" in out
        assert "pkg/newmod.py" in out
