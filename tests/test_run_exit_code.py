"""Regression tests: the ``run`` tool reports non-zero exit codes.

The model once tried to capture exit codes itself via broken shell pipelines
(``sys.exitcode`` does not exist) because the run-tool output never surfaced
them. The tool now appends ``[EXIT CODE: N]`` so the model can read it
directly.
"""

import asyncio
import sys

from agent import Agent, _shape_run_stderr


class TestExitCodeReporting:
    def test_success_output_unchanged(self) -> None:
        assert _shape_run_stderr("", "hello", 0) == "hello"

    def test_exit_code_appended_when_no_stderr(self) -> None:
        out = _shape_run_stderr("", "boom", 1)
        assert out == "boom\n[EXIT CODE: 1]"

    def test_exit_code_appended_when_stderr_present(self) -> None:
        out = _shape_run_stderr("bad", "output", 2)
        assert "[EXIT CODE: 2]" in out

    def test_exit_code_appended_end_after_stderr(self) -> None:
        out = _shape_run_stderr("bad", "output", 2)
        assert out.rstrip().endswith("[EXIT CODE: 2]")

    def test_none_returncode_adds_no_marker(self) -> None:
        assert _shape_run_stderr("", "output", None) == "output"

    def test_zero_returncode_adds_no_marker(self) -> None:
        out = _shape_run_stderr("", "output", 0)
        assert "[EXIT CODE" not in out

    def test_stderr_with_returncode_zero_no_marker(self) -> None:
        out = _shape_run_stderr("warn", "output", 0)
        assert "[EXIT CODE" not in out
        assert "warn" in out

    def test_255_no_output_keeps_behavior(self) -> None:
        # rc=255 with no output is the Windows cmd.exe whole-pipeline failure
        # path. The function must not crash and always returns a string.
        out = _shape_run_stderr("", "", 255)
        assert isinstance(out, str) and out
        # It surfaces the failure clearly: either the Unix hint (Windows) or
        # an exit-code marker.
        assert "unix" in out.lower() or "[EXIT CODE" in out


class TestExitCodeEndToEnd:
    def _run(self, tmp_path, code: str) -> str:
        bot = Agent(workspace=str(tmp_path))
        cmd = f'"{sys.executable}" -c "{code}"'
        return asyncio.run(bot._nlp_run({"command": cmd}))

    def test_nlp_run_surfaces_nonzero_exit_code(self, tmp_path) -> None:
        # python -c "sys.exit(3)" reliably returns exit code 3 on every OS.
        out = self._run(tmp_path, "import sys; sys.exit(3)")
        assert "[EXIT CODE: 3]" in out

    def test_nlp_run_success_has_no_marker(self, tmp_path) -> None:
        out = self._run(tmp_path, "print(12345)")
        assert "[EXIT CODE" not in out
        assert "12345" in out
