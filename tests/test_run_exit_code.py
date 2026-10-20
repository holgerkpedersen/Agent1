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

    def test_none_output_no_stderr_returns_empty(self) -> None:
        # Regression: communicate() returns None stdout when a process
        # writes nothing; _shape_run_stderr must not crash on None.
        out = _shape_run_stderr(None, None, 0)
        assert out == ""

    def test_none_output_with_exit_code_appends_marker(self) -> None:
        out = _shape_run_stderr(None, None, 1)
        assert out == "\n[EXIT CODE: 1]"

    def test_none_output_with_stderr_appends_both(self) -> None:
        out = _shape_run_stderr("bad", None, 2)
        assert "bad" in out
        assert "[EXIT CODE: 2]" in out


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

    def test_nlp_run_unknown_command_exit_code_and_hint(self, tmp_path) -> None:
        # CommandNotFoundException is a CMDLET-level failure: $LASTEXITCODE
        # stays $null, so the marker snippet must fall back to $? — otherwise
        # the code was 0 (the trailing Write-Host's) and the Hint never fired.
        import os
        import pytest
        from agent_core.shell_utils import detect_shell
        if os.name != "nt" or detect_shell() != "powershell":
            pytest.skip("PowerShell session only")
        bot = Agent(workspace=str(tmp_path))
        out = asyncio.run(
            bot._nlp_run({"command": "definitely_not_a_real_command_9k2f"}))
        assert "[EXIT CODE: 1]" in out
        assert "Hint" in out
        assert "__AGENT1_EXIT__" not in out


class TestPsExitMarker:
    """parse_ps_exit_marker: strip the PowerShell marker, recover the code."""

    def test_own_line_marker_stripped_with_code(self) -> None:
        from agent_core.shell_utils import parse_ps_exit_marker
        out, code = parse_ps_exit_marker("hello\n__AGENT1_EXIT__:3\n")
        assert out == "hello\n"
        assert code == 3

    def test_first_line_marker_stripped(self) -> None:
        from agent_core.shell_utils import parse_ps_exit_marker
        out, code = parse_ps_exit_marker("__AGENT1_EXIT__:1\nfoo")
        assert out == "foo"
        assert code == 1

    def test_marker_glued_to_output_keeps_output(self) -> None:
        # A child that writes bytes WITHOUT a trailing newline leaves the
        # marker glued on the same line ("中__AGENT1_EXIT__:0").  Only the
        # token+digits may be removed — never the content before them.
        from agent_core.shell_utils import parse_ps_exit_marker
        out, code = parse_ps_exit_marker("中__AGENT1_EXIT__:0\n")
        assert "中" in out
        assert code == 0

    def test_empty_marker_line_stripped_code_none(self) -> None:
        # $LASTEXITCODE was $null (no digits): the line must still go.
        from agent_core.shell_utils import parse_ps_exit_marker
        out, code = parse_ps_exit_marker("err text\n__AGENT1_EXIT__:\n")
        assert "__AGENT1_EXIT__" not in out
        assert "err text" in out
        assert code is None

    def test_no_marker_passthrough(self) -> None:
        from agent_core.shell_utils import parse_ps_exit_marker
        assert parse_ps_exit_marker("plain") == ("plain", None)


class TestShellHintWordings:
    """The command-not-found Hint must fire for BOTH cmd.exe and PowerShell,
    including PowerShell's console-width wraps that split wording mid-word."""

    def test_powershell_wording_hint(self) -> None:
        err = ("The term 'nope' is not recognized as the name of a cmdlet, "
               "function, script file, or operable program.")
        out = _shape_run_stderr(err, "", 1)
        assert "Hint" in out

    def test_powershell_wrapped_midword_hint(self) -> None:
        # PowerShell wraps stderr at console width: "of a c\nmdlet".
        err = ("The term 'nope' is not recognized as the name of a c\n"
               "mdlet, function, script file, or operable program.")
        out = _shape_run_stderr(err, "", 1)
        assert "Hint" in out

    def test_cmd_exe_wording_still_hinted(self) -> None:
        err = ("'nope' is not recognized as an internal or external command, "
               "operable program or batch file.")
        out = _shape_run_stderr(err, "", 1)
        assert "Hint" in out

    def test_no_hint_on_success(self) -> None:
        err = ("The term 'nope' is not recognized as the name of a cmdlet.")
        out = _shape_run_stderr(err, "", 0)
        assert "Hint" not in out
