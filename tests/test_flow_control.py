"""Tests for the flow-stop controls: stop tokens, Ctrl+C, and the
"issues this patch targets" display at patch-presentation time."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reset_stop_flag():
    from agent_core.commands.base import clear_stop
    clear_stop()
    yield
    clear_stop()


class TestReadChoice:
    def test_yes(self):
        from agent_core.commands.base import read_choice
        with patch("builtins.input", return_value="y"):
            assert read_choice("> ") is True

    def test_yes_caps(self):
        from agent_core.commands.base import read_choice
        with patch("builtins.input", return_value="YES"):
            assert read_choice("> ") is True

    def test_no(self):
        from agent_core.commands.base import read_choice, stop_requested
        with patch("builtins.input", return_value="n"):
            assert read_choice("> ") is False
        assert stop_requested() is False

    def test_enter(self):
        from agent_core.commands.base import read_choice
        with patch("builtins.input", return_value=""):
            assert read_choice("> ") is False

    @pytest.mark.parametrize("token", ["s", "stop", "q", "quit", "abort", "x", "Quit"])
    def test_stop_tokens_decline_and_request_stop(self, token):
        from agent_core.commands.base import read_choice, stop_requested
        with patch("builtins.input", return_value=token):
            assert read_choice("> ") is False
        assert stop_requested() is True

    def test_eof_declines_without_stop(self):
        from agent_core.commands.base import read_choice, stop_requested
        with patch("builtins.input", side_effect=EOFError):
            assert read_choice("> ") is False
        assert stop_requested() is False

    def test_keyboard_interrupt_declines_and_requests_stop(self):
        from agent_core.commands.base import read_choice, stop_requested
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert read_choice("> ") is False
        assert stop_requested() is True


class TestReadInput:
    def test_keyboard_interrupt_requests_stop(self):
        from agent_core.commands.base import read_input, stop_requested
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert read_input() == ""
        assert stop_requested() is True

    def test_eof_returns_empty_without_stop(self):
        from agent_core.commands.base import read_input, stop_requested
        with patch("builtins.input", side_effect=EOFError):
            assert read_input() == ""
        assert stop_requested() is False


class TestSaveFilePy:
    def test_stop_token_does_not_write(self, tmp_path, capsys):
        from agent_core.commands.base import save_file_py, stop_requested
        target = tmp_path / "a.py"
        target.write_text("old = 1\n", encoding="utf-8")
        with patch("builtins.input", return_value="q"):
            written = save_file_py(str(target), "new = 1\n", auto_yes=False)
        assert written is False
        assert target.read_text(encoding="utf-8") == "old = 1\n"
        assert stop_requested() is True
        assert "Stopping the flow" in capsys.readouterr().out


class TestChatStoppable:
    def test_cancelled_request_converts_to_flow_stopped(self):
        from agent_core.commands.base import chat_stoppable, FlowStopped, stop_requested

        async def boom(msgs, **kw):
            raise asyncio.CancelledError()

        async def run():
            try:
                await chat_stoppable(boom)([{}])
            except FlowStopped:
                return True
            return False

        assert asyncio.run(run()) is True
        assert stop_requested() is True

    def test_normal_result_passes_through(self):
        from agent_core.commands.base import chat_stoppable, stop_requested

        async def ok(msgs, **kw):
            return "fine"

        assert asyncio.run(chat_stoppable(ok)([{}])) == "fine"
        assert stop_requested() is False

    def test_flow_stopped_not_swallowed_by_exception_handlers(self):
        from agent_core.commands.base import FlowStopped
        with pytest.raises(FlowStopped):
            try:
                raise FlowStopped()
            except Exception:  # noqa: BLE001 - must NOT catch BaseException
                pytest.fail("except Exception swallowed FlowStopped")


class TestApplyPatchReason:
    def test_reason_shown_above_diff(self, tmp_path, capsys):
        from agent_core.commands.fix_cmd import FixCommand
        fc = FixCommand()
        target = tmp_path / "test.py"
        target.write_text("old = 1\n", encoding="utf-8")
        with patch("builtins.input", return_value="n"):
            fc._apply_patch(
                "@@ -1 +1 @@\n- old = 1\n+ new = 1\n",
                str(target), str(tmp_path),
                reason="test.py:1: error: assignment to undefined name  [name-defined]",
            )
        out = capsys.readouterr().out
        assert "Reason:" in out
        assert "name-defined" in out
        assert target.read_text(encoding="utf-8") == "old = 1\n"


class TestApplyFixBlocksContext:
    def test_targeted_issues_printed_above_patch(self, tmp_path, capsys):
        from agent_core.commands.base import stop_requested
        from agent_core.commands.fix_cmd import FixCommand
        target = tmp_path / "t.py"
        target.write_text("def f():\n    pass\n", encoding="utf-8")
        response = (
            "[PATCH: t.py]\n"
            "@@ -1,2 +1,2 @@\n"
            " def f():\n"
            "-    pass\n"
            "+    return 1\n"
        )
        fc = FixCommand()
        with patch("builtins.input", return_value="q"):
            n, failures = fc._apply_fix_blocks(
                response, str(tmp_path), "t.py", auto_yes=False,
                context_errors=["t.py:2: error: Missing return statement  [return]"],
            )
        out = capsys.readouterr().out
        assert "Issues this patch targets" in out
        assert "Missing return statement" in out
        assert n == 0
        assert stop_requested() is True
        assert target.read_text(encoding="utf-8") == "def f():\n    pass\n"


class TestShowPatchVerdict:
    """The pre-prompt verdict verifies a candidate patch against mypy so the
    y/N decision is informed (fixes N/M targeted, introduces K new)."""

    def test_verdict_printed_before_prompt(self, tmp_path, capsys):
        from agent_core.commands.fix_cmd import FixCommand
        target = tmp_path / "t.py"
        target.write_text("x: int = 'not an int'\n", encoding="utf-8")
        fc = FixCommand()
        fc._show_patch_verdict(
            str(target),
            "x: int = 5\n",
            "t.py",
            ["t.py:1: error: Incompatible types in assignment (expression has type \"str\", variable has type \"int\")  [assignment]"],
            str(tmp_path),
        )
        out = capsys.readouterr().out
        assert "[verify]" in out
        assert "targeted error" in out
        assert "→" in out

    def test_verdict_detects_new_errors(self, tmp_path, capsys):
        """A patch that merely silences one error but breaks another must be
        flagged as introducing new errors."""
        from agent_core.commands.fix_cmd import FixCommand
        target = tmp_path / "t.py"
        target.write_text("x: int = 'bad'\n", encoding="utf-8")
        fc = FixCommand()
        fc._show_patch_verdict(
            str(target),
            "x: int = 'bad'  # type: ignore[assignment]\n",
            "t.py",
            ["t.py:1: error: Incompatible types in assignment (expression has type \"str\", variable has type \"int\")  [assignment]"],
            str(tmp_path),
        )
        out = capsys.readouterr().out
        assert "introduces" in out or "→ y" in out

    def test_verdict_ignores_pre_existing_errors(self, tmp_path, capsys):
        """A patch that fixes the targeted error but leaves OTHER pre-existing
        errors in the file must NOT be flagged as introducing them."""
        from agent_core.commands.fix_cmd import FixCommand
        target = tmp_path / "t.py"
        target.write_text(
            "def f(a: int | None) -> int:\n"
            "    return a - 1\n"
            "def g() -> None:\n"
            "    x = nonexistent_attr\n"
            "    print(x)\n",
            encoding="utf-8",
        )
        fc = FixCommand()
        patched = (
            "def f(a: int | None) -> int:\n"
            "    assert a is not None\n"
            "    return a - 1\n"
            "def g() -> None:\n"
            "    x = nonexistent_attr\n"
            "    print(x)\n"
        )
        fc._show_patch_verdict(
            str(target),
            patched,
            "t.py",
            ['t.py:2: error: Unsupported operand types for - ("None" and "int")  [operator]'],
            str(tmp_path),
        )
        out = capsys.readouterr().out
        assert "fixes 1/1 targeted" in out
        assert "introduces" not in out
        assert "→ y" in out
