"""Tests for the LLM-free `run` REPL command."""
import asyncio
from unittest.mock import AsyncMock, patch

from agent_core.commands.run_cmd import RunCommand, _DEFAULT_TIMEOUT


class _FakeAgent:
    def __init__(self, result: str = "ok"):
        self._execute_tool_call = AsyncMock(return_value=result)


class TestRunCommand:
    def test_name_and_help(self):
        cmd = RunCommand()
        assert cmd.name == "run"
        assert "no LLM" in cmd.help_text

    def test_executes_via_guarded_tool_path(self):
        agent = _FakeAgent(result="hello world")
        cmd = RunCommand()
        with patch("builtins.print") as p:
            out = asyncio.run(cmd.execute(["python", "-m", "x", "--flag"], agent))
        agent._execute_tool_call.assert_awaited_once_with(
            "run", {"command": "python -m x --flag", "timeout": _DEFAULT_TIMEOUT}
        )
        assert out is True
        printed = " ".join(str(c.args) for c in p.call_args_list)
        assert "hello world" in printed

    def test_default_timeout_is_900(self):
        assert _DEFAULT_TIMEOUT == 900
        agent = _FakeAgent()
        cmd = RunCommand()
        with patch("builtins.print"):
            asyncio.run(cmd.execute(["python", "-m", "x"], agent))
        agent._execute_tool_call.assert_awaited_once_with(
            "run", {"command": "python -m x", "timeout": 900}
        )

    def test_timeout_flag_parsed(self):
        agent = _FakeAgent()
        cmd = RunCommand()
        with patch("builtins.print"):
            asyncio.run(cmd.execute(["--timeout", "900", "python", "-m", "x"], agent))
        agent._execute_tool_call.assert_awaited_once_with(
            "run", {"command": "python -m x", "timeout": 900}
        )

    def test_invalid_timeout_rejected(self):
        agent = _FakeAgent()
        cmd = RunCommand()
        with patch("builtins.print") as p:
            out = asyncio.run(cmd.execute(["--timeout", "abc", "x"], agent))
        agent._execute_tool_call.assert_not_awaited()
        assert out is True
        printed = " ".join(str(c.args) for c in p.call_args_list)
        assert "expects seconds" in printed

    def test_requires_command(self):
        agent = _FakeAgent()
        cmd = RunCommand()
        with patch("builtins.print"):
            out = asyncio.run(cmd.execute([], agent))
        agent._execute_tool_call.assert_not_awaited()
        assert out is True

    def test_timeout_flag_without_command_rejected(self):
        agent = _FakeAgent()
        cmd = RunCommand()
        with patch("builtins.print"):
            out = asyncio.run(cmd.execute(["--timeout", "60"], agent))
        agent._execute_tool_call.assert_not_awaited()
        assert out is True

    def test_blocked_command_error_passes_through(self):
        agent = _FakeAgent(result="Error: Dangerous command blocked")
        cmd = RunCommand()
        with patch("builtins.print") as p:
            asyncio.run(cmd.execute(["rm", "-rf", "/"], agent))
        agent._execute_tool_call.assert_awaited_once()
        printed = " ".join(str(c.args) for c in p.call_args_list)
        assert "blocked" in printed
