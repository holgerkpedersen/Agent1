"""Regression tests: one shell policy, shared by every execution path (plan #16).

Before this work there were TWO competing policies:

* the NLP ``run`` tool (``agent._nlp_run``) used a destructive-*blocklist*
  (``agent._DANGEROUS_SHELL_PATTERNS``) and executed with ``shell=True``;
* ``tool_router.ShellCommandHandler`` / ``tools/shell_ops.run_command`` used the
  ``agent_core.security.allowlist`` allow-list.

The plan converges them on "allow-list-with-fallbacks": the shared module owns
the policy (structural scan + destructive patterns + binary allow-list), and the
NLP path keeps its permissive fallback for non-allow-listed dev commands
(``findstr``/``more``/``pytest`` are not in ``SAFE_COMMANDS`` and must still run).

The hard constraint these tests pin: the NLP path must NOT start rejecting
metacharacters that merely appear INSIDE quotes — ``python -c "import sys;
sys.exit(3)"`` is a legitimate command whose exit code the tool reports.
"""

import asyncio
import subprocess
import sys

import pytest

from agent import Agent, _blocked_shell_command


class TestSingleSourceOfTruth:
    def test_destructive_patterns_live_in_the_shared_security_module(self) -> None:
        import agent
        from agent_core.security import allowlist

        # One owner for the policy: agent.py must not keep a private copy.
        assert agent._DANGEROUS_SHELL_PATTERNS is allowlist.DESTRUCTIVE_SHELL_PATTERNS

    def test_blocked_shell_command_delegates_to_shared_scanner(self) -> None:
        from agent_core.security.allowlist import find_destructive_shell_pattern

        # TIER-1 only: the NLP run tool shares the destructive block-list but
        # NOT the strict structural scan (it must still execute pipelines).
        for cmd in ("rm -rf /", "format c:", "python --version; rm -rf /"):
            shared = find_destructive_shell_pattern(cmd)
            assert _blocked_shell_command(cmd) == shared, cmd
            assert _blocked_shell_command(cmd) is not None, cmd

    def test_nlp_path_does_not_apply_the_strict_scan(self) -> None:
        """A pipe is NOT a violation for the NLP path (see the Hint feature)."""
        from agent_core.security.allowlist import scan_command

        pipe = "echo a 2>&1 | findstr a"
        assert scan_command(pipe) is not None  # strict callers reject it
        assert _blocked_shell_command(pipe) is None  # NLP dev-shell runs it


class TestSharedScanner:
    def test_rejects_unquoted_chaining(self) -> None:
        from agent_core.security.allowlist import scan_command

        assert scan_command("python --version; rm -rf /") is not None
        assert scan_command("cat a.txt && rm -rf /") is not None

    def test_rejects_destructive_patterns(self) -> None:
        from agent_core.security.allowlist import scan_command

        assert scan_command("rm -rf /") is not None
        assert scan_command("format c:") is not None

    def test_ignores_metacharacters_inside_quotes(self) -> None:
        """Quoted ``;``/``|`` are data, not shell operators."""
        from agent_core.security.allowlist import scan_command

        assert scan_command('python -c "import sys; sys.exit(3)"') is None
        assert scan_command('python -c "print(1 | 2)"') is None
        assert scan_command("git commit -m 'fix: a > b'") is None

    def test_allows_plain_commands(self) -> None:
        from agent_core.security.allowlist import scan_command

        assert scan_command("python -m pytest -q --no-cov") is None
        assert scan_command("git status") is None
        assert scan_command("echo hi") is None


class TestNlpRunUsesSharedPolicy:
    def test_blocks_unquoted_chaining(self, tmp_path) -> None:
        bot = Agent(workspace=str(tmp_path))
        out = asyncio.run(bot._nlp_run({"command": "python --version; rm -rf /"}))
        assert "blocked" in out.lower()

    def test_blocks_destructive_command(self, tmp_path) -> None:
        bot = Agent(workspace=str(tmp_path))
        out = asyncio.run(bot._nlp_run({"command": "format c:"}))
        assert "Dangerous command blocked" in out

    def test_quoted_semicolon_still_reports_exit_code(self, tmp_path) -> None:
        """Unification must not break quoted commands (allow-list-with-fallbacks)."""
        bot = Agent(workspace=str(tmp_path))
        cmd = f'"{sys.executable}" -c "import sys; sys.exit(3)"'
        out = asyncio.run(bot._nlp_run({"command": cmd}))
        assert "[EXIT CODE: 3]" in out

    def test_permissive_fallback_keeps_non_allowlisted_commands_running(
        self, tmp_path, monkeypatch
    ) -> None:
        """``findstr`` is NOT in SAFE_COMMANDS; the run tool must still run it."""
        calls: list = []

        class _Ok:
            returncode = 0

            def communicate(self, timeout=None):  # noqa: ANN001
                return ("ok", "")

        def fake_popen(cmd, **kwargs):  # noqa: ANN001
            calls.append((cmd, kwargs))
            return _Ok()

        # Build the agent BEFORE patching: Agent() itself shells out via
        # FullRunGate.changed_real_files, which needs the real subprocess.
        bot = Agent(workspace=str(tmp_path))
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        out = asyncio.run(bot._nlp_run({"command": "findstr /n x file.txt"}))
        assert calls, "Popen was not called — command was refused"
        assert "not allowed" not in out.lower()


class TestToolRouterDropsShellTrue:
    def test_tool_router_executes_without_a_shell(self, monkeypatch) -> None:
        """tool_router must not re-introduce shell interpretation (shell_ops parity)."""
        import tool_router

        seen: dict = {}

        class _Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_run(args, **kwargs):  # noqa: ANN001
            seen["args"] = args
            seen["kwargs"] = kwargs
            return _Result()

        monkeypatch.setattr(subprocess, "run", fake_run)
        handler = tool_router.ShellCommandHandler()
        result = handler.execute(
            tool_router.ShellCommandArgs(command="python --version")
        )
        assert result == {"stdout": "ok", "returncode": 0}
        assert seen["kwargs"]["shell"] is False
        assert isinstance(seen["args"], list), (
            "shell=False requires an argv list, not a command string"
        )

    def test_tool_router_still_rejects_disallowed_binary(self) -> None:
        import tool_router

        handler = tool_router.ShellCommandHandler()
        with pytest.raises(tool_router.ToolExecutionError):
            handler.execute(tool_router.ShellCommandArgs(command="curl http://x"))
