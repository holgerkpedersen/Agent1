"""Security hardening tests (plan tasks 1, 2, 5, 6):
path containment, shell metacharacter rejection, shell_ops delegation."""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from agent_core.security.allowlist import (
    find_unsafe_shell_pattern,
    is_command_allowed,
)
from agent_core.security.path_utils import SecurityViolationError, normalize_path
from agent_core.tools.shell_ops import run_command


class TestPathContainment:
    def _root(self):
        return Path(tempfile.mkdtemp(prefix="sec_"))

    def test_normal_relative_path_ok(self):
        root = self._root()
        (root / "sub").mkdir()
        assert normalize_path(root, "sub/file.txt") == (root / "sub/file.txt").resolve()

    def test_dotdot_escape_rejected(self):
        root = self._root()
        with pytest.raises(SecurityViolationError):
            normalize_path(root, "../outside.txt")

    def test_absolute_escape_rejected(self):
        root = self._root()
        with pytest.raises(SecurityViolationError):
            normalize_path(root, str(root.parent / "evil.txt"))

    def test_symlink_traversal_rejected(self):
        root = self._root()
        outside = root.parent / "symlink_target.txt"
        outside.write_text("x", encoding="utf-8")
        try:
            link = root / "link"
            link.symlink_to(outside, target_is_directory=False)
        except OSError:
            pytest.skip("symlinks not available on this platform")
        with pytest.raises(SecurityViolationError):
            normalize_path(root, "link")

    def test_empty_path_rejected(self):
        root = self._root()
        with pytest.raises(SecurityViolationError):
            normalize_path(root, "")


class TestShellMetacharRejection:
    def test_safe_command_passes(self):
        assert find_unsafe_shell_pattern("python -m pytest tests") is None
        assert find_unsafe_shell_pattern("git status") is None

    @pytest.mark.parametrize(
        "cmd",
        [
            "python; rm -rf /",
            "python && echo hi",
            "git status || echo fail",
            "cat file | grep x",
            "python > out.txt",
            "dir < in.txt",
            "echo `whoami`",
            "python -c \"$(whoami)\"",
            "echo a\nrm -rf /",
        ],
    )
    def test_unsafe_patterns_rejected(self, cmd):
        assert find_unsafe_shell_pattern(cmd) is not None


class TestShellOps:
    def test_allowlisted_command_runs(self, tmp_path):
        rc, out, err = run_command(tmp_path, "python --version")
        assert rc == 0
        assert "Python" in out

    def test_disallowed_binary_rejected(self, tmp_path):
        rc, out, err = run_command(tmp_path, "rm -rf /")
        assert rc == 1
        assert "not allowed" in err

    def test_chained_command_rejected_before_allowlist(self, tmp_path):
        # Even an allow-listed binary cannot smuggle a chained command.
        rc, out, err = run_command(tmp_path, "python --version; rm -rf /")
        assert rc == 1
        assert "chaining" in err or "rejected" in err or "separator" in err

    def test_pipe_rejected(self, tmp_path):
        rc, out, err = run_command(tmp_path, "python --version | head -1")
        assert rc == 1
        assert "pipe" in err

    def test_redirection_rejected(self, tmp_path):
        rc, out, err = run_command(tmp_path, "python --version > out.txt")
        assert rc == 1
        assert "redirection" in err

    def test_empty_command(self, tmp_path):
        rc, out, err = run_command(tmp_path, "   ")
        assert rc == 1


class TestToolRouterDelegation:
    def test_handler_uses_shared_allowlist(self):
        from tool_router import ShellCommandHandler, ShellCommandArgs, ToolExecutionError
        handler = ShellCommandHandler()
        with pytest.raises(ToolExecutionError):
            handler.execute(ShellCommandArgs(command="rm -rf /"))

    def test_handler_rejects_chaining(self):
        from tool_router import ShellCommandHandler, ShellCommandArgs, ToolExecutionError
        handler = ShellCommandHandler()
        with pytest.raises(ToolExecutionError):
            handler.execute(ShellCommandArgs(command="python; rm -rf /"))

    def test_handler_allows_allowlisted_binary(self):
        from tool_router import ShellCommandHandler, ShellCommandArgs
        handler = ShellCommandHandler()
        result = handler.execute(ShellCommandArgs(command="python --version"))
        assert isinstance(result, dict)
        assert result["returncode"] == 0

    def test_no_duplicate_allowlist_state(self):
        import inspect
        from tool_router import ShellCommandHandler
        src = inspect.getsource(ShellCommandHandler)
        assert "_allowed_binaries" not in src
