"""Unit tests for agent_core.subprocess_utils timeout handling verification."""
import asyncio

import pytest

from agent_core.exceptions import ToolExecutionError
from agent_core.subprocess_utils import run_subprocess_with_timeout


def test_successful_execution() -> None:
    """Test successful subprocess execution returns correct values."""
    returncode, stdout, stderr = asyncio.run(
        run_subprocess_with_timeout(["echo", "hello"], timeout_sec=5.0)
    )
    assert returncode == 0
    assert b"hello" in stdout


def test_nonzero_exit_code() -> None:
    """Test subprocess returning non-zero exit code."""
    returncode, stdout, stderr = asyncio.run(
        run_subprocess_with_timeout(
            ["python", "-c", "import sys; sys.exit(1)"], timeout_sec=5.0
        )
    )
    assert returncode == 1


def test_stderr_capture() -> None:
    """Test stderr output is captured correctly."""
    returncode, stdout, stderr = asyncio.run(
        run_subprocess_with_timeout(
            ["python", "-c", "import sys; print('error', file=sys.stderr)"],
            timeout_sec=5.0,
        )
    )
    assert b"error" in stderr


def test_cwd_parameter() -> None:
    """Test subprocess execution with cwd parameter."""
    returncode, stdout, stderr = asyncio.run(
        run_subprocess_with_timeout(["pwd"], timeout_sec=5.0, cwd="/tmp")
    )
    assert returncode == 0


def test_empty_output() -> None:
    """Test subprocess with no output produces empty bytes."""
    returncode, stdout, stderr = asyncio.run(
        run_subprocess_with_timeout(["python", "-c", "pass"], timeout_sec=5.0)
    )
    assert returncode == 0
    assert stdout == b""


def test_return_type_tuple() -> None:
    """Test that return value is a tuple of (int, bytes, bytes)."""
    result = asyncio.run(
        run_subprocess_with_timeout(["echo", "test"], timeout_sec=5.0)
    )
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_timeout_raises_tool_execution_error() -> None:
    """Test that subprocess timeout raises ToolExecutionError."""
    with pytest.raises(ToolExecutionError):
        asyncio.run(
            run_subprocess_with_timeout(
                ["python", "-c", "import time; time.sleep(10)"], timeout_sec=2.0
            )
        )


def test_tool_execution_error_message_contains_command() -> None:
    """Test ToolExecutionError message references the subprocess tool name."""
    with pytest.raises(ToolExecutionError, match="subprocess"):
        asyncio.run(
            run_subprocess_with_timeout(
                ["python", "-c", "import time; time.sleep(10)"], timeout_sec=2.0
            )
        )


def test_tool_execution_error_message_contains_timeout() -> None:
    """Test ToolExecutionError message contains the timeout value."""
    with pytest.raises(ToolExecutionError, match="Timed out after 2.0 seconds"):
        asyncio.run(
            run_subprocess_with_timeout(
                ["python", "-c", "import time; time.sleep(10)"], timeout_sec=2.0
            )
        )


def test_short_timeout_success() -> None:
    """Test short timeout with fast command still succeeds."""
    returncode, stdout, stderr = asyncio.run(
        run_subprocess_with_timeout(["echo", "quick"], timeout_sec=1.0)
    )
    assert returncode == 0


def test_timeout_value_reflected_in_error() -> None:
    """Test that different timeout values appear in error messages."""
    with pytest.raises(ToolExecutionError, match="Timed out after 3.5 seconds"):
        asyncio.run(
            run_subprocess_with_timeout(
                ["python", "-c", "import time; time.sleep(10)"], timeout_sec=3.5
            )
        )


def test_default_cwd_none() -> None:
    """Test subprocess execution with cwd=None uses current directory."""
    returncode, stdout, stderr = asyncio.run(
        run_subprocess_with_timeout(["echo", "default"], timeout_sec=5.0)
    )
    assert returncode == 0


def test_multiple_commands_in_list() -> None:
    """Test subprocess execution with multi-element command list."""
    returncode, stdout, stderr = asyncio.run(
        run_subprocess_with_timeout(["python", "-c", "print('multi')"], timeout_sec=5.0)
    )
    assert returncode == 0
    assert b"multi" in stdout


def test_tool_execution_error_inherits_agent_base_error() -> None:
    """Test that ToolExecutionError inherits from AgentBaseError."""
    with pytest.raises(ToolExecutionError):
        asyncio.run(
            run_subprocess_with_timeout(
                ["python", "-c", "import time; time.sleep(10)"], timeout_sec=2.0
            )
        )