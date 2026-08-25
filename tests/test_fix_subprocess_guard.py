"""Regression tests: fix_cmd child processes are wall-clock capped.

History (2026-08-25): a ``fix`` tool call appeared to hang the turn with no
model traffic because every mypy/py_compile child ran WITHOUT a timeout while
the NLP fix handler captured stdout — a cold mypy cache or a wedged child
(file lock from a Ctrl-C'd session) pinned the turn silently.  Now every
child goes through ``_run_capped``, which kills on expiry, reports on stderr
(NOT captured by redirect_stdout) and returns an empty failure result.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import agent_core.commands.fix_cmd as fix_cmd
from agent_core.commands.fix_cmd import (
    MYPY_TIMEOUT_S,
    PY_COMPILE_TIMEOUT_S,
    _run_capped,
)


class TestRunCapped:
    def test_quick_command_succeeds(self) -> None:
        r = _run_capped([sys.executable, "-c", "print('hello')"], timeout_s=30)
        assert r.returncode == 0
        assert "hello" in r.stdout

    def test_wedged_child_is_killed_not_hanging(self) -> None:
        r = _run_capped(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_s=1,
        )
        assert r.returncode != 0
        assert r.stdout == ""

    def test_timeout_reports_on_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        _run_capped(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_s=1,
        )
        err = capsys.readouterr().err
        assert "timed out" in err
        # The report must go to stderr so the NLP fix handler's stdout
        # capture cannot hide it.
        assert "[fix]" in err


class TestAllChildrenRoutedThroughCap:
    @staticmethod
    def _source() -> str:
        return Path(fix_cmd.__file__).read_text(encoding="utf-8")

    def test_only_helper_calls_subprocess_run(self) -> None:
        src = self._source()
        body = src.split("def _run_capped", 1)[1]
        uses = re.findall(r"subprocess\.run\(", body)
        assert len(uses) == 1, (
            "new direct subprocess.run calls bypass the wall-clock cap - "
            "route them through _run_capped"
        )

    def test_timeout_constants_exist(self) -> None:
        assert MYPY_TIMEOUT_S >= 60
        assert PY_COMPILE_TIMEOUT_S >= 10


@pytest.mark.parametrize("bad", ["timeout_s=None", "capture_output=False"])
def test_signature_keeps_capture_and_timeout(bad: str) -> None:
    """Guard against someone weakening the helper's contract."""
    import inspect

    sig = inspect.signature(_run_capped)
    param_names = list(sig.parameters)
    assert param_names[:3] == ["cmd", "cwd", "timeout_s"]
