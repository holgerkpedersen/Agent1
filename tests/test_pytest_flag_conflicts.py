"""Regression: known-bad pytest flag combos are refused before Popen.

History (2026-10-19): the chat model issued ``run(command="python -m pytest
--testmon -q --no-cov -p no:cacheprovider --tb=short")``.
pytest-testmon 2.2.0's ``pytest_configure`` does a hard ``options["lf"]``
lookup; ``lf`` is registered by the *cacheprovider* plugin, which
``-p no:cacheprovider`` removes — so the KeyError is re-raised by pytest as
``INTERNALERROR`` before any test runs.  Not related to ``--tb=short``.  The
run tool now refuses the combo with an actionable message instead of burning
a turn on a third-party stack trace.
"""

from __future__ import annotations

import asyncio

import pytest

import agent
from agent import Agent
from agent_core.pytest_cmd_policy import (
    _testmon_cacheprovider_conflict,
    pytest_flag_conflict,
)


# ---------------------------------------------------------------------------
# pytest_flag_conflict — pure detector
# ---------------------------------------------------------------------------


class TestPytestFlagConflict:
    @pytest.mark.parametrize(
        "command",
        [
            "python -m pytest --testmon -q --no-cov -p no:cacheprovider --tb=short",
            "python -m pytest -p no:cacheprovider --testmon -q",
            "python -m pytest --testmon -p=no:cacheprovider",
            "python -m pytest --testmon -pno:cacheprovider",
            "cd /d C:\\Dev\\Agent1 && python -m pytest --testmon "
            "-p no:cacheprovider -q",
        ],
    )
    def test_bad_combo_detected(self, command: str) -> None:
        msg = pytest_flag_conflict(command)
        assert "INTERNALERROR" in msg
        assert "no:cacheprovider" in msg

    @pytest.mark.parametrize(
        "command",
        [
            # Each flag alone is fine; cacheprovider does not conflict with
            # testmon, so -p no:cacheprovider is only bad IN COMBINATION.
            "python -m pytest --testmon -q --no-cov",
            "python -m pytest -q -p no:cacheprovider --no-cov",
            "python -m pytest tests/test_x.py -q",
            "python -m pytest --lf -q",
            "echo pytest rules",
            "python -c \"print('pytest')\"",
            # --tb=short is irrelevant to the crash; alone it is obviously fine.
            "python -m pytest --tb=short -q",
        ],
    )
    def test_ok_commands_return_empty(self, command: str) -> None:
        assert pytest_flag_conflict(command) == ""

    def test_non_pytest_segment_ignored(self) -> None:
        """A bad combo only matters inside a segment that actually runs pytest."""
        assert pytest_flag_conflict("echo --testmon -p no:cacheprovider") == ""
        assert (
            pytest_flag_conflict("git status && echo --testmon -p no:cacheprovider")
            == ""
        )

    def test_direct_token_helper(self) -> None:
        assert _testmon_cacheprovider_conflict(["--testmon"]) is None
        assert _testmon_cacheprovider_conflict(["-p", "no:cacheprovider"]) is None
        assert _testmon_cacheprovider_conflict(["--testmon", "-p", "no:cacheprovider"])


# ---------------------------------------------------------------------------
# _nlp_run wiring — the refusal must fire BEFORE subprocess.Popen
# ---------------------------------------------------------------------------


class _ExplodingPopen:
    """Popen stand-in that fails the test if it is ever reached."""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - must not run
        raise AssertionError("Popen must not be called for a refused command")


class TestRunToolRefusesBadCombo:
    @pytest.fixture()
    def bot(self) -> Agent:
        return Agent(workspace=".")

    def test_run_refuses_and_never_spawns(
        self,
        bot: Agent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(agent.subprocess, "Popen", _ExplodingPopen)

        async def go() -> str:
            return await bot._nlp_run(
                {
                    "command": (
                        "python -m pytest --testmon -q --no-cov "
                        "-p no:cacheprovider --tb=short"
                    ),
                    "timeout": 600,
                }
            )

        out = asyncio.run(go())
        assert out.startswith("Error: ")
        assert "INTERNALERROR" in out
        assert "no:cacheprovider" in out

    def test_run_still_executes_safe_pytest(
        self,
        bot: Agent,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Legitimate --testmon (or -p no:cacheprovider) runs still execute."""
        calls: list[float] = []

        class _FakeProc:
            def __init__(self, *args, **kwargs) -> None:
                self.returncode = 0

            def communicate(self, timeout: float | None = None):
                calls.append(timeout or 0.0)
                return ("ok", "")

        monkeypatch.setattr(agent.subprocess, "Popen", _FakeProc)
        monkeypatch.setattr(agent, "_pytest_full_suite_timeout", lambda: 900.0)
        monkeypatch.setenv("AGENT_PYTEST_FULL_RUN_GATE", "off")

        async def go() -> str:
            return await bot._nlp_run(
                {
                    "command": "python -m pytest --testmon -q --no-cov",
                    "timeout": 300,
                }
            )

        out = asyncio.run(go())
        assert "ok" in out
        assert calls, "safe --testmon command must still reach Popen"
