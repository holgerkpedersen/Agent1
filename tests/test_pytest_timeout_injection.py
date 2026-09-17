"""Regression tests: the run/tests tools honour PYTEST_FULL_SUITE_TIMEOUT.

History: the model routinely issued ``run(command="python -m pytest -q
--no-cov", timeout=600)``.  On slower machines the full suite needs longer than
600s, so the run tool killed the process tree mid-suite (and conftest's own
watchdog budget — read from ``PYTEST_FULL_SUITE_TIMEOUT`` in ``.env`` — was
never consulted).  The whole run was wasted every time.

Now a detected FULL pytest invocation takes at least
``PYTEST_FULL_SUITE_TIMEOUT`` seconds, uncapped by ``_MAX_RUN_TIMEOUT_S``.
Targeted runs keep the existing cap.
"""

from __future__ import annotations

import asyncio

import pytest

import agent
from agent import Agent


@pytest.fixture()
def bot() -> Agent:
    return Agent(workspace=".")


class _FakeProc:
    """Minimal Popen stand-in that records the timeout passed to communicate."""

    instances: list["_FakeProc"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.returncode = 0
        self.timeout_seen: float | None = None
        _FakeProc.instances.append(self)

    def communicate(self, timeout: float | None = None):
        self.timeout_seen = timeout
        return ("ok", "")


# ---------------------------------------------------------------------------
# _is_full_pytest_command
# ---------------------------------------------------------------------------


class TestFullPytestDetection:
    @pytest.mark.parametrize(
        "command",
        [
            "python -m pytest -q --no-cov",
            "python -m pytest -q --no-cov 2>&1",
            "pytest -q",
            "python -m pytest",
            "set PYTHONPATH=. && python -m pytest -q --no-cov",
            "python -m pytest -q >nul",
            "python -m pytest --ignore=tests/test_x.py -q",
        ],
    )
    def test_full_runs_detected(self, command: str) -> None:
        assert agent._is_full_pytest_command(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            "python -m pytest tests/test_hue_integration.py -q --no-cov",
            "python -m pytest tests -q",
            "pytest agent_core/tests -q",
            "python -m pytest tests/a.py tests/b.py",
            "echo pytest rules",
            "python -c \"print('pytest')\"",
        ],
    )
    def test_targeted_runs_not_misdetected(self, command: str) -> None:
        assert agent._is_full_pytest_command(command) is False


# ---------------------------------------------------------------------------
# _pytest_full_suite_timeout resolution
# ---------------------------------------------------------------------------


class TestBudgetResolution:
    def test_env_var_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PYTEST_FULL_SUITE_TIMEOUT", "1234")
        assert agent._pytest_full_suite_timeout() == 1234.0

    def test_repo_env_file_fallback(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("PYTEST_FULL_SUITE_TIMEOUT", raising=False)
        env = tmp_path / ".env"
        env.write_text(
            "# comment\nGITHUB_TOKEN=abc\nPYTEST_FULL_SUITE_TIMEOUT=1800\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(agent, "_ENV_FILE_PATH", str(env))
        assert agent._pytest_full_suite_timeout() == 1800.0

    def test_last_run_margin_matches_watchdog(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Budget mirrors conftest: max(floor, last * 1.2)."""
        monkeypatch.delenv("PYTEST_FULL_SUITE_TIMEOUT", raising=False)
        monkeypatch.delenv("PYTEST_LAST_FULL_RUN_SECONDS", raising=False)
        env = tmp_path / ".env"
        env.write_text(
            "PYTEST_FULL_SUITE_TIMEOUT=100\nPYTEST_LAST_FULL_RUN_SECONDS=1000\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(agent, "_ENV_FILE_PATH", str(env))
        assert agent._pytest_full_suite_timeout() == 1200.0

    def test_process_env_wins_over_repo_env(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PYTEST_FULL_SUITE_TIMEOUT", "777")
        env = tmp_path / ".env"
        env.write_text("PYTEST_FULL_SUITE_TIMEOUT=100\n", encoding="utf-8")
        monkeypatch.setattr(agent, "_ENV_FILE_PATH", str(env))
        assert agent._pytest_full_suite_timeout() == 777.0

    def test_missing_file_falls_back_to_cap(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("PYTEST_FULL_SUITE_TIMEOUT", raising=False)
        monkeypatch.setattr(agent, "_ENV_FILE_PATH", str(tmp_path / "absent.env"))
        assert agent._pytest_full_suite_timeout() == float(agent._MAX_RUN_TIMEOUT_S)

    def test_garbage_value_falls_back_to_cap(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PYTEST_FULL_SUITE_TIMEOUT", "not-a-number")
        assert agent._pytest_full_suite_timeout() == float(agent._MAX_RUN_TIMEOUT_S)


# ---------------------------------------------------------------------------
# _nlp_run wiring
# ---------------------------------------------------------------------------


class TestRunToolTimeout:
    def _run(self, bot: Agent, args: dict) -> _FakeProc:
        _FakeProc.instances.clear()

        async def go() -> str:
            return await bot._nlp_run(args)

        asyncio.run(go())
        assert _FakeProc.instances, "Popen was not called"
        return _FakeProc.instances[-1]

    def test_full_pytest_uses_injected_budget(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(agent.subprocess, "Popen", _FakeProc)
        monkeypatch.setattr(agent, "_pytest_full_suite_timeout", lambda: 1800.0)
        proc = self._run(
            bot,
            {"command": "python -m pytest -q --no-cov", "timeout": 600},
        )
        assert proc.timeout_seen == 1800.0

    def test_model_timeout_never_lowers_the_injected_budget(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(agent.subprocess, "Popen", _FakeProc)
        monkeypatch.setattr(agent, "_pytest_full_suite_timeout", lambda: 900.0)
        proc = self._run(
            bot,
            {"command": "python -m pytest -q --no-cov", "timeout": 30},
        )
        assert proc.timeout_seen == 900.0

    def test_non_pytest_command_still_capped(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(agent.subprocess, "Popen", _FakeProc)
        proc = self._run(bot, {"command": "echo hi", "timeout": 10 ** 6})
        assert proc.timeout_seen == agent._MAX_RUN_TIMEOUT_S

    def test_targeted_pytest_still_capped(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(agent.subprocess, "Popen", _FakeProc)
        monkeypatch.setattr(agent, "_pytest_full_suite_timeout", lambda: 1800.0)
        proc = self._run(
            bot,
            {
                "command": "python -m pytest tests/test_x.py -q",
                "timeout": 10 ** 6,
            },
        )
        assert proc.timeout_seen == agent._MAX_RUN_TIMEOUT_S
