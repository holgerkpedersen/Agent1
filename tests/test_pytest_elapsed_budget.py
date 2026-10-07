"""A pytest run must not be killed at a guessed timeout.

Regression for ``run(command="python -m pytest --lf -q --no-cov", timeout=300)``
being killed after 300s: ``--lf`` is deliberately NOT a full run (the
full-run gate stays off), so it got neither the gate's budget nor the
full-run escape hatch and died at whatever number the model happened to pass.

Two rules now hold:

1. **The budget bounds any pytest run.**  A subset selector (``--lf`` /
   ``--nf`` / ``--testmon``) can select most of the suite, so it gets the same
   timeout floor as a full run.  Targeted path runs and non-pytest commands
   stay capped by ``_MAX_RUN_TIMEOUT_S``.
2. **The budget is measured, not guessed.**  A completed full run writes its
   elapsed seconds to ``PYTEST_LAST_FULL_RUN_SECONDS``; the next budget is
   ``max(PYTEST_FULL_SUITE_TIMEOUT, elapsed * 1.5)``.  A run killed by a
   timeout is recorded as a LOWER BOUND so the budget grows past the point
   that killed it instead of losing the same run again.  The floor itself
   comes from measurement (a real full run took 861.5s here, 608s in CI).
"""
from __future__ import annotations

import asyncio
import subprocess

import pytest

import agent
import agent_core.pytest_gate as agent_core_pytest_gate
from agent import Agent
from agent_core.pytest_gate import (
    DEFAULT_FULL_SUITE_TIMEOUT,
    full_suite_timeout,
    record_full_run_seconds,
)

#: Measured on this box (repo ``.env``, PYTEST_LAST_FULL_RUN_SECONDS).
MEASURED_FULL_RUN_SECONDS = 861.5


@pytest.fixture()
def bot(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Agent:
    # Never let a test write timings into the LIVE repo `.env`: the recorded
    # full-run measurement there is real machine history.
    monkeypatch.setattr(
        agent_core_pytest_gate, "_ENV_FILE_PATH", str(tmp_path / "pytest.env"),
    )
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


class _FakeProcTimesOut:
    """Popen stand-in whose first communicate() times out, second returns.

    Mirrors the real flow: the tool kills the tree on TimeoutExpired and then
    drains the pipes once more - that second call must succeed.
    """

    instances: list["_FakeProcTimesOut"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.returncode = -1
        self.timeout_seen: float | None = None
        self._raised = False
        _FakeProcTimesOut.instances.append(self)

    def communicate(self, timeout: float | None = None):
        self.timeout_seen = timeout
        if not self._raised:
            self._raised = True
            raise subprocess.TimeoutExpired(cmd="pytest", timeout=timeout)
        return ("", "")


def _run(bot: Agent, args: dict, fake=_FakeProc) -> _FakeProc:
    fake.instances.clear()

    async def go() -> str:
        return await bot._nlp_run(args)

    asyncio.run(go())
    assert fake.instances, "Popen was not called"
    return fake.instances[-1]


class TestSubsetSelectorGetsSuiteBudget:
    """The reported bug: ``pytest --lf`` died at the model's guessed 300s."""

    @pytest.mark.parametrize(
        "command",
        [
            "python -m pytest --lf -q --no-cov",
            "python -m pytest --nf -q --no-cov",
            "python -m pytest --testmon -q --no-cov",
            "python -m pytest --last-failed -q",
            "python -m pytest --lf -q --no-cov 2>&1 | findstr /c:\"passed\"",
        ],
    )
    def test_subset_selectors_get_the_budget(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch, command: str,
    ) -> None:
        monkeypatch.setattr(agent.subprocess, "Popen", _FakeProc)
        monkeypatch.setattr(agent, "_pytest_full_suite_timeout", lambda: 1800.0)
        proc = _run(bot, {"command": command, "timeout": 300})
        assert proc.timeout_seen == 1800.0

    def test_full_run_still_gets_the_budget(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(agent.subprocess, "Popen", _FakeProc)
        monkeypatch.setattr(agent, "_pytest_full_suite_timeout", lambda: 1800.0)
        monkeypatch.setenv(agent_core_pytest_gate.GATE_ENV, "off")
        proc = _run(bot, {"command": "python -m pytest -q --no-cov", "timeout": 300})
        assert proc.timeout_seen == 1800.0

    def test_targeted_run_stays_capped(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A single-file run must not inherit the whole-suite ceiling."""
        monkeypatch.setattr(agent.subprocess, "Popen", _FakeProc)
        monkeypatch.setattr(agent, "_pytest_full_suite_timeout", lambda: 1800.0)
        proc = _run(
            bot,
            {"command": "python -m pytest tests/test_x.py -q", "timeout": 10 ** 6},
        )
        assert proc.timeout_seen == agent._MAX_RUN_TIMEOUT_S

    def test_non_pytest_command_stays_capped(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(agent.subprocess, "Popen", _FakeProc)
        proc = _run(bot, {"command": "echo hi", "timeout": 10 ** 6})
        assert proc.timeout_seen == agent._MAX_RUN_TIMEOUT_S

    def test_timeout_message_names_the_budget(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(agent.subprocess, "Popen", _FakeProcTimesOut)
        monkeypatch.setattr(agent, "_kill_process_tree", lambda proc: None)
        monkeypatch.setattr(agent, "_pytest_full_suite_timeout", lambda: 1800.0)

        async def go() -> str:
            return await bot._nlp_run(
                {"command": "python -m pytest --lf -q --no-cov", "timeout": 300},
            )

        text = asyncio.run(go())
        assert "Measured full-suite budget" in text
        assert "1800s" in text


class TestBudgetIsMeasuredNotGuessed:
    def test_measurement_wins_over_the_floor(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("PYTEST_FULL_SUITE_TIMEOUT", raising=False)
        monkeypatch.delenv("PYTEST_LAST_FULL_RUN_SECONDS", raising=False)
        monkeypatch.setattr(
            agent_core_pytest_gate, "_ENV_FILE_PATH", str(tmp_path / ".env"),
        )
        (tmp_path / ".env").write_text(
            "PYTEST_FULL_SUITE_TIMEOUT=100\nPYTEST_LAST_FULL_RUN_SECONDS=1000\n",
            encoding="utf-8",
        )
        assert full_suite_timeout() == 1500.0  # 1000 * 1.5

    def test_floor_wins_when_nothing_measured(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PYTEST_FULL_SUITE_TIMEOUT", "700")
        monkeypatch.delenv("PYTEST_LAST_FULL_RUN_SECONDS", raising=False)
        monkeypatch.setattr(
            agent_core_pytest_gate, "_ENV_FILE_PATH", str(tmp_path / "absent.env"),
        )
        assert full_suite_timeout() == 700.0

    def test_default_floor_exceeds_the_real_run_cost(self) -> None:
        """The old 600s floor was below the measured 861.5s full run."""
        assert DEFAULT_FULL_SUITE_TIMEOUT > MEASURED_FULL_RUN_SECONDS
        assert agent._FULL_SUITE_MARGIN == agent_core_pytest_gate.FULL_SUITE_MARGIN


class TestElapsedRecording:
    def test_completed_run_overwrites_the_measurement(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = tmp_path / ".env"
        monkeypatch.setattr(
            agent_core_pytest_gate, "_ENV_FILE_PATH", str(env),
        )
        record_full_run_seconds(500.0)
        record_full_run_seconds(321.5)
        assert "PYTEST_LAST_FULL_RUN_SECONDS=321.5" in env.read_text(encoding="utf-8")

    def test_killed_run_grows_the_budget(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = tmp_path / ".env"
        monkeypatch.setattr(
            agent_core_pytest_gate, "_ENV_FILE_PATH", str(env),
        )
        record_full_run_seconds(200.0)
        # Killed at 800s: the true cost is >= 800, so the budget must grow.
        record_full_run_seconds(800.0, completed=False)
        assert "PYTEST_LAST_FULL_RUN_SECONDS=800.0" in env.read_text(encoding="utf-8")

    def test_killed_run_never_shrinks_the_budget(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = tmp_path / ".env"
        monkeypatch.setattr(
            agent_core_pytest_gate, "_ENV_FILE_PATH", str(env),
        )
        record_full_run_seconds(900.0)
        record_full_run_seconds(120.0, completed=False)
        assert "PYTEST_LAST_FULL_RUN_SECONDS=900.0" in env.read_text(encoding="utf-8")

    def test_recorded_value_drives_the_next_budget(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("PYTEST_FULL_SUITE_TIMEOUT", raising=False)
        monkeypatch.delenv("PYTEST_LAST_FULL_RUN_SECONDS", raising=False)
        monkeypatch.setattr(
            agent_core_pytest_gate, "_ENV_FILE_PATH", str(tmp_path / ".env"),
        )
        record_full_run_seconds(1000.0)
        assert full_suite_timeout() == 1500.0

    def test_recording_preserves_other_env_keys(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = tmp_path / ".env"
        monkeypatch.setattr(
            agent_core_pytest_gate, "_ENV_FILE_PATH", str(env),
        )
        env.write_text(
            "# comment\nGITHUB_TOKEN=secret\nPYTEST_FULL_SUITE_TIMEOUT=2400\n",
            encoding="utf-8",
        )
        record_full_run_seconds(300.0)
        text = env.read_text(encoding="utf-8")
        assert "GITHUB_TOKEN=secret" in text
        assert "PYTEST_FULL_SUITE_TIMEOUT=2400" in text
        assert "# comment" in text
        assert "PYTEST_LAST_FULL_RUN_SECONDS=300.0" in text


class TestRunToolRecordsElapsed:
    def test_completed_full_run_records_its_elapsed(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: list[float] = []
        monkeypatch.setattr(agent.subprocess, "Popen", _FakeProc)
        monkeypatch.setenv(agent_core_pytest_gate.GATE_ENV, "off")
        monkeypatch.setattr(
            agent,
            "record_full_run_seconds",
            lambda seconds, completed=True: seen.append(seconds),
        )
        _FakeProc.instances.clear()

        async def go() -> str:
            return await bot._nlp_run(
                {"command": "python -m pytest -q --no-cov", "timeout": 300},
            )

        asyncio.run(go())
        assert seen, "elapsed was not recorded for a completed full run"
        assert seen[0] >= 0.0

    def test_killed_full_run_records_a_lower_bound(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: list[tuple[float, bool]] = []
        monkeypatch.setattr(agent.subprocess, "Popen", _FakeProcTimesOut)
        monkeypatch.setattr(agent, "_kill_process_tree", lambda proc: None)
        monkeypatch.setenv(agent_core_pytest_gate.GATE_ENV, "off")
        monkeypatch.setattr(
            agent,
            "record_full_run_seconds",
            lambda seconds, completed=True: seen.append((seconds, completed)),
        )

        async def go() -> str:
            return await bot._nlp_run(
                {"command": "python -m pytest -q --no-cov", "timeout": 300},
            )

        asyncio.run(go())
        assert seen, "a killed full run must still record its lower bound"
        assert seen[0][1] is False, "a killed run is a lower bound, not a measurement"

    def test_non_pytest_run_records_nothing(
        self, bot: Agent, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: list[float] = []
        monkeypatch.setattr(agent.subprocess, "Popen", _FakeProc)
        monkeypatch.setattr(
            agent,
            "record_full_run_seconds",
            lambda seconds, completed=True: seen.append(seconds),
        )
        _run(bot, {"command": "echo hi", "timeout": 10})
        assert seen == []
