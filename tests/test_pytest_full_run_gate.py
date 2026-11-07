"""Regression tests: FULL pytest runs must be EARNED and RARE.

History (2026-09-30): answering the read-only question "what's next in the
repo?" ended with ``run(command="python -m pytest -q --no-header --no-cov")`` -
a ~2 minute full-suite run - even though the only file the turn had touched was
a scratch probe (``_tmp_probe.py``).  Two rules now guard this
(``agent_core.pytest_gate``):

1. a full run is refused unless a REAL tracked source file changed since the
   session baseline (scratch paths never count);
2. at most ``AGENT_MAX_FULL_PYTEST_RUNS`` (default 1) full run per session, so
   the one run is naturally spent near the END of the work.
"""

from __future__ import annotations

import os
import subprocess

import pytest

import agent
from agent import Agent
from agent_core import pytest_gate
from agent_core.pytest_gate import (
    DEFAULT_MAX_FULL_RUNS,
    GATE_ENV,
    MAX_FULL_RUNS_ENV,
    FullRunGate,
    changed_real_files,
    is_scratch_path,
    is_testable_source,
    max_full_runs_from_env,
)


# ---------------------------------------------------------------------------
# path classification
# ---------------------------------------------------------------------------


class TestPathClassification:
    @pytest.mark.parametrize(
        "path",
        [
            "_tmp_probe.py",
            "agent_core/_tmp_x.py",
            "tests/tmp_helper.py",
            "scratch_plan.md",
            "tmp/output.txt",
            "temp/notes.md",
            "agent.py.bak",
            "build.py.orig",
            "trace.log",
            "reports/traces/a.jsonl",
            ".docs/2026-01-01/plan.md",
            "backups/run/x.py",
            "agent_core/__pycache__/x.pyc",
        ],
    )
    def test_scratch_paths_excluded(self, path: str) -> None:
        assert is_scratch_path(path) is True
        assert is_testable_source(path) is False

    @pytest.mark.parametrize(
        "path",
        [
            "agent.py",
            "agent_core/llm/tool_loop.py",
            "pyproject.toml",
            "tests/test_x.py",
        ],
    )
    def test_real_paths_count(self, path: str) -> None:
        assert is_scratch_path(path) is False
        assert is_testable_source(path) is True

    def test_non_testable_extension_ignored(self) -> None:
        assert is_testable_source("logo.svg") is False


# ---------------------------------------------------------------------------
# git-derived change set
# ---------------------------------------------------------------------------


class _Git:
    """Stub for ``git status --porcelain -uall``."""

    def __init__(self, output: str = "", returncode: int = 0) -> None:
        self.output = output
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):  # noqa: ANN001 - subprocess.run stub
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, self.returncode, self.output, "")

    def install(self, monkeypatch) -> None:  # noqa: ANN001
        monkeypatch.setattr("agent_core.pytest_gate.subprocess.run", self)


class TestChangedRealFiles:
    def test_scratch_and_untracked_excluded(self, monkeypatch) -> None:
        _Git("?? _tmp_probe.py\n M agent_core/llm/tool_loop.py\n").install(monkeypatch)
        assert changed_real_files(".") == {"agent_core/llm/tool_loop.py"}

    def test_rename_uses_new_path(self, monkeypatch) -> None:
        _Git("R  agent_core/old.py -> agent_core/new.py\n").install(monkeypatch)
        assert changed_real_files(".") == {"agent_core/new.py"}

    def test_deleted_file_counts(self, monkeypatch) -> None:
        _Git(" D agent.py\n").install(monkeypatch)
        assert changed_real_files(".") == {"agent.py"}

    def test_git_failure_is_soft(self, monkeypatch) -> None:
        _Git("", returncode=128).install(monkeypatch)
        assert changed_real_files(".") == set()

    def test_missing_git_is_soft(self, monkeypatch) -> None:
        def _boom(*a, **k):  # noqa: ANN001
            raise OSError("git not found")

        monkeypatch.setattr("agent_core.pytest_gate.subprocess.run", _boom)
        assert changed_real_files(".") == set()


class TestSubsetFlagsAreNotFullRuns:
    """`--lf` / `--nf` / `--testmon` run a SUBSET - they must never be treated as
    the full suite (they would get the suite budget and the full-run gate)."""

    @pytest.mark.parametrize(
        "command",
        [
            "python -m pytest --lf -q --no-cov",
            "python -m pytest --lfnf -q --no-cov",
            "python -m pytest --nf -q",
            "python -m pytest --new-first",
            "python -m pytest --failed-first -q",
            "python -m pytest --testmon -q --no-cov",
            "pytest -q --lf tests/",
        ],
    )
    def test_tool_classifier_says_not_full(self, command: str) -> None:
        assert agent._is_full_pytest_command(command) is False


# ---------------------------------------------------------------------------
# the gate itself
# ---------------------------------------------------------------------------


class TestFullRunGate:
    @pytest.fixture(autouse=True)
    def _pin_budget(self, monkeypatch) -> None:
        """Pin the documented default so these tests assert on the CAP itself.

        The repo-root ``.env`` may carry a deliberately raised
        ``AGENT_MAX_FULL_PYTEST_RUNS``; without this pin the suite would assert
        on whatever the local config happens to say. Tests that care about a
        specific budget set the variable themselves and win.
        """
        monkeypatch.setenv(MAX_FULL_RUNS_ENV, str(DEFAULT_MAX_FULL_RUNS))

    def test_refuses_without_real_change(self, monkeypatch) -> None:
        git = _Git(" M agent_core/llm/tool_loop.py\n")
        git.install(monkeypatch)
        gate = FullRunGate(".", baseline={"agent_core/llm/tool_loop.py"})
        refusal = gate.check()
        assert refusal is not None
        assert "REFUSED" in refusal

    def test_scratch_change_does_not_earn_a_run(self, monkeypatch) -> None:
        _Git(" M agent_core/llm/tool_loop.py\n?? _tmp_probe.py\n").install(monkeypatch)
        gate = FullRunGate(".", baseline={"agent_core/llm/tool_loop.py"})
        assert gate.pending_real_changes() == set()
        assert gate.check() is not None

    def test_allows_once_after_a_real_change(self, monkeypatch) -> None:
        _Git(" M agent_core/llm/tool_loop.py\n").install(monkeypatch)
        gate = FullRunGate(".", baseline=set())
        assert gate.check() is None
        gate.record_full_run()
        assert gate.runs == 1
        # Budget spent and no NEW change since: refuse the second full run.
        assert gate.check() is not None

    def test_new_change_does_not_bypass_the_cap(self, monkeypatch) -> None:
        """The budget is a HARD per-session cap: the cheap lanes handle iteration."""
        _Git(" M agent_core/llm/tool_loop.py\n").install(monkeypatch)
        gate = FullRunGate(".", baseline=set())
        assert gate.check() is None
        gate.record_full_run()
        _Git(" M agent_core/llm/tool_loop.py\n M agent.py\n").install(monkeypatch)
        assert gate.check() is not None

    def test_env_budget_two_allows_two(self, monkeypatch) -> None:
        monkeypatch.setenv(MAX_FULL_RUNS_ENV, "2")
        _Git(" M agent.py\n").install(monkeypatch)
        gate = FullRunGate(".", baseline=set())
        assert gate.check() is None
        gate.record_full_run()
        assert gate.check() is None
        gate.record_full_run()
        assert gate.check() is not None

    def test_env_budget_zero_disables_full_runs(self, monkeypatch) -> None:
        monkeypatch.setenv(MAX_FULL_RUNS_ENV, "0")
        assert max_full_runs_from_env() == 0
        _Git(" M agent.py\n").install(monkeypatch)
        gate = FullRunGate(".", baseline=set())
        assert gate.check() is not None  # a real change is not enough

    def test_env_budget_invalid_falls_back(self, monkeypatch) -> None:
        monkeypatch.setenv(MAX_FULL_RUNS_ENV, "not-a-number")
        assert max_full_runs_from_env() == DEFAULT_MAX_FULL_RUNS

    def test_gate_can_be_disabled(self, monkeypatch) -> None:
        monkeypatch.setenv(GATE_ENV, "off")
        _Git(" M agent.py\n").install(monkeypatch)
        gate = FullRunGate(".", baseline={"agent.py"})
        gate.record_full_run()
        gate.record_full_run()
        assert gate.check() is None


# ---------------------------------------------------------------------------
# cross-repo work: the EARNED rule follows the repo the command runs in
# ---------------------------------------------------------------------------


class TestCrossRepoGate:
    """History (2026-11-07): an hour-long Kanban session (a SECOND repo,
    reached via ``cd C:\\Dev\\Kanban; python -m pytest ...``) was refused a full
    suite run FOUR times — the gate checked Agent1's git status (clean) while
    the real edits sat in Kanban's tree.  The refusal is deterministic, so the
    model retried the same doomed command each time.  The gate must judge
    dirtiness against the repo the command actually executes in."""

    def test_foreign_repo_change_earns_a_run(self, monkeypatch, tmp_path) -> None:
        """Session repo clean, Kanban dirty -> the Kanban run is allowed."""
        session = tmp_path / "agent1"
        foreign = tmp_path / "kanban"
        session.mkdir()
        foreign.mkdir()

        def _changed(root: str) -> set[str]:
            return {"src/app.py"} if str(root) == str(foreign) else set()

        monkeypatch.setattr(pytest_gate, "changed_real_files", _changed)
        gate = FullRunGate(str(session), baseline=set())
        assert gate.check(str(session)) is not None  # session: nothing earned
        assert gate.check(str(foreign)) is None  # Kanban: real change -> allowed

    def test_foreign_dirt_earls_immediately_no_lazy_baseline(
        self, monkeypatch, tmp_path
    ) -> None:
        """A foreign repo's pre-existing tracked changes earn the run at the
        FIRST check — the Kanban session was dirty before the gate ever saw the
        repo, so a lazily-captured baseline would have refused it forever.
        The budget (one per session) is the protection against waste."""
        session = tmp_path / "agent1"
        foreign = tmp_path / "kanban"
        session.mkdir()
        foreign.mkdir()

        def _changed(root: str) -> set[str]:
            return {"src/security.py"} if str(root) == str(foreign) else set()

        monkeypatch.setattr(pytest_gate, "changed_real_files", _changed)
        gate = FullRunGate(str(session), baseline=set())
        assert gate.check(str(foreign)) is None
        assert gate.check(str(foreign)) is None  # dirt unchanged, budget unspent

    def test_budget_is_session_wide(self, monkeypatch, tmp_path) -> None:
        """One full pass per session, whichever repo earned it (cheap lanes
        handle iteration)."""
        monkeypatch.setenv(MAX_FULL_RUNS_ENV, "1")
        foreign = tmp_path / "kanban"
        foreign.mkdir()
        monkeypatch.setattr(
            pytest_gate, "changed_real_files", lambda root: {"a.py"}
        )
        gate = FullRunGate(str(tmp_path / "agent1"), baseline=set())
        assert gate.check(str(foreign)) is None
        gate.record_full_run()
        assert gate.check(str(foreign)) is not None

    def test_check_without_root_keeps_legacy_behavior(self, monkeypatch) -> None:
        """check() with no root == the pre-cross-repo behavior (the primary
        baseline only) — the wiring tests above pin this contract."""
        _Git(" M agent_core/llm/tool_loop.py\n").install(monkeypatch)
        gate = FullRunGate(".", baseline=set())
        assert gate.check() is None


class TestEffectiveRepoRoot:
    """The command-to-repo resolver: follow the ``cd`` chain, fall back safely."""

    def test_plain_command_uses_start_repo(self, tmp_path) -> None:
        repo = tmp_path / "kanban"
        (repo / ".git").mkdir(parents=True)
        got = pytest_gate.effective_repo_root(
            "python -m pytest -q --no-cov", str(repo)
        )
        assert got == str(repo)

    def test_cd_chain_resolves_the_target_repo(self, tmp_path) -> None:
        start = tmp_path / "agent1"
        start.mkdir()
        kanban = tmp_path / "kanban"
        (kanban / ".git").mkdir(parents=True)
        cmd = f"cd {kanban}; python -m pytest tests/ -q --no-cov"
        assert pytest_gate.effective_repo_root(cmd, str(start)) == str(kanban)

    def test_powershell_cd_and_git_bash_mount(self, tmp_path) -> None:
        kanban = tmp_path / "kanban"
        (kanban / ".git").mkdir(parents=True)
        # PowerShell Set-Location with -Path, and a quoted target.
        cmd = f"Set-Location -Path '{kanban}'; python -m pytest -q"
        assert pytest_gate.effective_repo_root(cmd, str(tmp_path)) == str(kanban)

    def test_relative_cd_and_parent_walk(self, tmp_path) -> None:
        outer = tmp_path / "work"
        kanban = outer / "kanban"
        (kanban / ".git").mkdir(parents=True)
        start = kanban / "src"
        start.mkdir(parents=True)
        cmd = "cd ..; python -m pytest -q"
        assert pytest_gate.effective_repo_root(cmd, str(start)) == str(kanban)

    def test_missing_cd_target_keeps_previous_dir(self, tmp_path) -> None:
        start = tmp_path / "agent1"
        (start / ".git").mkdir(parents=True)
        cmd = "cd C:\\does\\not\\exist; python -m pytest -q"
        assert pytest_gate.effective_repo_root(cmd, str(start)) == str(start)

    def test_git_repo_root_walks_up(self, tmp_path) -> None:
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        nested = repo / "src" / "agent1"
        nested.mkdir(parents=True)
        assert pytest_gate.find_repo_root(str(nested)) == str(repo)

    def test_no_git_returns_start(self, tmp_path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        assert pytest_gate.find_repo_root(str(plain)) == str(plain)

    def test_git_bash_mount_style(self, tmp_path) -> None:
        """Git-Bash /c/... paths (the run tool hint shows them) resolve on Windows."""
        if os.name != "nt":
            # The mount translation is a Windows-only rule (see
            # _normalize_cd_target); on POSIX `/c/...` IS an absolute path.
            pytest.skip("Git-Bash mount style is a Windows-only shape")
        kanban = tmp_path / "kanban"
        (kanban / ".git").mkdir(parents=True)
        norm = os.path.normcase(str(kanban))  # c:\dev\kanban\...
        mount = "/" + norm[0] + "/" + norm[3:].replace("\\", "/")
        cmd = f"cd {mount}; python -m pytest -q"
        assert os.path.normcase(
            pytest_gate.effective_repo_root(cmd, str(tmp_path))
        ) == norm


class TestBudgetReadsDotenv:
    """The COUNT budget must be raisable without restarting the launch shell.

    History (2026-10-03): ``.env`` already carried ``PYTEST_FULL_SUITE_TIMEOUT``
    and the refusal text told the model to "raise AGENT_MAX_FULL_PYTEST_RUNS
    deliberately", but ``max_full_runs_from_env`` only looked at the process
    env — so the documented raise had nowhere to go. It now falls back to the
    repo-root ``.env``, exactly like ``agent._read_env_value`` does for the
    timeout.
    """

    @pytest.fixture(autouse=True)
    def _no_process_env(self, monkeypatch) -> None:
        monkeypatch.delenv(MAX_FULL_RUNS_ENV, raising=False)

    def test_absent_process_env_reads_dotenv(self, monkeypatch, tmp_path) -> None:
        dotenv = tmp_path / ".env"
        dotenv.write_text(
            "PYTEST_FULL_SUITE_TIMEOUT=600\nAGENT_MAX_FULL_PYTEST_RUNS=4\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(pytest_gate, "_ENV_FILE_PATH", str(dotenv))
        assert max_full_runs_from_env() == 4

    def test_process_env_wins_over_dotenv(self, monkeypatch, tmp_path) -> None:
        dotenv = tmp_path / ".env"
        dotenv.write_text("AGENT_MAX_FULL_PYTEST_RUNS=9\n", encoding="utf-8")
        monkeypatch.setattr(pytest_gate, "_ENV_FILE_PATH", str(dotenv))
        monkeypatch.setenv(MAX_FULL_RUNS_ENV, "2")
        assert max_full_runs_from_env() == 2

    def test_invalid_process_env_stays_default_not_dotenv(
        self, monkeypatch, tmp_path
    ) -> None:
        """A present-but-garbage env value must not silently read ``.env``."""
        dotenv = tmp_path / ".env"
        dotenv.write_text("AGENT_MAX_FULL_PYTEST_RUNS=9\n", encoding="utf-8")
        monkeypatch.setattr(pytest_gate, "_ENV_FILE_PATH", str(dotenv))
        monkeypatch.setenv(MAX_FULL_RUNS_ENV, "not-a-number")
        assert max_full_runs_from_env() == DEFAULT_MAX_FULL_RUNS

    def test_missing_dotenv_keeps_default(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            pytest_gate, "_ENV_FILE_PATH", str(tmp_path / "does-not-exist.env")
        )
        assert max_full_runs_from_env() == DEFAULT_MAX_FULL_RUNS

    def test_dotenv_zero_still_disables(self, monkeypatch, tmp_path) -> None:
        dotenv = tmp_path / ".env"
        dotenv.write_text("AGENT_MAX_FULL_PYTEST_RUNS=0\n", encoding="utf-8")
        monkeypatch.setattr(pytest_gate, "_ENV_FILE_PATH", str(dotenv))
        assert max_full_runs_from_env() == 0


# ---------------------------------------------------------------------------
# wiring into the run / tests tools
# ---------------------------------------------------------------------------


@pytest.fixture()
def bot(monkeypatch) -> Agent:
    monkeypatch.setenv(MAX_FULL_RUNS_ENV, "1")
    monkeypatch.setattr(agent, "_effective_ws_dir", lambda self: ".", raising=False)
    return Agent(workspace=".")


class TestRunToolGate:
    @pytest.fixture(autouse=True)
    def _stable_git(self, monkeypatch) -> None:
        """Pin the git view: the repo is dirty while we edit it, and a real
        change set would legitimately EARN a full run (see the last test)."""
        monkeypatch.setattr(
            "agent_core.pytest_gate.changed_real_files",
            lambda root: {"agent_core/llm/tool_loop.py"},
        )

    @pytest.mark.asyncio
    async def test_full_run_refused_after_scratch_file_only(self, bot: Agent) -> None:
        # Baseline: the session starts on an already-modified tracked file.
        bot._full_run_gate = FullRunGate(".", baseline={"agent_core/llm/tool_loop.py"})
        out = await bot._nlp_run({"command": "python -m pytest -q --no-cov"})
        assert "REFUSED" in out

    @pytest.mark.asyncio
    async def test_full_run_in_foreign_repo_earned_by_foreign_dirt(
        self, bot: Agent, monkeypatch, tmp_path
    ) -> None:
        """The Kanban scenario (2026-11-07): ``cd <other repo>; python -m pytest``
        must be judged on THAT repo's git status.  Session repo clean, foreign
        repo dirty -> the run is earned (it used to be refused forever)."""
        kanban = tmp_path / "kanban"
        (kanban / ".git").mkdir(parents=True)

        def _changed(root: str) -> set[str]:
            if os.path.normcase(str(root)) == os.path.normcase(str(kanban)):
                return {"src/security.py"}
            return set()

        monkeypatch.setattr(pytest_gate, "changed_real_files", _changed)
        bot._full_run_gate = FullRunGate(".", baseline=set())

        class _Ok:
            returncode = 0

            def communicate(self, timeout=None):  # noqa: ANN001
                return ("38 passed", "")

        monkeypatch.setattr(agent.subprocess, "Popen", lambda cmd, **kw: _Ok())
        # `cd` is a shell BUILTIN: it is not in the run allow-list, so under
        # AGENT_RUN_SANDBOX=strict (CI's default) the command never reaches
        # Popen and the gate records nothing.  This test pins the cross-repo
        # EARNED rule, not the sandbox; dev mode is the documented escape
        # hatch for a builtin-led command.
        monkeypatch.setenv("AGENT_RUN_SANDBOX", "dev")
        out = await bot._nlp_run(
            {"command": f"cd {kanban}; python -m pytest -q --no-cov"}
        )
        assert "REFUSED" not in out, out
        assert bot._full_run_gate.runs == 1

    @pytest.mark.asyncio
    async def test_full_run_in_clean_foreign_repo_not_earned_by_session_dirt(
        self, bot: Agent, monkeypatch, tmp_path
    ) -> None:
        """Mirror case: the session repo is dirty, the command's repo is clean.
        The EARNED rule is per-repo — foreign dirt must not be inferred from
        unrelated session changes."""
        kanban = tmp_path / "kanban"
        (kanban / ".git").mkdir(parents=True)

        def _changed(root: str) -> set[str]:
            if os.path.normcase(str(root)) == os.path.normcase(str(kanban)):
                return set()
            return {"agent_core/llm/tool_loop.py"}

        monkeypatch.setattr(pytest_gate, "changed_real_files", _changed)
        bot._full_run_gate = FullRunGate(".", baseline=set())
        out = await bot._nlp_run(
            {"command": f"cd {kanban}; python -m pytest -q --no-cov"}
        )
        assert "REFUSED" in out
        assert "checked repo" in out
        assert bot._full_run_gate.runs == 0

    @pytest.mark.asyncio
    async def test_full_run_allowed_and_counted(self, bot: Agent, monkeypatch) -> None:
        bot._full_run_gate = FullRunGate(".", baseline=set())

        started: list[float | None] = []

        class _Ok:
            returncode = 0

            def communicate(self, timeout=None):  # noqa: ANN001
                started.append(timeout)
                return ("2580 passed", "")

        def _popen(cmd, **kwargs):  # noqa: ANN001
            return _Ok()

        monkeypatch.setattr(agent.subprocess, "Popen", _popen)
        out = await bot._nlp_run({"command": "python -m pytest -q --no-cov"})
        assert "REFUSED" not in out
        assert started, "the allowed full run was actually executed"
        assert bot._full_run_gate.runs == 1
        # Second attempt in the same session: refused.
        again = await bot._nlp_run({"command": "python -m pytest -q --no-cov"})
        assert "REFUSED" in again

    @pytest.mark.asyncio
    async def test_targeted_runs_never_gated(self, bot: Agent, monkeypatch) -> None:
        bot._full_run_gate = FullRunGate(".", baseline={"agent_core/llm/tool_loop.py"})

        class _Ok:
            returncode = 0

            def __enter__(self):  # noqa: ANN204
                return self

            def __exit__(self, *exc):  # noqa: ANN002, ANN204
                return False

            def communicate(self, timeout=None):  # noqa: ANN001
                return ("2 passed", "")

        monkeypatch.setattr(agent.subprocess, "Popen", lambda cmd, **kw: _Ok())
        for cmd in (
            "python -m pytest --lf -q --no-cov",
            "python -m pytest --testmon -q --no-cov",
            "python -m pytest tests/test_x.py -q --no-cov",
        ):
            out = await bot._nlp_run({"command": cmd})
            assert "REFUSED" not in out, cmd
        assert bot._full_run_gate.runs == 0

    @pytest.mark.asyncio
    async def test_tests_tool_whole_workspace_refused(self, bot: Agent) -> None:
        bot._full_run_gate = FullRunGate(".", baseline={"agent_core/llm/tool_loop.py"})
        out = await bot._nlp_tests({"path": "."})
        assert "REFUSED" in out

    @pytest.mark.asyncio
    async def test_tests_tool_spends_budget_only_after_completion(
        self, bot: Agent, monkeypatch
    ) -> None:
        # Regression: the tests tool used to record the full run BEFORE the
        # subprocess started, so a timeout or launch failure permanently
        # burned the one-shot budget.  Only a completed run spends it
        # (parity with _nlp_run).
        bot._full_run_gate = FullRunGate(".", baseline=set())
        monkeypatch.setattr(
            "agent_core.handlers.nlp_tools._run_subprocess_captured",
            lambda cmd, cwd=".", timeout=60, label="Run": ("2580 passed", None),
        )
        out = await bot._nlp_tests({"path": "."})
        assert "REFUSED" not in out
        assert bot._full_run_gate.runs == 1
        # Second whole-workspace attempt in the same session: refused.
        again = await bot._nlp_tests({"path": "."})
        assert "REFUSED" in again

    @pytest.mark.asyncio
    async def test_tests_tool_timeout_does_not_spend_budget(
        self, bot: Agent, monkeypatch
    ) -> None:
        bot._full_run_gate = FullRunGate(".", baseline=set())
        monkeypatch.setattr(
            "agent_core.handlers.nlp_tools._run_subprocess_captured",
            lambda cmd, cwd=".", timeout=60, label="Run": (
                f"{label} timed out after {timeout}s",
                None,
            ),
        )
        out = await bot._nlp_tests({"path": "."})
        assert "timed out after" in out
        assert bot._full_run_gate.runs == 0

    @pytest.mark.asyncio
    async def test_tests_tool_launch_error_does_not_spend_budget(
        self, bot: Agent, monkeypatch
    ) -> None:
        bot._full_run_gate = FullRunGate(".", baseline=set())
        monkeypatch.setattr(
            "agent_core.handlers.nlp_tools._run_subprocess_captured",
            lambda cmd, cwd=".", timeout=60, label="Run": (
                "",
                f"{label} error: python not found",
            ),
        )
        out = await bot._nlp_tests({"path": "."})
        assert "error:" in out
        assert bot._full_run_gate.runs == 0


# ---------------------------------------------------------------------------
# FullRunGate.reset — a cleared session must start fresh (plan-audit finding)
# ---------------------------------------------------------------------------


class TestGateReset:
    """The 2026-10 plan audit found the gate was ghost state: ``clear`` wiped
    chat history and the memory JSONs, yet a spent full-run budget silently
    carried into every later session (every subsequent full run refused).
    These tests pin the reset contract; the wiring through
    :meth:`Agent.clear_history` is pinned in tests/test_clear_history_gate_reset.py.

    Deliberately standalone: inside ``TestRunToolGate`` an autouse fixture
    replaces ``changed_real_files`` wholesale and pins the budget env, so these
    tests own their whole environment explicitly (a mutable fake-git view)
    instead of inheriting ambient shell/.env state.
    """

    @pytest.fixture(autouse=True)
    def _gate_env(self, monkeypatch) -> None:
        """Deterministic gate env: enabled, hard cap of one full run."""
        monkeypatch.delenv(GATE_ENV, raising=False)
        monkeypatch.setenv(MAX_FULL_RUNS_ENV, "1")

    @staticmethod
    def _fake_git(monkeypatch, state: dict) -> None:
        """Patch ``changed_real_files`` to a mutable fake git view."""
        monkeypatch.setattr(
            pytest_gate, "changed_real_files", lambda root: set(state["files"])
        )

    def test_reset_zeroes_spent_budget_and_rebaselines(self, monkeypatch) -> None:
        """A cleared session gets the budget back and a *current* baseline."""
        self._fake_git(monkeypatch, {"files": {"agent_core/llm/tool_loop.py"}})
        gate = FullRunGate(".", baseline={"stale_from_last_session.py"})
        assert "stale_from_last_session.py" in gate._baseline  # ghost pre-reset
        gate.record_full_run()
        refusal = gate.check() or ""
        assert "budget" in refusal.lower(), "spent budget refuses before reset"

        gate.reset()

        assert gate.runs == 0, "reset must hand the per-session budget back"
        assert gate._baseline == {"agent_core/llm/tool_loop.py"}, (
            "baseline re-read: current git view replaces the old session's"
        )
        assert gate.pending_real_changes() == set()

    def test_reset_lets_a_new_change_earn_a_full_run(self, monkeypatch) -> None:
        """End-to-end effect of the restored budget: a fresh change earns a run.

        Pre-fix this refused forever — every cleared session inherited the old
        session's spent budget (the ghost state the plan audit found).
        """
        state = {"files": set()}
        self._fake_git(monkeypatch, state)
        gate = FullRunGate(".", baseline=set())
        gate.record_full_run()  # spend the whole budget on this "session"
        gate.reset()

        state["files"] = {"agent.py"}  # a real change appears after clearing
        assert gate.check() is None, (
            f"fresh session + fresh change must earn a full run: {gate.check()}"
        )

    def test_reset_survives_missing_git(self, monkeypatch) -> None:
        """Baseline re-read is best effort — a git hiccup must not raise."""

        def _boom(*a, **k):  # noqa: ANN001
            raise FileNotFoundError("git")

        monkeypatch.setattr(pytest_gate.subprocess, "run", _boom)
        gate = FullRunGate(".", baseline={"stale.py"})
        gate.record_full_run()
        gate.reset()  # must not raise on a dead git
        assert gate.runs == 0
        assert gate._baseline == set(), "git failure degrades to an empty baseline"
