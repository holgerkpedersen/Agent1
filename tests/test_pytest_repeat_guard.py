"""Regression tests: an identical pytest run over an unchanged tree is refused.

History (2026-11-27): verification was being REPEATED for no reason - a green
targeted run over an untouched tree was run again, and again, because "let's be
sure".  A second identical run cannot produce a different result, so the tool
now hands back the PREVIOUS result instead of paying for the run twice
(``agent_core.pytest_repeat_guard``).  The rule it implements is the
state-change rule: verification stays valid until the repository changes.

These tests pin:

1. the fingerprint (content-based, housekeeping-proof, fail open);
2. the guard's allow/refuse matrix (first run, unchanged repeat, real change,
   different lane, disabled, no fingerprint);
3. the wiring into the ``run`` and ``tests`` tools;
4. the per-session lifetime (``clear`` forgets every record).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent_core import pytest_repeat_guard as guard_mod
from agent_core.handlers import nlp_tools
from agent_core.pytest_repeat_guard import (
    GUARD_ENV,
    PytestRepeatGuard,
    normalize_command,
    summarize_pytest_output,
    tree_fingerprint,
)

GIT = shutil.which("git")

needs_git = pytest.mark.skipif(GIT is None, reason="git is not installed")


# ---------------------------------------------------------------------------
# command normalization
# ---------------------------------------------------------------------------


class TestNormalizeCommand:
    def test_collapses_whitespace_and_case(self) -> None:
        assert normalize_command("  Python  -M  Pytest -q ") == "python -m pytest -q"

    def test_token_order_is_preserved(self) -> None:
        """``-k``/``-m`` pair a flag with its VALUE, so sorting would merge
        different selections into one key."""
        a = normalize_command("python -m pytest -k 'not slow' -q")
        b = normalize_command("python -m pytest -q -k 'not slow'")
        assert a != b

    def test_accepts_a_token_sequence(self) -> None:
        assert normalize_command([sys.executable, "-m", "pytest", "x.py"]) == (
            f"{sys.executable} -m pytest x.py".casefold()
        )


# ---------------------------------------------------------------------------
# result summaries (the model sees the OLD outcome instead of re-running)
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_final_summary_line_wins(self) -> None:
        out = (
            "tests/test_x.py::test_a PASSED\n"
            "==================== 1 failed, 12 passed in 0.83s ===================\n"
        )
        assert summarize_pytest_output(out) == "1 failed, 12 passed in 0.83s"

    def test_no_tests_ran(self) -> None:
        out = "collected 0 items\n\n============= no tests ran in 0.02s =============\n"
        assert summarize_pytest_output(out) == "no tests ran in 0.02s"

    def test_exit_code_fallback(self) -> None:
        assert "exit code 2" in summarize_pytest_output("boom", exit_code=2)

    def test_no_evidence_at_all(self) -> None:
        assert summarize_pytest_output("") == "no summary captured"


# ---------------------------------------------------------------------------
# fingerprint: content-based, housekeeping-proof, fail open
# ---------------------------------------------------------------------------


def _git(repo, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [GIT, "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )


@pytest.fixture()
def repo(tmp_path):
    """A real (throwaway) git repo with one committed source file."""
    if GIT is None:  # pragma: no cover - CI has git
        pytest.skip("git is not installed")
    work = tmp_path / "repo"
    work.mkdir()
    (work / "src.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert _git(work, "init").returncode == 0
    assert _git(work, "add", "-A").returncode == 0
    assert _git(work, "commit", "-m", "init").returncode == 0
    return work


@needs_git
class TestFingerprint:
    def test_clean_tree_is_stable(self, repo) -> None:
        first = tree_fingerprint(str(repo))
        assert first
        assert tree_fingerprint(str(repo)) == first

    def test_edit_changes_the_fingerprint(self, repo) -> None:
        before = tree_fingerprint(str(repo))
        (repo / "src.py").write_text("VALUE = 2\n", encoding="utf-8")
        after = tree_fingerprint(str(repo))
        assert after and after != before

    def test_same_content_rewrite_does_not_change_it(self, repo) -> None:
        """Content, not mtime: re-writing identical bytes is not a change."""
        before = tree_fingerprint(str(repo))
        (repo / "src.py").write_text("VALUE = 1\n", encoding="utf-8")
        os.utime(repo / "src.py", None)
        assert tree_fingerprint(str(repo)) == before

    def test_new_untracked_file_changes_it(self, repo) -> None:
        before = tree_fingerprint(str(repo))
        new_test = repo / "tests_new.py"
        new_test.write_text("def test_x():\n    pass\n", encoding="utf-8")
        assert tree_fingerprint(str(repo)) != before

    def test_deleted_file_changes_it(self, repo) -> None:
        before = tree_fingerprint(str(repo))
        (repo / "src.py").unlink()
        assert tree_fingerprint(str(repo)) != before

    def test_run_housekeeping_is_ignored(self, repo) -> None:
        """The run's own cache must never make a repeat look legitimate."""
        before = tree_fingerprint(str(repo))
        cache = repo / ".pytest_cache" / "v" / "cache"
        cache.mkdir(parents=True)
        (cache / "lastfailed").write_text("{}", encoding="utf-8")
        pycache = repo / "__pycache__"
        pycache.mkdir()
        (pycache / "src.cpython-314.pyc").write_bytes(b"x")
        assert tree_fingerprint(str(repo)) == before

    def test_non_git_workspace_still_fingerprints(self, tmp_path) -> None:
        work = tmp_path / "plain"
        work.mkdir()
        (work / "a.py").write_text("x = 1\n", encoding="utf-8")
        first = tree_fingerprint(str(work))
        assert first
        (work / "a.py").write_text("x = 2\n", encoding="utf-8")
        assert tree_fingerprint(str(work)) != first

    def test_missing_workspace_fails_open(self, tmp_path) -> None:
        assert tree_fingerprint(str(tmp_path / "nope")) is None

    def test_oversized_workspace_fails_open(self, repo, monkeypatch) -> None:
        monkeypatch.setattr(guard_mod, "_MAX_SCAN_ENTRIES", 0)
        monkeypatch.setattr(guard_mod, "_status_entries", lambda root: None)
        assert tree_fingerprint(str(repo)) is None


# ---------------------------------------------------------------------------
# the guard itself
# ---------------------------------------------------------------------------


class TestGuardMatrix:
    def _guard(self, state: dict, **kwargs) -> PytestRepeatGuard:
        return PytestRepeatGuard(
            ".",
            fingerprint_fn=lambda root: state["fp"],
            clock=lambda: "10:00:00",
            **kwargs,
        )

    def test_first_run_is_allowed(self) -> None:
        guard = self._guard({"fp": "a"})
        refusal, fingerprint = guard.check("python -m pytest tests/x.py -q")
        assert refusal is None
        assert fingerprint == "a"

    def test_identical_repeat_is_refused_with_the_old_result(self) -> None:
        guard = self._guard({"fp": "a"})
        _refusal, fingerprint = guard.check("python -m pytest tests/x.py -q")
        guard.record(
            "python -m pytest tests/x.py -q",
            fingerprint,
            output="============ 12 passed in 0.50s ============",
        )
        refusal, returned = guard.check("python -m pytest tests/x.py -q")
        assert returned is None
        assert refusal is not None
        assert "REFUSED" in refusal
        assert "12 passed in 0.50s" in refusal
        assert "10:00:00" in refusal
        assert GUARD_ENV in refusal

    def test_whitespace_variant_is_the_same_lane(self) -> None:
        guard = self._guard({"fp": "a"})
        _r, fingerprint = guard.check("python -m pytest tests/x.py -q")
        guard.record("python -m pytest tests/x.py -q", fingerprint, output="1 passed")
        refusal, _ = guard.check("  python -m   pytest   tests/x.py -q  ")
        assert refusal is not None

    def test_a_real_change_lifts_the_refusal(self) -> None:
        state = {"fp": "a"}
        guard = self._guard(state)
        _r, fingerprint = guard.check("python -m pytest tests/x.py -q")
        guard.record("python -m pytest tests/x.py -q", fingerprint, output="1 passed")
        state["fp"] = "b"
        refusal, new_fingerprint = guard.check("python -m pytest tests/x.py -q")
        assert refusal is None
        assert new_fingerprint == "b"

    def test_a_different_lane_is_not_refused(self) -> None:
        guard = self._guard({"fp": "a"})
        _r, fingerprint = guard.check("python -m pytest tests/x.py -q")
        guard.record("python -m pytest tests/x.py -q", fingerprint, output="1 passed")
        refusal, _ = guard.check("python -m pytest tests/y.py -q")
        assert refusal is None

    def test_fail_open_without_a_fingerprint(self, monkeypatch) -> None:
        guard = PytestRepeatGuard(".", fingerprint_fn=lambda root: None)
        refusal, fingerprint = guard.check("python -m pytest -q")
        assert refusal is None and fingerprint is None
        guard.record("python -m pytest -q", None, output="1 passed")
        assert guard.records == 0

    def test_broken_probe_fails_open(self) -> None:
        def _boom(root: str) -> str:
            raise RuntimeError("git exploded")

        guard = PytestRepeatGuard(".", fingerprint_fn=_boom)
        refusal, fingerprint = guard.check("python -m pytest -q")
        assert refusal is None and fingerprint is None

    def test_disabled_by_env(self, monkeypatch) -> None:
        guard = self._guard({"fp": "a"})
        _r, fingerprint = guard.check("python -m pytest -q")
        guard.record("python -m pytest -q", fingerprint, output="1 passed")
        monkeypatch.setenv(GUARD_ENV, "off")
        refusal, fingerprint = guard.check("python -m pytest -q")
        assert refusal is None and fingerprint is None
        guard.record("python -m pytest -q", fingerprint, output="1 passed")
        assert guard.records == 1  # nothing new was stored either

    def test_reset_forgets_everything(self) -> None:
        guard = self._guard({"fp": "a"})
        _r, fingerprint = guard.check("python -m pytest -q")
        guard.record("python -m pytest -q", fingerprint, output="1 passed")
        guard.reset()
        assert guard.records == 0
        assert guard.check("python -m pytest -q")[0] is None

    def test_records_are_bounded(self) -> None:
        guard = self._guard({"fp": "a"}, max_records=2)
        for command in ("a", "b", "c"):
            _r, fingerprint = guard.check(f"python -m pytest {command}.py -q")
            guard.record(
                f"python -m pytest {command}.py -q", fingerprint, output="1 passed"
            )
        assert guard.records == 2
        # The OLDEST entry was dropped, so it may run again.
        assert guard.check("python -m pytest a.py -q")[0] is None


# ---------------------------------------------------------------------------
# wiring: the run tool
# ---------------------------------------------------------------------------


class _FakeProc:
    returncode = 0

    def __init__(
        self, output: str = "============ 2 passed in 0.10s ============"
    ) -> None:
        self._output = output

    def communicate(self, timeout=None):  # noqa: ANN001
        return self._output, ""


@pytest.fixture()
def bot(tmp_path):
    from agent import Agent

    return Agent(workspace=str(tmp_path))


def _armed_guard(bot, state: dict) -> PytestRepeatGuard:
    guard = PytestRepeatGuard(
        bot.workspace,
        fingerprint_fn=lambda root: state["fp"],
        clock=lambda: "11:22:33",
    )
    bot._pytest_repeat_guard = guard
    return guard


class TestRunToolWiring:
    @pytest.mark.asyncio
    async def test_identical_repeat_is_not_executed(self, bot, monkeypatch) -> None:
        state = {"fp": "a"}
        guard = _armed_guard(bot, state)
        calls: list = []

        def _popen(cmd, **kwargs):  # noqa: ANN001
            calls.append(cmd)
            return _FakeProc()

        monkeypatch.setattr(nlp_tools.subprocess, "Popen", _popen)
        cmd = "python -m pytest tests/test_x.py -q --no-cov"

        first = await bot._nlp_run({"command": cmd})
        assert "2 passed" in first
        assert len(calls) == 1
        assert guard.records == 1

        second = await bot._nlp_run({"command": cmd})
        assert "REFUSED" in second
        assert "2 passed in 0.10s" in second
        assert len(calls) == 1, "the redundant run must not reach the shell"

        state["fp"] = "b"
        third = await bot._nlp_run({"command": cmd})
        assert "REFUSED" not in third
        assert len(calls) == 2

        fourth = await bot._nlp_run({"command": cmd})
        assert "REFUSED" in fourth
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_non_pytest_commands_are_never_refused(
        self, bot, monkeypatch
    ) -> None:
        state = {"fp": "a"}
        _armed_guard(bot, state)
        calls: list = []
        monkeypatch.setattr(
            nlp_tools.subprocess,
            "Popen",
            lambda cmd, **kw: calls.append(cmd) or _FakeProc("ok"),
        )
        for _ in range(2):
            out = await bot._nlp_run({"command": "git status"})
            assert "REFUSED" not in out
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_failed_launch_is_not_recorded(self, bot, monkeypatch) -> None:
        """Only a COMPLETED run is evidence; a launch failure must stay
        retryable instead of being suppressed by the guard."""
        state = {"fp": "a"}
        _armed_guard(bot, state)

        def _boom(cmd, **kwargs):  # noqa: ANN001
            raise OSError("spawn failed")

        monkeypatch.setattr(nlp_tools.subprocess, "Popen", _boom)
        first = await bot._nlp_run({"command": "python -m pytest tests/x.py -q"})
        assert first.startswith("Error:")

        calls: list = []
        monkeypatch.setattr(
            nlp_tools.subprocess,
            "Popen",
            lambda cmd, **kw: calls.append(cmd) or _FakeProc("1 passed"),
        )
        second = await bot._nlp_run({"command": "python -m pytest tests/x.py -q"})
        assert "REFUSED" not in second
        assert calls, "the retry after a failed launch must actually run"


# ---------------------------------------------------------------------------
# wiring: the tests tool
# ---------------------------------------------------------------------------


class TestTestsToolWiring:
    @pytest.mark.asyncio
    async def test_identical_repeat_is_not_executed(self, bot, monkeypatch) -> None:
        state = {"fp": "a"}
        _armed_guard(bot, state)
        calls: list = []

        def _captured(cmd, cwd=".", timeout=60, label="Run"):  # noqa: ANN001
            calls.append(cmd)
            return "============ 7 passed in 0.40s ============", None

        monkeypatch.setattr(nlp_tools, "_run_subprocess_captured", _captured)

        first = await bot._nlp_tests({"path": "tests/test_x.py"})
        assert "7 passed" in first
        assert len(calls) == 1

        second = await bot._nlp_tests({"path": "tests/test_x.py"})
        assert "REFUSED" in second
        assert len(calls) == 1

        state["fp"] = "b"
        third = await bot._nlp_tests({"path": "tests/test_x.py"})
        assert "REFUSED" not in third
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_timed_out_run_is_not_recorded(self, bot, monkeypatch) -> None:
        state = {"fp": "a"}
        _armed_guard(bot, state)
        calls: list = []

        def _captured(cmd, cwd=".", timeout=60, label="Run"):  # noqa: ANN001
            calls.append(cmd)
            return f"{label} timed out after {timeout}s", None

        monkeypatch.setattr(nlp_tools, "_run_subprocess_captured", _captured)
        await bot._nlp_tests({"path": "tests/test_x.py"})
        again = await bot._nlp_tests({"path": "tests/test_x.py"})
        assert "REFUSED" not in again
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# per-session lifetime: ``clear`` must forget the records
# ---------------------------------------------------------------------------


class TestClearResets:
    def test_clear_history_forgets_every_record(
        self, bot, tmp_path, monkeypatch
    ) -> None:
        from agent_core.memory import persistence

        monkeypatch.setattr(
            persistence,
            "current_paths",
            lambda: SimpleNamespace(
                chat_history_path=str(tmp_path / "chat.json"),
                agent_memory_path=str(tmp_path / "memory.json"),
                prior_history_path=str(tmp_path / "prior.json"),
            ),
        )
        guard = PytestRepeatGuard(".", fingerprint_fn=lambda root: "a")
        bot._pytest_repeat_guard = guard
        guard.record("python -m pytest -q", "a", output="1 passed")
        assert guard.records == 1

        bot.clear_history()

        assert guard.records == 0
        assert guard.check("python -m pytest -q")[0] is None


if __name__ == "__main__":
    pytest.main([__file__])
