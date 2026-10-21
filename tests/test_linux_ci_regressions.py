"""Regression tests for three failures seen in the Linux CI run."""
import os
import signal

import pytest

from agent_core import run_utils
from agent_core.commands.plan_verifier import _norm


class TestNormKeepsAbsolutePaths:
    def test_absolute_posix_path_keeps_root_slash(self):
        assert _norm("/tmp/ws/existing.py") == "/tmp/ws/existing.py"

    def test_leading_dot_slash_is_stripped(self):
        assert _norm("./agent.py") == "agent.py"

    def test_hidden_directory_is_not_mangled(self):
        assert _norm(".github/workflows/ci.yml") == ".github/workflows/ci.yml"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups only")
class TestKillProcessTreeNeverSignalsOwnGroup:
    """A child that does not lead its own group must not trigger killpg,
    because getpgid() then returns the agent's own group."""

    def test_non_leader_child_is_killed_directly_not_via_killpg(self, monkeypatch):
        killed_groups: list[tuple[int, int]] = []
        direct_kills: list[int] = []

        class _Proc:
            pid = 4242

            def kill(self):
                direct_kills.append(self.pid)

        monkeypatch.setattr(run_utils.os, "getpgid", lambda pid: os.getpgrp())
        monkeypatch.setattr(
            run_utils.os, "killpg", lambda pgid, sig: killed_groups.append((pgid, sig))
        )

        run_utils.kill_process_tree(_Proc())

        assert killed_groups == [], "killpg must not target the agent's own group"
        assert direct_kills == [4242]

    def test_group_leader_child_gets_group_signal(self, monkeypatch):
        killed_groups: list[tuple[int, int]] = []

        class _Proc:
            pid = 5151

            def kill(self):
                pass

        monkeypatch.setattr(run_utils.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(
            run_utils.os, "killpg", lambda pgid, sig: killed_groups.append((pgid, sig))
        )

        run_utils.kill_process_tree(_Proc())

        assert killed_groups == [(5151, signal.SIGTERM)]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups only")
def test_real_child_in_agent_group_is_killed_and_agent_survives():
    """Real-process check (no mocks): a child sharing the test process's
    group must be killed while the test process itself keeps running."""
    import subprocess

    proc = subprocess.Popen(["sleep", "30"])
    try:
        assert os.getpgid(proc.pid) == os.getpgrp()
        run_utils.kill_process_tree(proc)
        proc.wait(timeout=5)
        assert proc.returncode == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    # Reaching this line proves the agent process was not signalled.
    assert os.getpgrp() > 0
