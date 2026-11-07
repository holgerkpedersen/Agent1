"""Regression tests for three failures seen in the Linux CI run."""
import os
import pathlib
import re
import signal

import pytest
import yaml

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


class TestKanbanCheckoutKeepsWorkspaceIntact:
    """CI run 42409020802: `actions/checkout@v4` with a relative `path`
    wipes its working directory (the workspace root) before cloning, and
    the repo root follows the existing `.git` - so the Kanban clone landed
    *inside* the Agent1 checkout after deleting the Agent1 tree.  The
    install step then died on a missing `pyproject.toml` and the composite
    action's Post step could not find its own `action.yml`.
    The sibling layout is load-bearing, so pin it in CI config."""

    ACTION = (
        pathlib.Path(__file__).resolve().parent.parent
        / ".github" / "actions" / "kanban-checkout" / "action.yml"
    )

    def test_action_definition_exists(self):
        assert self.ACTION.is_file(), f"missing {self.ACTION}"

    def test_default_path_is_a_sibling_not_inside_the_workspace(self):
        text = self.ACTION.read_text(encoding="utf-8")
        default = re.search(
            r"^  path:\n(?:.*\n)*?    default:\s*(\S.*)$", text, re.MULTILINE
        )
        assert default is not None, "the action must declare a path default"
        value = default.group(1).strip().strip('"').strip("'")
        assert value == "../Kanban", (
            "the Kanban checkout must land next to the workspace; an empty "
            "or workspace-relative path makes actions/checkout delete the "
            f"Agent1 tree (saw: {value!r})"
        )

    def test_ci_installs_the_package_after_the_kanban_step(self):
        """Order guard: a checkout that wipes the tree must never run after
        the step that reads pyproject.toml."""
        ci = (
            pathlib.Path(__file__).resolve().parent.parent
            / ".github" / "workflows" / "ci.yml"
        )
        job = yaml.safe_load(ci.read_text(encoding="utf-8"))["jobs"]["test"]
        steps = [s.get("name", "") for s in job["steps"]]
        kanban = next(i for i, n in enumerate(steps) if "Check out Kanban" in n)
        install = next(i for i, n in enumerate(steps) if "Install package" in n)
        assert kanban < install, "Kanban checkout must precede the pip install"
