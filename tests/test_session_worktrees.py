"""Per-session git worktrees: binding, lifecycle, startup, and walk sweeps.

Integration tests drive REAL git in throwaway repos under tmp_path; the
production code runs unmodified.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent as agent_module
from agent_core import file_protection
from agent_core.agent import AgentCore
from agent_core.commands.analyze_cmd import _collect_directory_files
from agent_core.file_searcher import FileSearcher
from agent_core.session import (
    create_session,
    delete_session,
    load_meta,
    rename_session,
    reset,
    worktrees,
)
from agent_core.session.cli import parse_and_activate
from agent_core.session.models import SESSION_VERSION, SessionMeta

pytestmark = pytest.mark.integration


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    init = subprocess.run(
        ["git", "init", "-b", "main", str(root)],
        capture_output=True,
        text=True,
    )
    if init.returncode != 0:
        subprocess.run(["git", "init", str(root)], capture_output=True, check=True)
        _git(root, "checkout", "-b", "main")
    (root / "hello.txt").write_text("v1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.email=t@example.com",
        "-c",
        "user.name=test",
        "commit",
        "-m",
        "init",
    )
    return root


@pytest.fixture(autouse=True)
def _session_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENT_SESSION_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setenv(worktrees.WORKTREE_ROOT_ENV, str(tmp_path / "wt"))
    monkeypatch.setenv(worktrees.WORKTREES_ENV, "1")
    monkeypatch.setattr(agent_module, "LLMClient", _FakeClient)
    reset()
    yield
    reset()


class _FakeClient:
    def __init__(self, model_name=None, api_key=None, **kwargs):
        self._model_name = model_name or "fake"
        self._provider = SimpleNamespace(context_limit=None)
        self._profile_name = None
        self.api_key = ""

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value


class TestEnvGate:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1", True),
            ("true", True),
            ("YES", True),
            ("On", True),
            ("0", False),
            ("false", False),
            ("", False),
        ],
    )
    def test_gate_values(self, value: str, expected: bool, monkeypatch):
        monkeypatch.setenv(worktrees.WORKTREES_ENV, value)
        assert worktrees.worktrees_enabled() is expected

    def test_gate_off_by_default(self, monkeypatch):
        monkeypatch.delenv(worktrees.WORKTREES_ENV, raising=False)
        assert worktrees.worktrees_enabled() is False


class TestBind:
    def test_bind_creates_registered_worktree(self, repo: Path):
        wt = worktrees.bind("alpha", repo=repo)
        assert wt.name == "alpha"
        assert wt.is_dir()
        assert (wt / "hello.txt").is_file()
        branches = {e["branch"] for e in worktrees.list_worktrees(repo)}
        assert "session/alpha" in branches
        assert worktrees.current_branch(wt) == "session/alpha"

    def test_bind_twice_fails(self, repo: Path):
        worktrees.bind("alpha", repo=repo)
        with pytest.raises(worktrees.WorktreeError):
            worktrees.bind("alpha", repo=repo)

    @pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "-x"])
    def test_bind_rejects_bad_ids(self, bad: str, repo: Path):
        with pytest.raises(worktrees.WorktreeError):
            worktrees.bind(bad, repo=repo)

    def test_bind_non_git_repo_fails(self, tmp_path: Path):
        plain = tmp_path / "plain"
        plain.mkdir()
        with pytest.raises(worktrees.WorktreeError):
            worktrees.bind("alpha", repo=plain)


class TestUnbind:
    def test_clean_removal(self, repo: Path):
        wt = worktrees.bind("alpha", repo=repo)
        assert worktrees.unbind("alpha", repo=repo) is True
        assert not wt.exists()
        assert not worktrees.is_registered("alpha", repo)

    def test_missing_worktree_is_noop(self, repo: Path):
        assert worktrees.unbind("ghost", repo=repo) is False

    def test_dirty_refused_without_force(self, repo: Path):
        wt = worktrees.bind("alpha", repo=repo)
        (wt / "dirty.txt").write_text("keep me\n", encoding="utf-8")
        with pytest.raises(worktrees.WorktreeError):
            worktrees.unbind("alpha", repo=repo)
        assert wt.is_dir()
        assert worktrees.unbind("alpha", repo=repo, force=True) is True
        assert not wt.exists()


class TestMovePrune:
    def test_move_relocates_worktree_and_branch(self, repo: Path):
        old = worktrees.bind("alpha", repo=repo)
        new = worktrees.move_worktree("alpha", "beta", repo=repo)
        assert new.name == "beta"
        assert not old.exists()
        assert (new / "hello.txt").is_file()
        assert worktrees.current_branch(new) == "session/beta"

    def test_move_existing_destination_refused(self, repo: Path):
        worktrees.bind("alpha", repo=repo)
        worktrees.bind("gamma", repo=repo)
        with pytest.raises(worktrees.WorktreeError):
            worktrees.move_worktree("alpha", "gamma", repo=repo)

    def test_move_missing_worktree_refused(self, repo: Path):
        with pytest.raises(worktrees.WorktreeError):
            worktrees.move_worktree("ghost", "beta", repo=repo)

    def test_prune_removes_dangling_registration(self, repo: Path):
        wt = worktrees.bind("alpha", repo=repo)
        shutil.rmtree(wt)
        worktrees.prune(repo=repo)
        assert not worktrees.is_registered("alpha", repo)


class TestMetaV2:
    def test_schema_version_is_two(self):
        assert SESSION_VERSION == 2

    def test_worktree_roundtrips(self):
        meta = SessionMeta(id="x", worktree="C:/wt/x")
        assert meta.to_dict()["worktree"] == "C:/wt/x"
        assert SessionMeta.from_dict(meta.to_dict()).worktree == "C:/wt/x"

    def test_v1_meta_defaults_to_shared(self):
        v1 = {
            "schema_version": 1,
            "id": "x",
            "title": "x",
            "workspace": "W",
            "model": "",
            "provider": "",
            "mode": "build",
            "created_at": "t",
            "updated_at": "t",
            "turn_count": 0,
            "active": False,
        }
        assert SessionMeta.from_dict(v1).worktree == ""


class TestStoreLifecycle:
    def test_create_persists_worktree(self, repo: Path):
        wt = str(worktrees.bind("alpha", repo=repo))
        create_session("alpha", workspace=wt, worktree=wt, repo=str(repo))
        meta = load_meta("alpha")
        assert meta.worktree == wt
        assert meta.workspace == wt
        assert os.path.abspath(meta.repo) == os.path.abspath(str(repo))

    def test_delete_unbinds_the_worktree(self, repo: Path):
        wt = worktrees.bind("alpha", repo=repo)
        create_session(
            "alpha", workspace=str(wt), worktree=str(wt), repo=str(repo)
        )
        delete_session("alpha")
        assert not wt.exists()

    def test_delete_dirty_refused_fail_closed(self, repo: Path):
        wt = worktrees.bind("alpha", repo=repo)
        create_session(
            "alpha", workspace=str(wt), worktree=str(wt), repo=str(repo)
        )
        (wt / "dirty.txt").write_text("keep me\n", encoding="utf-8")
        with pytest.raises(Exception):
            delete_session("alpha")
        assert wt.is_dir()

    def test_rename_moves_worktree(self, repo: Path):
        wt = worktrees.bind("alpha", repo=repo)
        create_session(
            "alpha", workspace=str(wt), worktree=str(wt), repo=str(repo)
        )
        meta = rename_session("alpha", "beta")
        assert meta.id == "beta"
        assert Path(meta.worktree).name == "beta"
        assert Path(meta.workspace).name == "beta"


class TestStartup:
    def test_new_session_startup_binds_worktree(self, repo: Path):
        selection = parse_and_activate(
            ["--new-session", "alpha"], workspace=str(repo)
        )
        assert selection.should_exit is False
        assert selection.session_id == "alpha"
        assert Path(selection.workspace).name == "alpha"
        assert selection.worktree == selection.workspace
        assert load_meta("alpha").worktree == selection.workspace

    def test_no_worktree_keeps_shared_workspace(self, repo: Path):
        selection = parse_and_activate(
            ["--new-session", "alpha", "--no-worktree"], workspace=str(repo)
        )
        assert selection.should_exit is False
        assert os.path.abspath(selection.workspace) == os.path.abspath(str(repo))
        assert selection.worktree == ""
        assert load_meta("alpha").worktree == ""

    def test_non_git_workspace_falls_back_with_notice(
        self, tmp_path: Path, capsys
    ):
        plain = tmp_path / "plain"
        plain.mkdir()
        selection = parse_and_activate(
            ["--new-session", "alpha"], workspace=str(plain)
        )
        assert selection.should_exit is False
        assert os.path.abspath(selection.workspace) == os.path.abspath(str(plain))
        assert "not a git repo" in capsys.readouterr().out

    def test_resume_lands_in_the_bound_worktree(self, repo: Path):
        first = parse_and_activate(
            ["--new-session", "alpha"], workspace=str(repo)
        )
        assert first.should_exit is False
        from agent_core.session import deactivate_session

        deactivate_session()
        second = parse_and_activate(["--session", "alpha"], workspace=str(repo))
        assert second.should_exit is False
        assert Path(second.workspace).name == "alpha"

    def test_resume_missing_worktree_falls_back_with_notice(
        self, repo: Path, capsys
    ):
        selection = parse_and_activate(
            ["--new-session", "alpha"], workspace=str(repo)
        )
        from agent_core.session import deactivate_session

        deactivate_session()
        shutil.rmtree(selection.worktree)
        second = parse_and_activate(["--session", "alpha"], workspace=str(repo))
        assert os.path.abspath(second.workspace) == os.path.abspath(str(repo))
        assert "missing" in capsys.readouterr().out


class TestSwitchRebinding:
    def test_switch_rebinds_workspace_objects(self, repo: Path):
        wt = str(worktrees.bind("alpha", repo=repo))
        create_session("alpha", workspace=wt, worktree=wt)
        agent = AgentCore(workspace=str(repo))
        assert os.path.abspath(agent.workspace) == os.path.abspath(str(repo))

        agent.switch_session("alpha")

        assert os.path.abspath(agent.workspace) == os.path.abspath(wt)
        assert os.path.abspath(agent.fs.workspace) == os.path.abspath(wt)
        assert os.path.abspath(agent.searcher.workspace) == os.path.abspath(wt)
        assert agent._full_run_gate is not None
        assert os.path.abspath(agent._full_run_gate.root) == os.path.abspath(wt)
        assert agent._nlp_workspace is None

    def test_switch_missing_worktree_warns_and_stays(
        self, repo: Path, capsys
    ):
        wt = str(worktrees.bind("alpha", repo=repo))
        create_session("alpha", workspace=wt, worktree=wt)
        agent = AgentCore(workspace=str(repo))
        shutil.rmtree(wt)

        agent.switch_session("alpha")

        assert "missing" in capsys.readouterr().out
        assert os.path.abspath(agent.workspace) == os.path.abspath(str(repo))


class TestWalkSweep:
    def test_searcher_skips_session_and_worktree_dirs(self, tmp_path: Path):
        (tmp_path / "top.py").write_text("SENTINELTOP\n", encoding="utf-8")
        for sub in (".worktrees/alpha", ".sessions/x", "reports/traces"):
            d = tmp_path / sub
            d.mkdir(parents=True)
            (d / "copy.py").write_text("SENTINELTOP\n", encoding="utf-8")
        searcher = FileSearcher(str(tmp_path))
        out = asyncio.run(searcher.search("SENTINELTOP", str(tmp_path)))
        assert "top.py" in out
        for hidden in (".worktrees", ".sessions", "reports"):
            assert hidden not in out

    def test_analyze_skips_worktree_tree(self, tmp_path: Path):
        (tmp_path / "top.py").write_text("x\n", encoding="utf-8")
        hidden = tmp_path / ".worktrees" / "alpha"
        hidden.mkdir(parents=True)
        (hidden / "copy.py").write_text("x\n", encoding="utf-8")
        files = _collect_directory_files(str(tmp_path))
        assert any(f.endswith("top.py") for f in files)
        assert not any(".worktrees" in f for f in files)


class TestProtection:
    def test_session_trees_are_protected(self, tmp_path: Path):
        assert file_protection.is_protected(
            str(tmp_path / ".worktrees" / "alpha" / "x.py"), str(tmp_path)
        )
        assert file_protection.is_protected(
            str(tmp_path / ".sessions" / "alpha" / "y.py"), str(tmp_path)
        )
        assert not file_protection.is_protected(
            str(tmp_path / "src" / "z.py"), str(tmp_path)
        )


class TestGitignore:
    def test_repo_ignores_worktree_root(self):
        text = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(
            encoding="utf-8"
        )
        lines = {line.strip() for line in text.splitlines()}
        assert ".worktrees/" in lines
        assert ".sessions/" in lines
