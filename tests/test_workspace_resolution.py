"""Regression tests for workspace resolution and its system-prompt block.

Before :mod:`agent_core.workspace` existed the answer to "which directory is
the agent working on?" was derived from the *code*: ``AgentCore.DEFAULT_
WORKSPACE = dirname(dirname(abspath(__file__)))`` — Agent1's own checkout.  So
starting the agent inside another repository silently aimed every tool at
Agent1, and three layers disagreed about the root:

* ``AgentCore.DEFAULT_WORKSPACE`` (what the REPL used),
* ``AgentConfig.workspace_root`` (``AGENT_WORKSPACE_ROOT`` or cwd, never read),
* ``session.store.repo_root()`` (what startup pinned).

These tests pin the single precedence chain (explicit -> env -> project cwd ->
repo root) and the block that finally TELLS the model where it works.
"""

from __future__ import annotations

import os

import pytest

from agent_core import workspace as ws_mod
from agent_core.run_utils import (
    WORKSPACE_MARKER,
    strip_dynamic_system_blocks,
)
from agent_core.workspace import (
    WORKSPACE_ENV,
    looks_like_repo,
    resolve_workspace,
)


@pytest.fixture()
def proj(tmp_path):
    """A directory that looks like a project root."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    return root


@pytest.fixture()
def plain(tmp_path):
    """A directory WITHOUT any project marker."""
    root = tmp_path / "plain"
    root.mkdir()
    return root


def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


# ── looks_like_repo ─────────────────────────────────────────────────────────


def test_looks_like_repo_true_for_marker(proj) -> None:
    assert looks_like_repo(proj) is True


def test_looks_like_repo_false_without_marker(plain) -> None:
    assert looks_like_repo(plain) is False


def test_looks_like_repo_false_for_missing_path(tmp_path) -> None:
    assert looks_like_repo(tmp_path / "nope") is False


def test_looks_like_repo_never_raises_on_garbage() -> None:
    assert looks_like_repo(None) is False  # type: ignore[arg-type]


def test_looks_like_repo_git_dir_counts(tmp_path) -> None:
    root = tmp_path / "checkout"
    (root / ".git").mkdir(parents=True)
    assert looks_like_repo(root) is True


# ── resolve_workspace precedence ────────────────────────────────────────────


def test_explicit_path_wins_over_env_and_cwd(proj, plain, tmp_path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    got = resolve_workspace(str(other), env={WORKSPACE_ENV: str(proj)}, cwd=str(plain))
    assert _norm(got) == _norm(other)


def test_env_wins_over_project_cwd(proj, plain) -> None:
    got = resolve_workspace(None, env={WORKSPACE_ENV: str(proj)}, cwd=str(plain))
    assert _norm(got) == _norm(proj)


def test_project_cwd_wins_over_repo_root(proj) -> None:
    got = resolve_workspace(None, env={}, cwd=str(proj))
    assert _norm(got) == _norm(proj)


def test_plain_cwd_falls_back_to_repo_root(plain) -> None:
    got = resolve_workspace(None, env={}, cwd=str(plain))
    assert _norm(got) == _norm(ws_mod._repo_root())
    # A stray directory (e.g. C:\Windows) must never become the workspace.
    assert _norm(got) != _norm(plain)


def test_bad_explicit_falls_through_to_env(proj, tmp_path) -> None:
    missing = tmp_path / "does-not-exist"
    got = resolve_workspace(
        str(missing), env={WORKSPACE_ENV: str(proj)}, cwd=str(tmp_path)
    )
    assert _norm(got) == _norm(proj)


def test_bad_env_falls_through_to_project_cwd(proj, tmp_path) -> None:
    got = resolve_workspace(
        None, env={WORKSPACE_ENV: str(tmp_path / "gone")}, cwd=str(proj)
    )
    assert _norm(got) == _norm(proj)


def test_empty_and_whitespace_are_ignored(proj) -> None:
    got = resolve_workspace("   ", env={WORKSPACE_ENV: "  "}, cwd=str(proj))
    assert _norm(got) == _norm(proj)


def test_always_returns_absolute_existing_dir(tmp_path) -> None:
    got = resolve_workspace(None, env={}, cwd=str(tmp_path))
    assert os.path.isabs(got)
    assert os.path.isdir(got)


def test_relative_explicit_is_absolutised(proj) -> None:
    rel = os.path.relpath(str(proj), os.getcwd())
    got = resolve_workspace(rel, env={}, cwd=str(proj))
    assert os.path.isabs(got)
    assert _norm(got) == _norm(proj)


# ── AgentCore wiring ────────────────────────────────────────────────────────


def test_agent_uses_explicit_workspace(proj) -> None:
    import agent

    a = agent.Agent(workspace=str(proj))
    assert _norm(a.workspace) == _norm(proj)


def test_agent_workspace_is_not_the_code_checkout_when_given_another(proj) -> None:
    import agent

    a = agent.Agent(workspace=str(proj))
    assert _norm(a.workspace) != _norm(agent.AgentCore.DEFAULT_WORKSPACE)


def test_agent_env_override(proj, monkeypatch) -> None:
    import agent

    monkeypatch.setenv(WORKSPACE_ENV, str(proj))
    a = agent.Agent()
    assert _norm(a.workspace) == _norm(proj)


# ── the model-facing WORKSPACE block ────────────────────────────────────────


def test_block_names_the_absolute_root(proj) -> None:
    import agent

    a = agent.Agent(workspace=str(proj))
    block = a._workspace_block()
    assert block.startswith(WORKSPACE_MARKER)
    assert _norm(proj) in block.lower()
    # The model must be told how to switch, not only where it is.
    assert "workspace reset" in block


def test_block_empty_for_default_workspace() -> None:
    """A single-repo setup keeps a byte-identical prompt."""
    import agent

    a = agent.Agent(workspace=agent.AgentCore.DEFAULT_WORKSPACE)
    assert a._workspace_block() == ""


def test_block_reports_git_branch(proj) -> None:
    import agent

    (proj / ".git").mkdir()
    (proj / ".git" / "HEAD").write_text(
        "ref: refs/heads/feature-x\n", encoding="utf-8"
    )
    a = agent.Agent(workspace=str(proj))
    assert "feature-x" in a._workspace_block()


def test_block_marks_detached_head(proj) -> None:
    import agent

    (proj / ".git").mkdir()
    (proj / ".git" / "HEAD").write_text("a" * 40 + "\n", encoding="utf-8")
    a = agent.Agent(workspace=str(proj))
    assert "detached" in a._workspace_block()


def test_block_handles_non_git_root(plain) -> None:
    import agent

    a = agent.Agent(workspace=str(plain))
    assert "not a git working tree" in a._workspace_block()


def test_stripper_removes_workspace_block() -> None:
    """Rebuilding must replace the block, never accumulate copies."""
    prompt = "BASE PROMPT" + WORKSPACE_MARKER + "\nAbsolute working root: X\n"
    assert strip_dynamic_system_blocks(prompt) == "BASE PROMPT"


def test_rebuild_keeps_exactly_one_workspace_block(proj) -> None:
    import agent

    a = agent.Agent(workspace=str(proj))
    a._refresh_system_message("hello")
    a._refresh_system_message("hello again")
    content = a._chat_history[0]["content"]
    assert content.count(WORKSPACE_MARKER) == 1
    assert _norm(proj) in content.lower()
