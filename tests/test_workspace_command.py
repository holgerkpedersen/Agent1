"""Regression tests for the ``workspace`` REPL command and its guardrail.

The agent used to have exactly one way to aim its tools at another directory:
``paste --workspace``, which is not discoverable and does not tell the model.
:class:`agent_core.commands.workspace_cmd.WorkspaceCommand` makes the root
visible (show), overridable (switch) and reversible (reset), and rebuilds the
system prompt so the WORKSPACE block follows the switch instead of advertising
the previous root for the rest of the session.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from agent_core.commands.workspace_cmd import WorkspaceCommand
from agent_core.workspace import (
    describe_target_divergence,
    describe_workspace_mismatch,
)


@pytest.fixture()
def agent(tmp_path):
    import agent as agent_module

    root = tmp_path / "home"
    root.mkdir()
    return agent_module.Agent(workspace=str(root))


@pytest.fixture()
def target(tmp_path):
    path = tmp_path / "target"
    path.mkdir()
    (path / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    return path


def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _run(cmd: WorkspaceCommand, args, agent) -> bool:
    return asyncio.run(cmd.execute(list(args), agent))


def _effective(agent) -> str:
    from agent_core.handlers.nlp_tools import _effective_ws_dir

    return _effective_ws_dir(agent)


# ── show ────────────────────────────────────────────────────────────────────


def test_show_reports_workspace(agent, capsys) -> None:
    assert _run(WorkspaceCommand(), [], agent) is True
    out = capsys.readouterr().out
    assert "Workspace:" in out
    assert _norm(agent.workspace).split(os.sep)[-1] in out.lower()


def test_show_leaves_workspace_untouched(agent) -> None:
    before = _effective(agent)
    _run(WorkspaceCommand(), ["show"], agent)
    assert _effective(agent) == before


# ── switch ──────────────────────────────────────────────────────────────────


def test_switch_moves_effective_dir(agent, target) -> None:
    assert _run(WorkspaceCommand(), [str(target)], agent) is True
    assert _norm(_effective(agent)) == _norm(target)


def test_switch_rebuilds_prompt_block(agent, target) -> None:
    """The model must be told the NEW root, not the one it booted with."""
    from agent_core.run_utils import WORKSPACE_MARKER

    _run(WorkspaceCommand(), [str(target)], agent)
    content = str(agent._chat_history[0]["content"])
    assert content.count(WORKSPACE_MARKER) == 1
    assert _norm(target) in content.lower()


def test_switch_rejects_missing_dir(agent, tmp_path, capsys) -> None:
    before = _effective(agent)
    assert _run(WorkspaceCommand(), [str(tmp_path / "nope")], agent) is False
    assert "not a directory" in capsys.readouterr().out
    assert _effective(agent) == before


def test_switch_rejects_file(agent, target, capsys) -> None:
    before = _effective(agent)
    file_path = target / "pyproject.toml"
    assert _run(WorkspaceCommand(), [str(file_path)], agent) is False
    assert _effective(agent) == before


# ── reset ───────────────────────────────────────────────────────────────────


def test_reset_returns_to_boot_root(agent, target) -> None:
    boot = _effective(agent)
    _run(WorkspaceCommand(), [str(target)], agent)
    assert _norm(_effective(agent)) == _norm(target)
    assert _run(WorkspaceCommand(), ["reset"], agent) is True
    assert _norm(_effective(agent)) == _norm(boot)


def test_reset_clears_the_override_attribute(agent, target) -> None:
    _run(WorkspaceCommand(), [str(target)], agent)
    _run(WorkspaceCommand(), ["reset"], agent)
    assert agent._nlp_workspace is None


# ── the cwd/workspace mismatch guardrail ────────────────────────────────────


def test_mismatch_silent_when_equal() -> None:
    here = os.getcwd()
    assert describe_workspace_mismatch(here) == ""


def test_mismatch_warns_with_both_paths(tmp_path) -> None:
    warning = describe_workspace_mismatch(str(tmp_path), cwd=os.getcwd())
    assert "working in" in warning
    assert "started from" in warning


def test_mismatch_silent_for_empty_workspace() -> None:
    assert describe_workspace_mismatch("") == ""
    assert describe_workspace_mismatch(None) == ""


def test_mismatch_survives_garbage() -> None:
    assert describe_workspace_mismatch(object()) == ""  # type: ignore[arg-type]


# ── the command-target divergence guardrail ─────────────────────────────────
#
# ``implement`` and ``workflow`` accept their OWN ``--workspace``.  Interpreting
# it silently forked the command's world from the conversation's: the chat kept
# talking about one checkout while files landed in another.


def test_divergence_silent_when_equal(tmp_path) -> None:
    assert describe_target_divergence(str(tmp_path), str(tmp_path)) == ""


def test_divergence_silent_for_same_path_spelled_differently(tmp_path) -> None:
    spelled = str(tmp_path) + os.sep + "."
    assert describe_target_divergence(spelled, str(tmp_path)) == ""


def test_divergence_warns_with_both_roots(tmp_path) -> None:
    chat = tmp_path / "chat"
    other = tmp_path / "other"
    chat.mkdir()
    other.mkdir()
    warning = describe_target_divergence(str(other), str(chat))
    assert str(other) in warning
    assert str(chat) in warning
    assert "chat is working in" in warning


def test_divergence_silent_without_both_sides(tmp_path) -> None:
    assert describe_target_divergence(None, str(tmp_path)) == ""
    assert describe_target_divergence(str(tmp_path), None) == ""
    assert describe_target_divergence("", "") == ""


def test_divergence_survives_garbage() -> None:
    assert describe_target_divergence(object(), "x") == ""  # type: ignore[arg-type]
    assert describe_target_divergence("x", object()) == ""  # type: ignore[arg-type]


def test_implement_warns_when_workspace_forks_from_chat(tmp_path, capsys) -> None:
    """The real ``implement`` parse path must print the divergence warning."""
    from agent_core.commands.implement_cmd import ImplementCommand

    chat = tmp_path / "chat"
    other = tmp_path / "other"
    chat.mkdir()
    other.mkdir()

    class _Stub:
        workspace = str(chat)
        _nlp_workspace = None

    cmd = ImplementCommand()
    # ``--status`` returns before doing any work; the warning is emitted
    # straight after the argument parse, which is what this asserts.
    try:
        asyncio.run(cmd.execute(["--status", "--workspace", str(other)], _Stub()))
    except Exception:  # noqa: BLE001 - the pipeline is not under test here
        pass
    out = capsys.readouterr().out
    assert "chat is working in" in out
