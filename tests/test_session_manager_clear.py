"""Tests for ClearCommand - the CLI ``clear`` command over the legacy SessionManager.

Target: ``agent_core/cli/commands/clear.py``.  The named-session layer under
``agent_core/session/`` is a different subsystem (covered by
``tests/test_session_layer.py`` and friends); isolation cases at the bottom
prove this command never touches it.

Config loading in ``ClearCommand.__init__`` (``load_agent_settings`` /
``_validate_settings``) is stubbed at the ``agent_core.cli.commands.clear``
import path, so the suite never reads ``model.json``, ``.env`` or the network.

Contract pinned from source:

- ``display_stats()`` logs the message count, token estimate and last
  activity through the module logger; it never mutates the manager.
- ``prompt_user_confirmation()`` re-prompts on any answer other than the
  single letters ``y``/``n``; ``input()`` errors propagate untouched (the
  source has no try/except - a future guard must show up as a test change).
- ``run()`` returns ``"Clear aborted by user."`` without clearing when
  declined, and ``"Session cleared successfully."`` after clearing.
- ``create_clear_command`` wires the factory result to the manager it clears.
"""

import logging
from pathlib import Path

import pytest

from agent_core.cli.commands import clear as clear_mod
from agent_core.cli.commands.clear import ClearCommand, create_clear_command
from agent_core.cli.session.session_manager import SessionManager


@pytest.fixture(autouse=True)
def _stub_config(monkeypatch):
    """Remove the constructor's live config I/O side effect (no file reads)."""
    monkeypatch.setattr(clear_mod, "load_agent_settings", lambda: {}, raising=False)
    monkeypatch.setattr(
        clear_mod, "_validate_settings", lambda settings: None, raising=False
    )


def _cmd(manager=None):
    manager = manager if manager is not None else SessionManager()
    return ClearCommand(manager), manager


def _answers(monkeypatch, *responses):
    """Patch builtins.input to replay `responses`; record the prompts asked."""
    asked = []

    def fake_input(prompt=""):
        asked.append(prompt)
        if not responses:
            raise AssertionError("input() called more times than stubbed")
        return responses[len(asked) - 1]

    monkeypatch.setattr("builtins.input", fake_input)
    return asked


# --------------------------------------------------------------------------
# T4 - construction surface
# --------------------------------------------------------------------------

def test_clear_command_stores_the_injected_manager():
    cmd, manager = _cmd()
    assert cmd._session_manager is manager


def test_clear_command_constructor_reads_settings_once(monkeypatch):
    calls = []

    def fake_load():
        calls.append(1)
        return {}

    monkeypatch.setattr(clear_mod, "load_agent_settings", fake_load, raising=False)
    ClearCommand(SessionManager())
    assert len(calls) == 1


# --------------------------------------------------------------------------
# T5 - stats display
# --------------------------------------------------------------------------

def test_display_stats_logs_count_tokens_and_activity(caplog):
    cmd, manager = _cmd()
    manager.add_message({"role": "user", "content": "x" * 10})   # 10 // 4 == 2 tokens
    with caplog.at_level(logging.INFO, logger=clear_mod.__name__):
        cmd.display_stats()
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "Session stats:" in text
    assert "Messages: 1" in text
    assert "Tokens used: 2" in text
    assert f"Last activity: {manager.get_session_stats()['last_activity']}" in text


def test_display_stats_never_mutates_the_manager(caplog):
    cmd, manager = _cmd()
    manager.add_message({"role": "user", "content": "keep me"})
    before = manager.get_session_stats()
    with caplog.at_level(logging.INFO, logger=clear_mod.__name__):
        cmd.display_stats()
    assert manager.get_session_stats() == before


def test_display_stats_on_empty_manager_logs_zeroes(caplog):
    cmd, _manager = _cmd()
    with caplog.at_level(logging.INFO, logger=clear_mod.__name__):
        cmd.display_stats()
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "Messages: 0" in text
    assert "Tokens used: 0" in text


# --------------------------------------------------------------------------
# T6 - confirmation loop
# --------------------------------------------------------------------------

def test_confirmation_accepts_y(monkeypatch, caplog):
    cmd, _manager = _cmd()
    asked = _answers(monkeypatch, "y")
    with caplog.at_level(logging.INFO, logger=clear_mod.__name__):
        assert cmd.prompt_user_confirmation() is True
    assert len(asked) == 1
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "User confirmed session clear." in logged


def test_confirmation_accepts_n(monkeypatch, caplog):
    cmd, _manager = _cmd()
    asked = _answers(monkeypatch, "n")
    with caplog.at_level(logging.WARNING, logger=clear_mod.__name__):
        assert cmd.prompt_user_confirmation() is False
    assert len(asked) == 1
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "Clear aborted. Conversation context left intact." in logged


def test_confirmation_normalises_case_and_whitespace(monkeypatch):
    cmd, _manager = _cmd()
    asked = _answers(monkeypatch, "  Y  ", " N ")
    assert cmd.prompt_user_confirmation() is True
    assert cmd.prompt_user_confirmation() is False
    assert len(asked) == 2


def test_confirmation_reprompts_on_invalid_answers(monkeypatch, caplog):
    cmd, _manager = _cmd()
    asked = _answers(monkeypatch, "yes", "no", "maybe", "", "y")
    with caplog.at_level(logging.WARNING, logger=clear_mod.__name__):
        assert cmd.prompt_user_confirmation() is True
    # 4 invalid answers each re-prompt, then 'y' accepts: 5 prompts total.
    assert len(asked) == 5
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert text.count("Invalid input. Please respond with 'y' or 'n'.") == 4


def test_confirmation_accepts_only_single_letters(monkeypatch):
    cmd, _manager = _cmd()
    asked = _answers(monkeypatch, "yes", "no", "n")
    assert cmd.prompt_user_confirmation() is False
    assert len(asked) == 3


def test_confirmation_never_clears_on_its_own(monkeypatch):
    cmd, manager = _cmd()
    manager.add_message({"role": "user", "content": "context"})
    _answers(monkeypatch, "y")
    assert cmd.prompt_user_confirmation() is True
    assert manager.get_session_stats()["message_count"] == 1


def test_confirmation_input_errors_propagate_untouched(monkeypatch):
    # The source has no try/except around input(); pinning the raise-through
    # means a future guard shows up as a deliberate, visible test change.
    cmd, _manager = _cmd()

    def boom(prompt=""):
        raise EOFError("stubbed")

    monkeypatch.setattr("builtins.input", boom)
    with pytest.raises(EOFError):
        cmd.prompt_user_confirmation()


def test_confirmation_keyboard_interrupt_propagates_untouched(monkeypatch):
    cmd, _manager = _cmd()

    def boom(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", boom)
    with pytest.raises(KeyboardInterrupt):
        cmd.prompt_user_confirmation()


# --------------------------------------------------------------------------
# T7 - end-to-end run() and factory wiring
# --------------------------------------------------------------------------

def test_run_declined_returns_abort_message_and_keeps_messages(monkeypatch):
    cmd, manager = _cmd()
    manager.add_message({"role": "user", "content": "precious context"})
    _answers(monkeypatch, "n")
    assert cmd.run() == "Clear aborted by user."
    assert manager.get_session_stats()["message_count"] == 1


def test_run_accepted_clears_and_returns_success_message(monkeypatch, caplog):
    cmd, manager = _cmd()
    manager.add_message({"role": "user", "content": "throwaway"})
    _answers(monkeypatch, "y")
    with caplog.at_level(logging.INFO, logger=clear_mod.__name__):
        assert cmd.run() == "Session cleared successfully."
    assert manager.get_session_stats()["message_count"] == 0
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "Session cleared successfully." in logged


def test_run_displays_stats_before_prompting(monkeypatch):
    cmd, manager = _cmd()
    manager.add_message({"role": "user", "content": "me"})
    seen_counts = []

    def fake_input(prompt=""):
        # display_stats runs before the first prompt: stats must already be out.
        seen_counts.append(manager.get_session_stats()["message_count"])
        return "n"

    monkeypatch.setattr("builtins.input", fake_input)
    cmd.run()
    assert seen_counts == [1]


def test_create_clear_command_wires_the_injected_manager(monkeypatch):
    manager = SessionManager()
    cmd = create_clear_command(manager)
    assert isinstance(cmd, ClearCommand)
    assert cmd._session_manager is manager
    manager.add_message({"role": "user", "content": "cleared via factory"})
    _answers(monkeypatch, "y")
    assert cmd.run() == "Session cleared successfully."
    assert manager.get_session_stats()["message_count"] == 0


# --------------------------------------------------------------------------
# T8 - isolation
# --------------------------------------------------------------------------

def test_clearing_one_manager_leaves_a_second_intact(monkeypatch):
    a = SessionManager()
    b = SessionManager()
    a.add_message({"role": "user", "content": "in a"})
    b.add_message({"role": "user", "content": "in b"})
    cmd = ClearCommand(a)
    _answers(monkeypatch, "y")
    assert cmd.run() == "Session cleared successfully."
    assert a.get_session_stats()["message_count"] == 0
    assert b.get_session_stats()["message_count"] == 1


def test_clear_command_touches_no_named_session_state(tmp_path, monkeypatch):
    """The legacy manager owns no files; the named-session state is separate.

    Runs the full clear flow with cwd pointed at a fresh subdir of tmp_path
    and asserts (a) no file anywhere under that subdir was created, and
    (b) every filesystem call that reaches the real ``os`` layer stays inside
    tmp_path - i.e. nothing from ``agent_core/session/`` state is read or
    written by this command.

    The root conftest pre-creates ``tmp_path/runtime_state`` as the global
    runtime-state sandbox, so the observed dir is a fresh subdir; the
    os-level guard keeps its scope at tmp_path.
    """
    import os

    work = tmp_path / "clear_cwd"
    work.mkdir()
    root = str(tmp_path.resolve())

    def _inside(path):
        try:
            return Path(os.path.abspath(str(path))).is_relative_to(root)
        except (OSError, ValueError, TypeError):
            return False

    offenders = []
    real_open = open
    real_remove, real_rename = os.remove, os.rename
    real_replace = os.replace

    def guarded_open(file, *args, **kwargs):
        if not _inside(file):
            offenders.append(("open", str(file)))
        return real_open(file, *args, **kwargs)

    def guarded_remove(path, *args, **kwargs):
        if not _inside(path):
            offenders.append(("remove", str(path)))
        return real_remove(path, *args, **kwargs)

    def guarded_rename(src, dst, *args, **kwargs):
        if not (_inside(src) and _inside(dst)):
            offenders.append(("rename", f"{src} -> {dst}"))
        return real_rename(src, dst, *args, **kwargs)

    def guarded_replace(src, dst, *args, **kwargs):
        if not (_inside(src) and _inside(dst)):
            offenders.append(("replace", f"{src} -> {dst}"))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)
    monkeypatch.setattr(os, "remove", guarded_remove)
    monkeypatch.setattr(os, "rename", guarded_rename)
    monkeypatch.setattr(os, "replace", guarded_replace)
    monkeypatch.chdir(work)

    manager = SessionManager()
    manager.add_message({"role": "user", "content": "legacy only"})
    cmd = ClearCommand(manager)
    _answers(monkeypatch, "y")
    assert cmd.run() == "Session cleared successfully."

    assert offenders == []
    assert list(work.rglob("*")) == []
