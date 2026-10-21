"""Tests for the `display` REPL command (agent_core/commands/display_cmd.py).

The command persists the chosen mode to ``.env`` in the current working
directory, so every test runs inside a temporary directory to avoid touching
the real project ``.env``.
"""

import asyncio

import pytest

from agent_core.commands.display_cmd import DisplayCommand


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Run in an isolated cwd and clear any inherited display-mode env var."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_DISPLAY_MODE", raising=False)
    return tmp_path


def _run(args):
    return asyncio.run(DisplayCommand().execute(args, agent=None))


def test_name_and_help_mention_intent():
    cmd = DisplayCommand()
    assert cmd.name == "display"
    assert "intent" in cmd.help_text


def test_no_args_shows_current_mode_and_lists_intent(sandbox, capsys):
    assert _run([]) is True
    out = capsys.readouterr().out
    assert "Display mode is" in out
    assert "intent" in out


def test_set_intent_returns_true_and_persists_to_env(sandbox, capsys):
    assert _run(["intent"]) is True
    assert "Display mode set to 'intent'" in capsys.readouterr().out
    env_text = (sandbox / ".env").read_text(encoding="utf-8")
    assert "AGENT_DISPLAY_MODE=intent" in env_text


def test_set_intent_is_case_and_whitespace_insensitive(sandbox):
    assert _run(["  INTENT  "]) is True
    assert "AGENT_DISPLAY_MODE=intent" in (sandbox / ".env").read_text(encoding="utf-8")


def test_unknown_mode_is_rejected_and_env_not_written(sandbox, capsys):
    assert _run(["bogus"]) is False
    out = capsys.readouterr().out
    assert "Unknown mode 'bogus'" in out
    assert "intent" in out  # options list includes the new mode
    assert not (sandbox / ".env").exists()


def test_existing_display_line_is_replaced_not_duplicated(sandbox):
    (sandbox / ".env").write_text(
        "FOO=1\nAGENT_DISPLAY_MODE=quiet\nBAR=2\n", encoding="utf-8"
    )
    assert _run(["intent"]) is True
    lines = (sandbox / ".env").read_text(encoding="utf-8").splitlines()
    assert lines.count("AGENT_DISPLAY_MODE=intent") == 1
    assert not any(ln == "AGENT_DISPLAY_MODE=quiet" for ln in lines)
    assert "FOO=1" in lines and "BAR=2" in lines


def test_append_when_env_has_no_display_line(sandbox):
    (sandbox / ".env").write_text("FOO=1\n", encoding="utf-8")
    assert _run(["clean"]) is True
    text = (sandbox / ".env").read_text(encoding="utf-8")
    assert "FOO=1" in text
    assert text.count("AGENT_DISPLAY_MODE=clean") == 1


def test_current_mode_reflects_env_when_settings_fail(sandbox, monkeypatch, capsys):
    import agent_core.config as cfg

    def boom(*a, **k):
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(cfg, "load_agent_settings", boom)
    monkeypatch.setenv("AGENT_DISPLAY_MODE", "intent")
    assert _run([]) is True
    assert "Display mode is 'intent'." in capsys.readouterr().out


def test_current_mode_falls_back_to_verbose_on_invalid_env(
    sandbox, monkeypatch, capsys
):
    import agent_core.config as cfg

    def boom(*a, **k):
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(cfg, "load_agent_settings", boom)
    monkeypatch.setenv("AGENT_DISPLAY_MODE", "nonsense")
    assert _run([]) is True
    assert "Display mode is 'verbose'." in capsys.readouterr().out


class TestResolveDisplayMode:
    """Regression tests for ``agent_core.run_utils.resolve_display_mode``.

    ``chat_nlp`` feeds this value straight into ``ToolLoopRunner``, so a bad
    return silently forces VERBOSE for every mode while ``display`` keeps
    reporting the mode as set.  History: the t00 import-block extraction
    replaced agent.py's original ``_resolve_display_mode`` with
    ``getattr(settings, "display_mode", AgentDisplayMode.ANSI)`` — ``ANSI`` was
    never a member, arguments evaluate before ``getattr`` runs, so every call
    raised ``AttributeError``, was swallowed, and returned ``0`` (mapped to
    VERBOSE by the loop).  intent/quiet/clean were dead while every test kept
    passing because they monkeypatch the resolver instead of calling it.
    """

    def test_env_intent_resolves_to_intent_member(self, sandbox, monkeypatch):
        from agent_core.config import AgentDisplayMode
        from agent_core.run_utils import resolve_display_mode

        monkeypatch.setenv("AGENT_DISPLAY_MODE", "intent")
        assert resolve_display_mode() is AgentDisplayMode.INTENT

    def test_agent_shim_alias_returns_real_mode(self, sandbox, monkeypatch):
        import agent
        from agent_core.config import AgentDisplayMode

        monkeypatch.setenv("AGENT_DISPLAY_MODE", "intent")
        assert agent._resolve_display_mode() is AgentDisplayMode.INTENT

    def test_loop_runner_receives_intent_from_real_resolver(self, sandbox, monkeypatch):
        from agent_core.config import AgentDisplayMode
        from agent_core.llm.tool_loop import DisplayMode, ToolLoopRunner
        from agent_core.run_utils import resolve_display_mode

        monkeypatch.setenv("AGENT_DISPLAY_MODE", "intent")
        mode = resolve_display_mode()
        assert isinstance(mode, AgentDisplayMode)  # never a sentinel like 0
        loop = ToolLoopRunner(max_iterations=1, display_mode=mode)
        assert loop.display_mode is DisplayMode.INTENT

    def test_settings_failure_falls_back_to_env_not_sentinel(
        self, sandbox, monkeypatch
    ):
        import agent_core.config as cfg
        from agent_core.config import AgentDisplayMode
        from agent_core.run_utils import resolve_display_mode

        def boom(*a, **k):
            raise RuntimeError("settings unavailable")

        monkeypatch.setattr(cfg, "load_agent_settings", boom)
        monkeypatch.setenv("AGENT_DISPLAY_MODE", "quiet")
        assert resolve_display_mode() is AgentDisplayMode.QUIET

    def test_missing_and_invalid_env_default_to_verbose(self, sandbox, monkeypatch):
        from agent_core.config import AgentDisplayMode
        from agent_core.run_utils import resolve_display_mode

        # sandbox clears AGENT_DISPLAY_MODE and cwd has no .env
        assert resolve_display_mode() is AgentDisplayMode.VERBOSE
        monkeypatch.setenv("AGENT_DISPLAY_MODE", "nonsense")
        assert resolve_display_mode() is AgentDisplayMode.VERBOSE


class TestIntentSystemSuffix:
    """``intent_mode_system_suffix`` steers the model to narrate a one-line
    intent with every tool-call round when (and only when) INTENT mode is on.

    History: the display showed whatever narration the model produced — but
    models rarely write content with tool_calls, so the intent block was
    usually a bare call list.  The suffix nudges the model; the display's
    per-call ``why:`` fallback covers rounds that still arrive bare.
    """

    def test_suffix_present_when_intent_mode(self, sandbox, monkeypatch):
        from agent_core.run_utils import intent_mode_system_suffix

        monkeypatch.setenv("AGENT_DISPLAY_MODE", "intent")
        text = intent_mode_system_suffix()
        assert "INTENT DISPLAY MODE" in text
        assert "ONE short" in text

    def test_suffix_empty_in_other_modes(self, sandbox, monkeypatch):
        from agent_core.run_utils import intent_mode_system_suffix

        monkeypatch.setenv("AGENT_DISPLAY_MODE", "verbose")
        assert intent_mode_system_suffix() == ""
        monkeypatch.setenv("AGENT_DISPLAY_MODE", "clean")
        assert intent_mode_system_suffix() == ""
        monkeypatch.setenv("AGENT_DISPLAY_MODE", "quiet")
        assert intent_mode_system_suffix() == ""

    def test_strip_removes_intent_suffix_no_accumulation(self, sandbox, monkeypatch):
        from agent_core.run_utils import (
            _INTENT_SYSTEM_SUFFIX,
            strip_dynamic_system_blocks,
        )

        monkeypatch.setenv("AGENT_DISPLAY_MODE", "intent")
        text = "BASE PROMPT" + _INTENT_SYSTEM_SUFFIX
        assert strip_dynamic_system_blocks(text) == "BASE PROMPT"
