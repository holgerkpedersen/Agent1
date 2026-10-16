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
    assert not any(l == "AGENT_DISPLAY_MODE=quiet" for l in lines)
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


def test_current_mode_falls_back_to_verbose_on_invalid_env(sandbox, monkeypatch, capsys):
    import agent_core.config as cfg

    def boom(*a, **k):
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(cfg, "load_agent_settings", boom)
    monkeypatch.setenv("AGENT_DISPLAY_MODE", "nonsense")
    assert _run([]) is True
    assert "Display mode is 'verbose'." in capsys.readouterr().out
