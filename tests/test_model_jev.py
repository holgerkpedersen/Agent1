"""`model jev` — first-class setup for the DEDICATED Jev decision model.

The Jev model must be independent of the selected chat model and persist
across sessions: `model jev <name>` writes model.json + .env
(AGENT_JEV_MODEL), and `load_agent_settings` reads it back without the user
hand-editing .env.
"""

import asyncio
import json
from pathlib import Path

import pytest

from agent_core.constants import persist_jev_model


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Never touch the real model.json / .env from these tests."""
    import agent_core.constants as constants

    monkeypatch.setattr(constants, "MODEL_JSON_PATH", str(tmp_path / "model.json"))
    monkeypatch.delenv("AGENT_JEV_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    # No LM Studio network calls.
    import agent_core.llm.lmstudio as lmstudio

    monkeypatch.setattr(lmstudio, "get_models_status", lambda: [])
    yield tmp_path


def _run(args):
    from agent_core.commands.model_cmd import ModelCommand

    return asyncio.run(ModelCommand().execute(args, object()))


def test_model_help_mentions_jev():
    from agent_core.commands.model_cmd import ModelCommand

    assert "jev" in ModelCommand().help_text


def test_model_jev_shows_current_model(_isolate, capsys):
    assert _run(["jev"]) is True
    out = capsys.readouterr().out
    assert "Jev model: qwen2.5-coder-1.5b-instruct" in out
    assert "provider=LMStudioProvider" in out
    assert "Independent of the selected chat model" in out
    assert "model jev <name>" in out


def test_model_jev_sets_and_persists(_isolate, capsys):
    assert _run(["jev", "qwen2.5-coder-1.5b-instruct"]) is True
    out = capsys.readouterr().out
    assert "Jev model set: qwen2.5-coder-1.5b-instruct" in out
    assert "AGENT_JEV_MODEL" in out

    model_json = json.loads((_isolate / "model.json").read_text(encoding="utf-8"))
    assert model_json["jev_model"] == "qwen2.5-coder-1.5b-instruct"
    env_text = (_isolate / ".env").read_text(encoding="utf-8")
    assert "AGENT_JEV_MODEL=qwen2.5-coder-1.5b-instruct" in env_text


def test_persist_jev_model_updates_existing_env_line(_isolate):
    (_isolate / ".env").write_text(
        "AGENT_MODEL=keep-me\nAGENT_JEV_MODEL=old-model\n", encoding="utf-8",
    )
    persist_jev_model("new-model")
    env_text = (_isolate / ".env").read_text(encoding="utf-8")
    assert "AGENT_MODEL=keep-me" in env_text
    assert "AGENT_JEV_MODEL=new-model" in env_text
    assert "old-model" not in env_text


def test_load_agent_settings_reads_persisted_jev_model(_isolate):
    from agent_core.config import load_agent_settings

    persist_jev_model("qwen2.5-coder-1.5b-instruct")
    settings = load_agent_settings(env_path=Path(_isolate) / "no-such.env")
    assert settings.jev_model == "qwen2.5-coder-1.5b-instruct"


def test_env_beats_persisted_jev_model(_isolate, monkeypatch):
    from agent_core.config import load_agent_settings

    persist_jev_model("persisted-model")
    monkeypatch.setenv("AGENT_JEV_MODEL", "env-model")
    settings = load_agent_settings(env_path=Path(_isolate) / "no-such.env")
    assert settings.jev_model == "env-model"
