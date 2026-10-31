"""In-place Agent session switching."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import agent as agent_module
from agent_core.agent import AgentCore
from agent_core.session import (
    activate_session,
    create_session,
    current_session_id,
    reset,
)


class FakeClient:
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


@pytest.fixture(autouse=True)
def _session_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENT_SESSION_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setattr(agent_module, "LLMClient", FakeClient)
    reset()
    yield
    reset()


def test_switch_session_saves_and_reloads_history(tmp_path: Path) -> None:
    create_session("one", workspace=str(tmp_path))
    create_session("two", workspace=str(tmp_path))
    activate_session("one", workspace=str(tmp_path))
    agent = AgentCore(workspace=str(tmp_path))
    agent._chat_history = [
        {"role": "user", "content": "only in one"},
        {"role": "assistant", "content": "ok"},
    ]

    agent.switch_session("two")
    assert current_session_id() == "two"
    assert agent._chat_history == []

    agent._chat_history = [
        {"role": "user", "content": "only in two"},
        {"role": "assistant", "content": "ok"},
    ]
    agent.switch_session("one")
    assert current_session_id() == "one"
    assert agent._chat_history[-2]["content"] == "only in one"


def test_switch_applies_the_session_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_session("one", workspace=str(tmp_path))
    create_session("two", workspace=str(tmp_path), model="x/y", provider="p")
    activate_session("one", workspace=str(tmp_path))
    agent = AgentCore(workspace=str(tmp_path))
    calls: list[tuple[str, str | None]] = []

    def _build(settings, name, provider_override=None):
        calls.append((name, provider_override))
        return SimpleNamespace(context_limit=None)

    monkeypatch.setattr("agent_core.llm.provider.build_provider", _build)

    agent.switch_session("two")
    assert agent.model_name == "x/y"
    assert agent.llm.model_name == "x/y"
    assert calls == [("x/y", "p")]


def test_switch_to_default_restores_the_global_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_session("one", workspace=str(tmp_path), model="x/y", provider="p")
    activate_session("one", workspace=str(tmp_path))
    agent = AgentCore(workspace=str(tmp_path))
    monkeypatch.setattr(
        "agent_core.constants.resolve_model", lambda name=None: "global/model"
    )
    monkeypatch.setattr(
        "agent_core.llm.provider.build_provider",
        lambda settings, name, provider_override=None: SimpleNamespace(),
    )

    agent.switch_session("default")
    assert current_session_id() == "default"
    assert agent.model_name == "global/model"
    assert agent.llm.model_name == "global/model"


def test_agent_init_repairs_a_stale_session_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_session(
        "corrupt",
        workspace=str(tmp_path),
        model="vendor/local-model",
        provider="opencode",
    )
    activate_session("corrupt", workspace=str(tmp_path))
    captured: dict[str, object] = {}

    class _CaptureClient(FakeClient):
        def __init__(self, model_name=None, api_key=None, **kwargs):
            captured.update(kwargs)
            captured["model"] = model_name
            super().__init__(model_name=model_name, api_key=api_key, **kwargs)

    monkeypatch.setattr(agent_module, "LLMClient", _CaptureClient)
    AgentCore(workspace=str(tmp_path))

    assert captured["model"] == "vendor/local-model"
    assert captured["provider_override"] == "lmstudio"


def test_switch_to_same_session_is_noop() -> None:
    create_session("one")
    activate_session("one", workspace="C:/ws")
    agent = AgentCore(workspace=".")
    same = agent.switch_session("one")
    assert same is not None
    assert current_session_id() == "one"
