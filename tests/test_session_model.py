"""Per-session model persistence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_core.constants as constants
from agent_core.agent import AgentCore
from agent_core.llm.lmstudio import LMStudioProvider
from agent_core.llm.provider import FailoverProvider
from agent_core.session import activate_session, create_session, load_meta, reset


@pytest.fixture(autouse=True)
def _session_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("AGENT_SESSION_ROOT", str(tmp_path / "sessions"))
    monkeypatch.chdir(tmp_path)
    reset()
    yield
    reset()


def test_named_session_model_is_persisted_in_session_meta(tmp_path: Path) -> None:
    create_session("work")
    activate_session("work", workspace="C:/ws")
    model_json = tmp_path / "model.json"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(constants, "MODEL_JSON_PATH", str(model_json))
    try:
        constants.persist_model_choice("opencode-go/example", provider="opencode")
    finally:
        monkeypatch.undo()

    meta = load_meta("work")
    assert meta.model == "opencode-go/example"
    assert meta.provider == "opencode"
    assert not model_json.exists()
    assert not (tmp_path / ".env").exists()


def test_default_session_keeps_global_model_persistence(tmp_path: Path) -> None:
    activate_session("default", workspace="C:/ws")
    model_json = tmp_path / "model.json"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(constants, "MODEL_JSON_PATH", str(model_json))
    try:
        constants.persist_model_choice("vendor/default-model", provider="lmstudio")
    finally:
        monkeypatch.undo()

    assert model_json.is_file()
    assert load_meta("default").model == ""


class TestSessionProviderName:
    def test_bare_lmstudio_model_records_lmstudio_not_the_default_cloud(
        self,
    ) -> None:
        provider = LMStudioProvider(model_name="vendor/local-model")
        agent = SimpleNamespace(
            llm=SimpleNamespace(_provider=provider),
            model_name="vendor/local-model",
        )
        assert AgentCore._session_provider_name(agent) == "lmstudio"

    def test_failover_wrapper_records_the_provider_serving_the_model(self) -> None:
        lms = LMStudioProvider(model_name="vendor/local-model")
        other = LMStudioProvider(model_name="other")
        provider = FailoverProvider([lms, other], "vendor/local-model")
        agent = SimpleNamespace(
            llm=SimpleNamespace(_provider=provider),
            model_name="vendor/local-model",
        )
        assert AgentCore._session_provider_name(agent) == "lmstudio"

    @pytest.mark.parametrize(
        ("model_name", "expected"),
        [
            ("vendor/local-model", "lmstudio"),
            ("another/agent-model", "lmstudio"),
            ("llama/local-backend-a", "llama"),
            ("llama/local-backend-b", "llama"),
        ],
    )
    def test_stale_opencode_provider_for_local_model_is_repaired(
        self, model_name: str, expected: str
    ) -> None:
        assert (
            AgentCore._validated_session_provider(model_name, "opencode")
            == expected
        )

    @pytest.mark.parametrize(
        ("model_name", "expected"),
        [
            ("opencode-go/example", "opencode"),
            ("opencode-zen/example", "opencode"),
            ("openrouter/example", "openrouter"),
            ("lemonade/example", "lemonade"),
        ],
    )
    def test_explicit_non_lmstudio_route_is_preserved(
        self, model_name: str, expected: str
    ) -> None:
        assert (
            AgentCore._validated_session_provider(model_name, "opencode")
            == expected
        )

    def test_matching_lmstudio_provider_is_kept(self) -> None:
        assert (
            AgentCore._validated_session_provider(
                "vendor/local-model", "lmstudio"
            )
            == "lmstudio"
        )

    def test_build_transport_keeps_the_session_provider(self) -> None:
        from agent_core.llm.client import LLMClient

        client = LLMClient("vendor/local-model", provider_override="lmstudio")
        transport = AgentCore.build_transport(None, client)  # type: ignore[arg-type]
        assert transport is client._provider
        assert type(transport.providers[0]).__name__ == "LMStudioProvider"
