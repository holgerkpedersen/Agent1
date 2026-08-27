"""Tests for the `model unload` command with the llama provider."""
from types import SimpleNamespace

import pytest

from agent_core.commands.model_cmd import ModelCommand
from agent_core.llm.provider import build_provider


def _settings(provider="llama", llama_base_url="http://127.0.0.1:8080/v1"):
    return SimpleNamespace(
        llm_provider=provider,
        opencode_server_url="http://127.0.0.1:4096",
        opencode_password="",
        opencode_api_url="https://opencode.ai/zen/go/v1",
        opencode_api_key="",
        llama_base_url=llama_base_url,
    )


def _agent(model_name="llama/laguna"):
    """Build a minimal agent-like object with a real LlamaProvider."""
    settings = _settings()
    agent = SimpleNamespace()
    agent.llm = SimpleNamespace()
    agent.llm.model_name = model_name
    agent.llm._provider = build_provider(settings, model_name)
    return agent


class TestUnloadLlamaProvider:
    """`model unload <name>` should route to llama-server shutdown when the
    current provider is llama, not LM Studio."""

    @pytest.mark.anyio
    async def test_unload_routes_to_llama_shutdown(self, monkeypatch):
        cmd = ModelCommand()
        agent = _agent("llama/laguna")

        captured: dict = {}

        def _fake_shutdown():
            captured["called"] = True
            return True, "llama-server shut down (model unloaded)"

        # Monkey-patch shutdown on the provider instance.
        agent.llm._provider.shutdown = _fake_shutdown

        await cmd._unload_model(["laguna"], agent)

        assert captured.get("called") is True

    @pytest.mark.anyio
    async def test_unload_with_no_query_shuts_down_server(self, monkeypatch):
        """`model unload` (no args) still shuts down the llama-server."""
        cmd = ModelCommand()
        agent = _agent("llama/laguna")

        captured: dict = {}

        def _fake_shutdown():
            captured["called"] = True
            return True, "llama-server shut down (model unloaded)"

        agent.llm._provider.shutdown = _fake_shutdown

        await cmd._unload_model([], agent)

        assert captured.get("called") is True

    @pytest.mark.anyio
    async def test_unload_ignores_query_for_llama(self, monkeypatch, capsys):
        """llama-server holds a single model; the query target is ignored."""
        cmd = ModelCommand()
        agent = _agent("llama/laguna")

        def _fake_shutdown():
            return True, "llama-server shut down (model unloaded)"

        agent.llm._provider.shutdown = _fake_shutdown

        await cmd._unload_model(["some-other-model"], agent)

        out = capsys.readouterr().out
        assert "Shutting down" in out
