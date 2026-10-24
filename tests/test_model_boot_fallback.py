"""Regression tests: boot-fallback / Jev-poll / model-name honesty (2026-10-24).

Live incident: a REPL session persisted ``llama/swift-1.5-iq3_s`` but
``resolve_model()`` gated it on the LM-Studio-era ``KNOWN_MODELS`` shortlist
(it is absent), silently discarded it, and the first-run live-poll fallback
then adopted the Jev decision model (``qwen2.5-coder-1.5b-instruct``, a 1.5B
non-agent model the Jev engine had loaded into VRAM) as the chat model.  The
session lost tool calling entirely.  A second trace-honesty bug compounded
diagnosis: ``Agent.model_name`` was frozen at boot while ``model`` switches
only updated ``agent.llm.model_name``, so every trace named the wrong model.

Fix contracts pinned here:

1. A persisted NON-lmstudio model (``llama/``, ``lemonade/``, ``openrouter/``
   ...) is authoritative — returned even when absent from ``KNOWN_MODELS``,
   before the live poll is consulted.
2. The live-poll first-run fallback NEVER adopts the configured Jev model
   (it cannot drive the tool loop); it takes the next loaded model or falls
   through to ``DEFAULT_MODEL``.
3. ``AgentCore.model_name`` is a live property delegating to ``llm.model_name``
   (falling back to the boot value before the client exists), so trace meta
   stamped per turn names the model actually answering.
4. ``chat_nlp`` warns loudly when a pasted image is trimmed out of context by
   the history budget (the model will NOT see it), instead of answering as if
   the image were visible.
"""

from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_models(*loaded_keys: str) -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "display_name": "Fake",
            "params_string": "9B",
            "size_bytes": 1_000_000,
            "loaded": True,
            "instance_id": f"inst-{i}",
        }
        for i, key in enumerate(loaded_keys)
    ]


def _settings(provider: str = "lmstudio", jev: str = "qwen2.5-coder-1.5b-instruct"):
    return type(
        "S",
        (),
        {
            "llm_provider": provider,
            "opencode_model": "opencode-go/deepseek-v4-flash",
            "jev_model": jev,
        },
    )()


# ---------------------------------------------------------------------------
# 1. Persisted non-lmstudio models survive the KNOWN_MODELS gate
# ---------------------------------------------------------------------------


class TestPersistedProviderNamespacedModelWins:
    @staticmethod
    def _patch(monkeypatch, persisted: dict[str, Any]) -> None:
        import agent_core.constants as const

        monkeypatch.setattr(const, "load_model_json", lambda: persisted)
        monkeypatch.setattr(
            "agent_core.config.load_agent_settings", lambda: _settings()
        )

    def test_persisted_llama_model_not_in_known_models_is_kept(self, monkeypatch):
        """The exact 2026-10-24 incident: llama/swift-1.5-iq3_s persisted,
        Jev model loaded in VRAM — boot must return the llama model."""
        import agent_core.constants as const

        self._patch(
            monkeypatch,
            {"model": "llama/swift-1.5-iq3_s", "provider": "llama"},
        )
        # Live poll would show the Jev model loaded (the trap that bit us).
        monkeypatch.setattr(
            "agent_core.llm.lmstudio.get_models_status",
            lambda: _fake_models("qwen2.5-coder-1.5b-instruct"),
        )
        assert "llama/swift-1.5-iq3_s" not in const.KNOWN_MODELS
        assert const.resolve_model(None) == "llama/swift-1.5-iq3_s"

    def test_persisted_lemonade_model_is_kept(self, monkeypatch):
        import agent_core.constants as const

        self._patch(
            monkeypatch,
            {"model": "lemonade/qwen3-4b", "provider": "lemonade"},
        )
        assert const.resolve_model(None) == "lemonade/qwen3-4b"

    def test_persisted_openrouter_model_is_kept(self, monkeypatch):
        import agent_core.constants as const

        self._patch(
            monkeypatch,
            {"model": "openrouter/qwen/qwen3-coder", "provider": "openrouter"},
        )
        assert const.resolve_model(None) == "openrouter/qwen/qwen3-coder"

    def test_known_lmstudio_persisted_model_still_wins(self, monkeypatch):
        """No regression of the multi-shell pin: KNOWN_MODELS entries keep
        beating the live poll as before."""
        import agent_core.constants as const

        self._patch(
            monkeypatch, {"model": "laguna-s-2.1", "provider": "lmstudio"}
        )
        monkeypatch.setattr(
            "agent_core.llm.lmstudio.get_models_status",
            lambda: _fake_models("qwen3.5-9b-mtp"),
        )
        assert const.resolve_model(None) == "laguna-s-2.1"


# ---------------------------------------------------------------------------
# 2. Live-poll fallback never adopts the Jev model
# ---------------------------------------------------------------------------


class TestLivePollSkipsJevModel:
    @staticmethod
    def _patch(monkeypatch, persisted: dict[str, Any], jev: str) -> None:
        import agent_core.constants as const

        monkeypatch.setattr(const, "load_model_json", lambda: persisted)
        monkeypatch.setattr(
            "agent_core.config.load_agent_settings", lambda: _settings(jev=jev)
        )

    def test_poll_adopts_next_loaded_model_not_jev(self, monkeypatch):
        """Jev model loaded first, real model second — boot takes the real
        one, not the 1.5B decision model."""
        import agent_core.constants as const

        self._patch(monkeypatch, {}, "qwen2.5-coder-1.5b-instruct")
        monkeypatch.setattr(
            "agent_core.llm.lmstudio.get_models_status",
            lambda: _fake_models(
                "qwen2.5-coder-1.5b-instruct", "qwen/qwen3.8-27b"
            ),
        )
        assert const.resolve_model(None) == "qwen/qwen3.8-27b"

    def test_poll_with_only_jev_loaded_falls_through_to_default(self, monkeypatch):
        """Nothing but the Jev model in VRAM → never adopt it; fall through
        to the catalog default."""
        import agent_core.constants as const

        self._patch(monkeypatch, {}, "qwen2.5-coder-1.5b-instruct")
        monkeypatch.setattr(
            "agent_core.llm.lmstudio.get_models_status",
            lambda: _fake_models("qwen2.5-coder-1.5b-instruct"),
        )
        assert const.resolve_model(None) == const.DEFAULT_MODEL

    def test_jev_match_is_prefix_agnostic(self, monkeypatch):
        """A provider-namespaced Jev id (lemonade/qwen2.5-coder-1.5b-instruct)
        is still excluded via the bare-name tail comparison."""
        import agent_core.constants as const

        self._patch(monkeypatch, {}, "lemonade/qwen2.5-coder-1.5b-instruct")
        monkeypatch.setattr(
            "agent_core.llm.lmstudio.get_models_status",
            lambda: _fake_models(
                "lmstudio/qwen2.5-coder-1.5b-instruct", "google/gemma-4-12b"
            ),
        )
        assert const.resolve_model(None) == "google/gemma-4-12b"


# ---------------------------------------------------------------------------
# 3. Agent.model_name is live (trace-meta honesty)
# ---------------------------------------------------------------------------


class TestAgentModelNameIsLive:
    @staticmethod
    def _bare_agent():
        """AgentCore without running __init__ (no providers, no history)."""
        from agent_core.agent import AgentCore

        return object.__new__(AgentCore)

    def test_falls_back_to_boot_value_before_llm_exists(self):
        agent = self._bare_agent()
        agent.model_name = "boot-model"
        assert agent.model_name == "boot-model"

    def test_follows_llm_after_switch(self):
        agent = self._bare_agent()
        agent.model_name = "boot-model"
        agent.llm = type("L", (), {"model_name": "switched-model"})()
        assert agent.model_name == "switched-model"

    def test_empty_llm_model_name_falls_back_to_boot(self):
        agent = self._bare_agent()
        agent.model_name = "boot-model"
        agent.llm = type("L", (), {"model_name": ""})()
        assert agent.model_name == "boot-model"

    def test_property_stored_via_setter_not_plain_attribute(self):
        """A plain instance attribute would shadow the property; the setter
        must route through ``_model_name_boot``."""
        agent = self._bare_agent()
        agent.model_name = "x"
        assert agent.__dict__.get("model_name") is None
        assert agent._model_name_boot == "x"


# ---------------------------------------------------------------------------
# 4. Oversized pasted images are reported, not silently dropped
# ---------------------------------------------------------------------------


class TestImageDropIsReported:
    def test_helper_detects_multimodal_image_blocks(self):
        from agent_core.agent import _has_image_block

        assert _has_image_block(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAA"},
                    },
                ],
            }
        )
        assert not _has_image_block({"role": "user", "content": "plain text"})
        assert not _has_image_block(
            {"role": "user", "content": [{"type": "text", "text": "only text"}]}
        )

    def test_oversized_image_trimmed_with_warning(self, capsys):
        """A paste larger than the history budget is dropped by
        ``_trim_chat_history``; chat_nlp must say so (the model cannot see
        it) instead of answering as if the image were visible."""
        from unittest.mock import patch

        from agent import Agent

        agent = Agent(workspace=".")
        big = "data:image/png;base64," + "A" * 20_000

        async def fake_run(self, messages, **kwargs):
            return "ok", list(messages)

        with patch(
            "agent_core.llm.tool_loop.ToolLoopRunner.run", new=fake_run
        ):
            import asyncio

            asyncio.run(agent.chat_nlp("look at this", images=[big]))

        out = capsys.readouterr().out
        assert "[image] dropped from context" in out
        # The image block is gone from the surviving history.
        from agent_core.agent import _has_image_block

        assert not any(_has_image_block(m) for m in agent._chat_history)

    def test_small_image_survives_without_warning(self, capsys):
        from unittest.mock import patch

        from agent import Agent

        agent = Agent(workspace=".")
        small = "data:image/png;base64,AAA"

        async def fake_run(self, messages, **kwargs):
            return "ok", list(messages)

        with patch(
            "agent_core.llm.tool_loop.ToolLoopRunner.run", new=fake_run
        ):
            import asyncio

            asyncio.run(agent.chat_nlp("look at this", images=[small]))

        out = capsys.readouterr().out
        assert "[image] dropped from context" not in out
        from agent_core.agent import _has_image_block

        assert any(_has_image_block(m) for m in agent._chat_history)
