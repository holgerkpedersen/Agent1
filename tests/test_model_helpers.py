"""Tests for model command fuzzy matching and model resolution logic."""
import os
import tempfile

import pytest

from agent_core.commands.model_cmd import ModelCommand
from agent_core.constants import (
    resolve_model,
    persist_model_choice,
    load_model_json,
    save_model_json,
    KNOWN_MODELS,
    MODEL_JSON_PATH,
)


class TestModelCommandProviderSync:
    """`model list` / `model reload` must never sync an OPENCODE model to the
    LM Studio-loaded model (the user's chosen model was being silently
    replaced and provider=lmstudio persisted)."""

    @staticmethod
    def _opencode_agent():
        from agent_core.llm.opencode_provider import OpencodeProvider

        provider = OpencodeProvider("opencode-go/deepseek-v4-flash", read_store=False)
        llm = type("L", (), {
            "model_name": "opencode-go/deepseek-v4-flash",
            "_provider": provider,
        })()
        return type("A", (), {"llm": llm})()

    @staticmethod
    def _settings(provider: str):
        return type("S", (), {
            "llm_provider": provider,
            "opencode_server_url": "http://127.0.0.1:4096",
            "opencode_password": "",
            "opencode_api_url": "https://opencode.ai/zen/go/v1",
            "opencode_api_key": "",
        })()

    def test_list_models_skips_autosync_for_opencode(self, monkeypatch, capsys):
        from agent_core.llm.opencode_provider import OpencodeProvider

        cmd = ModelCommand()
        agent = self._opencode_agent()
        fake_models = [{
            "key": "qwen/qwen3.8-27b", "display_name": "Qwen3.8 27B",
            "params_string": "27B", "size_bytes": 100, "loaded": True,
            "instance_id": "i1",
        }]
        monkeypatch.setattr("agent_core.commands.model_cmd._lms.get_models_status", lambda: fake_models)
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: self._settings("opencode"))
        monkeypatch.setattr(
            "agent_core.constants.load_model_json",
            lambda: {"model": "opencode-go/deepseek-v4-flash", "provider": "opencode"},
        )
        monkeypatch.setattr(
            OpencodeProvider, "list_models",
            lambda self: ["opencode-go/deepseek-v4-flash"],
        )
        monkeypatch.setattr(
            "agent_core.commands.model_cmd.persist_model_choice",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not persist on opencode list")),
        )

        cmd._list_models(agent)
        out = capsys.readouterr().out
        assert "switching" not in out.lower()
        assert agent.llm.model_name == "opencode-go/deepseek-v4-flash"

    def test_reload_skips_sync_for_opencode(self, monkeypatch, capsys):
        cmd = ModelCommand()
        agent = self._opencode_agent()
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: self._settings("opencode"))
        monkeypatch.setattr(
            "agent_core.constants.load_model_json",
            lambda: {"model": "opencode-go/deepseek-v4-flash", "provider": "opencode"},
        )

        cmd._sync_with_lmstudio(agent)
        out = capsys.readouterr().out
        assert "no LM Studio sync needed" in out
        assert agent.llm.model_name == "opencode-go/deepseek-v4-flash"

    def test_sync_still_switches_when_lmstudio_active(self, monkeypatch, capsys):
        from agent_core.llm.lmstudio import LMStudioProvider

        provider = LMStudioProvider(model_name="laguna-s-2.1")
        llm = type("L", (), {"model_name": "laguna-s-2.1", "_provider": provider})()
        agent = type("A", (), {"llm": llm})()
        fake_models = [{
            "key": "qwen/qwen3.8-27b", "display_name": "Qwen3.8 27B",
            "params_string": "27B", "size_bytes": 100, "loaded": True,
            "instance_id": "i1",
        }]
        monkeypatch.setattr("agent_core.commands.model_cmd._lms.get_models_status", lambda: fake_models)
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: self._settings("lmstudio"))
        monkeypatch.setattr(
            "agent_core.constants.load_model_json",
            lambda: {"model": "laguna-s-2.1", "provider": "lmstudio"},
        )
        monkeypatch.setattr("agent_core.commands.model_cmd.persist_model_choice", lambda *a, **k: None)

        cmd = ModelCommand()
        cmd._sync_with_lmstudio(agent)
        out = capsys.readouterr().out
        assert "Syncing agent" in out
        assert agent.llm.model_name == "qwen/qwen3.8-27b"


class TestModelCommandResolveMatch:
    """Test _resolve_match without needing LM Studio running."""

    def setup_method(self):
        self.cmd = ModelCommand()
        self.fake_models = [
            {"key": "laguna-s-2.1", "display_name": "Laguna S 2.1 UD", "params_string": "8B-MoE"},
            {"key": "qwen3.5-9b", "display_name": "Qwen3.5 9B", "params_string": "9B"},
            {"key": "qwen3.6-27b-mtp", "display_name": "Qwen3.6 27B", "params_string": "27B"},
            {"key": "google/gemma-4-31b", "display_name": "Gemma 4 31B", "params_string": "31B"},
            {"key": "qwen3.6-35b-a3b-mtp", "display_name": "Qwen3.6 35B A3B", "params_string": "35B-A3B"},
        ]

    def test_exact_key_match(self):
        assert self.cmd._resolve_match("laguna-s-2.1", self.fake_models) == "laguna-s-2.1"

    def test_exact_display_name_match(self):
        assert self.cmd._resolve_match("Qwen3.5 9B", self.fake_models) == "qwen3.5-9b"

    def test_params_string_match_9b(self):
        assert self.cmd._resolve_match("9b", self.fake_models) == "qwen3.5-9b"

    def test_params_string_match_27b(self):
        assert self.cmd._resolve_match("27b", self.fake_models) == "qwen3.6-27b-mtp"

    def test_params_string_match_31b(self):
        assert self.cmd._resolve_match("31b", self.fake_models) == "google/gemma-4-31b"

    def test_params_string_match_35b(self):
        assert self.cmd._resolve_match("35b", self.fake_models) == "qwen3.6-35b-a3b-mtp"

    def test_substring_key_match(self):
        assert self.cmd._resolve_match("gemma", self.fake_models) == "google/gemma-4-31b"

    def test_substring_display_match(self):
        assert self.cmd._resolve_match("laguna", self.fake_models) == "laguna-s-2.1"

    def test_difflib_fallback(self):
        assert self.cmd._resolve_match("qwen", self.fake_models) == "qwen3.5-9b"

    def test_no_match_returns_none(self):
        assert self.cmd._resolve_match("nonexistent-model-xyz", self.fake_models) is None

    def test_empty_query_returns_none(self):
        assert self.cmd._resolve_match("", self.fake_models) is None

    def test_case_insensitive(self):
        assert self.cmd._resolve_match("LAGUNA", self.fake_models) == "laguna-s-2.1"

    def test_unique_display_substring(self):
        assert self.cmd._resolve_match("35B A3B", self.fake_models) == "qwen3.6-35b-a3b-mtp"


class TestResolveOpencodeMatch:
    """Regression: partial opencode names must resolve to the opencode
    catalog (e.g. `model nemotron-3.5-lightning-free`), not be hijacked to an
    LM Studio model.  Covers the user report where `model list` hid the
    opencode models and `model <partial>` switched to LM Studio instead."""

    def setup_method(self):
        self.cmd = ModelCommand()
        self.catalog = [
            "opencode-go/hy3",
            "opencode-go/nemotron-3.5-lightning-free",
            "opencode-go/deepseek-v4-flash",
            "opencode-go/glm-5.2",
        ]

    def test_exact_prefixed_match(self):
        assert self.cmd._resolve_opencode_match(
            "opencode-go/hy3", self.catalog) == "opencode-go/hy3"

    def test_exact_unprefixed_tail_match(self):
        assert self.cmd._resolve_opencode_match(
            "hy3", self.catalog) == "opencode-go/hy3"

    def test_partial_free_model_resolves(self):
        # The exact failing query from the user report.
        assert self.cmd._resolve_opencode_match(
            "nemotron-3.5-lightning-free", self.catalog
        ) == "opencode-go/nemotron-3.5-lightning-free"

    def test_partial_free_model_short(self):
        assert self.cmd._resolve_opencode_match(
            "lightning-free", self.catalog
        ) == "opencode-go/nemotron-3.5-lightning-free"

    def test_substring_unique_tail(self):
        assert self.cmd._resolve_opencode_match(
            "glm", self.catalog) == "opencode-go/glm-5.2"

    def test_empty_catalog_returns_none(self):
        assert self.cmd._resolve_opencode_match("hy3", []) is None

    def test_empty_query_returns_none(self):
        assert self.cmd._resolve_opencode_match("", self.catalog) is None

    def test_ambiguous_substring_returns_none(self):
        # "e" matches multiple tails → no unique match (avoid wrong switch).
        assert self.cmd._resolve_opencode_match("e", self.catalog) is None


class TestModelListShowsOpencodeCatalog:
    """`model list` must show the real opencode catalog (via the agent's
    provider with the resolved API key), not a placeholder with an empty key
    that silently drops every opencode model."""

    @staticmethod
    def _opencode_agent():
        from agent_core.llm.opencode_provider import OpencodeProvider

        provider = OpencodeProvider(
            "opencode-go/nemotron-3.5-lightning-free", api_key="sk-test", read_store=False,
        )
        llm = type("L", (), {
            "model_name": "opencode-go/nemotron-3.5-lightning-free",
            "_provider": provider,
        })()
        return type("A", (), {"llm": llm})()

    @staticmethod
    def _settings():
        return type("S", (), {
            "llm_provider": "opencode",
            "opencode_server_url": "http://127.0.0.1:4096",
            "opencode_password": "",
            "opencode_api_url": "https://opencode.ai/zen/go/v1",
            "opencode_api_key": "sk-test",
        })()

    def test_list_includes_opencode_models_from_real_provider(self, monkeypatch, capsys):
        from agent_core.commands.model_cmd import ModelCommand

        cmd = ModelCommand()
        agent = self._opencode_agent()
        fake_models = [{
            "key": "qwen/qwen3.8-27b", "display_name": "Qwen3.8 27B",
            "params_string": "27B", "size_bytes": 100, "loaded": True,
            "instance_id": "i1",
        }]
        monkeypatch.setattr(
            "agent_core.commands.model_cmd._lms.get_models_status", lambda: fake_models)
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: self._settings())
        monkeypatch.setattr(
            "agent_core.constants.load_model_json",
            lambda: {"model": "opencode-go/nemotron-3.5-lightning-free", "provider": "opencode"},
        )
        # The real provider's list_models is hit; mock the HTTP fetch to a
        # catalog that includes the free model the user wanted.
        monkeypatch.setattr(
            type(agent.llm._provider), "list_models",
            lambda self: [
                "opencode-go/hy3",
                "opencode-go/nemotron-3.5-lightning-free",
                "opencode-go/deepseek-v4-flash",
            ],
        )

        cmd._list_models(agent)
        out = capsys.readouterr().out
        assert "opencode-go/nemotron-3.5-lightning-free" in out
        assert "opencode-go/hy3" in out
        # The current model is marked.
        assert "*" in out


class TestSwitchModelPrefersOpencode:
    """`model <partial opencode name>` must switch to the opencode model,
    never to an LM Studio substring match."""

    @staticmethod
    def _lmstudio_agent_current():
        from agent_core.llm.lmstudio import LMStudioProvider

        provider = LMStudioProvider(model_name="opencode-go/hy3")
        llm = type("L", (), {
            "model_name": "opencode-go/hy3",
            "_provider": provider,
            "_profile_name": None,
        })()
        return type("A", (), {"llm": llm})()

    def test_partial_free_name_switches_to_opencode(self, monkeypatch):
        from agent_core.commands.model_cmd import ModelCommand
        from agent_core.config import load_agent_settings

        cmd = ModelCommand()
        agent = self._lmstudio_agent_current()
        # LM Studio has a nemotron loaded — the buggy code matched this.
        fake_models = [{
            "key": "nvidia/nemotron-3-nano-4b", "display_name": "Nemotron 3 Nano 4B",
            "params_string": "4B", "size_bytes": 100, "loaded": True,
            "instance_id": "i1",
        }]
        monkeypatch.setattr(
            "agent_core.commands.model_cmd._lms.get_models_status", lambda: fake_models)
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: type("S", (), {
            "llm_provider": "opencode",
            "opencode_server_url": "http://127.0.0.1:4096",
            "opencode_password": "",
            "opencode_api_url": "https://opencode.ai/zen/go/v1",
            "opencode_api_key": "sk-test",
        })())
        monkeypatch.setattr(
            "agent_core.constants.load_model_json",
            lambda: {"model": "opencode-go/hy3", "provider": "opencode"},
        )
        # Build a fresh opencode provider that returns the free model.
        monkeypatch.setattr(
            "agent_core.commands.model_cmd.ModelCommand._opencode_catalog",
            lambda self, agent: (["opencode-go/nemotron-3.5-lightning-free",
                                  "opencode-go/hy3"], [], True),
        )
        # Capture the provider that gets built for the switch.
        built: dict[str, object] = {}
        import agent_core.llm.provider as prov_mod
        real_build = prov_mod.build_provider

        def fake_build(settings, model_name):
            built["model"] = model_name
            return real_build(settings, model_name)

        monkeypatch.setattr(prov_mod, "build_provider", fake_build)
        persisted: dict[str, str] = {}
        monkeypatch.setattr(
            "agent_core.commands.model_cmd.persist_model_choice",
            lambda name, provider=None: persisted.setdefault("model", name),
        )

        import asyncio
        asyncio.run(cmd._switch_model(["nemotron-3.5-lightning-free"], agent))

        assert agent.llm.model_name == "opencode-go/nemotron-3.5-lightning-free"
        assert built["model"] == "opencode-go/nemotron-3.5-lightning-free"
        assert persisted["model"] == "opencode-go/nemotron-3.5-lightning-free"


class TestModelListShowsZenFreeTier:
    """Regression: the keyless opencode-zen FREE tier must appear in
    `model list` even without an API key (the user reported free models like
    nemotron-3.5-lightning-free were missing — they live under
    opencode-zen, not opencode-go)."""

    @staticmethod
    def _opencode_agent():
        from agent_core.llm.opencode_provider import OpencodeProvider

        provider = OpencodeProvider(
            "opencode-go/hy3", api_key="sk-test", read_store=False,
        )
        llm = type("L", (), {
            "model_name": "opencode-go/hy3",
            "_provider": provider,
        })()
        return type("A", (), {"llm": llm})()

    def test_list_includes_zen_free_models(self, monkeypatch, capsys):
        from agent_core.commands.model_cmd import ModelCommand

        cmd = ModelCommand()
        agent = self._opencode_agent()
        fake_models = [{
            "key": "qwen/qwen3.8-27b", "display_name": "Qwen3.8 27B",
            "params_string": "27B", "size_bytes": 100, "loaded": True,
            "instance_id": "i1",
        }]
        monkeypatch.setattr(
            "agent_core.commands.model_cmd._lms.get_models_status", lambda: fake_models)
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: type("S", (), {
            "llm_provider": "opencode",
            "opencode_server_url": "http://127.0.0.1:4096",
            "opencode_password": "",
            "opencode_api_url": "https://opencode.ai/zen/go/v1",
            "opencode_api_key": "sk-test",
        })())
        monkeypatch.setattr(
            "agent_core.constants.load_model_json",
            lambda: {"model": "opencode-go/hy3", "provider": "opencode"},
        )
        # Keyed opencode-go catalog + keyless zen free catalog.
        monkeypatch.setattr(
            "agent_core.commands.model_cmd.ModelCommand._opencode_catalog",
            lambda self, agent: (
                ["opencode-go/hy3", "opencode-go/glm-5.2"],
                ["opencode-zen/hy3-free", "opencode-zen/nemotron-3.5-lightning-free",
                 "opencode-zen/laguna-s-2.1-free"],
                True,
            ),
        )

        cmd._list_models(agent)
        out = capsys.readouterr().out
        assert "[opencode-zen]" in out
        assert "opencode-zen/nemotron-3.5-lightning-free" in out
        assert "opencode-zen/hy3-free" in out
        assert "no API key" in out


class TestSwitchModelPrefersZenFree:
    """`model <partial free name>` must switch to the keyless opencode-zen
    free model, never to a paid opencode-go substring or an LM Studio model."""

    @staticmethod
    def _agent():
        from agent_core.llm.lmstudio import LMStudioProvider

        provider = LMStudioProvider(model_name="opencode-go/hy3")
        llm = type("L", (), {
            "model_name": "opencode-go/hy3",
            "_provider": provider,
            "_profile_name": None,
        })()
        return type("A", (), {"llm": llm})()

    def test_partial_free_name_switches_to_zen(self, monkeypatch):
        from agent_core.commands.model_cmd import ModelCommand
        import agent_core.llm.provider as prov_mod

        cmd = ModelCommand()
        agent = self._agent()
        fake_models = [{
            "key": "nvidia/nemotron-3-nano-4b", "display_name": "Nemotron 3 Nano 4B",
            "params_string": "4B", "size_bytes": 100, "loaded": True,
            "instance_id": "i1",
        }]
        monkeypatch.setattr(
            "agent_core.commands.model_cmd._lms.get_models_status", lambda: fake_models)
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: type("S", (), {
            "llm_provider": "opencode",
            "opencode_server_url": "http://127.0.0.1:4096",
            "opencode_password": "",
            "opencode_api_url": "https://opencode.ai/zen/go/v1",
            "opencode_api_key": "sk-test",
        })())
        monkeypatch.setattr(
            "agent_core.constants.load_model_json",
            lambda: {"model": "opencode-go/hy3", "provider": "opencode"},
        )
        # Keyed tier has NO free model; the free one is only in the zen list.
        monkeypatch.setattr(
            "agent_core.commands.model_cmd.ModelCommand._opencode_catalog",
            lambda self, agent: (
                ["opencode-go/hy3", "opencode-go/glm-5.2"],
                ["opencode-zen/hy3-free", "opencode-zen/nemotron-3.5-lightning-free"],
                True,
            ),
        )
        built: dict[str, object] = {}
        real_build = prov_mod.build_provider

        def fake_build(settings, model_name):
            built["model"] = model_name
            return real_build(settings, model_name)

        monkeypatch.setattr(prov_mod, "build_provider", fake_build)
        persisted: dict[str, str] = {}
        monkeypatch.setattr(
            "agent_core.commands.model_cmd.persist_model_choice",
            lambda name, provider=None: persisted.setdefault("model", name),
        )

        import asyncio
        asyncio.run(cmd._switch_model(["nemotron-3.5-lightning-free"], agent))

        assert agent.llm.model_name == "opencode-zen/nemotron-3.5-lightning-free"
        assert built["model"] == "opencode-zen/nemotron-3.5-lightning-free"
        assert persisted["model"] == "opencode-zen/nemotron-3.5-lightning-free"

    def test_explicit_zen_prefix_switches(self, monkeypatch):
        from agent_core.commands.model_cmd import ModelCommand
        import agent_core.llm.provider as prov_mod

        cmd = ModelCommand()
        agent = self._agent()
        monkeypatch.setattr(
            "agent_core.commands.model_cmd._lms.get_models_status", lambda: [])
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: type("S", (), {
            "llm_provider": "opencode",
            "opencode_server_url": "http://127.0.0.1:4096",
            "opencode_password": "",
            "opencode_api_url": "https://opencode.ai/zen/go/v1",
            "opencode_api_key": "sk-test",
        })())
        monkeypatch.setattr(
            "agent_core.constants.load_model_json",
            lambda: {"model": "opencode-go/hy3", "provider": "opencode"},
        )
        built: dict[str, object] = {}
        real_build = prov_mod.build_provider

        def fake_build(settings, model_name):
            built["model"] = model_name
            return real_build(settings, model_name)

        monkeypatch.setattr(prov_mod, "build_provider", fake_build)
        persisted: dict[str, str] = {}
        monkeypatch.setattr(
            "agent_core.commands.model_cmd.persist_model_choice",
            lambda name, provider=None: persisted.setdefault("model", name),
        )

        import asyncio
        asyncio.run(cmd._switch_model(["opencode-zen/hy3-free"], agent))
        assert agent.llm.model_name == "opencode-zen/hy3-free"
        assert built["model"] == "opencode-zen/hy3-free"



class TestResolveModel:
    def test_explicit_argument_takes_priority(self):
        result = resolve_model(explicit="qwen3.6-27b-mtp")
        assert result == "qwen3.6-27b-mtp"

    def test_returns_known_model(self):
        # When no LM Studio, no model.json, should fallback to DEFAULT_MODEL
        # Note: if LM Studio is running, this returns whatever model is loaded
        result = resolve_model(explicit=None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_explicit_none_still_resolves(self):
        result = resolve_model()  # no explicit argument
        assert isinstance(result, str)
        assert len(result) > 0


class TestPersistModelChoice:
    def test_load_empty_returns_dict(self):
        data = load_model_json()
        assert isinstance(data, dict)

    def test_save_and_load_roundtrip(self):
        original = load_model_json()
        try:
            test_data = {"model": "laguna-s-2.1", "test_marker": "pytest"}
            save_model_json(test_data)
            loaded = load_model_json()
            assert loaded["model"] == "laguna-s-2.1"
            assert loaded["test_marker"] == "pytest"
        finally:
            save_model_json(original)
