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
