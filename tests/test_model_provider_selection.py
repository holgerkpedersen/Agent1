"""Regression tests for explicit provider selection via `model <name> --provider <p>`.

Covers the user report: a model name like `laguna-s-2.1` is normally routed to
LM Studio (prefix-based), but the user wants to override that and route it to
opencode instead — and vice-versa for an opencode name routed to LM Studio.

The `--provider`/`-p` flag bypasses prefix-based routing so the user can pick
the provider explicitly.  Without the flag, routing is unchanged.
"""
import pytest

from agent_core.commands.model_cmd import ModelCommand


def _settings(provider: str = "lmstudio"):
    return type("S", (), {
        "llm_provider": provider,
        "opencode_server_url": "http://127.0.0.1:4096",
        "opencode_password": "",
        "opencode_api_url": "https://opencode.ai/zen/go/v1",
        "opencode_api_key": "",
    })


def _lmstudio_agent(model_name: str = "laguna-s-2.1"):
    from agent_core.llm.lmstudio import LMStudioProvider
    provider = LMStudioProvider(model_name=model_name)
    llm = type("L", (), {"model_name": model_name, "_provider": provider})()
    return type("A", (), {"llm": llm})


def _opencode_agent(model_name: str = "opencode-go/deepseek-v4-flash"):
    from agent_core.llm.opencode_provider import OpencodeProvider
    provider = OpencodeProvider(model_name, read_store=False)
    llm = type("L", (), {"model_name": model_name, "_provider": provider})()
    return type("A", (), {"llm": llm})


# ---------------------------------------------------------------------------
#  Flag parsing
# ---------------------------------------------------------------------------

class TestParseProviderFlag:
    def setup_method(self):
        self.cmd = ModelCommand()

    def test_long_flag_stripped(self):
        args, override = self.cmd._parse_provider_flag(["laguna-s-2.1", "--provider", "opencode"])
        assert args == ["laguna-s-2.1"]
        assert override == "opencode"

    def test_short_flag_stripped(self):
        args, override = self.cmd._parse_provider_flag(["laguna-s-2.1", "-p", "lmstudio"])
        assert args == ["laguna-s-2.1"]
        assert override == "lmstudio"

    def test_equals_form(self):
        args, override = self.cmd._parse_provider_flag(["laguna-s-2.1", "--provider=opencode"])
        assert args == ["laguna-s-2.1"]
        assert override == "opencode"

    def test_no_flag_returns_none(self):
        args, override = self.cmd._parse_provider_flag(["laguna-s-2.1"])
        assert args == ["laguna-s-2.1"]
        assert override is None

    def test_flag_lowercased(self):
        args, override = self.cmd._parse_provider_flag(["laguna-s-2.1", "--provider", "OpenCode"])
        assert override == "opencode"

    def test_flag_at_start(self):
        args, override = self.cmd._parse_provider_flag(["-p", "opencode", "laguna-s-2.1"])
        assert args == ["laguna-s-2.1"]
        assert override == "opencode"

    def test_flag_no_value_left_in_args(self):
        """A flag with no following value should NOT set an override."""
        args, override = self.cmd._parse_provider_flag(["laguna-s-2.1", "--provider"])
        assert override is None
        assert "--provider" in args


# ---------------------------------------------------------------------------
#  build_provider override
# ---------------------------------------------------------------------------

class TestBuildProviderOverride:
    def test_override_routes_lmstudio_name_to_opencode(self, monkeypatch):
        """A `laguna-*` model name, when given provider_override='opencode',
        must build an OpencodeProvider instead of LMStudioProvider."""
        from agent_core.llm.opencode_provider import OpencodeProvider
        from agent_core.llm.lmstudio import LMStudioProvider

        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: _settings("lmstudio"))
        monkeypatch.setattr(
            "agent_core.constants.load_model_json",
            lambda: {"model": "laguna-s-2.1", "provider": "lmstudio"},
        )

        prov = build_provider(_settings("lmstudio"), "laguna-s-2.1", provider_override="opencode")
        assert isinstance(prov, OpencodeProvider)
        assert prov.model_name == "laguna-s-2.1"
        assert prov.zen_mode is False  # not a zen prefix → opencode-go mode

    def test_override_routes_opencode_name_to_lmstudio(self, monkeypatch):
        """An `opencode-go/...` model name, when given provider_override='lmstudio',
        must build an LMStudioProvider instead of OpencodeProvider."""
        from agent_core.llm.opencode_provider import OpencodeProvider
        from agent_core.llm.lmstudio import LMStudioProvider

        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: _settings("lmstudio"))
        monkeypatch.setattr(
            "agent_core.constants.load_model_json",
            lambda: {"model": "opencode-go/hy3", "provider": "opencode"},
        )

        prov = build_provider(_settings("lmstudio"), "opencode-go/hy3", provider_override="lmstudio")
        assert isinstance(prov, LMStudioProvider)
        assert prov.model_name == "opencode-go/hy3"

    def test_no_override_keeps_prefix_routing(self, monkeypatch):
        """Without override, prefix-based routing must be unchanged."""
        from agent_core.llm.opencode_provider import OpencodeProvider
        from agent_core.llm.lmstudio import LMStudioProvider

        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: _settings("lmstudio"))
        monkeypatch.setattr("agent_core.constants.load_model_json", lambda: {})

        # laguna-* → lmstudio (prefix rule)
        prov = build_provider(_settings("lmstudio"), "laguna-s-2.1")
        assert isinstance(prov, LMStudioProvider)

        # opencode-go/* → opencode (prefix rule)
        prov = build_provider(_settings("lmstudio"), "opencode-go/hy3")
        assert isinstance(prov, OpencodeProvider)

    def test_override_none_falls_back_to_normal_routing(self, monkeypatch):
        from agent_core.llm.lmstudio import LMStudioProvider

        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: _settings("lmstudio"))
        monkeypatch.setattr("agent_core.constants.load_model_json", lambda: {})

        prov = build_provider(_settings("lmstudio"), "laguna-s-2.1", provider_override=None)
        assert isinstance(prov, LMStudioProvider)

    def test_invalid_override_ignored(self, monkeypatch):
        """An unrecognized override value must fall back to normal routing."""
        from agent_core.llm.lmstudio import LMStudioProvider

        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: _settings("lmstudio"))
        monkeypatch.setattr("agent_core.constants.load_model_json", lambda: {})

        prov = build_provider(_settings("lmstudio"), "laguna-s-2.1", provider_override="bogus")
        assert isinstance(prov, LMStudioProvider)


# ---------------------------------------------------------------------------
#  _switch_model with --provider
# ---------------------------------------------------------------------------

class TestSwitchModelWithProvider:
    def setup_method(self):
        self.cmd = ModelCommand()
        self.fake_models = [{
            "key": "laguna-s-2.1-ud",
            "display_name": "Laguna S 2.1 UD",
            "params_string": "8B-MoE",
            "size_bytes": 100,
            "loaded": True,
            "instance_id": "i1",
        }]
        # Controlled opencode catalogs — no network access.
        self.go_catalog = ["opencode-go/deepseek-v4-flash", "opencode-go/hy3"]
        self.zen_catalog = ["opencode-zen/nemotron-3.5-lightning-free"]

    def _patch_opencode_catalog(self, monkeypatch):
        """Mock _opencode_catalog and _zen_free_catalog to avoid network."""
        monkeypatch.setattr(
            self.cmd, "_opencode_catalog",
            lambda agent: (self.go_catalog, self.zen_catalog, True),
        )
        monkeypatch.setattr(
            self.cmd, "_zen_free_catalog",
            lambda: self.zen_catalog,
        )

    def test_laguna_routed_to_opencode_with_flag(self, monkeypatch, capsys):
        """`model laguna-s-2.1-ud --provider opencode` routes to opencode,
        bypassing the LM Studio prefix rule."""
        from agent_core.llm.opencode_provider import OpencodeProvider

        agent = _lmstudio_agent("qwen3.5-9b")
        monkeypatch.setattr("agent_core.commands.model_cmd._lms.get_models_status", lambda: self.fake_models)
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: _settings("lmstudio"))
        monkeypatch.setattr("agent_core.constants.load_model_json", lambda: {"model": "qwen3.5-9b", "provider": "lmstudio"})
        monkeypatch.setattr("agent_core.constants.persist_model_choice", lambda *a, **k: None)
        self._patch_opencode_catalog(monkeypatch)

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self.cmd._switch_model(["laguna-s-2.1-ud", "--provider", "opencode"], agent)
        )

        assert isinstance(agent.llm._provider, OpencodeProvider)
        # No opencode catalog match for "laguna-s-2.1-ud" → bare name → opencode-go/ prefix
        assert agent.llm.model_name == "opencode-go/laguna-s-2.1-ud"
        out = capsys.readouterr().out
        assert "provider=opencode" in out

    def test_opencode_routed_to_lmstudio_with_flag(self, monkeypatch, capsys):
        """`model laguna-s-2.1-ud --provider lmstudio` routes to LM Studio even
        though the name could be an opencode catalog member."""
        from agent_core.llm.lmstudio import LMStudioProvider

        agent = _opencode_agent("opencode-go/hy3")
        monkeypatch.setattr("agent_core.commands.model_cmd._lms.get_models_status", lambda: self.fake_models)
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: _settings("lmstudio"))
        monkeypatch.setattr("agent_core.constants.load_model_json", lambda: {"model": "opencode-go/hy3", "provider": "opencode"})
        monkeypatch.setattr("agent_core.constants.persist_model_choice", lambda *a, **k: None)
        self._patch_opencode_catalog(monkeypatch)

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self.cmd._switch_model(["laguna-s-2.1-ud", "--provider", "lmstudio"], agent)
        )

        assert isinstance(agent.llm._provider, LMStudioProvider)
        assert agent.llm.model_name == "laguna-s-2.1-ud"
        out = capsys.readouterr().out
        assert "Switched" in out

    def test_invalid_provider_rejected(self, monkeypatch, capsys):
        agent = _lmstudio_agent("qwen3.5-9b")
        monkeypatch.setattr("agent_core.commands.model_cmd._lms.get_models_status", lambda: self.fake_models)
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: _settings("lmstudio"))
        monkeypatch.setattr("agent_core.constants.load_model_json", lambda: {})
        monkeypatch.setattr("agent_core.constants.persist_model_choice", lambda *a, **k: None)
        self._patch_opencode_catalog(monkeypatch)

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self.cmd._switch_model(["laguna-s-2.1-ud", "--provider", "bogus"], agent)
        )

        out = capsys.readouterr().out
        assert "Unknown provider" in out
        assert agent.llm.model_name == "qwen3.5-9b"  # unchanged

    def test_no_flag_keeps_prefix_routing(self, monkeypatch, capsys):
        """Without --provider, `model laguna-s-2.1-ud` still routes to LM Studio."""
        from agent_core.llm.lmstudio import LMStudioProvider

        agent = _lmstudio_agent("qwen3.5-9b")
        monkeypatch.setattr("agent_core.commands.model_cmd._lms.get_models_status", lambda: self.fake_models)
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: _settings("lmstudio"))
        monkeypatch.setattr("agent_core.constants.load_model_json", lambda: {"model": "qwen3.5-9b", "provider": "lmstudio"})
        monkeypatch.setattr("agent_core.constants.persist_model_choice", lambda *a, **k: None)
        self._patch_opencode_catalog(monkeypatch)

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self.cmd._switch_model(["laguna-s-2.1-ud"], agent)
        )

        assert isinstance(agent.llm._provider, LMStudioProvider)
        assert agent.llm.model_name == "laguna-s-2.1-ud"

    def test_short_flag_also_works(self, monkeypatch, capsys):
        from agent_core.llm.opencode_provider import OpencodeProvider

        agent = _lmstudio_agent("qwen3.5-9b")
        monkeypatch.setattr("agent_core.commands.model_cmd._lms.get_models_status", lambda: self.fake_models)
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: _settings("lmstudio"))
        monkeypatch.setattr("agent_core.constants.load_model_json", lambda: {"model": "qwen3.5-9b", "provider": "lmstudio"})
        monkeypatch.setattr("agent_core.constants.persist_model_choice", lambda *a, **k: None)
        self._patch_opencode_catalog(monkeypatch)

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self.cmd._switch_model(["laguna-s-2.1-ud", "-p", "opencode"], agent)
        )

        assert isinstance(agent.llm._provider, OpencodeProvider)
        assert agent.llm.model_name == "opencode-go/laguna-s-2.1-ud"

    def test_opencode_catalog_match_with_flag(self, monkeypatch, capsys):
        """`model hy3 --provider opencode` resolves to the opencode-go catalog
        match, not a bare-name prefix."""
        from agent_core.llm.opencode_provider import OpencodeProvider

        agent = _lmstudio_agent("qwen3.5-9b")
        monkeypatch.setattr("agent_core.commands.model_cmd._lms.get_models_status", lambda: self.fake_models)
        monkeypatch.setattr("agent_core.config.load_agent_settings", lambda: _settings("lmstudio"))
        monkeypatch.setattr("agent_core.constants.load_model_json", lambda: {"model": "qwen3.5-9b", "provider": "lmstudio"})
        monkeypatch.setattr("agent_core.constants.persist_model_choice", lambda *a, **k: None)
        self._patch_opencode_catalog(monkeypatch)

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self.cmd._switch_model(["hy3", "--provider", "opencode"], agent)
        )

        assert isinstance(agent.llm._provider, OpencodeProvider)
        assert agent.llm.model_name == "opencode-go/hy3"
        out = capsys.readouterr().out
        assert "provider=opencode" in out


# ---------------------------------------------------------------------------
#  build_provider import (needed by TestBuildProviderOverride)
# ---------------------------------------------------------------------------

from agent_core.llm.provider import build_provider  # noqa: E402
