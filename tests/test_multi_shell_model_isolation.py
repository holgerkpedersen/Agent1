"""Regression tests: two concurrent agent.py shells must not hijack each
other's model (user report 2026-08-21).

Three leak paths existed:

1. ``model list`` auto-sync — if another shell had loaded a different model,
   listing in this shell SILENTLY switched the session to it AND persisted it
   to model.json/.env.
2. ``resolve_model()`` live-poll priority — at startup the live LM Studio
   poll outranked the persisted choice, so a new session adopted whatever
   the other shell had in VRAM instead of its own persisted model.
3. No pinning at all — nothing kept a running session on its chosen model.

Fix contract: a session keeps ITS model.  Listing is read-only (advisory
warning only); the persisted choice outranks the live poll; adoption of what
LM Studio currently has loaded is explicit via ``model reload`` / ``model
<name>``.  On-demand auto-reload of the pinned model inside
``LMStudioProvider._make_request`` is intentionally preserved — that is the
recovery path, not the bug.

Hermetic: every LM Studio / settings touchpoint is monkeypatched, so no real
server or model.json is touched.
"""
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _fake_models(loaded_key: str) -> list[dict[str, Any]]:
    return [{
        "key": loaded_key,
        "display_name": "Fake Model",
        "params_string": "9B",
        "size_bytes": 1_000_000,
        "loaded": True,
        "instance_id": "inst-1",
    }]


@pytest.fixture()
def lmstudio_env(monkeypatch):
    """Patch every LM Studio/settings touchpoint used by model_cmd/constants.

    Returns a settable holder: ``state.loaded`` is what LM Studio reports as
    loaded; ``state.persisted`` is what model.json would return.
    """
    from agent_core.commands import model_cmd

    class _State:
        loaded = "laguna-s-2.1"
        persisted: dict[str, Any] = {"model": "laguna-s-2.1", "provider": "lmstudio"}

    monkeypatch.setattr(
        "agent_core.commands.model_cmd._lms.get_models_status",
        lambda: _fake_models(_State.loaded),
    )
    monkeypatch.setattr(
        "agent_core.constants.load_model_json",
        lambda: dict(_State.persisted),
    )
    monkeypatch.setattr(
        "agent_core.config.load_agent_settings",
        lambda: type("S", (), {
            "llm_provider": "lmstudio",
            "opencode_model": "opencode-go/deepseek-v4-flash",
        })(),
    )
    # Never write real files from these tests.
    saved: dict[str, Any] = {}
    monkeypatch.setattr(
        "agent_core.commands.model_cmd.persist_model_choice",
        lambda name, provider=None: saved.setdefault("persisted", name),
    )

    def _make_agent(current: str):
        provider = type("P", (), {
            "model_name": current,
            "_profile_name": None,
        })()
        llm = type("L", (), {"model_name": current, "_provider": provider})()
        return type("A", (), {"llm": llm})()

    return _State, saved, _make_agent


# ---------------------------------------------------------------------------
# Leak 1: `model list` must never switch or persist
# ---------------------------------------------------------------------------

class TestModelListIsReadOnly:
    def test_list_does_not_switch_when_another_shell_loaded_otherwise(
        self, lmstudio_env, capsys
    ):
        state, saved, make_agent = lmstudio_env
        from agent_core.commands.model_cmd import ModelCommand

        agent = make_agent("laguna-s-2.1")
        state.loaded = "qwen3.5-9b-mtp"  # shell 2 loaded something else

        ModelCommand()._list_models(agent)

        assert agent.llm.model_name == "laguna-s-2.1"
        assert "persisted" not in saved  # no silent persist either
        out = capsys.readouterr().out
        assert "keeps laguna-s-2.1" in out
        assert "switching" not in out.lower().replace("switching to", "")

    def test_list_advises_explicit_adoption(self, lmstudio_env, capsys):
        state, _, make_agent = lmstudio_env
        from agent_core.commands.model_cmd import ModelCommand

        agent = make_agent("laguna-s-2.1")
        state.loaded = "qwen3.5-9b-mtp"

        ModelCommand()._list_models(agent)
        out = capsys.readouterr().out
        assert "model reload" in out or "model qwen3.5-9b-mtp" in out

    def test_list_stays_quiet_when_session_model_is_the_loaded_one(
        self, lmstudio_env, capsys
    ):
        state, _, make_agent = lmstudio_env
        from agent_core.commands.model_cmd import ModelCommand

        agent = make_agent("laguna-s-2.1")
        state.loaded = "laguna-s-2.1"

        ModelCommand()._list_models(agent)
        out = capsys.readouterr().out
        assert "⚠" not in out


# ---------------------------------------------------------------------------
# Leak 2: resolve_model priority — persisted choice beats the live poll
# ---------------------------------------------------------------------------

class TestResolveModelPrefersPersistedOverLivePoll:
    @staticmethod
    def _patch_common(monkeypatch, persisted: dict[str, Any]) -> None:
        import agent_core.constants as const

        monkeypatch.setattr(const, "load_model_json", lambda: persisted)
        monkeypatch.setattr(
            "agent_core.config.load_agent_settings",
            lambda: type("S", (), {
                "llm_provider": "lmstudio",
                "opencode_model": "opencode-go/deepseek-v4-flash",
            })(),
        )

    def test_new_session_keeps_persisted_model_not_other_shells_vram(
        self, monkeypatch
    ):
        import agent_core.constants as const

        self._patch_common(
            monkeypatch, {"model": "laguna-s-2.1", "provider": "lmstudio"}
        )
        # Another shell put qwen in VRAM after model.json said laguna.
        monkeypatch.setattr(
            "agent_core.llm.lmstudio.get_models_status",
            lambda: _fake_models("qwen3.5-9b-mtp"),
        )

        assert const.resolve_model(None) == "laguna-s-2.1"

    def test_first_run_fallback_still_adopts_loaded_model(self, monkeypatch):
        import agent_core.constants as const

        self._patch_common(monkeypatch, {})
        monkeypatch.setattr(
            "agent_core.llm.lmstudio.get_models_status",
            lambda: _fake_models("qwen3.5-9b-mtp"),
        )

        # Nothing persisted yet → adopting the loaded model is correct.
        assert const.resolve_model(None) == "qwen3.5-9b-mtp"

    def test_unknown_persisted_model_falls_through_to_live_poll(
        self, monkeypatch
    ):
        import agent_core.constants as const

        self._patch_common(monkeypatch, {"model": "not-a-real-model", "provider": "lmstudio"})
        monkeypatch.setattr(
            "agent_core.llm.lmstudio.get_models_status",
            lambda: _fake_models("qwen3.5-9b-mtp"),
        )

        assert const.resolve_model(None) == "qwen3.5-9b-mtp"


# ---------------------------------------------------------------------------
# Leak 3: end-to-end through the REAL LLMClient constructor
# ---------------------------------------------------------------------------

class TestSessionPinThroughRealLLMClient:
    def test_llmclient_startup_pins_persisted_model_despite_foreign_vram(
        self, monkeypatch
    ):
        """The exact user scenario: shell 2 loads qwen; a NEW shell 1 must
        still come up on its own persisted model."""
        import agent as agent_mod

        monkeypatch.setattr(
            "agent_core.constants.load_model_json",
            lambda: {"model": "laguna-s-2.1", "provider": "lmstudio"},
        )
        monkeypatch.setattr(
            "agent_core.config.load_agent_settings",
            lambda: type("S", (), {
                "llm_provider": "lmstudio",
                "opencode_model": "opencode-go/deepseek-v4-flash",
            })(),
        )
        monkeypatch.setattr(
            "agent_core.llm.lmstudio.get_models_status",
            lambda: _fake_models("qwen3.5-9b-mtp"),
        )

        client = agent_mod.LLMClient()
        assert client.model_name == "laguna-s-2.1"
