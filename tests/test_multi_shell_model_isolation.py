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


# ---------------------------------------------------------------------------
# Live-poll cost regression (2026-10-01)
#
# The live LM Studio poll is a blocking HTTP round-trip (~2s against a busy
# server) that runs from EVERY provider construction — ``Agent.__init__`` calls
# ``resolve_model`` four times, and ``tests/`` builds ~370 Agents.  Uncached,
# one ``Agent()`` cost 4.5s of pure socket wait and a single full pytest run
# spent minutes inside it (observed: a ``harnessfix.loop --auto-approve`` run
# that looked hung was really blocked in ``socket.connect`` via this poll).
#
# Contract: the poll runs AT MOST ONCE per process, and the cache is
# resettable so the isolation tests above still see their own poll.
# ---------------------------------------------------------------------------

class TestLivePollIsCached:
    def test_poll_runs_once_for_many_resolutions(self, monkeypatch) -> None:
        import agent_core.constants as const

        const.reset_live_poll_cache()
        monkeypatch.setattr(const, "load_model_json", lambda: {})
        monkeypatch.setattr(
            "agent_core.config.load_agent_settings",
            lambda: type("S", (), {
                "llm_provider": "lmstudio",
                "opencode_model": "opencode-go/deepseek-v4-flash",
            })(),
        )
        calls = {"n": 0}

        def _poll() -> list[dict[str, Any]]:
            calls["n"] += 1
            return _fake_models("qwen3.5-9b-mtp")

        monkeypatch.setattr(
            "agent_core.llm.lmstudio.get_models_status", _poll
        )

        for _ in range(5):
            assert const.resolve_model(None) == "qwen3.5-9b-mtp"
        assert calls["n"] == 1, "live poll must be cached, not re-run per call"

    def test_reset_makes_the_next_call_poll_again(self, monkeypatch) -> None:
        import agent_core.constants as const

        monkeypatch.setattr(const, "load_model_json", lambda: {})
        monkeypatch.setattr(
            "agent_core.config.load_agent_settings",
            lambda: type("S", (), {
                "llm_provider": "lmstudio",
                "opencode_model": "opencode-go/deepseek-v4-flash",
            })(),
        )
        seen: list[str] = []

        def _poll() -> list[dict[str, Any]]:
            key = f"model-{len(seen)}"
            seen.append(key)
            return _fake_models(key)

        monkeypatch.setattr("agent_core.llm.lmstudio.get_models_status", _poll)

        const.reset_live_poll_cache()
        assert const.resolve_model(None) == "model-0"
        const.reset_live_poll_cache()
        assert const.resolve_model(None) == "model-1"

    def test_autouse_fixture_clears_the_cache_between_tests(
        self, monkeypatch
    ) -> None:
        """Pin the conftest autouse reset itself.

        Without it the isolation tests above pass only by luck: they all fake
        the SAME model, so a stale cache is indistinguishable from a fresh
        poll.  This test deliberately caches ``leaked-from-a-prior-test`` and
        asserts the next test starts from a clean slate.
        """
        import agent_core.constants as const

        const.reset_live_poll_cache()
        monkeypatch.setattr(const, "load_model_json", lambda: {})
        monkeypatch.setattr(
            "agent_core.config.load_agent_settings",
            lambda: type("S", (), {
                "llm_provider": "lmstudio",
                "opencode_model": "opencode-go/deepseek-v4-flash",
            })(),
        )
        monkeypatch.setattr(
            "agent_core.llm.lmstudio.get_models_status",
            lambda: _fake_models("leaked-from-a-prior-test"),
        )
        # Deliberately DO NOT reset: poison the process-wide cache.
        assert const.resolve_model(None) == "leaked-from-a-prior-test"

    def test_cache_does_not_survive_into_the_next_test(
        self, monkeypatch
    ) -> None:
        """Runs AFTER ``test_autouse_fixture_clears_the_cache_between_tests``.

        That test leaves ``leaked-from-a-prior-test`` in the process-wide
        cache on purpose.  If the conftest autouse reset were removed, this
        test would still see it and fail.
        """
        import agent_core.constants as const

        assert const._LIVE_POLL_CACHE is None, (
            "the autouse fixture must clear the live-poll cache between tests; "
            f"leaked value: {const._LIVE_POLL_CACHE!r}"
        )

        monkeypatch.setattr(const, "load_model_json", lambda: {})
        monkeypatch.setattr(
            "agent_core.config.load_agent_settings",
            lambda: type("S", (), {
                "llm_provider": "lmstudio",
                "opencode_model": "opencode-go/deepseek-v4-flash",
            })(),
        )
        monkeypatch.setattr(
            "agent_core.llm.lmstudio.get_models_status",
            lambda: _fake_models("fresh-poll"),
        )
        assert const.resolve_model(None) == "fresh-poll"

    def test_poll_failure_is_cached_as_empty_not_retried_forever(
        self, monkeypatch
    ) -> None:
        """A dead LM Studio must not be re-probed on every construction."""
        import agent_core.constants as const

        const.reset_live_poll_cache()
        monkeypatch.setattr(const, "load_model_json", lambda: {})
        monkeypatch.setattr(
            "agent_core.config.load_agent_settings",
            lambda: type("S", (), {
                "llm_provider": "lmstudio",
                "opencode_model": "opencode-go/deepseek-v4-flash",
            })(),
        )
        calls = {"n": 0}

        def _poll() -> list[dict[str, Any]]:
            calls["n"] += 1
            raise OSError("connection refused")

        monkeypatch.setattr("agent_core.llm.lmstudio.get_models_status", _poll)

        for _ in range(3):
            const.resolve_model(None)
        assert calls["n"] == 1, "a failed poll must not be retried per call"
