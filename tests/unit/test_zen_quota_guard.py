"""Regression tests for the cached free-tier quota guard.

Companion to ``tests/test_zen_quota_classification.py`` (which covers Tasks 1-3:
classify the 429, stop retrying it, open the failover gate).  This suite covers
Tasks 4-7 — the parts that make a dead free tier CHEAP instead of merely
correct:

* ``QuotaGuard`` remembers an exhausted tier with an expiry,
* the provider answers from that memory with ZERO HTTP calls (the 0 s path),
* the pointless free-model rotation is skipped for quota but kept for outages,
* the keyless catalog lists only ids the keyless tier can actually call.

All hermetic: no network, no sleeps (the guard takes an injected clock).
"""

from __future__ import annotations

import asyncio
import io
import urllib.error

import pytest

from agent_core.llm.opencode_provider import (
    ZEN_QUOTA_TIER,
    OpencodeProvider,
    _quota_exhausted_message,
)
from agent_core.llm.provider import is_connection_failure
from agent_core.llm.quota_guard import QuotaGuard, quota_guard

#: The exact gateway body observed live for an exhausted free tier.
LIVE_QUOTA_BODY = (
    '{"type":"error","error":{"type":"FreeUsageLimitError","message":'
    '"Free usage limit reached. Free models work in OpenCode '
    '(https://opencode.ai); subscribe to OpenCode Go to keep going: '
    'https://opencode.ai/go"}}'
)


class FakeClock:
    """Manually advanced monotonic clock — expiry without ``sleep``."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    """Build a real HTTPError carrying *body* (``fp`` mirrors a live socket)."""
    return urllib.error.HTTPError(
        url="https://opencode.ai/zen/v1/chat/completions",
        code=code,
        msg="Too Many Requests",
        hdrs=None,
        fp=io.BytesIO(body.encode("utf-8")),
    )


def _zen_provider(
    guard: QuotaGuard, model: str = "opencode-zen/beam-free"
) -> OpencodeProvider:
    """A keyless zen provider sharing *guard*, with no credential store read."""
    return OpencodeProvider(model, read_store=False, quota=guard, max_retries=0)


# ----------------------------------------------------------------------
# Task 4 — the guard itself
# ----------------------------------------------------------------------


class TestQuotaGuard:
    def test_mark_then_is_exhausted(self):
        guard = QuotaGuard(clock=FakeClock(), cooldown=60.0)
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is False
        guard.mark_exhausted(ZEN_QUOTA_TIER)
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is True

    def test_expiry_uses_the_injected_clock(self):
        clock = FakeClock()
        guard = QuotaGuard(clock=clock, cooldown=60.0)
        guard.mark_exhausted(ZEN_QUOTA_TIER)

        clock.advance(59.0)
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is True
        # An expired tier is forgotten, so the next turn probes it again.
        clock.advance(2.0)
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is False
        assert guard.remaining(ZEN_QUOTA_TIER) == 0.0

    def test_retry_after_overrides_the_default_cooldown(self):
        clock = FakeClock()
        guard = QuotaGuard(clock=clock, cooldown=900.0)
        guard.mark_exhausted(ZEN_QUOTA_TIER, retry_after=30.0)

        clock.advance(29.0)
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is True
        clock.advance(2.0)
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is False

    def test_invalid_retry_after_falls_back_to_the_cooldown(self):
        clock = FakeClock()
        guard = QuotaGuard(clock=clock, cooldown=10.0)
        guard.mark_exhausted(ZEN_QUOTA_TIER, retry_after="not-a-number")

        clock.advance(9.0)
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is True

    def test_key_is_the_tier_not_the_model(self):
        """Quota is ACCOUNT-wide: a different free MODEL stays blocked."""
        guard = QuotaGuard(clock=FakeClock(), cooldown=60.0)
        guard.mark_exhausted(ZEN_QUOTA_TIER)

        other_model = _zen_provider(guard, "opencode-zen/flint-40b-free")
        assert other_model.zen_mode is True
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is True

    def test_reset_clears_one_tier_or_all(self):
        guard = QuotaGuard(clock=FakeClock(), cooldown=60.0)
        guard.mark_exhausted("opencode-zen")
        guard.mark_exhausted("openrouter")

        guard.reset("openrouter")
        assert guard.is_exhausted("openrouter") is False
        assert guard.is_exhausted("opencode-zen") is True

        guard.reset()
        assert guard.is_exhausted("opencode-zen") is False

    def test_cooldown_env_override(self, monkeypatch):
        monkeypatch.setenv("AGENT_QUOTA_COOLDOWN_SECONDS", "5")
        clock = FakeClock()
        guard = QuotaGuard(clock=clock)  # no explicit cooldown -> env
        guard.mark_exhausted(ZEN_QUOTA_TIER)

        clock.advance(4.0)
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is True
        clock.advance(2.0)
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is False

    def test_process_guard_is_a_singleton(self):
        assert quota_guard() is quota_guard()


# ----------------------------------------------------------------------
# Task 5 — consult the guard before spending a request (the 0 s path)
# ----------------------------------------------------------------------


class TestExhaustedFastPath:
    def test_exhausted_tier_makes_no_http_call(self, monkeypatch):
        """The whole point: a primed guard must cost 0 HTTP requests."""
        guard = QuotaGuard(clock=FakeClock(), cooldown=60.0)
        guard.mark_exhausted(ZEN_QUOTA_TIER)
        provider = _zen_provider(guard)

        def _explode(*args, **kwargs):
            raise AssertionError("network touched on an exhausted tier")

        monkeypatch.setattr(provider, "_request", _explode)

        result = asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))

        # The turn answers from memory with an actionable message, and that
        # message must still trip the provider failover gate so the harness
        # hands over to the next provider instead of ending the turn.
        assert is_connection_failure(result) is True
        assert "beam-free" in result
        assert "opencode.ai/go" in result
        assert "model list" in result

    def test_message_names_the_model_and_the_way_out(self):
        msg = _quota_exhausted_message("opencode-zen/beam-free")
        assert "beam-free" in msg
        assert ZEN_QUOTA_TIER in msg
        assert "https://opencode.ai/go" in msg
        assert msg.startswith("[Error:")

    def test_observed_quota_body_primes_the_guard(self, monkeypatch):
        """A live 429 quota body must be remembered, not just reported."""
        guard = QuotaGuard(clock=FakeClock(), cooldown=60.0)
        provider = _zen_provider(guard)
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is False

        def _raise(*args, **kwargs):
            raise _http_error(429, LIVE_QUOTA_BODY)

        monkeypatch.setattr(provider, "_request", _raise)

        result = asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))

        # The observed body primed the guard AND the turn reports it.
        assert is_connection_failure(result) is True
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is True

        # Second turn now short-circuits: no HTTP at all.
        def _explode(*args, **kwargs):
            raise AssertionError("network touched on the second turn")

        monkeypatch.setattr(provider, "_request", _explode)
        second = asyncio.run(provider.chat([{"role": "user", "content": "again"}]))
        assert is_connection_failure(second) is True
        assert "beam-free" in second

    def test_retry_after_header_is_honoured(self, monkeypatch):
        guard = QuotaGuard(clock=FakeClock(), cooldown=900.0)
        provider = _zen_provider(guard)

        def _raise(*args, **kwargs):
            exc = _http_error(429, LIVE_QUOTA_BODY)
            exc.headers = {"Retry-After": "12"}  # type: ignore[assignment]
            raise exc

        monkeypatch.setattr(provider, "_request", _raise)
        asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))

        remaining = guard.remaining(ZEN_QUOTA_TIER)
        assert 0.0 < remaining <= 12.0


# ----------------------------------------------------------------------
# Task 6 — collapse the pointless free-model rotation
# ----------------------------------------------------------------------


class TestRotationIsSuppressedForQuota:
    def test_quota_result_does_not_rotate_models(self, monkeypatch):
        """Every free model shares the quota, so rotating cannot help."""
        guard = QuotaGuard(clock=FakeClock(), cooldown=60.0)
        provider = _zen_provider(guard)
        rotations: list[str] = []

        async def _rotated(*args, **kwargs):
            rotations.append("rotated")
            return "rotated"

        monkeypatch.setattr(provider, "_zen_free_fallback", _rotated)

        async def _chat_api(*args, **kwargs):
            # Mirror the real ``_chat_api``: an observed quota body primes the
            # guard and is rewritten into the actionable message.
            guard.mark_exhausted(ZEN_QUOTA_TIER)
            return _quota_exhausted_message(provider.model_name)

        monkeypatch.setattr(provider, "_chat_api", _chat_api)

        result = asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))

        # No rotation: every free model shares the account quota.
        assert rotations == []
        assert "beam-free" in result
        assert is_connection_failure(result) is True

    def test_backend_outage_still_rotates(self, monkeypatch):
        """The rotation keeps its original purpose: 5xx / timeouts."""
        guard = QuotaGuard(clock=FakeClock(), cooldown=60.0)
        provider = _zen_provider(guard)
        rotations: list[str] = []

        async def _rotated(*args, **kwargs):
            rotations.append("rotated")
            return "rotated-ok"

        monkeypatch.setattr(provider, "_zen_free_fallback", _rotated)

        async def _chat_api(*args, **kwargs):
            # A real backend-down body (marker: "service unavailable").
            return (
                "[Error: opencode API request failed: "
                "HTTP Error 503: Service Unavailable]"
            )

        monkeypatch.setattr(provider, "_chat_api", _chat_api)

        result = asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))

        assert rotations == ["rotated"]
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is False
        assert result == "rotated-ok"

    def test_non_zen_provider_never_consults_the_zen_tier(self, monkeypatch):
        guard = QuotaGuard(clock=FakeClock(), cooldown=60.0)
        guard.mark_exhausted(ZEN_QUOTA_TIER)
        provider = OpencodeProvider(
            "opencode-go/deepseek-v4.1-flash",
            read_store=False,
            api_key="sk-test",
            quota=guard,
            max_retries=0,
        )
        assert provider.zen_mode is False

        async def _chat_api(*args, **kwargs):
            return "go tier answer"

        monkeypatch.setattr(provider, "_chat_api", _chat_api)

        result = asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))
        assert result == "go tier answer"


# ----------------------------------------------------------------------
# Task 7 — make the keyless catalog honest
# ----------------------------------------------------------------------


class TestHonestZenCatalog:
    CATALOG = {
        "data": [
            {"id": "beam-free"},
            {"id": "flint-40b-free"},
            {"id": "claude-opus-5"},  # paid: needs a Go subscription
            {"id": "gpt-6"},  # paid
        ]
    }

    def test_keyless_catalog_lists_only_free_ids(self, monkeypatch):
        provider = _zen_provider(QuotaGuard(clock=FakeClock(), cooldown=60.0))
        monkeypatch.setattr(provider, "_request", lambda *a, **k: self.CATALOG)

        ids = provider.list_models()

        assert ids == ["opencode-zen/beam-free", "opencode-zen/flint-40b-free"]
        assert all(i.endswith("-free") for i in ids)

    def test_paid_ids_stay_in_the_full_catalog(self, monkeypatch):
        """Keyed/go callers and the raw zen fetch keep the full list."""
        provider = _zen_provider(QuotaGuard(clock=FakeClock(), cooldown=60.0))
        monkeypatch.setattr(provider, "_request", lambda *a, **k: self.CATALOG)

        full = provider._zen_free_models()

        assert "opencode-zen/claude-opus-5" in full
        assert "opencode-zen/beam-free" in full

    def test_catalog_note_reports_exhausted_quota(self, monkeypatch):
        guard = QuotaGuard(clock=FakeClock(), cooldown=60.0)
        provider = _zen_provider(guard)
        monkeypatch.setattr(provider, "_request", lambda *a, **k: self.CATALOG)

        provider.list_models()
        assert provider.zen_catalog_note == ""

        guard.mark_exhausted(ZEN_QUOTA_TIER)
        ids = provider.list_models()

        assert "quota currently exhausted" in provider.zen_catalog_note
        # The note must never leak into selectable ids.
        assert all("/" in i for i in ids)
        assert not any("(" in i for i in ids)

    def test_unreachable_catalog_is_empty_not_a_crash(self, monkeypatch):
        provider = _zen_provider(QuotaGuard(clock=FakeClock(), cooldown=60.0))

        def _boom(*args, **kwargs):
            raise OSError("network disabled in test")

        monkeypatch.setattr(provider, "_request", _boom)
        assert provider.list_models() == []


@pytest.mark.parametrize("status", [429, 503])
def test_bare_transient_status_still_retries(status):
    """A plain 429 (real rate limiting) must stay retryable — Task 2 guard."""
    from agent_core.llm.opencode_provider import _is_transient_http_error

    assert _is_transient_http_error(_http_error(status, "{}")) is True


# ----------------------------------------------------------------------
# Task 8 — the verdict survives a restart
# ----------------------------------------------------------------------


class WallClock:
    """Manually advanced wall clock (persisted deadlines use epoch seconds)."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestPersistedQuotaState:
    """A restart must not spend another request re-probing a dead tier."""

    def test_memory_only_guard_writes_no_file(self, tmp_path, monkeypatch):
        """The default stays pure in-memory: no path, no filesystem side effect."""
        # Observe the resolver instead of the directory: an autouse conftest
        # fixture already populates ``tmp_path`` (runtime_state/), so "the dir
        # is empty" would be measuring the harness, not the guard.
        monkeypatch.setenv("AGENT_QUOTA_COOLDOWN_SECONDS", "1")
        guard = QuotaGuard(clock=FakeClock(), cooldown=60.0)
        guard.mark_exhausted(ZEN_QUOTA_TIER)
        guard.reset()

        assert guard.is_exhausted(ZEN_QUOTA_TIER) is False
        assert not list(tmp_path.glob("**/quota_state.json"))

    def test_mark_survives_a_fresh_process(self, tmp_path):
        """A new guard (new process) restores the block instead of re-probing."""
        state = tmp_path / "quota_state.json"
        mono, wall = FakeClock(), WallClock()

        first = QuotaGuard(
            clock=mono, wall_clock=wall, cooldown=900.0, state_path=lambda: str(state)
        )
        first.mark_exhausted(ZEN_QUOTA_TIER)
        assert state.exists()

        # Simulate a restart: fresh monotonic clock, same wall clock instant.
        restarted = QuotaGuard(
            clock=FakeClock(),
            wall_clock=wall,
            cooldown=900.0,
            state_path=lambda: str(state),
        )

        assert restarted.is_exhausted(ZEN_QUOTA_TIER) is True
        assert 890.0 < restarted.remaining(ZEN_QUOTA_TIER) <= 900.0

    def test_persisted_verdict_expires_in_the_new_process(self, tmp_path):
        """The deadline is honoured across processes, not restarted at load."""
        state = tmp_path / "quota_state.json"
        wall = WallClock()
        QuotaGuard(
            clock=FakeClock(), wall_clock=wall, cooldown=60.0,
            state_path=lambda: str(state),
        ).mark_exhausted(ZEN_QUOTA_TIER)

        wall.advance(61.0)  # the cooldown elapses while the process is down
        restarted = QuotaGuard(
            clock=FakeClock(), wall_clock=wall, cooldown=60.0,
            state_path=lambda: str(state),
        )

        assert restarted.is_exhausted(ZEN_QUOTA_TIER) is False

    def test_reset_clears_the_persisted_verdict(self, tmp_path):
        """Re-selecting a model must clear the on-disk block too."""
        state = tmp_path / "quota_state.json"
        wall = WallClock()

        def _guard() -> QuotaGuard:
            return QuotaGuard(
                clock=FakeClock(), wall_clock=wall, cooldown=600.0,
                state_path=lambda: str(state),
            )

        guard = _guard()
        guard.mark_exhausted(ZEN_QUOTA_TIER)
        guard.reset()

        assert _guard().is_exhausted(ZEN_QUOTA_TIER) is False

    def test_corrupt_state_file_is_ignored_not_fatal(self, tmp_path):
        """Garbage on disk must degrade to 'unknown', never crash a turn."""
        state = tmp_path / "quota_state.json"
        state.write_text("{not json", encoding="utf-8")

        guard = QuotaGuard(
            clock=FakeClock(), wall_clock=WallClock(), cooldown=60.0,
            state_path=lambda: str(state),
        )

        assert guard.is_exhausted(ZEN_QUOTA_TIER) is False
        guard.mark_exhausted(ZEN_QUOTA_TIER)  # still writable afterwards
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is True

    def test_unresolvable_path_never_raises(self):
        """A broken path resolver must not break the fast path."""

        def _boom() -> str:
            raise RuntimeError("no session")

        guard = QuotaGuard(
            clock=FakeClock(), cooldown=60.0, state_path=_boom
        )

        assert guard.is_exhausted(ZEN_QUOTA_TIER) is False
        guard.mark_exhausted(ZEN_QUOTA_TIER)
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is True

    def test_restart_makes_zero_http_calls(self, tmp_path, monkeypatch):
        """End-to-end: a restored block answers without touching the network."""
        state = tmp_path / "quota_state.json"
        wall = WallClock()
        QuotaGuard(
            clock=FakeClock(), wall_clock=wall, cooldown=900.0,
            state_path=lambda: str(state),
        ).mark_exhausted(ZEN_QUOTA_TIER)

        restored = QuotaGuard(
            clock=FakeClock(), wall_clock=wall, cooldown=900.0,
            state_path=lambda: str(state),
        )
        provider = _zen_provider(restored)
        calls = []

        def _explode(*args, **kwargs):
            calls.append(args)
            raise AssertionError("a restored quota block must not hit the network")

        monkeypatch.setattr(provider, "_request", _explode)
        text = asyncio.run(provider.chat([{"role": "user", "content": "hi"}]))

        assert calls == []
        assert "FreeUsageLimitError" not in text
        assert "opencode.ai/go" in text

    def test_session_state_path_is_next_to_the_chat_history(self):
        """The real resolver targets the session dir, not the repo root."""
        from agent_core.session.models import SessionPaths

        paths = SessionPaths(
            session_id="default",
            state_dir=__import__("pathlib").Path("/tmp/state"),
            chat_history_path="/tmp/state/chat_history.json",
            agent_memory_path="/tmp/state/agent_memory.json",
            turn_log_path="/tmp/state/turn_log.jsonl",
        )

        assert str(paths.quota_state_path) == str(
            __import__("pathlib").Path("/tmp/state/quota_state.json")
        )


class TestUserReselectClearsQuotaBlock:
    """Regression: a persisted block must have a way OUT.

    Persisting the verdict (Tasks 10) means a user who subscribes mid-day stays
    quarantined across restarts unless re-selecting a model clears it.  These
    tests drive the REAL ``ModelCommand`` entry points, not a simulation, and
    pin the deliberate asymmetry: a user switch clears the block, while the
    automatic failover path (``llm/provider.py``) must never do so.
    """

    @pytest.fixture(autouse=True)
    def _clean_process_guard(self):
        """Leave the process-wide singleton un-blocked after every test.

        These tests mark the REAL ``quota_guard()`` exhausted, which both lives
        in memory for the whole session and persists ``quota_state.json``.  Left
        behind, that leaks into unrelated suites (``tests/test_opencode_provider.py``
        ``TestZenFree*`` starting failing only when run in the same process) and
        into the developer's own session state.
        """
        from agent_core.llm.quota_guard import reset_quota_guard

        reset_quota_guard()
        yield
        reset_quota_guard()

    @staticmethod
    def _agent():
        from agent_core.llm.lmstudio import LMStudioProvider

        provider = LMStudioProvider(model_name="local-model")
        llm = type(
            "L",
            (),
            {
                "model_name": "local-model",
                "_provider": provider,
                "_profile_name": None,
            },
        )()
        return type("A", (), {"llm": llm})()

    @staticmethod
    def _settings():
        return type(
            "S",
            (),
            {
                "llm_provider": "lmstudio",
                "opencode_model": "opencode-go/kimi-k2",
                "opencode_server_url": "http://127.0.0.1:4096",
                "opencode_password": "",
                "opencode_api_url": "https://opencode.ai/zen/go/v1",
                "opencode_api_key": "sk-test",
                "openrouter_model": "openrouter/free-model",
                "lemonade_model": "",
            },
        )()

    def _switch_stubs(self, monkeypatch):
        """Stub everything the switch needs EXCEPT the quota reset."""
        import agent_core.llm.provider as prov_mod

        monkeypatch.setattr(
            "agent_core.commands.model_cmd._lms.get_models_status", lambda: []
        )
        monkeypatch.setattr(
            "agent_core.config.load_agent_settings", lambda: self._settings()
        )
        monkeypatch.setattr(
            "agent_core.constants.load_model_json",
            lambda: {"model": "local-model", "provider": "lmstudio"},
        )
        monkeypatch.setattr(
            "agent_core.commands.model_cmd.persist_model_choice",
            lambda name, provider=None: None,
        )
        monkeypatch.setattr(
            prov_mod, "build_provider", lambda s, m, provider_override=None: object()
        )

    def test_reselecting_a_model_clears_a_persisted_block(self, monkeypatch):
        from agent_core.commands.model_cmd import ModelCommand

        guard = quota_guard()
        guard.mark_exhausted(ZEN_QUOTA_TIER)
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is True

        self._switch_stubs(monkeypatch)
        agent = self._agent()
        asyncio.run(ModelCommand()._switch_model(["opencode-zen/hy3-free"], agent))

        assert agent.llm.model_name == "opencode-zen/hy3-free"
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is False

    def test_model_switch_reset_also_clears_the_state_file(self, monkeypatch):
        """The file itself is rewritten empty, so a restart does not reload it."""
        import json

        from agent_core.commands.model_cmd import ModelCommand
        from agent_core.llm.quota_guard import _session_state_path

        guard = quota_guard()
        guard.mark_exhausted(ZEN_QUOTA_TIER)
        state = _session_state_path()
        if not state or not __import__("os").path.exists(state):
            pytest.skip("no session state path in this environment")

        self._switch_stubs(monkeypatch)
        cmd = ModelCommand()
        asyncio.run(cmd._switch_model(["opencode-zen/hy3-free"], self._agent()))

        with open(state, "r", encoding="utf-8") as fh:
            persisted = json.load(fh)
        assert ZEN_QUOTA_TIER not in (persisted.get("tiers") or {})

    def test_model_provider_switch_clears_the_block_too(self, monkeypatch):
        """`model provider opencode` is a user decision — same escape hatch."""
        from agent_core.commands.model_cmd import ModelCommand

        guard = quota_guard()
        guard.mark_exhausted(ZEN_QUOTA_TIER)
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is True

        self._switch_stubs(monkeypatch)
        asyncio.run(ModelCommand()._handle_provider(["opencode"], self._agent()))

        assert guard.is_exhausted(ZEN_QUOTA_TIER) is False

    def test_failover_does_not_clear_the_block(self, monkeypatch, tmp_path):
        """The asymmetry that makes persistence safe: only a USER clears it.

        ``llm/provider.py`` calls the REAL ``persist_model_choice`` after a
        failover, so this drives that real function (not a stub of it).  If the
        persistence path reset the guard, every successful failover would
        silently un-quarantine the dead tier and re-introduce the wasted probe.
        """
        import json

        import agent_core.constants as constants
        from agent_core.constants import persist_model_choice

        guard = quota_guard()
        guard.mark_exhausted(ZEN_QUOTA_TIER)

        # Point the REAL persistence at a temp dir — never the repo's .env.
        monkeypatch.setattr(
            constants, "MODEL_JSON_PATH", str(tmp_path / "model.json")
        )
        monkeypatch.chdir(tmp_path)

        persist_model_choice("some-working-model", provider="lmstudio")

        written = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
        assert written["model"] == "some-working-model"
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is True

    # -- `model reset-quota`: the explicit escape hatch ---------------------

    def test_reset_quota_verb_clears_a_persisted_block(self, monkeypatch):
        """The named escape hatch works WITHOUT changing the selection.

        A user who subscribes mid-session may not want to switch models; before
        this verb the only way to drop a persisted block was to re-pick one.
        """
        from agent_core.commands.model_cmd import ModelCommand

        guard = quota_guard()
        guard.mark_exhausted(ZEN_QUOTA_TIER)
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is True

        out = io.StringIO()
        monkeypatch.setattr("sys.stdout", out)
        asyncio.run(ModelCommand().execute(["reset-quota"], self._agent()))

        assert guard.is_exhausted(ZEN_QUOTA_TIER) is False
        assert "Cleared" in out.getvalue()

    def test_reset_quota_is_routed_as_a_verb_not_a_model_name(self, monkeypatch):
        """Regression: without the dispatch branch 'reset-quota' became a MODEL.

        It would fall through to ``_switch_model``, which searched the catalogs
        for a model called "reset-quota" and cleared the block as a side effect
        of the switch attempt — silently changing the user's model selection.
        """
        from agent_core.commands.model_cmd import ModelCommand

        guard = quota_guard()
        guard.mark_exhausted(ZEN_QUOTA_TIER)

        called: list[list[str]] = []

        async def _boom(args, agent):
            called.append(list(args))

        monkeypatch.setattr(ModelCommand, "_switch_model", _boom)
        monkeypatch.setattr("sys.stdout", io.StringIO())
        asyncio.run(ModelCommand().execute(["reset-quota"], self._agent()))

        assert called == [], "reset-quota must not be treated as a model name"
        assert guard.is_exhausted(ZEN_QUOTA_TIER) is False

    def test_reset_quota_names_a_specific_tier(self, monkeypatch):
        from agent_core.commands.model_cmd import ModelCommand

        guard = quota_guard()
        guard.mark_exhausted(ZEN_QUOTA_TIER)
        guard.mark_exhausted("other-tier")

        monkeypatch.setattr("sys.stdout", io.StringIO())
        asyncio.run(
            ModelCommand().execute(["reset-quota", ZEN_QUOTA_TIER], self._agent())
        )

        assert guard.is_exhausted(ZEN_QUOTA_TIER) is False
        assert guard.is_exhausted("other-tier") is True

    def test_reset_quota_reports_when_nothing_was_blocked(self, monkeypatch):
        from agent_core.commands.model_cmd import ModelCommand

        out = io.StringIO()
        monkeypatch.setattr("sys.stdout", out)
        asyncio.run(ModelCommand().execute(["reset-quota"], self._agent()))

        assert "No " in out.getvalue() or "nothing to clear" in out.getvalue()

    def test_reset_quota_clears_the_state_file(self, monkeypatch):
        """The file must be rewritten, or a restart reloads the stale block."""
        import json

        from agent_core.commands.model_cmd import ModelCommand
        from agent_core.llm.quota_guard import _session_state_path

        guard = quota_guard()
        guard.mark_exhausted(ZEN_QUOTA_TIER)
        state = _session_state_path()
        if not state or not __import__("os").path.exists(state):
            pytest.skip("no session state path in this environment")

        monkeypatch.setattr("sys.stdout", io.StringIO())
        asyncio.run(ModelCommand().execute(["reset-quota"], self._agent()))

        with open(state, "r", encoding="utf-8") as fh:
            persisted = json.load(fh)
        assert ZEN_QUOTA_TIER not in (persisted.get("tiers") or {})
