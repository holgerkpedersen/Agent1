"""Regression test for LLMClient.__init__ profile-restore failure handling.

Bug: a broken/corrupt model.json (or missing get_profile) was swallowed by a
bare ``print("Warning: silenced exception in agent.py:86")`` with no traceback,
leaving _profile_name in an indeterminate state and hiding the root cause from
logs.  Fix: log the full traceback via logger.warning and reset
_profile_name to None so downstream code can detect "no active profile".

Hermetic: patches load_model_json / get_profile so no real model.json is read;
asserts the logged message contains a traceback frame and that _profile_name
is None on failure.  Also asserts the happy path still restores the profile.
"""
import logging

import pytest


@pytest.fixture()
def _capture_warnings(caplog):
    """Capture WARNING-level logs from the agent module logger."""
    caplog.set_level(logging.WARNING, logger="agent")
    return caplog


def test_profile_restore_failure_is_logged_with_traceback(_capture_warnings, monkeypatch):
    """A failing profile restore must log a traceback and reset _profile_name.

    build_provider also calls load_model_json internally (before the restore
    block), so we patch build_provider to return a fake provider — isolating
    the failure to the profile-restore try/except at agent.py:192-205 only.
    """
    import agent as agent_mod  # the real module — not a copy

    class _FakeProvider:
        model_name = "test-model-failure"
        _profile_name = None
        temperature = 0.7
        max_tokens = 512

    def _fake_build_provider(settings, model_name):
        return _FakeProvider()

    monkeypatch.setattr("agent_core.llm.provider.build_provider", _fake_build_provider)
    # load_model_json returns a profile name so the restore block reaches get_profile;
    # then make get_profile raise to trigger __init__'s except branch.
    def _ok_load():
        return {"profile": "default"}

    monkeypatch.setattr("agent_core.constants.load_model_json", _ok_load)

    def _fail_get_profile(name):  # noqa: ARG001
        raise RuntimeError("corrupt model.json: missing 'temperature'")

    monkeypatch.setattr(
        "agent_core.llm.model_profiles.get_profile", _fail_get_profile,
    )

    client = agent_mod.LLMClient(model_name="test-model-failure")
    assert client._profile_name is None  # fix: explicit reset, not indeterminate
    warnings_text = "\n".join(r.getMessage() for r in _capture_warnings.records)
    assert "Failed to restore active profile from model.json" in warnings_text
    assert "RuntimeError" in warnings_text  # traceback content captured
    assert "corrupt model.json: missing 'temperature'" in warnings_text


def test_profile_restore_success_sets_name(_capture_warnings, monkeypatch):
    """Happy path: a valid profile still restores _profile_name."""
    from agent_core.constants import load_model_json

    def _ok_load():
        return {"profile": "default"}

    # get_profile must resolve; use the real one (it exists in repo) patched to
    # not depend on disk state.
    class _FakeProfile:
        temperature = 0.7
        max_tokens = 512

    monkeypatch.setattr("agent_core.constants.load_model_json", _ok_load)
    monkeypatch.setattr(
        "agent_core.llm.model_profiles.get_profile", lambda name: _FakeProfile(),
    )

    import agent as agent_mod

    client = agent_mod.LLMClient(model_name="test-model-ok")
    assert client._profile_name == "default"
    # No failure warning should be emitted on the happy path.
    warnings_text = "\n".join(r.getMessage() for r in _capture_warnings.records)
    assert "Failed to restore active profile" not in warnings_text


def test_signal_break_handler_runs_cleanup_before_exit(monkeypatch):
    """safe_signal_break_handler saves memory + closes trace then exits."""
    import agent as agent_mod
    from agent_core.security.shutdown import safe_signal_break_handler

    calls = []

    def save_memory():
        calls.append("save")

    def close_trace():
        calls.append("trace_close")

    # Capture os._exit so the test process doesn't actually die.
    captured = {}
    monkeypatch.setattr(agent_mod.os, "_exit", lambda code: captured.setdefault("code", code))

    safe_signal_break_handler(
        memory_path="agent_memory.json",
        trace_writer_close=close_trace,
        save_memory_fn=save_memory,
    )
    assert calls == ["save", "trace_close"]  # order matters — memory first
    assert captured["code"] == 1


def test_signal_break_handler_swallows_hook_failure(monkeypatch):
    """A failing cleanup hook must not abort shutdown."""
    import agent as agent_mod
    from agent_core.security.shutdown import safe_signal_break_handler

    def save_memory():
        raise ValueError("disk full")

    captured = {}
    monkeypatch.setattr(agent_mod.os, "_exit", lambda code: captured.setdefault("code", code))

    # Should not raise — hook failure logged + continues to exit.
    safe_signal_break_handler(
        memory_path="agent_memory.json",
        trace_writer_close=None,
        save_memory_fn=save_memory,
    )
    assert captured["code"] == 1


def test_register_signal_break_handler_returns_bool(monkeypatch):
    """register_signal_break_handler installs and reports success/failure."""
    from agent_core.security.shutdown import register_signal_break_handler

    installed = {}

    def fake_signal(sig, handler):
        installed["sig"] = sig
        installed["handler"] = handler

    monkeypatch.setattr("agent.signal.signal", fake_signal)
    result = register_signal_break_handler(
        memory_path="m.json", save_memory_fn=lambda: None, trace_writer_close=None,
    )
    assert result is True
    assert "sig" in installed and callable(installed["handler"])


def test_register_signal_break_handler_false_on_error(monkeypatch):
    """A signal API error yields False (graceful degradation)."""
    from agent_core.security.shutdown import register_signal_break_handler

    def raising_signal(sig, handler):  # noqa: ARG001
        raise ValueError("signal not allowed in this thread")

    monkeypatch.setattr("agent.signal.signal", raising_signal)
    result = register_signal_break_handler(memory_path="m.json")
    assert result is False


def test_install_with_agent_wires_callbacks(monkeypatch):
    """_install_signal_handlers(agent=...) wires _save_memory + trace close."""
    import agent as agent_mod

    class FakeAgent:
        def __init__(self):
            self._active_trace_writer = None

        def _save_memory(self):
            pass

    # Avoid actually registering a signal (test runner thread context).
    monkeypatch.setattr("agent.signal.signal", lambda sig, h: None)

    agent_mod._install_signal_handlers(agent=FakeAgent())
    assert callable(agent_mod._shutdown_save_memory)
    assert callable(agent_mod._shutdown_trace_close)
