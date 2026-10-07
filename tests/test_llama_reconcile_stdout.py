"""Regression: llama-server model reconciliation must never write to stdout.

Root cause of a real bug (batch 2, ``_warn_uncommitted`` tests): the
diagnostics in ``LLMClient._reconcile_llama_model`` used ``print()``, so
constructing an ``Agent``/``LLMClient`` polluted stdout.  Anything that
captures stdout — the REPL banner, ``_warn_uncommitted``'s clean-repo
assertion, tool-output parsing — then saw model-loading chatter.

The messages are diagnostics, so they belong on stderr (stdout is the
program's real output channel).  These tests drive the REAL
``LLMClient._reconcile_llama_model`` and assert stdout stays empty while the
information is still surfaced on stderr.
"""
from __future__ import annotations

from typing import Any

import pytest

import agent as agent_module
from agent import LLMClient


class LlamaProvider:
    """Stand-in whose *class name* matches the real llama provider.

    ``_reconcile_llama_model`` dispatches on ``type(provider).__name__``, so
    the name is the whole contract here.
    """

    def __init__(self, api_url: str | None = "http://127.0.0.1:1234/v1") -> None:
        self.api_url = api_url
        self._cached_server_model_id: str | None = None


def _make_client(model_name: str, provider: Any) -> LLMClient:
    """Build a client without running ``__init__`` (no provider/network setup)."""
    client = LLMClient.__new__(LLMClient)
    client._model_name = model_name
    client._provider = provider
    client.api_key = ""
    return client


@pytest.fixture()
def llama_env(monkeypatch: pytest.MonkeyPatch):
    """Force the llama path so ``_reconcile_llama_model`` does real work."""
    monkeypatch.setattr(
        "agent_core.llm.provider.provider_for", lambda *a, **k: "llama"
    )
    monkeypatch.setattr("agent_core.constants.load_model_json", lambda: {})
    return monkeypatch


def test_already_served_writes_nothing_to_stdout(
    llama_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The model is already served: info goes to stderr, stdout stays clean."""
    monkeypatch = llama_env
    from agent_core.llm import llama_server

    provider = LlamaProvider()
    client = _make_client("llama/Bonsai-27B-Q1_0", provider)
    monkeypatch.setattr(
        llama_server, "list_served_models", lambda api_url: ["Bonsai-27B-Q1_0"]
    )

    client._reconcile_llama_model(None)

    captured = capsys.readouterr()
    assert captured.out == "", f"stdout must stay clean, got: {captured.out!r}"
    assert "[llama]" in captured.err, "diagnostic must still be visible on stderr"
    assert provider._cached_server_model_id == "Bonsai-27B-Q1_0"


def test_ensure_path_writes_nothing_to_stdout(
    llama_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A different model is served and must be swapped: still no stdout."""
    monkeypatch = llama_env
    from agent_core.llm import llama_server

    provider = LlamaProvider()
    client = _make_client("llama/Bonsai-27B-Q1_0", provider)
    monkeypatch.setattr(
        llama_server, "list_served_models", lambda api_url: ["Some-Other-Model"]
    )
    monkeypatch.setattr(
        llama_server, "ensure_model_served", lambda api_url, name: (True, "ok")
    )

    client._reconcile_llama_model(None)

    captured = capsys.readouterr()
    assert captured.out == "", f"stdout must stay clean, got: {captured.out!r}"
    assert "[llama]" in captured.err


def test_no_model_served_writes_nothing_to_stdout(
    llama_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing served at all: the failure path is stderr-only too."""
    monkeypatch = llama_env
    from agent_core.llm import llama_server

    provider = LlamaProvider()
    client = _make_client("llama/Bonsai-27B-Q1_0", provider)
    monkeypatch.setattr(llama_server, "list_served_models", lambda api_url: [])
    monkeypatch.setattr(
        llama_server, "ensure_model_served", lambda api_url, name: (False, "nope")
    )

    client._reconcile_llama_model(None)

    captured = capsys.readouterr()
    assert captured.out == "", f"stdout must stay clean, got: {captured.out!r}"
    assert "WARNING" in captured.err


def test_non_llama_model_is_a_silent_noop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-llama model returns before any diagnostic is emitted."""
    client = _make_client("lmstudio/some-model", LlamaProvider())

    client._reconcile_llama_model(None)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_reconcile_is_never_reached_without_api_url(
    llama_env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A provider without an api_url bails out silently (no stdout)."""
    client = _make_client("llama/Bonsai-27B-Q1_0", LlamaProvider(api_url=None))

    client._reconcile_llama_model(None)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
