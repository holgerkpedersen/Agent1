"""Regression tests: opencode-zen FREE model names are configurable via .env.

These tests pin the contracts introduced to eliminate hardcoded LLM names
for the opencode-zen FREE tier:

1. ``AgentSettings.zen_free_default`` reads ``AGENT_ZEN_FREE_DEFAULT``.
2. ``AgentSettings.zen_free_fallbacks`` reads/parses
   ``AGENT_ZEN_FREE_FALLBACKS`` (comma-separated).
3. ``_zen_free_fallbacks()`` in opencode_provider reads from settings and
   discovers the live catalog when no list is configured (no hardcoded
   models).
4. ``_zen_free_catalog()`` in model_cmd uses the configurable default.
5. ``DEFAULT_OPENCODE_MODEL`` reads ``AGENT_OPENCODE_MODEL`` from .env.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_core.config import AgentSettings
from agent_core.constants import DEFAULT_OPENCODE_MODEL


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A temporary workspace root that satisfies _validate_settings."""
    return tmp_path


# --- AgentSettings fields --------------------------------------------------


def test_zen_free_default_reads_env(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """AGENT_ZEN_FREE_DEFAULT is mapped to settings.zen_free_default."""
    monkeypatch.setenv("AGENT_ZEN_FREE_DEFAULT", "opencode-zen/newmodel-free")
    settings = AgentSettings(workspace_root=workspace)
    assert settings.zen_free_default == "opencode-zen/newmodel-free"


def test_zen_free_default_fallback(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """When AGENT_ZEN_FREE_DEFAULT is unset, defaults to empty (not a model)."""
    monkeypatch.delenv("AGENT_ZEN_FREE_DEFAULT", raising=False)
    settings = AgentSettings(workspace_root=workspace)
    assert settings.zen_free_default == ""


def test_zen_free_fallbacks_reads_env(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """AGENT_ZEN_FREE_FALLBACKS is parsed into an ordered tuple."""
    monkeypatch.setenv(
        "AGENT_ZEN_FREE_FALLBACKS",
        "opencode-zen/a-free, opencode-zen/b-free, opencode-zen/c-free",
    )
    settings = AgentSettings(workspace_root=workspace)
    assert settings.zen_free_fallbacks == (
        "opencode-zen/a-free",
        "opencode-zen/b-free",
        "opencode-zen/c-free",
    )


def test_zen_free_fallbacks_default(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """When AGENT_ZEN_FREE_FALLBACKS is unset, defaults to empty (no model list)."""
    monkeypatch.delenv("AGENT_ZEN_FREE_FALLBACKS", raising=False)
    settings = AgentSettings(workspace_root=workspace)
    assert settings.zen_free_fallbacks == ()


def test_zen_free_fallbacks_strips_empties(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """Empty entries in the comma-separated list are dropped."""
    monkeypatch.setenv(
        "AGENT_ZEN_FREE_FALLBACKS",
        "opencode-zen/a-free,,opencode-zen/b-free,",
    )
    settings = AgentSettings(workspace_root=workspace)
    assert settings.zen_free_fallbacks == (
        "opencode-zen/a-free",
        "opencode-zen/b-free",
    )


# --- _zen_free_fallbacks() reads settings ----------------------------------


def test_zen_free_fallbacks_reads_settings(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """_zen_free_fallbacks() returns the settings-configured list."""
    from agent_core.llm.opencode_provider import _zen_free_fallbacks

    fake_settings = AgentSettings(
        workspace_root=workspace,
        zen_free_fallbacks=(
            "opencode-zen/custom1-free",
            "opencode-zen/custom2-free",
        ),
    )
    # load_agent_settings is imported inside _zen_free_fallbacks(), so we
    # patch it at the agent_core.config module level.
    with patch("agent_core.config.load_agent_settings", return_value=fake_settings):
        result = _zen_free_fallbacks()
    assert result == ["opencode-zen/custom1-free", "opencode-zen/custom2-free"]


def test_zen_free_fallbacks_fallback_on_settings_error(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """_zen_free_fallbacks() returns an empty list when settings and discovery fail."""
    from agent_core.llm.opencode_provider import _zen_free_fallbacks, OpencodeProvider

    with patch(
        "agent_core.config.load_agent_settings",
        side_effect=RuntimeError("settings broken"),
    ), patch.object(OpencodeProvider, "list_models", return_value=[]):
        result = _zen_free_fallbacks()
    assert result == []


# --- _zen_free_catalog() uses configurable default -------------------------


def test_zen_free_catalog_uses_configurable_default(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """model_cmd._zen_free_catalog() passes settings.zen_free_default to the provider."""
    from agent_core.commands.model_cmd import ModelCommand

    cmd = ModelCommand.__new__(ModelCommand)

    fake_settings = AgentSettings(
        workspace_root=workspace,
        zen_free_default="opencode-zen/custom-free",
    )

    captured: dict = {}

    class _FakeProvider:
        def __init__(self, model_name: str, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

        def list_models(self):
            return ["opencode-zen/custom-free"]

    with patch(
        "agent_core.llm.opencode_provider.OpencodeProvider", _FakeProvider
    ), patch(
        "agent_core.config.load_agent_settings", return_value=fake_settings
    ):
        result = cmd._zen_free_catalog()

    assert result == ["opencode-zen/custom-free"]
    assert captured["model_name"] == "opencode-zen/custom-free"
    assert captured["kwargs"]["read_store"] is False


def test_zen_free_catalog_falls_back_on_settings_error(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """model_cmd._zen_free_catalog() uses a generic placeholder when settings fail."""
    from agent_core.commands.model_cmd import ModelCommand

    cmd = ModelCommand.__new__(ModelCommand)

    captured: dict = {}

    class _FakeProvider:
        def __init__(self, model_name: str, **kwargs):
            captured["model_name"] = model_name

        def list_models(self):
            return []

    with patch(
        "agent_core.llm.opencode_provider.OpencodeProvider", _FakeProvider
    ), patch(
        "agent_core.config.load_agent_settings",
        side_effect=RuntimeError("settings broken"),
    ):
        result = cmd._zen_free_catalog()

    assert result == []
    assert captured["model_name"] == "opencode-zen/free"


# --- DEFAULT_OPENCODE_MODEL reads env --------------------------------------


def test_default_opencode_model_reads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEFAULT_OPENCODE_MODEL reflects AGENT_OPENCODE_MODEL from .env."""
    monkeypatch.setenv("AGENT_OPENCODE_MODEL", "opencode-go/custom-model")
    # Re-import to pick up the env change at module level.
    import importlib

    import agent_core.constants as constants_mod

    importlib.reload(constants_mod)
    assert constants_mod.DEFAULT_OPENCODE_MODEL == "opencode-go/custom-model"
    # Restore: reload without the env var to get the real default back.
    monkeypatch.delenv("AGENT_OPENCODE_MODEL", raising=False)
    importlib.reload(constants_mod)
    assert constants_mod.DEFAULT_OPENCODE_MODEL == DEFAULT_OPENCODE_MODEL
