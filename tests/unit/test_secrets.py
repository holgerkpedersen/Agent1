"""Secret-manager tests (plan OPS items 3-4): encrypted-file backend with a
redirectable store dir, env-first resolution, and config.py integration.

The secure store requires an optional backend (``keyring`` or
``cryptography`` — ``pip install -e .[secrets]``); tests that write secrets
skip with a clear reason when neither is importable instead of erroring.
"""
import os

import pytest

from agent_core.security import secrets

try:
    from cryptography.fernet import Fernet  # noqa: F401

    _HAS_BACKEND = True
except ImportError:
    _HAS_BACKEND = False

needs_backend = pytest.mark.skipif(
    not _HAS_BACKEND,
    reason="no secret backend — install 'keyring' or 'cryptography' "
           "(pip install -e .[secrets])",
)


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT1_SECRETS_DIR", str(tmp_path))
    yield tmp_path


class TestSecretStore:
    @needs_backend
    def test_set_and_get_roundtrip(self, store_dir):
        secrets.set_secret("OPENCODE_API_KEY", "sk-secret-value")
        assert secrets.get_secret("OPENCODE_API_KEY") == "sk-secret-value"
        assert secrets.has_secret("OPENCODE_API_KEY") is True

    def test_get_missing_returns_default(self, store_dir):
        assert secrets.get_secret("NOPE") == ""
        assert secrets.get_secret("NOPE", "fallback") == "fallback"
        assert secrets.has_secret("NOPE") is False

    @needs_backend
    def test_delete(self, store_dir):
        secrets.set_secret("TMP_KEY", "v")
        assert secrets.delete_secret("TMP_KEY") is True
        assert secrets.get_secret("TMP_KEY") == ""
        assert secrets.delete_secret("TMP_KEY") is False

    @needs_backend
    def test_store_is_encrypted_on_disk(self, store_dir):
        secrets.set_secret("API_KEY", "super-secret")
        raw = (store_dir / "secrets.enc").read_bytes()
        assert b"super-secret" not in raw
        # decrypts back correctly
        assert secrets.get_secret("API_KEY") == "super-secret"

    def test_corrupt_store_returns_empty(self, store_dir, monkeypatch):
        (store_dir / "secrets.enc").write_bytes(b"garbage-not-fernertoken")
        assert secrets.get_secret("ANY") == ""

    def test_empty_name_rejected(self, store_dir):
        with pytest.raises(ValueError):
            secrets.set_secret("", "x")


class TestResolveSecret:
    @needs_backend
    def test_env_wins_over_store(self, store_dir, monkeypatch):
        secrets.set_secret("OPENCODE_API_KEY", "stored")
        monkeypatch.setenv("OPENCODE_API_KEY", "env-wins")
        assert secrets.resolve_secret("OPENCODE_API_KEY") == "env-wins"

    @needs_backend
    def test_store_fallback(self, store_dir, monkeypatch):
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        secrets.set_secret("OPENCODE_API_KEY", "stored")
        assert secrets.resolve_secret("OPENCODE_API_KEY") == "stored"

    def test_default_when_nothing(self, store_dir, monkeypatch):
        monkeypatch.delenv("SOME_KEY", raising=False)
        assert secrets.resolve_secret("SOME_KEY", "dflt") == "dflt"


class TestConfigIntegration:
    def _load_settings(self, monkeypatch):
        # Hermetic: ignore the workspace .env so the real key never leaks in.
        monkeypatch.setattr("agent_core.config._load_env_file", lambda *a, **k: {})
        from agent_core.config import load_agent_settings
        return load_agent_settings()

    @needs_backend
    def test_opencode_api_key_from_store(self, store_dir, monkeypatch):
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        secrets.set_secret("OPENCODE_API_KEY", "store-key")
        settings = self._load_settings(monkeypatch)
        assert settings.opencode_api_key == "store-key"

    @needs_backend
    def test_opencode_api_key_env_wins(self, store_dir, monkeypatch):
        monkeypatch.setenv("OPENCODE_API_KEY", "env-key")
        secrets.set_secret("OPENCODE_API_KEY", "store-key")
        settings = self._load_settings(monkeypatch)
        assert settings.opencode_api_key == "env-key"
