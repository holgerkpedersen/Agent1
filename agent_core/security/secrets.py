"""Secret manager — OS keyring with an encrypted-file fallback (plan OPS item 3).

Replaces plaintext ``.env`` API keys with a tiered store:

1. **OS keyring** — used when the ``keyring`` package is installed (backend
   independent, e.g. Windows Credential Manager / macOS Keychain).
2. **Encrypted file** — Fernet-encrypted ``secrets.enc`` under the agent data
   dir (``%APPDATA%/agent1`` on Windows, ``~/.agent1`` elsewhere), keyed by a
   machine-scoped ``secrets.key``.  Requires the ``cryptography`` package.

Resolution order used by callers (``config.py``): environment variable →
``.env`` value → this store.  ``set_secret`` writes ONLY to the secure store —
plaintext is never written back to ``.env``.

The store directory can be redirected with ``AGENT1_SECRETS_DIR`` (used by
tests and portable setups).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Env var redirecting the secret store + key location (tests, portability).
_ENV_OVERRIDE = "AGENT1_SECRETS_DIR"


def _store_dir() -> Path:
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        return Path(override)
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "agent1"
    return Path.home() / ".agent1"


def _store_path() -> Path:
    return _store_dir() / "secrets.enc"


def _key_path() -> Path:
    return _store_dir() / "secrets.key"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _keyring_backend() -> Any | None:
    """Return a callable keyring module, or None when not installed."""
    try:
        import keyring  # type: ignore[import-not-found]
        return keyring
    except ImportError:
        return None


def _crypto_backend() -> Any | None:
    """Return ``cryptography.fernet.Fernet``, or None when not installed."""
    try:
        from cryptography.fernet import Fernet  # type: ignore[import-not-found]
        return Fernet
    except ImportError:
        return None


def _load_or_create_key() -> bytes:
    """Load the machine-scoped Fernet key, creating it on first use."""
    from cryptography.fernet import Fernet  # type: ignore[import-not-found]

    key_path = _key_path()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if key_path.exists():
        key = key_path.read_bytes()
        if len(key) == 44:
            return key
        logger.warning("Stale secrets.key at %s — regenerating", key_path)
    key = Fernet.generate_key()
    key_path.write_bytes(key)
    logger.info("Created machine-scoped secrets key at %s", key_path)
    return key


def _encrypted_store() -> dict[str, str]:
    """Read the encrypted secrets file (empty dict when absent/unreadable)."""
    fernet_cls = _crypto_backend()
    store_path = _store_path()
    if fernet_cls is None or not store_path.exists():
        return {}
    try:
        fernet = fernet_cls(_load_or_create_key())
        payload = fernet.decrypt(store_path.read_bytes()).decode("utf-8")
        data = json.loads(payload)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001 — corrupt store must not crash callers
        logger.warning("Could not decrypt secret store %s: %s", store_path, exc)
        return {}


def _write_encrypted_store(data: dict[str, str]) -> None:
    """Persist *data* to the encrypted file."""
    fernet_cls = _crypto_backend()
    if fernet_cls is None:
        raise RuntimeError(
            "No secret backend available: install 'keyring' or 'cryptography'"
        )
    fernet = fernet_cls(_load_or_create_key())
    store_path = _store_path()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    token = fernet.encrypt(json.dumps(data).encode("utf-8"))
    store_path.write_bytes(token)
    try:
        os.chmod(store_path, 0o600)
    except OSError:
        pass  # Windows ACLs — best effort


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def has_secret(name: str) -> bool:
    """True when *name* exists in the secure store (keyring or encrypted file)."""
    if not name:
        return False
    keyring = _keyring_backend()
    if keyring is not None:
        try:
            return keyring.get_password("agent1", name) is not None
        except Exception as exc:  # noqa: BLE001
            logger.debug("keyring lookup failed for %s: %s", name, exc)
    return name in _encrypted_store()


def get_secret(name: str, default: str = "") -> str:
    """Return the stored value for *name*, or *default* when absent.

    Keyring is consulted first; the encrypted file is the fallback.  The
    value is never logged.
    """
    if not name:
        return default
    keyring = _keyring_backend()
    if keyring is not None:
        try:
            value = keyring.get_password("agent1", name)
            if value:
                return str(value)
        except Exception as exc:  # noqa: BLE001
            logger.debug("keyring lookup failed for %s: %s", name, exc)
    return _encrypted_store().get(name, default)


def set_secret(name: str, value: str) -> None:
    """Store *value* for *name* in the secure store — never plaintext.

    Raises RuntimeError when no backend is available.
    """
    if not name:
        raise ValueError("secret name must not be empty")
    keyring = _keyring_backend()
    if keyring is not None:
        try:
            keyring.set_password("agent1", name, value)
            logger.debug("Stored secret %s via OS keyring", name)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("keyring write failed for %s (%s) — falling back to encrypted file", name, exc)
    data = _encrypted_store()
    data[name] = value
    _write_encrypted_store(data)
    logger.debug("Stored secret %s in encrypted file", name)


def delete_secret(name: str) -> bool:
    """Remove *name* from the store.  Returns True when it existed."""
    if not name:
        return False
    keyring = _keyring_backend()
    removed = False
    if keyring is not None:
        try:
            removed = keyring.delete_password("agent1", name)
        except Exception:  # noqa: BLE001 — not present or backend limitation
            removed = False
    data = _encrypted_store()
    if name in data:
        del data[name]
        _write_encrypted_store(data)
        removed = True
    return removed


# ---------------------------------------------------------------------------
# config.py integration
# ---------------------------------------------------------------------------

def resolve_secret(name: str, default: str = "") -> str:
    """Environment-first resolution used by ``config.py``:

    ``os.environ[name]`` (or the merged ``.env`` value) wins; the secure
    store is the fallback.  This keeps existing setups working while making
    plaintext ``.env`` optional.
    """
    value = os.environ.get(name)
    if value:
        return value
    try:
        return get_secret(name, default)
    except Exception as exc:  # noqa: BLE001 — never break settings load
        logger.debug("Secret resolution failed for %s: %s", name, exc)
        return default
