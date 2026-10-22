"""Supermemory-backed :class:`~agent_core.memory.types.StorageBackend`.

This is a network storage backend for :class:`MemoryStore`, injected through
the existing ``StorageBackend`` seam (``store`` / ``retrieve`` / ``delete``)
so no new abstraction is needed::

    MemoryStore(backend=SupermemoryStorage(fallback=SQLiteStorage(...)))

THE ONE RULE THIS MODULE OBEYS
------------------------------
It NEVER raises.  A memory write/read is a support action on the turn path,
not the point of the turn; aborting a run because a SaaS endpoint is down,
the key is unset, or the socket timed out is the same class of defect as the
context-compaction crash that killed interactive runs with
``RuntimeError: prompt cannot fit the context budget``.  Every remote failure
is swallowed, logged once, and served from the local ``fallback`` backend.

That mirrors the provider pattern already in this repo:
``openrouter_provider.py`` returns a readable error string instead of raising
when ``OPENROUTER_API_KEY`` is unset, and ``opencode_provider.py`` derives
``api_mode = bool(self.api_key)`` so a missing key switches the feature off
rather than breaking startup.

Consequences of the rule, stated plainly:

* an unset key means ``enabled is False`` and every call is local-only;
* a remote write that fails still writes locally, so the local copy is never
  behind the remote by a failed call;
* a remote read that fails (or misses) is answered from the local copy;
* the HTTP call is bounded by a timeout (``httpx`` timeouts come from
  ``.env`` per SECURITY.md) — an unbounded request on a per-turn hot path is
  how an agent hangs.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Env var the key resolves from.  Never hardcoded, never committed.
API_KEY_ENV = "SUPERMEMORY_API_KEY"

#: Base URL of the REST API; overridable for self-hosting and tests.
DEFAULT_BASE_URL = "https://api.supermemory.ai"

#: Bounded, per SECURITY.md's configurable-timeout rule.  Memory calls sit on
#: the per-turn path, so this stays well under a human-patience threshold.
DEFAULT_TIMEOUT_SECONDS = 5.0


class SupermemoryStorage:
    """Remote memory storage with a mandatory local fallback.

    ``http`` is injectable so the fail-open behaviour is testable without
    network access; ``fallback`` is the backend that answers whenever the
    remote side is unavailable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        container_tag: str = "default",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        http: Optional[Callable[..., Any]] = None,
        fallback: Optional[Any] = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get(API_KEY_ENV, "")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._container_tag = container_tag
        self._timeout = timeout
        self._http = http
        self._fallback = fallback
        #: True only when a key is present.  Disabled is a normal state, not
        #: an error — the caller gets local-only storage and the run proceeds.
        self.enabled = bool(self._api_key)
        self._remote_failures = 0

    # ------------------------------------------------------------------ plumbing

    @property
    def remote_failures(self) -> int:
        """Count of swallowed remote failures (observability, never fatal)."""
        return self._remote_failures

    def _default_http(self) -> Callable[..., Any]:
        import httpx

        def call(method: str, path: str, **kw: Any) -> Any:
            url = f"{self._base_url}{path}"
            headers = kw.pop("headers", {})
            headers.setdefault("Authorization", f"Bearer {self._api_key}")
            headers.setdefault("Content-Type", "application/json")
            response = httpx.request(
                method, url, headers=headers, timeout=self._timeout, **kw
            )
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()

        return call

    def _call(self, method: str, path: str, **kw: Any) -> Any:
        """One remote call, bounded and never raising.

        Returns ``None`` on ANY failure — missing key, transport error,
        non-2xx, undecodable body.  Callers treat ``None`` as "the remote had
        nothing to say" and use the fallback.
        """
        if not self.enabled:
            return None
        http = self._http or self._default_http()
        try:
            return http(method, path, timeout=self._timeout, **kw)
        except Exception as exc:  # noqa: BLE001 - fail-open is the contract
            self._remote_failures += 1
            logger.debug(
                "supermemory %s %s failed (%s); using local fallback",
                method,
                path,
                exc,
            )
            return None

    def _fallback_call(self, name: str, *args: Any) -> Any:
        """Delegate to the local backend.  A missing fallback is tolerated."""
        backend = self._fallback
        if backend is None:
            return None
        try:
            return getattr(backend, name)(*args)
        except Exception as exc:  # noqa: BLE001 - fail-open is the contract
            logger.debug("supermemory fallback %s failed: %s", name, exc)
            return None

    # ---------------------------------------------------- StorageBackend seam

    def store(self, key: str, data: Dict[str, Any]) -> None:
        """Write to the remote store, then locally.  Raises nothing, ever."""
        payload: Dict[str, Any] = {
            "key": key,
            "content": json.dumps(data, default=str),
            "containerTag": self._container_tag,
        }
        self._call("POST", "/v3/documents", json=payload)
        # The local write is unconditional: a failed remote call must not
        # leave the local copy behind the remote.
        self._fallback_call("store", key, data)

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Read from the remote store; on miss/failure use the local copy."""
        found = self._call(
            "GET",
            "/v3/documents",
            params={"key": key, "containerTag": self._container_tag},
        )
        record = _first_record(found)
        if record is not None:
            data = _decode(record)
            if data is not None:
                return data
        local = self._fallback_call("retrieve", key)
        return local if isinstance(local, dict) else None

    def delete(self, key: str) -> bool:
        """Delete remotely (best effort) and locally.  Returns the local result."""
        self._call(
            "DELETE",
            f"/v3/documents/{key}",
            params={"containerTag": self._container_tag},
        )
        result = self._fallback_call("delete", key)
        return bool(result)


def _first_record(found: Any) -> Optional[Dict[str, Any]]:
    """Pull the first document out of a response of unknown shape."""
    if isinstance(found, dict):
        items = found.get("documents") or found.get("results") or found.get("data")
        if isinstance(items, list) and items:
            first = items[0]
            return first if isinstance(first, dict) else None
        return found if found else None
    if isinstance(found, list) and found:
        first = found[0]
        return first if isinstance(first, dict) else None
    return None


def _decode(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Turn a remote record into the dict the caller expects.

    The remote stores what :meth:`store` wrote (a JSON string under
    ``content``), so a round-trip must recover it; a record that is not
    decodable is reported as a miss so the fallback gets the chance.
    """
    content = record.get("content")
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            return {"content": content}
        return parsed if isinstance(parsed, dict) else {"content": parsed}
    if isinstance(record, dict):
        return record
    return None


__all__: List[str] = [
    "API_KEY_ENV",
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT_SECONDS",
    "SupermemoryStorage",
]
