"""Supermemory-backed :class:`~agent_core.memory.types.StorageBackend`.

This is a network storage backend for :class:`MemoryStore`, injected through
the existing ``StorageBackend`` seam (``store`` / ``retrieve`` / ``delete``)
so no new abstraction is needed::

    MemoryStore(backend=SupermemoryStorage(fallback=SQLiteStorage(...)))

API VERSION — READ THIS BEFORE CHANGING ANY ROUTE
-------------------------------------------------
Every route here is the **v5 (namespaced) contract**, verified against the
provider's published OpenAPI document (``GET /openapi.json``, ``version
5.0.0``, 22 paths).  The v5 spec contains **no ``/v3`` and no ``/v4``
routes**; the ``/v3`` paths that used to live in this module are legacy
aliases, and at least one of them is gone: ``GET /v3/documents`` answers
``404 No route matches``.  A key alone is not sufficient on those legacy
routes either — they demand ``userId``/``orgId`` binding
(``"Either userId or orgId not found"``), whereas in v5 the namespace in the
path *is* the identity.

Because the unit suite mocks the transport, it asserts our assumed contract
and can never see a vendor-side change.  ``tests/test_supermemory_live.py``
is the check that talks to the real service; it is skipped unless
``SUPERMEMORY_API_KEY`` is set.

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

#: Namespace pattern the v5 spec enforces on the path parameter:
#: ``^[a-zA-Z0-9_:-]+$``, at most 100 characters.  Enforced locally so a bad
#: namespace fails into the fallback instead of costing a round trip.
_NAMESPACE_MAX = 100
_NAMESPACE_ALLOWED = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:-"
)

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
        namespace: str = "default",
        container_tag: str = "default",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        http: Optional[Callable[..., Any]] = None,
        fallback: Optional[Any] = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get(API_KEY_ENV, "")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        # v5 puts the identity in the path (/ns/{namespace}/...), which is
        # what replaced the v3/v4 ``containerTag`` request field.  The old
        # keyword is accepted and mapped so existing callers keep working.
        # ``namespace`` is the v5 name and wins; the default is not a value,
        # so an explicit ``container_tag`` is never shadowed by it.
        use_namespace = namespace and namespace != "default"
        chosen = namespace if use_namespace else (container_tag or "default")
        self._namespace = _safe_namespace(chosen)
        self._timeout = timeout
        self._http = http
        self._fallback = fallback
        #: True only when a key is present.  Disabled is a normal state, not
        #: an error — the caller gets local-only storage and the run proceeds.
        self.enabled = bool(self._api_key)
        self._remote_failures = 0

    @property
    def namespace(self) -> str:
        """The v5 namespace every route is scoped to."""
        return self._namespace

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
        """Write to the remote store, then locally.  Raises nothing, ever.

        v5 requires ``content`` (string); ``id`` is the caller-defined
        document id, which is how a :class:`MemoryStore` key is recoverable
        with ``GET /ns/{namespace}/document/{id}``.
        """
        payload: Dict[str, Any] = {
            "content": json.dumps(data, default=str),
            "id": key,
            "metadata": {"key": key},
            "taskType": "memory",
        }
        self._call("POST", f"/ns/{self._namespace}/document", json=payload)
        # The local write is unconditional: a failed remote call must not
        # leave the local copy behind the remote.
        self._fallback_call("store", key, data)

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Read from the remote store; on miss/failure use the local copy.

        Uses ``GET /ns/{namespace}/document/{id}`` — the only read-by-key
        verb v5 offers.  The legacy ``GET /v3/documents`` filter form is
        gone (it answers 404), so a query-string read cannot be used.  A
        list-shaped body is tolerated so a server that wraps its record in
        an array still round-trips.
        """
        found = self._call("GET", f"/ns/{self._namespace}/document/{key}")
        if isinstance(found, list):
            found = found[0] if found and isinstance(found[0], dict) else None
        data = _decode(found)
        if data is not None:
            return data
        local = self._fallback_call("retrieve", key)
        return local if isinstance(local, dict) else None

    def delete(self, key: str) -> bool:
        """Delete remotely (best effort) and locally.  Returns the local result.

        v5 deletes by id list, not by path segment.
        """
        self._call(
            "DELETE",
            f"/ns/{self._namespace}/document",
            json={"ids": [key]},
        )
        result = self._fallback_call("delete", key)
        return bool(result)

    # ------------------------------------------------------- v5 read surfaces

    def search(self, query: str, *, limit: int = 5) -> Optional[Dict[str, Any]]:
        """Semantic search over this namespace.  ``query`` is the v5 field.

        Returns the parsed body (``{"results": [...], "searchTime": ...}``)
        or ``None`` when the remote had nothing to say.
        """
        return self._call(
            "POST",
            f"/ns/{self._namespace}/search",
            json={"query": query, "limit": limit, "searchMode": "hybrid"},
        )

    def profile(self) -> Optional[Dict[str, Any]]:
        """The namespace profile (``{"profile": {static, dynamic, buckets}}``)."""
        return self._call("POST", f"/ns/{self._namespace}/profile", json={})

    def list_documents(self, *, limit: int = 100) -> Optional[Dict[str, Any]]:
        """List documents in this namespace (``POST /ns/{ns}/list/documents``)."""
        return self._call(
            "POST",
            f"/ns/{self._namespace}/list/documents",
            json={"include": {"content": True}},
            params={"limit": limit},
        )


def _safe_namespace(value: str) -> str:
    """Coerce ``value`` into the v5 namespace pattern.

    The spec enforces ``^[a-zA-Z0-9_:-]+$`` with a 100-character cap.  A key
    is caller-supplied and can contain anything, so characters outside the
    pattern are replaced rather than sent; this keeps a namespace from
    becoming an injection point in the request path.
    """
    cleaned = "".join(
        ch if ch in _NAMESPACE_ALLOWED else "-" for ch in (value or "")
    )
    return (cleaned or "default")[:_NAMESPACE_MAX]


def _decode(record: Any) -> Optional[Dict[str, Any]]:
    """Turn a remote record into the dict the caller expects.

    The remote stores what :meth:`store` wrote (a JSON string under
    ``content``), so a round-trip must recover it; a record that is not
    decodable is reported as a miss so the fallback gets the chance.
    """
    if not isinstance(record, dict):
        return None
    content = record.get("content")
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            return {"content": content}
        return parsed if isinstance(parsed, dict) else {"content": parsed}
    return record


__all__: List[str] = [
    "API_KEY_ENV",
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT_SECONDS",
    "SupermemoryStorage",
]
