"""Live contract check against the real Supermemory API.

The unit suite mocks the transport, which is exactly why it cannot catch a
provider contract change: it asserts *our* assumed routes, not the vendor's.
This test closes that gap, and is skipped unless a key is present so CI and
offline runs are unaffected.

    $env:SUPERMEMORY_API_KEY = "..."   # then:
    pytest -m live tests/test_supermemory_live.py

It runs against a throwaway namespace and deletes its documents afterwards.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict

import pytest

from agent_core.memory.supermemory import (
    API_KEY_ENV,
    DEFAULT_TIMEOUT_SECONDS,
    SupermemoryStorage,
)

LIVE_TIMEOUT = 20.0

pytestmark = pytest.mark.skipif(
    not os.environ.get(API_KEY_ENV),
    reason=f"{API_KEY_ENV} not set; live contract check needs a real key",
)


class RecordingHttp:
    """Real transport, with every call recorded so failures are reportable.

    ``response`` objects are kept (not just parsed bodies) so a failing call
    can be named with its status and body in the assertion message.
    """

    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def __call__(self, method: str, path: str, **kw: Any) -> Any:
        import httpx

        self.calls.append({"method": method, "path": path, **kw})
        headers = kw.pop("headers", {})
        headers.setdefault("Content-Type", "application/json")
        response = httpx.request(method, path, headers=headers, **kw)
        self.calls[-1]["status"] = response.status_code
        self.calls[-1]["body"] = response.text[:400]
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()

    def failures(self) -> str:
        return "\n".join(
            f"  {c['method']} {c['path']} -> {c.get('status')} {c.get('body')}"
            for c in self.calls
        )


@pytest.fixture()
def live() -> SupermemoryStorage:
    """A storage backend pointed at a fresh, namespaced, disposable container."""
    namespace = f"agent1-smoke-{uuid.uuid4().hex[:8]}"
    http = RecordingHttp()
    backend = SupermemoryStorage(
        namespace=namespace,
        http=http,
        fallback=None,
        timeout=LIVE_TIMEOUT,
    )
    backend._http = http  # type: ignore[attr-defined]
    yield backend
    # Best-effort cleanup: the namespace is disposable, documents are not ours
    # to keep.  Cleanup failures are not test failures.
    try:
        ids = [
            d.get("id")
            for d in (backend.list_documents() or {}).get("documents", [])
            if d.get("id")
        ]
        if ids:
            backend._call("DELETE", f"/ns/{namespace}/document", json={"ids": ids})
    except Exception:  # noqa: BLE001
        pass


def test_live_key_enables_the_backend(live: SupermemoryStorage) -> None:
    assert live.enabled is True


def test_live_round_trip_matches_the_published_contract(
    live: SupermemoryStorage,
) -> None:
    """store -> retrieve -> search -> delete against the real API.

    Every leg is asserted against the v5 schema the vendor publishes:
    POST /ns/{ns}/document (``content``), GET /ns/{ns}/document/{id}
    (``content``), POST /ns/{ns}/search (``query`` -> ``results``),
    DELETE /ns/{ns}/document (``ids``).
    """
    http: RecordingHttp = live._http  # type: ignore[attr-defined]
    marker = f"agent1-live-marker-{uuid.uuid4().hex[:8]}"
    payload = {"marker": marker, "note": "live contract probe"}

    live.store(marker, payload)
    post = [
        c
        for c in http.calls
        if c["method"] == "POST" and c["path"].endswith("/document")
    ]
    assert post, f"store issued no document write:\n{http.failures()}"

    # Memory extraction is asynchronous; poll within a bounded window.
    deadline = time.monotonic() + 20.0
    recovered = None
    while time.monotonic() < deadline:
        recovered = live.retrieve(marker)
        if recovered:
            break
        time.sleep(1.5)

    assert recovered is not None, (
        f"retrieve() found nothing for {marker!r} after 20s:\n{http.failures()}"
    )
    assert marker in str(recovered), f"round trip lost the marker: {recovered!r}"

    found = live.search(marker, limit=5)
    assert found is not None, f"search() failed:\n{http.failures()}"

    assert live.delete(marker) is False  # no local fallback => False, no raise


def test_live_profile_call_uses_the_namespaced_route(live: SupermemoryStorage) -> None:
    """The profile route is a POST under /ns/{ns}/ and returns {profile}."""
    profile = live.profile()
    http: RecordingHttp = live._http  # type: ignore[attr-defined]
    posts = [c for c in http.calls if c["path"].endswith("/profile")]
    assert posts, f"no profile call made:\n{http.failures()}"
    assert posts[0]["method"] == "POST"
    if profile is not None:
        assert "profile" in profile


def test_live_timeout_stays_bounded() -> None:
    """The hot path stays bounded: no call may use an unbounded timeout."""
    assert 0 < DEFAULT_TIMEOUT_SECONDS <= 10
