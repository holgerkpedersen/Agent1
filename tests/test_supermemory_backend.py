"""SupermemoryStorage: a StorageBackend that fails OPEN.

The rule these tests pin down: a missing API key, a timeout, a 4xx/5xx or a
malformed response must NEVER propagate out of store/retrieve/delete.  The
turn path must keep running on the local fallback, exactly like the provider
pattern in openrouter_provider.py (readable string, no raise) and
opencode_provider.py (no key => feature switches itself off).

TestContract additionally pins the ROUTES to the provider's published v5
OpenAPI contract.  These assertions are what the fail-open tests cannot
provide: they mock the transport, so a route that no longer exists on the
vendor side still passes them.  tests/test_supermemory_live.py is the check
against the real service.
"""

from typing import Any, Dict, List, Optional

import pytest

from agent_core.memory.supermemory import SupermemoryStorage


class FakeFallback:
    """Local SQLiteStorage stand-in that records every fallback call."""

    def __init__(self) -> None:
        self.store_calls: List[tuple] = []
        self.retrieve_calls: List[str] = []
        self.delete_calls: List[str] = []
        self._data: Dict[str, Any] = {}

    def store(self, key: str, data: Dict[str, Any]) -> None:
        self.store_calls.append((key, data))
        self._data[key] = data

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        self.retrieve_calls.append(key)
        return self._data.get(key)

    def delete(self, key: str) -> bool:
        self.delete_calls.append(key)
        return self._data.pop(key, None) is not None


class FakeHttp:
    """Stand-in for the injected HTTP callable.  Raises if ``error`` is set."""

    def __init__(
        self,
        responses: Optional[Dict[str, Any]] = None,
        error: Exception = None,
    ):
        self.responses = responses or {}
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, method: str, path: str, **kw: Any) -> Any:
        self.calls.append({"method": method, "path": path, **kw})
        if self.error is not None:
            raise self.error
        return self.responses.get(path, self.responses.get("*"))


def make(
    http: Optional[FakeHttp] = None,
    key: str = "k",
    fallback=None,
) -> SupermemoryStorage:
    return SupermemoryStorage(
        api_key=key,
        http=http,
        fallback=fallback if fallback is not None else FakeFallback(),
    )


class TestDisabledWithoutKey:
    def test_no_key_is_disabled_not_fatal(self):
        fb = FakeFallback()
        store = SupermemoryStorage(api_key="", fallback=fb)
        assert store.enabled is False

    def test_store_falls_back_locally(self):
        fb = FakeFallback()
        store = SupermemoryStorage(api_key="", fallback=fb)
        store.store("k1", {"a": 1})
        assert fb.store_calls == [("k1", {"a": 1})]

    def test_retrieve_falls_back_locally(self):
        fb = FakeFallback()
        fb.store("k1", {"a": 1})
        store = SupermemoryStorage(api_key="", fallback=fb)
        assert store.retrieve("k1") == {"a": 1}

    def test_delete_falls_back_locally(self):
        fb = FakeFallback()
        fb.store("k1", {"a": 1})
        store = SupermemoryStorage(api_key="", fallback=fb)
        assert store.delete("k1") is True
        assert fb.delete_calls == ["k1"]

    def test_env_key_is_read_not_hardcoded(self, monkeypatch):
        monkeypatch.setenv("SUPERMEMORY_API_KEY", "from-env")
        store = SupermemoryStorage(fallback=FakeFallback())
        assert store.enabled is True


class TestFailOpenOnRemoteErrors:
    @pytest.mark.parametrize(
        "exc",
        [
            TimeoutError("read timed out"),
            OSError("no route to host"),
            ValueError("bad json"),
        ],
    )
    def test_store_swallows_and_falls_back(self, exc):
        fb = FakeFallback()
        http = FakeHttp(error=exc)
        store = make(http, fallback=fb)
        store.store("k1", {"a": 1})  # must not raise
        assert fb.store_calls == [("k1", {"a": 1})]

    @pytest.mark.parametrize("exc", [TimeoutError("t"), OSError("x")])
    def test_retrieve_swallows_and_falls_back(self, exc):
        fb = FakeFallback()
        fb.store("k1", {"a": 1})
        http = FakeHttp(error=exc)
        store = make(http, fallback=fb)
        assert store.retrieve("k1") == {"a": 1}

    @pytest.mark.parametrize("exc", [TimeoutError("t"), OSError("x")])
    def test_delete_swallows_and_falls_back(self, exc):
        fb = FakeFallback()
        http = FakeHttp(error=exc)
        store = make(http, fallback=fb)
        assert store.delete("k1") is False  # local had nothing; no raise

    def test_remote_write_failure_keeps_local_copy(self):
        fb = FakeFallback()
        http = FakeHttp(error=OSError("boom"))
        store = make(http, fallback=fb)
        store.store("k1", {"a": 1})
        assert store.retrieve("k1") == {"a": 1}


class TestHappyPath:
    def test_store_posts_and_writes_fallback(self):
        fb = FakeFallback()
        http = FakeHttp(responses={"*": {"id": "remote-1"}})
        store = make(http, fallback=fb)
        store.store("k1", {"a": 1})
        assert any(c["method"].upper() == "POST" for c in http.calls)
        assert fb.store_calls == [("k1", {"a": 1})]

    def test_retrieve_maps_remote_payload(self):
        fb = FakeFallback()
        http = FakeHttp(responses={"*": [{"key": "k1", "content": "hello"}]})
        store = make(http, fallback=fb)
        out = store.retrieve("k1")
        assert out is not None
        assert out.get("content") == "hello"

    def test_retrieve_miss_falls_back(self):
        fb = FakeFallback()
        fb.store("k1", {"a": 1})
        http = FakeHttp(responses={"*": []})
        store = make(http, fallback=fb)
        assert store.retrieve("k1") == {"a": 1}

    def test_delete_hits_remote_then_local(self):
        fb = FakeFallback()
        http = FakeHttp(responses={"*": {}})
        store = make(http, fallback=fb)
        store.delete("k1")
        assert any("k1" in str(c) for c in http.calls)
        assert fb.delete_calls == ["k1"]

    def test_send_timeout_is_bounded(self):
        http = FakeHttp(responses={"*": {}})
        store = make(http)
        store.store("k1", {"a": 1})
        for call in http.calls:
            timeout = call.get("timeout")
            assert timeout is not None
            assert 0 < timeout <= 10


class TestContract:
    def test_satisfies_storage_backend_protocol(self):
        from agent_core.memory import StorageBackend  # noqa: F401

        store = SupermemoryStorage(api_key="", fallback=FakeFallback())
        for name in ("store", "retrieve", "delete"):
            assert callable(getattr(store, name))

    def test_memory_store_accepts_it_as_backend(self):
        from agent_core.memory import MemoryStore

        fb = FakeFallback()
        mem = MemoryStore(backend=SupermemoryStorage(api_key="", fallback=fb))
        mem.save_memory("k1", {"a": 1})
        assert mem.load_memory("k1") == {"a": 1}
        assert mem.forget_memory("k1") is True


class TestV5Routes:
    """The routes, verbs and bodies must match the published v5 OpenAPI spec.

    The v5 spec has NO /v3 and NO /v4 paths.  ``GET /v3/documents`` is gone
    (it answers ``404 No route matches``), so a read-by-key must use
    ``GET /ns/{namespace}/document/{id}``.  These tests exist because the
    fail-open suite passed against routes the provider no longer serves.
    """

    def _calls(self, http: FakeHttp) -> List[Dict[str, Any]]:
        return http.calls

    def test_store_writes_to_namespaced_document_route(self):
        http = FakeHttp(responses={"*": {"id": "remote-1", "status": "ok"}})
        store = make(http)
        store.store("k1", {"a": 1})
        posts = [c for c in self._calls(http) if c["method"] == "POST"]
        assert len(posts) == 1
        assert posts[0]["path"] == "/ns/default/document"
        body = posts[0]["json"]
        # v5 requires "content" (string); "id" is the caller-defined document
        # id that makes the key recoverable on read.
        assert "content" in body and isinstance(body["content"], str)
        assert body["id"] == "k1"
        assert "containerTag" not in body, "containerTag is v3/v4, not v5"

    def test_retrieve_reads_by_id_not_by_query_filter(self):
        http = FakeHttp(responses={"*": {"id": "k1", "content": '{"a": 1}'}})
        store = make(http)
        assert store.retrieve("k1") == {"a": 1}
        gets = [c for c in self._calls(http) if c["method"] == "GET"]
        assert len(gets) == 1
        assert gets[0]["path"] == "/ns/default/document/k1"
        assert "params" not in gets[0], "v5 has no read-by-filter verb"

    def test_delete_sends_ids_array(self):
        http = FakeHttp(responses={"*": {}})
        store = make(http)
        store.delete("k1")
        dels = [c for c in self._calls(http) if c["method"] == "DELETE"]
        assert len(dels) == 1
        assert dels[0]["path"] == "/ns/default/document"
        assert dels[0]["json"] == {"ids": ["k1"]}

    def test_search_uses_query_field(self):
        http = FakeHttp(responses={"*": {"results": [], "searchTime": 1}})
        store = make(http)
        store.search("hello", limit=3)
        posts = [c for c in self._calls(http) if c["path"].endswith("/search")]
        assert len(posts) == 1
        assert posts[0]["method"] == "POST"
        assert posts[0]["json"]["query"] == "hello"
        assert posts[0]["json"]["limit"] == 3

    def test_profile_uses_namespaced_post(self):
        http = FakeHttp(
            responses={"*": {"profile": {"static": [], "dynamic": [], "buckets": {}}}}
        )
        store = make(http)
        out = store.profile()
        posts = [c for c in self._calls(http) if c["path"].endswith("/profile")]
        assert posts[0]["method"] == "POST"
        assert posts[0]["path"] == "/ns/default/profile"
        assert out == {"profile": {"static": [], "dynamic": [], "buckets": {}}}

    def test_no_route_in_the_module_is_a_v3_or_v4_path(self):
        """Guard against re-introducing a legacy alias."""
        import inspect
        import re

        src = inspect.getsource(
            __import__("agent_core.memory.supermemory", fromlist=["x"])
        )
        offenders = re.findall(r'["\']/v[34]/', src)
        assert offenders == [], f"legacy routes in module source: {offenders}"

    def test_namespace_replaces_path_unsafe_characters(self):
        """A caller-supplied namespace cannot inject the request path."""
        http = FakeHttp(responses={"*": {"id": "x", "status": "ok"}})
        store = SupermemoryStorage(
            api_key="k", http=http, namespace="../../admin", fallback=FakeFallback()
        )
        store.store("k1", {"a": 1})
        path = self._calls(http)[0]["path"]
        assert ".." not in path
        assert "/" not in path.removeprefix("/ns/").split("/document")[0]
        assert path == f"/ns/{store.namespace}/document"

    def test_namespace_is_capped_at_the_spec_maximum(self):
        store = SupermemoryStorage(api_key="k", namespace="x" * 500)
        assert len(store.namespace) <= 100

    def test_legacy_container_tag_keyword_still_maps(self):
        """Callers written against the old keyword keep working."""
        store = SupermemoryStorage(api_key="k", container_tag="tenant-7")
        assert store.namespace == "tenant-7"


class TestDecodeRoundTrip:
    def test_non_dict_remote_response_is_a_miss(self):
        fb = FakeFallback()
        fb.store("k1", {"a": 1})
        http = FakeHttp(responses={"*": None})
        store = make(http, fallback=fb)
        assert store.retrieve("k1") == {"a": 1}

    def test_non_json_content_is_returned_as_content(self):
        http = FakeHttp(responses={"*": {"id": "k1", "content": "plain text"}})
        store = make(http)
        assert store.retrieve("k1") == {"content": "plain text"}

    def test_json_scalar_content_is_wrapped(self):
        http = FakeHttp(responses={"*": {"id": "k1", "content": "42"}})
        store = make(http)
        assert store.retrieve("k1") == {"content": 42}
