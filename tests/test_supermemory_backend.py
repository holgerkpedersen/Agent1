"""SupermemoryStorage: a StorageBackend that fails OPEN.

The rule these tests pin down: a missing API key, a timeout, a 4xx/5xx or a
malformed response must NEVER propagate out of store/retrieve/delete.  The
turn path must keep running on the local fallback, exactly like the provider
pattern in openrouter_provider.py (readable string, no raise) and
opencode_provider.py (no key => feature switches itself off).
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
